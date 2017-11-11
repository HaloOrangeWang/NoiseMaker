from datainputs.melody import MelodyPatternEncode
from dataoutputs.predictions import MusicPatternPrediction
from models.LstmPipeline import BaseLstmPipeline
from datainputs.fill import FillTrainData, OneSongRelNoteList, MelodyCoreNote
from models.configs import FillConfig
from settings import *
import random


def ImitatePatternDecode(cur_step, imitate_note_diff, init_time_diff, imitate_spd_ratio, melody_rel_note_list, tone, root_note):
    if tone == TONE_MAJOR:
        rel_list = [0, 2, 4, 5, 7, 9, 11]
    else:
        rel_list = [0, 2, 3, 5, 7, 8, 10]
    abs_note_output = [0 for t in range(8)]
    final_time_diff = init_time_diff + 8 * (1 - imitate_spd_ratio)
    imitate_time_diff = init_time_diff
    for note_iterator, note in enumerate(melody_rel_note_list[cur_step - init_time_diff: cur_step + 8 - final_time_diff]):
        if note != 0 and note_iterator + imitate_time_diff == int(note_iterator + imitate_time_diff):  # 这个时间步长的主旋律有音符
            rel_root_notelist = [[t[0] + imitate_note_diff, t[1]] for t in note]
            abs_note_list = [12 * (t[0] // 7) + rel_list[t[0] % 7] + t[1] + root_note for t in rel_root_notelist]
            abs_note_output[note_iterator] = abs_note_list
        imitate_time_diff -= (1 - imitate_spd_ratio)
    return abs_note_output


def GetAbsNoteList(cur_step, rel_note, melody_core_note_list, tone, root_note):
    if tone == TONE_MAJOR:
        rel_list = [0, 2, 4, 5, 7, 9, 11]
    else:
        rel_list = [0, 2, 3, 5, 7, 8, 10]
    rel_root_notelist = [[t[0] + melody_core_note_list[cur_step][0][0], t[1] + melody_core_note_list[cur_step][0][1]] for t in rel_note]
    abs_note_list = [12 * (t[0] // 7) + rel_list[t[0] % 7] + t[1] + root_note for t in rel_root_notelist]
    return abs_note_list


class FillPipeline(BaseLstmPipeline):

    def __init__(self, melody_data, melody_pattern_data, continuous_bar_number_data):
        self.train_data = FillTrainData(melody_data, melody_pattern_data, continuous_bar_number_data)
        self.flatten_spr_list = [[], []]
        for key, value in self.train_data.speed_ratio_dict.items():
            self.flatten_spr_list[0].append(key)
            self.flatten_spr_list[1].append(value)
        super().__init__()

    def prepare(self):
        self.config = FillConfig()  # 训练和弦所用的训练模型配置
        self.test_config = FillConfig()  # 测试生成的和弦所用的配置和训练的配置大体相同，但是batch_size为1
        self.test_config.batch_size = 1
        self.variable_scope_name = 'FillModel'

    def cal_fill_prob(self, melody_output):
        judge_fill_list = []
        # 连续两拍没有主旋律 这两拍加花的概率为70%
        # 当前拍只有一个音符 且下一拍没有主旋律 这两拍加花的概率为60%
        # 当前拍没有主旋律 且上一拍有多个音符 加花的概率为35%
        # 当前拍只有一个音符 且是一个小节的最后一拍 加花的概率为25%
        # 当前拍只有一个音符 且下一拍有主旋律 加花的概率为20%
        # 每个小节的第一拍 加花概率为15%
        # 其余时间 加花的概率为5%
        # 如果前一拍有加花 本拍加花的概率增加10%
        beat_iterator = 0
        while beat_iterator <= len(melody_output) - 16:
            if beat_iterator != 0 and judge_fill_list[-1] == 1:
                prob = 0.1
            else:
                prob = 0
            if melody_output[beat_iterator: beat_iterator + 16] == [0 for t in range(16)]:
                prob += 0.7
                judge_fill_list += [1 if random.random() <= prob else 0] * 2
                beat_iterator += 16
                continue
            elif melody_output[beat_iterator + 1: beat_iterator + 16] == [0 for t in range(15)]:
                prob += 0.6
                judge_fill_list += [1 if random.random() <= prob else 0] * 2
                beat_iterator += 16
                continue
            elif beat_iterator >= 8 and melody_output[beat_iterator: beat_iterator + 8] == [0 for t in range(8)]:
                prob += 0.35
                judge_fill_list.append(1 if random.random() <= prob else 0)
                beat_iterator += 8
                continue
            elif beat_iterator % 32 == 24 and melody_output[beat_iterator + 1: beat_iterator + 8] == [0 for t in range(7)]:
                prob += 0.25
                judge_fill_list.append(1 if random.random() <= prob else 0)
                beat_iterator += 8
                continue
            elif melody_output[beat_iterator + 1: beat_iterator + 8] == [0 for t in range(7)]:
                prob += 0.2
                judge_fill_list.append(1 if random.random() <= prob else 0)
                beat_iterator += 8
                continue
            elif beat_iterator % 32 == 0:
                prob += 0.1
                judge_fill_list.append(1 if random.random() <= prob else 0)
                beat_iterator += 8
                continue
            else:
                prob += 0.05
                judge_fill_list.append(1 if random.random() <= prob else 0)
                beat_iterator += 8
        judge_fill_list += [0]  # 最后一拍不加花
        return judge_fill_list

    def generate(self, session, melody_output, common_melody_patterns, tone):
        # # 1.数据预处理 将melody_output转化为pattern，并获取骨干音列表
        fill_pattern_output = []  # 鼓机输出的pattern形式
        fill_output = []  # 鼓机输出的note list形式
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dict = MelodyPatternEncode(common_melody_patterns, melody_pattern_input, 0.125, 1).music_pattern_dict
        root_note = 72 if tone == TONE_MAJOR else 69
        melody_rel_note_list = OneSongRelNoteList(melody_output, tone, root_note)
        melody_core_note_list = MelodyCoreNote(melody_rel_note_list)
        melody_output_pattern = []
        generate_fail_time = 0  # 被打回重新生成的次数
        for key in range(round(len(melody_output) / 32)):
            melody_output_pattern += melody_pattern_dict[key]
        # # 2.获取哪几拍加花 哪几拍不加花
        judge_fill_list = self.cal_fill_prob(melody_output)
        print('judge_fill_list', judge_fill_list)
        # print('melody_rel_note_list', melody_rel_note_list)
        # print('melody_core_note_list', melody_core_note_list)
        # # 3.逐拍生成数据
        melody_iterator = 0
        while melody_iterator < len(melody_output_pattern):  # 遍历整个主旋律 步长是1拍
            if judge_fill_list[melody_iterator] == 0:
                fill_output += [0 for t in range(8)]
                fill_pattern_output.append(0)
                melody_iterator += 1
                continue
            fill_prediction_input = [melody_iterator % 8] + [0 for t in range(8)]
            # 2.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_FILL_GENERATE_FAIL_TIME:
                print('fill restart ', generate_fail_time)
                return None
            # 2.3.生成输入数据
            if melody_iterator <= 3:
                fill_prediction_input += [0 for t in range(4 - melody_iterator)] + melody_output_pattern[:(melody_iterator + 3)]
            elif melody_iterator >= len(melody_output_pattern) - 2:
                fill_prediction_input += melody_output_pattern[melody_iterator - 4:] + [0 for t in range(melody_iterator - len(melody_output_pattern) + 3)]
            else:
                fill_prediction_input += melody_output_pattern[melody_iterator - 4: melody_iterator + 3]
            if melody_iterator <= 2:
                fill_prediction_input += [0 for t in range(3 - melody_iterator)] + fill_pattern_output[:melody_iterator]
            else:
                fill_prediction_input += fill_pattern_output[melody_iterator - 3: melody_iterator]
            for lookback_iterator in range(melody_iterator - 4, -1, -1):
                if fill_pattern_output[lookback_iterator] != 0:
                    if lookback_iterator <= 2:
                        fill_prediction_input[4 - lookback_iterator: 5] = melody_output_pattern[:lookback_iterator + 1]
                        fill_prediction_input[8 - lookback_iterator: 9] = fill_pattern_output[:lookback_iterator + 1]
                    else:
                        fill_prediction_input[1: 5] = melody_output_pattern[lookback_iterator - 3: lookback_iterator + 1]
                        fill_prediction_input[5: 9] = fill_pattern_output[lookback_iterator - 3: lookback_iterator + 1]
                    break
            # print('fill_prediction_input', melody_iterator, fill_prediction_input)
            # 2.4.生成输出数据
            fill_predict = self.predict(session, [fill_prediction_input])
            fill_out = MusicPatternPrediction(fill_predict, 1, COMMON_FILL_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的鼓点组合drum_out
            fill_pattern_output.append(fill_out)  # 添加到最终的和弦输出列表中
            # print('fill_predict', melody_iterator, list(fill_predict[-1, :]))
            # print('fill_pattern_output', melody_iterator, fill_pattern_output)

            rel_note_list = self.train_data.common_fill_patterns[fill_out]
            abs_note_list = []  # 这一拍的绝对音高列表
            flag_fail = False
            for rel_note_iterator, rel_note in enumerate(rel_note_list):
                if rel_note == 0:
                    abs_note_list.append(0)
                    continue
                elif rel_note == 1:
                    try:
                        imitate_time_diff += 8 * (1 - imitate_spd_ratio)
                        abs_note_list = ImitatePatternDecode(melody_iterator * 8 + rel_note_iterator, imitate_note_diff, imitate_time_diff, imitate_spd_ratio, melody_rel_note_list, tone, root_note)
                    except NameError:
                        flag_fail = True
                        break
                    except IndexError:
                        flag_fail = True
                        break
                    break
                if rel_note[0] == 1:  # 有模仿的表现手法
                    imitate_note_diff = rel_note[1][0]
                    imitate_time_diff = rel_note[1][1]
                    imitate_spd_ratio = int(1 / self.flatten_spr_list[0][self.flatten_spr_list[1].index(rel_note[1][2])])
                    abs_note_list = ImitatePatternDecode(melody_iterator * 8 + rel_note_iterator, imitate_note_diff, imitate_time_diff, imitate_spd_ratio, melody_rel_note_list, tone, root_note)
                    break
                else:
                    abs_note_list.append(GetAbsNoteList(melody_iterator * 8 + rel_note_iterator, rel_note[1:], melody_core_note_list, tone, root_note))
            if flag_fail is False:
                fill_output = fill_output + abs_note_list
            else:
                fill_output = fill_output + [0 for t in range(8)]
            melody_iterator += 1
        return fill_output
