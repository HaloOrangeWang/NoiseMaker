from settings import *
from datainputs.fill import FillTrainData, FillTestData, get_freq_dx
from interfaces.utils import DiaryLog
import numpy as np
import random
import copy


def choose_1fill(melody_data, chord_data, fill_pat_ary, fill_type, fill_avr_note=-1, last_fill_avr_note=-1):
    choose_pat_dx = -1
    choose_pat_score = np.inf
    score_ary = []

    for pat_it in range(len(fill_pat_ary)):
        # 1.计算加花和同时期主旋律的音符差异和相似情况。对于过渡型来说，这个分数为加花最后一个音符和接下来的主旋律的音高差异；对于强调型来说，这个分数为零；对于点缀型来说，这个分数为加花和过去几拍主旋律的音符差异
        note_diff_score = 0
        if fill_type == 1:
            for note_it in range(len(fill_pat_ary[pat_it]) - 1, -1, -1):
                if fill_pat_ary[pat_it][note_it] != 0:
                    for note_it2 in range(len(melody_data) - 8 - (len(fill_pat_ary[pat_it]) - note_it), len(melody_data)):
                        if melody_data[note_it2] != 0:
                            note_diff_score = sum([(abs(t % 12 - melody_data[note_it2] % 12) / 7) for t in fill_pat_ary[pat_it][note_it]]) / len(fill_pat_ary[pat_it][note_it])
                            break
                    break
        elif fill_type == 2:
            note_diff_score = 0
        else:
            fill_note_set = set()  # 将加花数据取集合
            for note_it in range(len(fill_pat_ary[pat_it])):
                if fill_pat_ary[pat_it][note_it] != 0:
                    fill_note_set = fill_note_set.union(set([t % 12 for t in fill_pat_ary[pat_it][note_it]]))
            melody_note_set = set([t % 12 for t in melody_data[:-8] if t != 0])  # 将同期主旋律数据取集合
            diff_note_count = len(fill_note_set) - len(fill_note_set & melody_note_set)  # 在加花数据中却不在主旋律数据中
            note_diff_score = (diff_note_count / len(fill_note_set))
        # 2.计算加花数据和同时期和弦的匹配关系
        chord_score_params = [1.6, 2, 1.2]
        note_time_not_in_chord = 0  # 记录不在和弦范围内的音符数量，并计算它们占总加花时长的比例。七音算0.25
        last_note_matching_score = 0
        for note_it in range(len(fill_pat_ary[pat_it])):
            if chord_data[note_it // 8] == 0:
                continue
            if fill_pat_ary[pat_it][note_it] == 0:
                note_time_not_in_chord += last_note_matching_score
            else:
                chord7_note_ary = set()
                if 1 <= chord_data[note_it // 8] <= 72 and chord_data[note_it // 8] % 6 == 1:  # 大三和弦 chord_set增加大七度和小七度
                    chord7_note_ary.add((chord_data[note_it // 8] // 6 + 10) % 12)
                    chord7_note_ary.add((chord_data[note_it // 8] // 6 + 11) % 12)
                if 1 <= chord_data[note_it // 8] <= 72 and chord_data[note_it // 8] % 6 == 2:  # 小三和弦 chord_set增加小七度
                    chord7_note_ary.add((chord_data[note_it // 8] // 6 + 10) % 12)
                note_matching_score = 0
                for note in fill_pat_ary[pat_it][note_it]:
                    if note % 12 in chord7_note_ary:
                        note_matching_score += 0.25
                    if note % 12 not in chord7_note_ary and note % 12 not in CHORD_LIST[chord_data[note_it // 8]]:
                        note_matching_score += 1
                note_matching_score /= len(fill_pat_ary[pat_it][note_it])
                note_time_not_in_chord += note_matching_score
                last_note_matching_score = note_matching_score
        chord_diff_score = chord_score_params[fill_type - 1] * (note_time_not_in_chord / len(fill_pat_ary[pat_it]))
        # 3.本次加花和前几次加花的平均音高的差异得分
        sum_note_pitch = 0  # 本次加花的音符音高总和
        note_count = 0  # 本次加花的音符总量
        for note_it in range(len(fill_pat_ary[pat_it])):
            if fill_pat_ary[pat_it][note_it] != 0:
                sum_note_pitch += sum(fill_pat_ary[pat_it][note_it])
                note_count += len(fill_pat_ary[pat_it][note_it])
        if note_count != 0:
            avr_note_pitch = sum_note_pitch / note_count
            if last_fill_avr_note != -1 and fill_avr_note != -1:  # 和上次加花与此前加花的平均音高相差一个八度 则差异分记为1分
                fill_hist_diff = (abs(avr_note_pitch - fill_avr_note) + abs(avr_note_pitch - last_fill_avr_note)) / 24
            elif fill_avr_note != -1:
                fill_hist_diff = abs(avr_note_pitch - fill_avr_note) / 12
            else:
                fill_hist_diff = 0
        else:
            raise ValueError
        # 4.计算加花和同期主旋律的按键差异得分 对于强调型加花 应保证加花的按键位置和主旋律的按键位置尽量相同 而过渡型和点缀型则应当尽量使按键位置不同
        keypress_same_note_count = 0  # 按键位置相同的音符数量（如果一个时间步长有多个音符，则按一个音符数量计算）
        keypress_diff_note_count = 0  # 按键位置不同的音符数量（如果一个时间步长有多个音符，则按一个音符数量计算）
        for note_it in range(len(fill_pat_ary[pat_it])):
            if bool(fill_pat_ary[pat_it][note_it]) ^ bool(melody_data[len(melody_data) - 8 - (len(fill_pat_ary[pat_it]) - note_it)]):
                keypress_diff_note_count += 1
            elif fill_pat_ary[pat_it][note_it] != 0 and melody_data[len(melody_data) - 8 - (len(fill_pat_ary[pat_it]) - note_it)] != 0:
                keypress_same_note_count += 1
        if fill_type == 2:
            keypress_diff_score = keypress_diff_note_count / (keypress_same_note_count + keypress_diff_note_count)
        else:
            keypress_diff_score = keypress_same_note_count / (keypress_same_note_count + keypress_diff_note_count)
        # 5.计算总差异得分 并将得分最小的当做这一拍的加花内容
        total_score = note_diff_score * note_diff_score + chord_diff_score * chord_diff_score + fill_hist_diff * fill_hist_diff + keypress_diff_score * keypress_diff_score  # 总差异得分是各项得分的平方和
        score_ary.append(total_score)
        if total_score < choose_pat_score:
            choose_pat_score = total_score
            choose_pat_dx = pat_it

    if choose_pat_score != np.inf:
        return choose_pat_dx
    else:
        raise ValueError


class FillPipeline:

    def __init__(self, is_train, *args):
        if is_train:
            # 训练时的情况
            # args = (raw_melody_data, section_data, continuous_bar_data)
            self.train_data = FillTrainData(*args)
        else:
            self.train_data = FillTestData()
        self.fill_prob_base = 0.05  # 加花的比例是多少

    def judge_fill(self, melody_out_notes, sec_data):
        judge_fill_list = [0 for t in range(len(melody_out_notes) // 8)]  # 每拍是否要加花的数组
        fill_type_1beat_old = 0  # 上一拍的加花情况
        fill_rep_beats = 0  # 此前已经有多少拍的同类型加花
        section_fill_count = [0, 0, 0, 0]
        sec_dx_old = -1
        section_beats = 0

        for beat_it in range(0, len(melody_out_notes) // 8):
            # 1.先获取当前拍所在的段落
            sec_dx = -1
            for sec_it in range(0, len(sec_data)):
                if sec_data[sec_it][0] > beat_it // 4:
                    sec_dx = sec_it - 1
                    break
            if sec_dx == -1:
                sec_dx = len(sec_data) - 1  # 属于这首歌的最后一个区段
            if sec_data[sec_dx][1] == "empty":  # 空小节不加花
                continue
            # 2.获取时间编码
            if (sec_dx == len(sec_data) - 1 and beat_it // 4 + 1 == len(melody_out_notes) // 32) or (sec_dx != len(sec_data) - 1 and beat_it // 4 + 1 == sec_data[sec_dx + 1][0]):
                flag_in_sec_end = True  # 当前小节是否为某一个乐段的最后一小节
            else:
                flag_in_sec_end = False
            timecode = 4 * int(flag_in_sec_end) + beat_it % 4  # 当前拍的时间编码
            # 3.获取主旋律的按键编码
            melody_mark = 0
            for step_it in range(beat_it * 8 - 8, beat_it * 8 + 8, 4):
                if step_it >= 0 and melody_out_notes[step_it: step_it + 4] != [0, 0, 0, 0]:
                    melody_mark += pow(2, (step_it - beat_it * 8 + 8) // 4)
            # 4.处理当前乐段的各类加花数量的计数器。如果变更乐段（有乐段的情况）或下一小节为空小节（无乐段的情况）则计数器清零
            if sec_dx != sec_dx_old:
                sec_dx_old = sec_dx
                section_fill_count = [0, 0, 0, 0]
                section_beats = 0
            # 5.加花不能跨越小节的第二拍。且单次加花不能超过4拍。如果到小节第二拍有前小节遗留下来的加花 则直接按停
            if (beat_it // 4 == 1 and fill_type_1beat_old in [1, 3] and fill_rep_beats >= 1) or fill_rep_beats >= 3:
                fill_type_1beat_old = 0
                fill_rep_beats = 0
            # 6.用Naive Bayes计算此时各种类型加花的概率
            if fill_type_1beat_old == 0:
                # 6.1.上一拍没有加花的情况
                fill_prob_ary = [0, 0, 0]  # 这个数组的内容是这一拍加花类型的。数组中的三项分别为过渡型, 强调型, 点缀型
                for type_it in range(1, 4):
                    fill_freq = (25 * section_fill_count[type_it] + 1) / (25 * (section_beats + 1))
                    fill_index = get_freq_dx(fill_freq)

                    fill_prob_ary[type_it - 1] = (self.train_data.all_fill_ary[type_it] + 1) / (sum(self.train_data.all_fill_ary) + 4)
                    fill_prob_ary[type_it - 1] *= (self.train_data.timecode_fill_ary[type_it][timecode] + 1) / (sum(self.train_data.timecode_fill_ary[type_it]) + 8)
                    fill_prob_ary[type_it - 1] /= sum(self.train_data.timecode_fill_ary[:, timecode] + 1) / (sum(sum(self.train_data.timecode_fill_ary)) + 8)
                    fill_prob_ary[type_it - 1] *= (self.train_data.keypress_fill_ary[type_it][melody_mark] + 1) / (sum(self.train_data.keypress_fill_ary[type_it]) + 16)
                    fill_prob_ary[type_it - 1] /= sum(self.train_data.keypress_fill_ary[:, melody_mark] + 1) / (sum(sum(self.train_data.keypress_fill_ary)) + 16)
                    fill_prob_ary[type_it - 1] *= (self.train_data.sameinsec_fill_ary[type_it - 1][fill_index] + 1) / (sum(self.train_data.sameinsec_fill_ary[type_it - 1]) + 6)
                    fill_prob_ary[type_it - 1] /= (self.train_data.sameinsec_fill_ary[type_it - 1][fill_index] + self.train_data.sec_nfill_ary[type_it - 1][fill_index] + 1) / (sum(self.train_data.sameinsec_fill_ary[type_it - 1]) + sum(self.train_data.sec_nfill_ary[type_it - 1]) + 6)
            else:
                # 6.2.上一拍有加花的情况
                timecode_rep = (3 if fill_rep_beats >= 3 else fill_rep_beats) * 8 + timecode

                fill_prob = (self.train_data.all_fill_rep_ary[fill_type_1beat_old + 2] + 1) / (self.train_data.all_fill_rep_ary[fill_type_1beat_old - 1] + self.train_data.all_fill_rep_ary[fill_type_1beat_old + 2] + 2)  # 计算这一拍的加花是否仍然沿用前一拍的加花内容
                fill_prob *= (self.train_data.timecode_fill_rep_ary[fill_type_1beat_old + 2][timecode_rep] + 1) / (sum(self.train_data.timecode_fill_rep_ary[fill_type_1beat_old + 2]) + 32)
                fill_prob *= (self.train_data.keypress_fill_rep_ary[fill_type_1beat_old + 2][melody_mark] + 1) / (sum(self.train_data.keypress_fill_rep_ary[fill_type_1beat_old + 2]) + 16)
                fill_prob /= sum(self.train_data.timecode_fill_rep_ary[[fill_type_1beat_old - 1, fill_type_1beat_old + 2], timecode_rep] + 1) / (sum(sum(self.train_data.timecode_fill_rep_ary[[fill_type_1beat_old - 1, fill_type_1beat_old + 2], :])) + 32)
                fill_prob /= sum(self.train_data.keypress_fill_rep_ary[[fill_type_1beat_old - 1, fill_type_1beat_old + 2], melody_mark] + 1) / (sum(sum(self.train_data.keypress_fill_rep_ary[[fill_type_1beat_old - 1, fill_type_1beat_old + 2], :])) + 16)
            # 7.根据概率生成这一拍的加花内容
            fill_type_1beat = 0
            if fill_type_1beat_old == 0:
                # 7.1.上一拍没有加花的情况
                random_value = random.random()
                for type_it in range(len(fill_prob_ary)):
                    random_value -= fill_prob_ary[type_it]
                    if random_value <= 0:
                        fill_type_1beat = type_it + 1  # 这里要加一
                        break
            else:
                # 7.2.上一拍有加花的情况
                random_value = random.random()
                if random_value < fill_prob:  # 是否沿用上一拍的加花内容
                    fill_type_1beat = fill_type_1beat_old
                    fill_rep_beats += 1
                else:
                    fill_rep_beats = 0
            judge_fill_list[beat_it] = fill_type_1beat
            # 8.一次循环的收尾部分
            section_fill_count[fill_type_1beat] += 1  # 本段落内该种类型加花计数器加一
            fill_type_1beat_old = fill_type_1beat
            section_beats += 1
        DiaryLog.warn('加花的输出判定为: ' + repr(judge_fill_list))
        return judge_fill_list

    def generate(self, melody_output, chord_output, fill_judge_ary):
        """
        根据“是否加花”的列表来生成一段加花
        :param melody_output: 主旋律的输出
        :param chord_output: 和弦的输出
        :param fill_judge_ary: 是否加花的列表 以拍为单位
        :return:
        """
        fill_output = []
        beat_dx = 0  # 当前第几拍
        fill_avr_note = -1  # 加花此前的平均音高
        last_fill_avr_note = -1  # 上一轮加花的平均音高
        fill_note_count = 0  # 加花此前的音符数量（如果一个时间步长有多个音符，则按真实音符数量计算）
        while True:
            # 1.如果这一拍判定为不加花 则加花直接增加一个空小节
            if fill_judge_ary[beat_dx] == 0:
                fill_output.extend([0 for t in range(8)])
                beat_dx += 1
                continue
            # 2.确定这一轮加花最多可以持续多少拍
            length_temp = 1  # 本轮加花最多连续多少拍
            cross_bar_beat_dx = np.inf  # 在加花的第几拍会出现跨小节的情况（如果出现跨小节 则加花的最后一拍必须只有1个音符）
            fill_type = fill_judge_ary[beat_dx]  # 本轮加花的类型
            for beat_it2 in range(beat_dx + 1, len(fill_judge_ary)):
                if fill_judge_ary[beat_it2] != 0:
                    length_temp += 1
                    if fill_type == 3 and length_temp == 3:
                        break  # 点缀型的最多连续三拍
                    if beat_it2 % 4 == 0:
                        cross_bar_beat_dx = length_temp - 1
                        break
                else:
                    break
            # 3.生成一拍或多拍的加花
            for fill_length in range(length_temp, 0, -1):
                pats_ary = []  # 生成可供选择的加花列表
                # 3.1.确定生成加花的拍数
                for pat_it in range(len(self.train_data.fill_type_pat_cls.classified_fill_pats[fill_type - 1])):
                    if fill_length * 8 - 8 < len(self.train_data.fill_type_pat_cls.classified_fill_pats[fill_type - 1][pat_it]) <= fill_length * 8:
                        if cross_bar_beat_dx > fill_length:
                            pats_ary.append(self.train_data.fill_type_pat_cls.classified_fill_pats[fill_type - 1][pat_it])
                        else:
                            pats_ary.append(self.train_data.fill_type_pat_cls.classified_fill_pats[fill_type - 1][pat_it][:cross_bar_beat_dx * 8 + 1])
                if len(pats_ary) <= 4:  # 可供选择的加花至少要有5个 否则向下减一拍
                    continue
                # 3.2.生成加花内容
                try:
                    melody_data = melody_output[max(0, (beat_dx - 7) * 8): min(len(melody_output), (beat_dx + fill_length + 1) * 8)]
                    if beat_dx + fill_length + 1 > len(melody_output):
                        melody_data.append([0] * (beat_dx + fill_length + 1 - len(melody_output)) * 8)

                    choose_pat_dx = choose_1fill(melody_data, chord_output[beat_dx: (beat_dx + fill_length)], pats_ary, fill_type, fill_avr_note, last_fill_avr_note)
                except ValueError:
                    continue
                fill_output_1pat = copy.deepcopy(pats_ary[choose_pat_dx])
                fill_output_1pat.extend([0] * (fill_length * 8 - len(fill_output_1pat)))  # 补齐整一拍
                # 3.3.把加花的内容移到整拍或整半拍上
                for note_it in range(len(fill_output_1pat)):
                    if fill_output_1pat[note_it] != 0 and note_it % 4 != 0:
                        if (note_it >= len(fill_output_1pat) - 4 or note_it % 4 in [1, 2]) and fill_output_1pat[note_it - note_it % 4] == 0:  # 位于前两个八分之一音符的情况 将其向前挪至整半拍位置
                            fill_output_1pat[note_it - note_it % 4] = fill_output_1pat[note_it]
                            fill_output_1pat[note_it] = 0
                        elif (note_it < len(fill_output_1pat) - 4 and note_it % 4 == 3) and fill_output_1pat[note_it + 4 - note_it % 4] == 0:  # 位于第三个八分之一音符位置 将其向后挪至整半拍的位置
                            fill_output_1pat[note_it + 4 - note_it % 4] = fill_output_1pat[note_it]
                            fill_output_1pat[note_it] = 0
                fill_output.extend(fill_output_1pat)
                beat_dx += fill_length
                # 3.4.计算生成加花结束后这个加花内容的平均音高，并更新本曲加花的平均音高
                note_count_1pat = 0
                sum_note_pitch = 0
                for note_it in range(len(fill_output_1pat)):
                    if fill_output_1pat[note_it] != 0:
                        sum_note_pitch += sum(fill_output_1pat[note_it])
                        note_count_1pat += len(fill_output_1pat[note_it])
                last_fill_avr_note = sum_note_pitch / note_count_1pat
                if fill_avr_note != -1:
                    fill_avr_note = (fill_avr_note * fill_note_count + last_fill_avr_note * note_count_1pat) / (fill_note_count + note_count_1pat)
                else:
                    fill_avr_note = last_fill_avr_note
                fill_note_count += note_count_1pat
                break
            if beat_dx >= len(fill_judge_ary):
                DiaryLog.warn('加花的最终输出: ' + repr(fill_output))
                return fill_output
