from interfaces.utils import get_dict_max_key
from settings import *
import copy
from interfaces.sql.sqlite import get_raw_song_data_from_dataset
from datainputs.melody import MelodyPatternEncode


def adjust_scale(raw_melody_data, raw_intro_data):
    """
    根据一首歌的主旋律音高来修正其前奏/间奏的音高，使前奏/间奏的平均音高与主旋律的平均音高相差在[-4, 8) * sgn(i - m) 之间
    :param raw_melody_data: 一首歌的主旋律数据
    :param raw_intro_data: 这首歌的前奏/间奏数据
    :return:
    """
    intro_sum_note_high = 0
    intro_note_count = 0
    melody_sum_note_high = 0
    melody_note_count = 0
    for key in raw_melody_data:  # 遍历主旋律的所有音符, 求出这些音符的平均音高
        for note_it in range(len(raw_melody_data[key])):
            if raw_melody_data[key][note_it] != 0:
                melody_note_count += 1
                melody_sum_note_high += raw_melody_data[key][note_it]
    for key in raw_intro_data:  # 遍历前奏/间奏的所有音符, 求出这些音符的平均音高
        for note_it in range(len(raw_intro_data[key])):
            if raw_intro_data[key][note_it] != 0:
                intro_note_count += 1
                intro_sum_note_high += raw_intro_data[key][note_it]
    if melody_note_count != 0 and intro_note_count != 0:
        intro_avr_note_high = intro_sum_note_high / intro_note_count  # 前奏/间奏的平均音高
        melody_avr_note_high = melody_sum_note_high / melody_note_count  # 主旋律的平均音高
        if intro_avr_note_high > melody_avr_note_high:
            adjust_value = 12 * ((intro_avr_note_high - melody_avr_note_high + 4) // 12)  # 允许的前奏/间奏和主旋律之间的音高差异在-4 ~ 8之间
        else:
            adjust_value = 12 * ((intro_avr_note_high - melody_avr_note_high + 8) // 12)  # 允许的前奏/间奏和主旋律之间的音高差异在-8 ~ 4之间
        if adjust_value != 0:
            intro_data = copy.deepcopy(raw_intro_data)  # 调整音高，将所有音符的音高下移adjust_value
            for key in intro_data:
                for note_it in range(len(intro_data[key])):
                    if intro_data[key][note_it] != 0:
                        intro_data[key][note_it] -= adjust_value
            return intro_data
        else:
            return raw_intro_data  # 不调整音高的情况 直接返回原先的前奏/间奏数据
    else:
        return raw_intro_data


def get_scale_shift_value(raw_melody_data, raw_intro_data, continuous_bar_data):
    score_ary = []
    for bar_it in range(len(continuous_bar_data)):
        if continuous_bar_data[bar_it] != 0 and (bar_it != 0 and continuous_bar_data[bar_it - 1] == 0):  # 主旋律开始的小节,说明前奏/间奏在这一小节结束了
            melody_start_step_dx = -1  # 主旋律准确开始的拍
            for step_it in range(32):
                if raw_melody_data[bar_it][step_it] != 0 and (step_it == 0 or raw_melody_data[bar_it][step_it] == 0):
                    melody_start_step_dx = step_it + bar_it * 32
                    break
            last_note = -1  # 上一个音符的音高
            last_note_step = -1  # 上一个音符所在的位置
            note_count = 0  # 一共多少音符
            shift_score = 0
            for step_it in range(melody_start_step_dx - 32, melody_start_step_dx):
                try:
                    if raw_intro_data[step_it // 32][step_it % 32] != 0:  # 计算变化分
                        if last_note > 0:
                            step_diff = step_it - last_note_step
                            shift_score += (raw_intro_data[step_it // 32][step_it % 32] - last_note) * (raw_intro_data[step_it // 32][step_it % 32] - last_note) / (step_diff * step_diff)
                        last_note = raw_intro_data[step_it // 32][step_it % 32]
                        last_note_step = step_it
                        note_count += 1
                except KeyError:
                    pass
            if last_note > 0:  # 添加前奏最后一个音符和主旋律第一个音符的音高差异分
                step_diff = melody_start_step_dx - last_note_step
                shift_score += (raw_melody_data[melody_start_step_dx // 32][melody_start_step_dx % 32] - last_note) * (raw_melody_data[melody_start_step_dx // 32][melody_start_step_dx % 32] - last_note) / (step_diff * step_diff)
                note_count += 1
            if note_count == 1:  # 只有一个音符的情况下，音高差异分为0分
                score_ary.append(0)
            elif note_count > 1:
                score_ary.append(shift_score / (note_count - 1))
    return score_ary


class IntroTrainData:
    """获取前奏、间奏训练模型的输入输出数据"""

    def __init__(self, raw_melody_data, melody_pat_data, common_melody_pats, section_data, continuous_bar_data):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取intro/interlude信息
        raw_intro_data = get_raw_song_data_from_dataset('intro')
        raw_interlude_data = get_raw_song_data_from_dataset('interlude')
        self.raw_intro_data = copy.deepcopy(raw_intro_data)
        self.raw_interlude_data = copy.deepcopy(raw_interlude_data)
        # 2.对数据进行处理，调整前奏/间奏音符的音高 防止出现前奏/间奏
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.raw_intro_data[song_it] != {} and raw_melody_data[song_it] != {}:
                self.raw_intro_data[song_it] = adjust_scale(raw_melody_data[song_it], self.raw_intro_data[song_it])
            if self.raw_interlude_data[song_it] != {} and raw_melody_data[song_it] != {}:
                self.raw_interlude_data[song_it] = adjust_scale(raw_melody_data[song_it], self.raw_interlude_data[song_it])
        # 3.获取前奏模型的输入输出数据
        self.intro_pat_data = [{} for t in range(TRAIN_FILE_NUMBERS)]  # intro数据
        for song_it in range(len(self.raw_intro_data)):
            if self.raw_intro_data[song_it] != {} and melody_pat_data[song_it] != {}:
                self.intro_pat_data[song_it] = MelodyPatternEncode(common_melody_pats, self.raw_intro_data[song_it], 0.125, 1).music_pattern_dic
                self.get_intro_model_io_data(self.intro_pat_data[song_it], melody_pat_data[song_it], continuous_bar_data[song_it], section_data[song_it])
        # 4.获取间奏模型的输入输出数据
        self.interlude_pat_data = [{} for t in range(TRAIN_FILE_NUMBERS)]  # Interlude数据
        for song_it in range(len(self.raw_interlude_data)):
            if self.raw_interlude_data[song_it] != {} and melody_pat_data[song_it] != {}:
                self.interlude_pat_data[song_it] = MelodyPatternEncode(common_melody_pats, self.raw_interlude_data[song_it], 0.125, 1).music_pattern_dic  # 前奏和间奏的common patterns沿用主旋律的
                self.get_interlude_model_io_data(self.interlude_pat_data[song_it], melody_pat_data[song_it], continuous_bar_data[song_it])
        # 5.生成每首歌的旋律变化累积幅度数据
        shift_score_ary = []
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.raw_intro_data[song_it] != {}:
                shift_score_ary.extend(get_scale_shift_value(raw_melody_data[song_it], self.raw_intro_data[song_it], continuous_bar_data[song_it]))
            if self.raw_interlude_data[song_it] != {}:
                shift_score_ary.extend(get_scale_shift_value(raw_melody_data[song_it], self.raw_interlude_data[song_it], continuous_bar_data[song_it]))
        # 6.找出前90%所在位置
        shift_score_ary = sorted(shift_score_ary)
        prob_09_dx = int(len(shift_score_ary) * 0.9 + 1)
        self.ShiftConfidenceLevel = shift_score_ary[prob_09_dx]

    def get_intro_model_io_data(self, intro_pat_data, melody_pat_data, continuous_bar_data, section_data):
        """
        训练前奏数据。为保证前奏和正式主旋律的连贯性，在生成模型的输入输出数据时，在前面加上第一段主旋律主歌的最后部分
        :param intro_pat_data: 一首歌的前奏数据
        :param melody_pat_data: 一首歌的主旋律数据（pattern形式）
        :param continuous_bar_data: 一首歌的连续小节数据
        :param section_data: 一首歌的乐段数据
        """
        # 1.计算应该把一首歌的第一段主歌放到前奏数据中的哪个位置
        intro_start_beat = 0
        flag_start_beat_found = False
        for bar_it in range(get_dict_max_key(intro_pat_data)):
            if flag_start_beat_found:
                break
            for beat_it in range(4):
                if intro_pat_data[bar_it][beat_it] != 0:
                    intro_start_beat = bar_it * 4 + beat_it  # 前奏从第几个步长开始
                    flag_start_beat_found = True
                    break
        flag_end_beat_found = False
        melody_end_beat = -1
        if section_data:  # 这首歌有乐段
            end_beat_temp = -1
            for sec_it in range(len(section_data) - 1):
                if section_data[sec_it][2] == SECTION_MAIN and section_data[sec_it + 1][2] != SECTION_MAIN:
                    end_beat_temp = int(section_data[sec_it + 1][0] * 4 + section_data[sec_it + 1][1] - 1)  # 第一段主歌结束于哪个步长
                    break
            if end_beat_temp == -1:
                end_beat_temp = len(melody_pat_data) * 4 - 1
            for beat_it in range(end_beat_temp, -1, -1):
                if melody_pat_data[beat_it // 4][beat_it % 4] != 0:
                    melody_end_beat = beat_it  # 第一段主歌准确地结束于哪个步长
                    flag_end_beat_found = True
                    break
        else:  # 这首歌没有乐段
            end_beat_temp = -1
            for bar_it in range(len(continuous_bar_data) - 1):
                if continuous_bar_data[bar_it] != 0 and continuous_bar_data[bar_it + 1] == 0:
                    end_beat_temp = (bar_it + 1) * 4 - 1
                    break
            if end_beat_temp == -1:
                end_beat_temp = len(melody_pat_data) * 4 - 1
            for beat_it in range(end_beat_temp, -1, -1):
                if melody_pat_data[beat_it // 4][beat_it % 4] != 0:
                    melody_end_beat = beat_it  # 第一段主歌准确地结束于哪个步长
                    flag_end_beat_found = True
                    break
        if flag_end_beat_found:  # 主旋律结束的那一拍应该移到前奏中的什么位置
            melody_end_beat_in_intro = intro_start_beat - (melody_end_beat % 4 - intro_start_beat % 4) - 4 * int(melody_end_beat % 4 <= intro_start_beat % 4)
        else:
            melody_end_beat_in_intro = -1
        # 2.生成前奏训练模型的输入输出数据
        for beat_it in range(intro_start_beat - 16, len(intro_pat_data) * 4 - 16):
            try:
                # 1.添加当前时间
                time_in_bar = beat_it % 4
                input_time_data = [time_in_bar]
                output_time_data = [time_in_bar]
                # 2.添加4小节的主旋律/前奏
                for beat_it2 in range(beat_it, beat_it + 17):
                    if beat_it2 <= melody_end_beat_in_intro:  # 应该添加一拍主旋律组合
                        melody_insert_beat = beat_it2 + (melody_end_beat - melody_end_beat_in_intro)  # 这里插入主旋律的哪个步长
                        if melody_insert_beat >= 0:
                            if beat_it2 != beat_it + 16:
                                input_time_data.append(melody_pat_data[melody_insert_beat // 4][melody_insert_beat % 4])
                            if beat_it != beat_it2:
                                output_time_data.append(melody_pat_data[melody_insert_beat // 4][melody_insert_beat % 4])
                        else:
                            if beat_it2 != beat_it + 16:
                                input_time_data.append(0)
                            if beat_it != beat_it2:
                                output_time_data.append(0)
                    else:  # 应该添加一拍前奏组合
                        if beat_it2 != beat_it:
                            output_time_data.append(intro_pat_data[beat_it2 // 4][beat_it2 % 4])
                        if beat_it2 != beat_it + 16:
                            input_time_data.append(intro_pat_data[beat_it2 // 4][beat_it2 % 4])
                # 3.当输出数据所在的小节不为空时，该数据收录进训练集
                if intro_pat_data[(beat_it + 16) // 4] != [0 for t in range(4)]:
                    self.input_data.append(input_time_data)
                    self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass

    def get_interlude_model_io_data(self, interlude_pat_data, melody_pat_data, continuous_bar_data):
        """
        获取间奏模型的输入输出数据。为保证正式主旋律和间奏的连贯性，在生成模型的输入输出数据时，在前面加上第一段主旋律副歌的最后部分
        :param interlude_pat_data:
        :param melody_pat_data:
        :param continuous_bar_data:
        """
        for beat_it in range(len(interlude_pat_data) * 4 - 16):
            try:
                # 1.添加当前时间
                time_in_bar = beat_it % 4
                input_time_data = [time_in_bar]
                output_time_data = [time_in_bar]
                # 2.添加4小节的主旋律/间奏
                flag_interlude_start = False
                for beat_it2 in range(beat_it, beat_it + 17):
                    if flag_interlude_start is False:
                        if interlude_pat_data[beat_it2 // 4][beat_it2 % 4] == 0:  # 间奏还没开始
                            if beat_it2 != beat_it + 16:
                                if len(melody_pat_data) >= beat_it2 // 4 + 1:
                                    input_time_data.append(melody_pat_data[beat_it2 // 4][beat_it2 % 4])
                                else:
                                    input_time_data.append(0)
                            if beat_it2 != beat_it:
                                if len(melody_pat_data) >= beat_it2 // 4 + 1:
                                    output_time_data.append(melody_pat_data[beat_it2 // 4][beat_it2 % 4])
                                else:
                                    output_time_data.append(0)
                        else:  # 间奏从这一拍开始
                            if beat_it != beat_it2:
                                output_time_data.append(interlude_pat_data[beat_it2 // 4][beat_it2 % 4])
                            if beat_it2 != beat_it + 16:
                                input_time_data.append(interlude_pat_data[beat_it2 // 4][beat_it2 % 4])
                            flag_interlude_start = True
                    else:
                        if beat_it != beat_it2:
                            output_time_data.append(interlude_pat_data[beat_it2 // 4][beat_it2 % 4])
                        if beat_it2 != beat_it + 16:
                            input_time_data.append(interlude_pat_data[beat_it2 // 4][beat_it2 % 4])
                # 3.当输出数据所在的小节不为空 且在当前拍尚未开始第二段主旋律时，该数据收录进训练集
                if interlude_pat_data[(beat_it + 16) // 4] != [0 for t in range(4)]:
                    if melody_pat_data[(beat_it + 16) // 4][:(beat_it + 16) % 4] == [0] * ((beat_it + 16) % 4):
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass
