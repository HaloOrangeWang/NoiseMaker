from settings import *
from interfaces.sql.sqlite import get_raw_song_data_from_dataset
from interfaces.utils import flat_array, DiaryLog
from datainputs.melody import MelodyPatternEncode
from validations.intro import IntroShiftConfidenceCheck
import numpy as np
import copy
import math
import os


def adjust_intro_pitch(raw_melody_data, raw_intro_data):
    """
    根据一首歌的主旋律音高来修正其前奏/间奏的音高，使前奏/间奏的平均音高与主旋律的平均音高相差在[-4, 8) * sgn(i - m) 之间
    :param raw_melody_data: 一首歌的主旋律数据
    :param raw_intro_data: 这首歌的前奏/间奏数据
    :return:
    """
    intro_pitch_sum = 0
    intro_note_count = 0
    melody_pitch_sum = 0
    melody_note_count = 0
    for step_it in range(len(raw_melody_data)):  # 遍历主旋律的所有音符, 求出这些音符的平均音高
        if raw_melody_data[step_it] != 0:
            melody_note_count += 1
            melody_pitch_sum += raw_melody_data[step_it]
    for step_it in range(len(raw_intro_data)):  # 遍历前奏/间奏的所有音符, 求出这些音符的平均音高
        if raw_intro_data[step_it] != 0:
            intro_note_count += 1
            intro_pitch_sum += raw_intro_data[step_it]
    if melody_note_count != 0 and intro_note_count != 0:
        intro_pitch_avr = intro_pitch_sum / intro_note_count  # 前奏/间奏的平均音高
        melody_pitch_avr = melody_pitch_sum / melody_note_count  # 主旋律的平均音高
        if intro_pitch_avr > melody_pitch_avr:
            pitch_adj = int(12 * ((intro_pitch_avr - melody_pitch_avr + 4) // 12))  # 允许的前奏/间奏和主旋律之间的音高差异在-4 ~ 8之间
        else:
            pitch_adj = int(12 * ((intro_pitch_avr - melody_pitch_avr + 8) // 12))  # 允许的前奏/间奏和主旋律之间的音高差异在-8 ~ 4之间
        if pitch_adj != 0:
            intro_data = copy.deepcopy(raw_intro_data)  # 调整音高，将所有音符的音高下移adjust_value
            for step_it in range(len(intro_data)):
                if intro_data[step_it] != 0:
                    intro_data[step_it] -= pitch_adj
            return intro_data
        else:
            return raw_intro_data  # 不调整音高的情况 直接返回原先的前奏/间奏数据
    else:
        return raw_intro_data


class IntroTrainData:
    """获取前奏、间奏训练模型的输入输出数据"""

    def __init__(self, raw_melody_data, melody_pat_data, common_melody_pats, section_data, continuous_bar_data):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        self.ShiftConfidence = IntroShiftConfidenceCheck()

        # 1.从数据集中读取intro/interlude信息，并变更为以音符步长为单位的列表
        raw_intro_data = get_raw_song_data_from_dataset('intro')
        raw_interlude_data = get_raw_song_data_from_dataset('interlude')
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_intro_data[song_it] != dict():
                raw_intro_data[song_it] = flat_array(raw_intro_data[song_it])
            else:
                raw_intro_data[song_it] = []  # 对于没有intro/interlude的歌曲，将格式转化为list格式
            if raw_interlude_data[song_it] != dict():
                raw_interlude_data[song_it] = flat_array(raw_interlude_data[song_it])
            else:
                raw_interlude_data[song_it] = []

        # 2.对数据进行处理，调整前奏/间奏音符的音高 防止出现前奏/间奏和主旋律音高差异过大的问题
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_intro_data[song_it] and raw_melody_data[song_it]:
                raw_intro_data[song_it] = adjust_intro_pitch(raw_melody_data[song_it], raw_intro_data[song_it])
            if raw_interlude_data[song_it] and raw_melody_data[song_it]:
                raw_interlude_data[song_it] = adjust_intro_pitch(raw_melody_data[song_it], raw_interlude_data[song_it])

        # 3.生成每首歌的旋律变化累积幅度数据
        # 3.1.生成每段前奏/间奏的音高变化情况
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_intro_data[song_it]:
                self.ShiftConfidence.train_1song(raw_melody_data=raw_melody_data[song_it], raw_intro_data=raw_intro_data[song_it], continuous_bar_data=continuous_bar_data[song_it])
            if raw_interlude_data[song_it]:
                self.ShiftConfidence.train_1song(raw_melody_data=raw_melody_data[song_it], raw_intro_data=raw_interlude_data[song_it], continuous_bar_data=continuous_bar_data[song_it])
        # 3.2.找出旋律变化和段落内差异前90%所在位置
        self.ShiftConfidence.calc_confidence_level(0.9)
        self.ShiftConfidence.store('intro_shift')

        # 4.获取前奏模型的输入输出数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_intro_data[song_it] and melody_pat_data[song_it]:
                intro_pat_data = MelodyPatternEncode(common_melody_pats, raw_intro_data[song_it], 0.125, 1).music_pattern_list
                self.get_intro_model_io_data(intro_pat_data, melody_pat_data[song_it], continuous_bar_data[song_it], section_data[song_it])

        # 5.获取间奏模型的输入输出数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_interlude_data[song_it] and melody_pat_data[song_it]:
                interlude_pat_data = MelodyPatternEncode(common_melody_pats, raw_interlude_data[song_it], 0.125, 1).music_pattern_list  # 前奏和间奏的common patterns沿用主旋律的
                self.get_interlude_model_io_data(interlude_pat_data, melody_pat_data[song_it], continuous_bar_data[song_it])
        np.save(os.path.join(PATH_PATTERNLOG, 'IntroInputData.npy'), self.input_data)  # 在generate的时候要比较生成数据和训练集是否雷同，因此这个也要存储

        DiaryLog.warn('Generation of intro and interlude train data has finished!')

    def get_intro_model_io_data(self, intro_pat_data, melody_pat_data, continuous_bar_data, section_data):
        """
        训练前奏数据。为保证前奏和正式主旋律的连贯性，在生成模型的输入输出数据时，在前面加上第一段主旋律主歌的最后部分
        :param intro_pat_data: 一首歌的前奏数据（pattern形式）
        :param melody_pat_data: 一首歌的主旋律数据（pattern形式）
        :param continuous_bar_data: 一首歌的连续小节数据
        :param section_data: 一首歌的乐段数据
        """
        # 1.计算应该把一首歌的第一段主歌放到前奏数据中的哪个位置
        # 1.1.获取前奏的起始拍
        intro_start_beat = 0
        flag_start_beat_found = False
        for beat_it in range(len(intro_pat_data)):
            if intro_pat_data[beat_it] != 0:
                intro_start_beat = beat_it  # 前奏从第几个步长开始
                flag_start_beat_found = True
                break
        # 1.2.获取主旋律第一段主歌的结束拍
        flag_end_beat_found = False
        melody_end_beat = -1
        if section_data:  # 这首歌有乐段
            end_beat_temp = -1
            for sec_it in range(len(section_data) - 1):
                if section_data[sec_it][2] == DEF_SEC_MAIN and section_data[sec_it + 1][2] != DEF_SEC_MAIN:
                    end_beat_temp = int(section_data[sec_it + 1][0] * 4 + section_data[sec_it + 1][1] - 1)  # 第一段主歌结束于哪个步长
                    break
            if end_beat_temp == -1:
                end_beat_temp = len(melody_pat_data) - 1
            for beat_it in range(end_beat_temp, -1, -1):
                if melody_pat_data[beat_it] != 0:
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
                end_beat_temp = len(melody_pat_data) - 1
            for beat_it in range(end_beat_temp, -1, -1):
                if melody_pat_data[beat_it] != 0:
                    melody_end_beat = beat_it  # 第一段主歌准确地结束于哪个步长
                    flag_end_beat_found = True
                    break
        # 1.3.找出主旋律的第一段主歌应该映射到前奏的什么位置
        if flag_end_beat_found:  # 主旋律结束的那一拍应该移到前奏中的什么位置
            # melody_end_beat_in_intro = intro_start_beat - (melody_end_beat % 4 - intro_start_beat % 4) - 4 * int(melody_end_beat % 4 <= intro_start_beat % 4)
            melody_end_beat_in_intro = melody_end_beat - 4 * ((melody_end_beat - intro_start_beat) // 4) - 4
        else:
            melody_end_beat_in_intro = -math.inf

        # 2.生成前奏训练模型的输入输出数据
        for beat_it in range(intro_start_beat - 4 * TRAIN_INTRO_IO_BARS, len(intro_pat_data) - 4 * TRAIN_INTRO_IO_BARS):
            try:
                # 2.1.获取当前所在的小节和所在小节的位置
                cur_bar = beat_it // 4  # 第几小节
                beat_in_bar = pat_step_in_bar = beat_it % 4
                input_time_data = [beat_in_bar]
                output_time_data = [beat_in_bar]
                # 2.2.添加4小节的主旋律/前奏。添加前奏/间奏的时候不考虑“如果一个小节为空则把前面的也置为空”的问题
                for ahead_beat_it in range(beat_it, beat_it + 4 * TRAIN_INTRO_IO_BARS + 1):
                    if ahead_beat_it <= melody_end_beat_in_intro:  # 应该添加一拍主旋律组合
                        melody_insert_beat = ahead_beat_it + (melody_end_beat - melody_end_beat_in_intro)  # 这里插入主旋律的哪个步长
                        if melody_insert_beat >= 0:
                            if ahead_beat_it != beat_it + 4 * TRAIN_INTRO_IO_BARS:
                                input_time_data.append(melody_pat_data[melody_insert_beat])
                            if ahead_beat_it != beat_it:
                                output_time_data.append(melody_pat_data[melody_insert_beat])
                        else:
                            if ahead_beat_it != beat_it + 4 * TRAIN_INTRO_IO_BARS:
                                input_time_data.append(0)
                            if ahead_beat_it != beat_it:
                                output_time_data.append(0)
                    else:  # 应该添加一拍前奏组合
                        if ahead_beat_it != beat_it:
                            output_time_data.append(intro_pat_data[ahead_beat_it])
                        if ahead_beat_it != beat_it + 4 * TRAIN_INTRO_IO_BARS:
                            input_time_data.append(intro_pat_data[ahead_beat_it])
                # 2.3.当输出数据所在的小节不为空时，该数据收录进训练集
                final_bar_intro_data = intro_pat_data[(cur_bar + TRAIN_INTRO_IO_BARS) * 4: (cur_bar + TRAIN_INTRO_IO_BARS + 1) * 4]
                if not set(final_bar_intro_data).issubset({0, COMMON_MELODY_PAT_NUM + 1}):
                    self.input_data.append(input_time_data)
                    self.output_data.append(output_time_data)
            except IndexError:
                pass

    def get_interlude_model_io_data(self, interlude_pat_data, melody_pat_data, continuous_bar_data):
        """
        获取间奏模型的输入输出数据。为保证正式主旋律和间奏的连贯性，在生成模型的输入输出数据时，在前面加上第一段主旋律副歌的最后部分
        :param interlude_pat_data:
        :param melody_pat_data:
        :param continuous_bar_data:
        """
        for beat_it in range(len(interlude_pat_data) - 4 * TRAIN_INTRO_IO_BARS):
            try:
                # 1.添加当前时间
                cur_bar = beat_it // 4  # 第几小节
                beat_in_bar = pat_step_in_bar = beat_it % 4
                input_time_data = [beat_in_bar]
                output_time_data = [beat_in_bar]

                # 2.添加4小节的主旋律/间奏
                flag_add_interlude = False
                for ahead_beat_it in range(beat_it, beat_it + 4 * TRAIN_INTRO_IO_BARS + 1):
                    if flag_add_interlude is False:
                        if interlude_pat_data[ahead_beat_it] == 0:  # 间奏还没开始
                            if ahead_beat_it != beat_it + 4 * TRAIN_INTRO_IO_BARS:
                                if len(melody_pat_data) >= ahead_beat_it:
                                    input_time_data.append(melody_pat_data[ahead_beat_it])
                                else:
                                    input_time_data.append(0)
                            if ahead_beat_it != beat_it:
                                if len(melody_pat_data) >= ahead_beat_it:
                                    output_time_data.append(melody_pat_data[ahead_beat_it])
                                else:
                                    output_time_data.append(0)
                        else:  # 间奏从这一拍开始
                            if beat_it != ahead_beat_it:
                                output_time_data.append(interlude_pat_data[ahead_beat_it])
                            if ahead_beat_it != beat_it + 4 * TRAIN_INTRO_IO_BARS:
                                input_time_data.append(interlude_pat_data[ahead_beat_it])
                            flag_add_interlude = True
                    else:
                        if ahead_beat_it != beat_it:
                            output_time_data.append(interlude_pat_data[ahead_beat_it])
                        if ahead_beat_it != beat_it + 4 * TRAIN_INTRO_IO_BARS:
                            input_time_data.append(interlude_pat_data[ahead_beat_it])

                # 3.当输出数据所在的小节不为空 且在当前拍尚未开始第二段主旋律时，该数据收录进训练集
                final_bar_interlude_data = interlude_pat_data[(cur_bar + TRAIN_INTRO_IO_BARS) * 4: (cur_bar + TRAIN_INTRO_IO_BARS + 1) * 4]
                if not set(final_bar_interlude_data).issubset({0, COMMON_MELODY_PAT_NUM + 1}):
                    if melody_pat_data[(cur_bar + TRAIN_INTRO_IO_BARS) * 4: (beat_it + 17)] == [0 for t in range(beat_it % 4 + 1)]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except IndexError:
                pass


class IntroTestData:

    def __init__(self):
        # 1.从sqlite中读取音高变化的情况
        self.ShiftConfidence = IntroShiftConfidenceCheck()
        self.ShiftConfidence.restore('intro_shift')

        # 2.获取前奏模型的输入数据，用于和生成结果进行比对
        self.input_data = np.load(os.path.join(PATH_PATTERNLOG, 'IntroInputData.npy')).tolist()  # 在generate的时候要比较生成数据和训练集是否雷同，因此这个也要存储

        DiaryLog.warn('Restoring of intro and interlude associated data has finished!')
