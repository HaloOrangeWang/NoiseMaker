from settings import *
from interfaces.sql.sqlite import NoteDict
from interfaces.utils import flat_array, get_dict_max_key
import copy
import numpy as np


def get_rel_notelist_melody(note_list, tone, root):
    """
    以某个音为基准　获取相对音高的音符列表(主旋律和ｆｉｌｌ通常使用这个)
    :param note_list: 原始音符列表
    :param tone: 节奏（０为大调　１为小调）
    :param root: 根音（通常情况下　０为６０　１为５７）
    :return: 相对音高的音符列表
    """
    # 1.获取相对音高对照表
    if tone == TONE_MAJOR:
        rel_note_ary = REL_NOTE_COMPARE_DICT['Major']
    elif tone == TONE_MINOR:
        rel_note_ary = REL_NOTE_COMPARE_DICT['Minor']
    # 2.转化为相对音高
    rel_note_list = []
    for note in sorted(note_list, reverse=True):  # 先从大到小排序
        rel_note_list.append([7 * ((note - root) // 12) + rel_note_ary[(note - root) % 12][0], rel_note_ary[(note - root) % 12][1]])
    return rel_note_list


def one_song_rel_notelist_melody(note_list, tone, root, use_note_dict=False):
    """
    以某个音为基准 获取一首歌的相对音高的音符列表
    :param use_note_dict: 原note_list中的音是note_dict中的音还是绝对音高
    :param note_list: 这首歌的原始音高列表
    :param tone: 节奏（０为大调　１为小调）
    :param root: 根音（０为72　１为69）
    :return: 这首歌的相对音高的音符列表
    """
    rel_note_list = []  # 转化为相对音高形式的音符列表
    for note in note_list:
        if note == 0:
            rel_note_list.append(0)
        else:
            if use_note_dict is True:
                rel_note_group = get_rel_notelist_melody(NoteDict[note], tone, root)
            else:
                rel_note_group = get_rel_notelist_melody([note], tone, root)
            rel_note_list.append(rel_note_group)
    return rel_note_list


def get_rel_notelist_chord(note_list, root, chord):
    """
    以同时期和弦的根音为基准 转换成相对音高列表。音高用音符的音高和根音的差值代替
    :param note_list: 音符列表
    :param root: 和弦的根音
    :param chord: 同时期的和弦
    :return: 相对音高列表
    """
    rootdict = [[0, 0], [1, -1], [1, 0], [2, -1], [2, 0], [3, 0], [3, 1], [4, 0], [5, -1], [5, 0], [6, -1], [6, 0]]
    stadard_namedict = [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6]  # 音名列表
    zeng_namedict = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
    jian_namedict = [0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 6]
    stadard_rel_list = [0, 2, 4, 5, 7, 9, 11]
    rel_note_list = []
    # 1.确定根音的音名
    rootname = rootdict[root % 12]
    # 2.判断该和弦是否为增减和弦
    if 1 <= chord <= 72 and (chord - 1) % 6 == 2:
        namelist = zeng_namedict
    elif 1 <= chord <= 72 and (chord - 1) % 6 == 3:
        namelist = jian_namedict
    else:
        namelist = stadard_namedict
    # 3.根据和弦和当前音的音高确定音名列表
    for note in sorted(note_list):
        namediff = 7 * ((note - root) // 12) + namelist[(note - root) % 12]
        notename = [namediff, (note % 12) - stadard_rel_list[(namediff + rootname[0]) % 7]]
        notename[1] -= 12 * round(notename[1] / 12)
        rel_note_list.append(notename)
    return rel_note_list


def one_song_rel_notelist_chord(raw_note_ary, root_data, chord_data, unit='bar', note_time_step=0.25):
    """
    以和弦根音为基准 获取一首歌的相对音高的音符列表
    :param note_time_step: 音符的时间步长
    :param raw_note_ary: 原始的音符列表
    :param root_data: 和弦的根音
    :param chord_data: 和弦数据
    :param unit: 以小节为单元还是以拍为单元
    :return:
    """
    time_step_ratio = int(1 / note_time_step)  # 和弦的时间步长与音符的时间步长的比值
    # 1.准备工作: 将原始的数据展成以拍为单位的音符列表
    rel_note_list = []  # 转化为相对音高形式的音符列表
    if unit == 'bar':
        note_list = flat_array(raw_note_ary)  # 把以小节为单位的列表降一维 变成以音符步长为单位的列表
    elif unit == 'step':
        note_list = copy.deepcopy(raw_note_ary)
    else:
        raise ValueError
    # 2.遍历所有音符 获相对音高列表
    for note_it, note in enumerate(note_list):
        if note == 0:
            rel_note_list.append(0)
        elif note_it // time_step_ratio >= len(root_data):  # 这一拍没有和弦 则相对音高为0
            rel_note_list.append(0)
        elif root_data[note_it // time_step_ratio] == 0:  # 这一拍没有根音 则相对音高为0
            rel_note_list.append(0)
        else:
            bar_dx = note_it // (4 * time_step_ratio)  # 第几个小节
            beat_in_bar = (note_it % (4 * time_step_ratio)) // time_step_ratio  # 小节内的第几拍
            rel_note_group = get_rel_notelist_chord(NoteDict[note], root_data[note_it // time_step_ratio], chord_data[bar_dx][beat_in_bar])
            rel_note_list.append(rel_note_group)
    return rel_note_list


class RhythmTrainData:
    """训练一种伴奏类型的节奏"""

    def __init__(self, keypress_pat_data, keypress_pat_dic, continuous_bar_data, time_step, pattern_time_step):
        """
        :param keypress_pat_data: 主旋律按键的组合数据
        :param keypress_pat_dic: 原始按键组合的列表
        :param continuous_bar_data: 连续小节计数列表
        :param time_step: 原始音符的时间步长
        :param pattern_time_step: 音符组合的时间步长
        """

        self.rhythm_data = self.get_raw_rhythm_data()
        # self.melody_rhythm_data = self.get_keypress_data()
        # 3.获取最常见的piano_guitar组合
        # self.common_pg_patterns, __ = CommonMusicPatterns(raw_pg_data, number=COMMON_PIANO_GUITAR_PATTERN_NUMBER, note_time_step=1 / 4, pattern_time_step=1)
        self.rhythm_patterns, self.rhythm_pattern_dict, self.rhythm_pattern_count = self.rhythm_patterns(self.rhythm_data, time_step, pattern_time_step)
        # self.melody_rhythm_patterns, __ = RhythmPatterns(self.rhythm_data, time_step, pattern_time_step)
        self.transfer_count = np.zeros([len(self.rhythm_pattern_count) - 1, len(self.rhythm_pattern_count) - 1])
        self.emission_count = np.zeros([len(keypress_pat_dic) - 1, len(self.rhythm_pattern_count) - 1])
        self.pi_count = np.zeros([len(self.rhythm_pattern_count) - 1])
        for part_it in range(len(self.rhythm_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if self.rhythm_data[part_it][song_it] != {} and keypress_pat_data[song_it] != []:
                    # self.rhythm_data[pg_part_iterator][song_iterator] = RhythmPatternEncode(self.rhythm_patterns, self.rhythm_data[pg_part_iterator][song_iterator], keypress_pattern_data[song_iterator], time_step, pattern_time_step).music_pattern_dict
                    self.count(self.rhythm_patterns[part_it][song_it], keypress_pat_data[song_it])
                    self.count_initial(self.rhythm_patterns[part_it][song_it], keypress_pat_data[song_it], continuous_bar_data[song_it], pattern_time_step)
        self.prob_convert()

    @staticmethod
    def rhythm_patterns(raw_music_data, time_step, pattern_time_step):
        """
        将原始的音符组合转成按键组合的列表
        :param raw_music_data: 原始的音符数据
        :param time_step: 音符的时间步长
        :param pattern_time_step: 音符组合的步长
        :return:
        """
        time_step_ratio = round(pattern_time_step / time_step)
        rhythm_pattern_ary = [[0 for t in range(time_step_ratio)]]  # 音符组合列表
        rhythm_pattern_data = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t1 in range(len(raw_music_data))]  # 三维数组 第一维是组合 第二维歌曲列表 第三维是按键的组合　以音符时间步长为单位
        rhythm_pattern_count = [0]  # 每种按键组合出现的次数
        for part_it, rhythm_part in enumerate(raw_music_data):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if rhythm_part[song_it] != {}:
                    # for key? in range(get_dict_max_key(rhythm_part[song_it]) + 1):
                    # rhythm_pattern_data[part_it][song_it].append(0)
                    rhythm_data_1song = [1 if t != 0 else 0 for t in rhythm_part[song_it]]
                    raw_pattern_list = [rhythm_data_1song[t: t + time_step_ratio] for t in range(0, len(rhythm_data_1song), time_step_ratio)]  # 以pattern的步长为步长 将音符进行分类
                    for pattern_step_it, raw_pattern in enumerate(raw_pattern_list):
                        if raw_pattern not in rhythm_pattern_ary:
                            rhythm_pattern_data[part_it][song_it].append(len(rhythm_pattern_ary))
                            rhythm_pattern_ary.append(raw_pattern)
                            rhythm_pattern_count.append(1)
                        else:
                            rhythm_pattern_data[part_it][song_it].append(rhythm_pattern_ary.index(raw_pattern))
                            rhythm_pattern_count[rhythm_pattern_ary.index(raw_pattern)] += 1
        return rhythm_pattern_data, rhythm_pattern_ary, rhythm_pattern_count

    def get_raw_rhythm_data(self):
        pass

    def count(self, rhythm_data, keypress_data):
        """
        对一首歌 对前一时刻的按键－这个时刻的按键 及 伴奏按键－主旋律按键的变换矩阵进行计数
        :param rhythm_data: 伴奏按键数据
        :param keypress_data: 主旋律的按键数据
        """
        for step_it in range(len(rhythm_data)):
            if step_it >= len(keypress_data):  # 这一小节没有和弦 直接跳过
                continue
            if step_it == 0:  # 不考虑第一小节的第一拍
                continue
            if step_it != 0:
                if (rhythm_data[step_it] == 0) or (rhythm_data[step_it - 1] == 0) or (keypress_data[step_it] == 0):  # 当前拍 上一拍的节奏编码不能是0或common+1 同时当前拍主旋律按键编码不能是0
                    continue
                self.transfer_count[rhythm_data[step_it - 1] - 1, rhythm_data[step_it] - 1] += 1
                self.emission_count[keypress_data[step_it] - 1, rhythm_data[step_it] - 1] += 1

    def count_initial(self, rhythm_data, keypress_data, continuous_bar_data, pattern_time_step):

        time_step_ratio = round(4 / pattern_time_step)  # 一个小节中有多少个按键组合
        for pat_step_dx in range(0, len(rhythm_data), time_step_ratio):  # 遍历每个小节
            if (pat_step_dx // time_step_ratio) >= len(continuous_bar_data):  # 这个小节没有对应的主旋律 直接跳过
                continue
            if pat_step_dx == 0:
                if continuous_bar_data[0] != 0:
                    if rhythm_data[0] != 0:
                        self.pi_count[rhythm_data[0] - 1] += 1
            else:
                if continuous_bar_data[pat_step_dx // time_step_ratio] != 0 and continuous_bar_data[(pat_step_dx // time_step_ratio) - 1] == 0:
                    if rhythm_data[pat_step_dx // time_step_ratio] != 0:
                        self.pi_count[rhythm_data[pat_step_dx // time_step_ratio] - 1] += 1

    def prob_convert(self):
        """将组合的变化规律由计数转变为概率"""
        self.transfer = np.zeros(self.transfer_count.shape)
        self.emission = np.zeros(self.emission_count.shape)
        self.pi = np.zeros(self.pi_count.shape)
        # 1.计算转移矩阵
        for row_it, row in enumerate(self.transfer_count):
            row_sum = sum(row)
            if row_sum == 0:
                self.transfer[row_it] = [1 / self.transfer.shape[1] for t in range(self.transfer.shape[1])]  # 如果数据集中这个状态没有接任何一个下一个状态，则概率均分
            else:
                self.transfer[row_it] = self.transfer_count[row_it] / row_sum
        # 2.计算观测矩阵
        for column_it in range(self.emission.shape[1]):
            column_sum = sum(self.emission_count[:, column_it])
            if column_sum == 0:
                self.emission[:, column_it] = [1 / self.emission.shape[0] for t in range(self.emission.shape[0])]  # 为空列则概率均分
            else:
                self.emission[:, column_it] = self.emission_count[:, column_it] / column_sum
        # 3.计算初始化转移向量
        self.pi = self.pi_count / sum(self.pi_count)


def melody_note_div_12(note_list):
    """
    将一个音符列表全部-n*12 但不能出现负数
    :param note_list: 音符列表(绝对音高形式)
    :return: 处理之后的音符列表
    """
    min_note = 999  # 音高最低的音符
    output_notelist = []
    for note in note_list:
        if note <= min_note and note != 0:
            min_note = note
    minus_note = 12 * ((min_note - 1) // 12)
    for note in note_list:
        if note == 0:
            output_notelist.append(0)
        else:
            output_notelist.append(note - minus_note)
    return output_notelist
