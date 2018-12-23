from settings import *
from interfaces.chord_parse import get_chord_root_pitch
from interfaces.music_patterns import BaseMusicPatterns
from interfaces.sql.sqlite import get_raw_song_data_from_dataset, get_tone_list
from interfaces.utils import flat_array, DiaryLog
import numpy as np
import os


def get_root_data_1song(chord_data, expect_root_base):
    """
    根据一首歌的和弦获取其根音 根音以拍为单位
    :param expect_root_base: 预期的根音平均值
    :param chord_data: 一首歌的和弦
    :return: 这首歌的根音
    """
    root_data = [0 for t in range(len(chord_data))]
    step_dx = 0
    while True:
        # 情况1：这首歌的根音已经寻找完毕 直接退出
        if step_dx >= len(chord_data):
            break
        # 情况2：处于一个小节的开端 且发现这个小节没有和弦 则认为这个小节也没有根音
        if step_dx % 4 == 0 and len(chord_data) - step_dx >= 4 and chord_data[step_dx: step_dx + 4] == [0, 0, 0, 0]:
            step_dx += 4
            continue
        # 情况3：正常拍 根据当前拍的和弦和上一拍的根音来决定这一拍的根音（第一拍的“上一拍根音”为零）
        if step_dx == 0 or root_data[step_dx - 1] == 0:
            root_data[step_dx] = get_chord_root_pitch(chord_data[step_dx], 0, expect_root_base)
        else:
            root_data[step_dx] = get_chord_root_pitch(chord_data[step_dx], root_data[step_dx - 1], expect_root_base)
        step_dx += 1

    return root_data


def get_root_chord_pattern(chord_data, root_data):
    """
    获取并编码和弦和根音的组合列表。如[40,1]代表根音为40,和弦编号为1
    :param root_data: 所有歌曲的根音列表
    :param chord_data: 和弦列表
    :return: 所有歌曲的根音-和弦组合，及其对照表
    """
    rc_pat_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 训练集中所有歌曲的根音－和弦组合
    all_rc_pats = [[-1]]  # 根音－和弦组合的统计结果
    rc_pat_count = [0]  # 各种根音－和弦组合的计数
    for song_it in range(TRAIN_FILE_NUMBERS):
        if chord_data[song_it]:
            # 1.找到一首歌的和弦长度，并初始化和弦-根音组合数据
            chord_len = len(chord_data[song_it])  # 这首歌的和弦的长度
            rc_pat_data[song_it] = [0 for t in range(chord_len)]  # 先全部初始化为0
            # 2.遍历所有的节拍，填写所有的和弦-根音组合
            for step_it in range(chord_len):
                if chord_data[song_it][step_it] != 0:  # 如果这个时间步长中和弦是0的话，就不用向下执行了
                    rc_pattern = [root_data[song_it][step_it], chord_data[song_it][step_it]]
                    try:
                        rc_pattern_dx = all_rc_pats.index(rc_pattern)  # 检查这个和弦-根音组合有没有被保存
                        rc_pat_count[rc_pattern_dx] += 1
                    except ValueError:
                        all_rc_pats.append(rc_pattern)  # 添加这个根音－和弦 组合
                        rc_pattern_dx = len(all_rc_pats) - 1
                        rc_pat_count.append(1)
                    rc_pat_data[song_it][step_it] = rc_pattern_dx  # 将这个音符保存起来
    return rc_pat_data, all_rc_pats, rc_pat_count


def get_chord_chord_pattern(chord_data):
    """
    获取并编码和弦-和弦的组合列表。如[1,1]代表第一拍和第二拍的和弦编号都为1
    :param chord_data: 所有歌曲的和弦列表
    :return: 所有歌曲的和弦-和弦组合，及其对照表
    """
    cc_pat_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 训练集中所有歌曲的和弦－和弦组合
    all_cc_pats = [[-1]]  # 和弦－和弦组合的统计结果
    cc_pat_count = [0]  # 各种和弦－和弦组合的计数
    for song_it in range(TRAIN_FILE_NUMBERS):
        if chord_data[song_it]:
            # 1.找到一首歌的和弦长度，并初始化和弦-和弦组合数据
            chord_len = len(chord_data[song_it])  # 这首歌的和弦的长度
            cc_pat_data[song_it] = [0 for t in range(chord_len // 2)]  # 先全部初始化为0
            # 2.逐两拍填写所有的和弦-根音组合
            for step_it in range(0, chord_len, 2):
                # 2.1.如果这两拍的和弦至少一个不为0的话，得到这两拍的和弦-和弦组合。否则跳过这两拍，组合为零。
                if chord_data[song_it][step_it] == 0 and chord_data[song_it][step_it + 1] == 0:
                    continue
                elif chord_data[song_it][step_it] != 0 and chord_data[song_it][step_it + 1] != 0:  # 两拍和弦都不是零的情况
                    cc_pattern = [chord_data[song_it][step_it], chord_data[song_it][step_it + 1]]
                elif chord_data[song_it][step_it] == 0:  # 第一拍和弦为零 用两个第二拍和弦替代
                    cc_pattern = [chord_data[song_it][step_it + 1], chord_data[song_it][step_it + 1]]
                else:  # 第二拍和弦为零 用两个第一拍和弦替代
                    cc_pattern = [chord_data[song_it][step_it], chord_data[song_it][step_it]]
                # 2.2.填写到pat_data和all_cc_pat变量中
                try:
                    cc_pattern_dx = all_cc_pats.index(cc_pattern)  # 检查这个和弦组合有没有被保存
                    cc_pat_count[cc_pattern_dx] += 1
                except ValueError:
                    all_cc_pats.append(cc_pattern)  # 添加这个根音－和弦 组合
                    cc_pattern_dx = len(all_cc_pats) - 1
                    cc_pat_count.append(1)
                cc_pat_data[song_it][step_it // 2] = cc_pattern_dx  # 将这个音符保存起来
    return cc_pat_data, all_cc_pats, cc_pat_count


class ChordTrainData:

    def __init__(self, melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        self.transfer_count = np.zeros([COMMON_CORE_NOTE_PAT_NUM * 2 + 2, len(CHORD_LIST) + 1], dtype=np.float32) + np.e ** (-3)  # 主旋律/调式与同时期和弦的转移矩阵 0是空主旋律 1-400是大调对应的主旋律 401-800是小调对应的主旋律 801对应的是罕见主旋律
        self.real_transfer_count = np.zeros([COMMON_CORE_NOTE_PAT_NUM * 2 + 2, len(CHORD_LIST) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦的转移矩阵 与上个变量的区别是不添加e**(-3)
        self.transfer = np.zeros([COMMON_CORE_NOTE_PAT_NUM * 2 + 2, len(CHORD_LIST) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦 概率取对数后的转移矩阵 这个转移矩阵的数字精度必须是float32的

        # 1.从数据集中读取所有歌曲的和弦数据，并变更为以音符步长为单位的列表
        self.chord_data = get_raw_song_data_from_dataset('chord', None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.chord_data[song_it] != dict():
                self.chord_data[song_it] = flat_array(self.chord_data[song_it])
            else:
                self.chord_data[song_it] = []  # 对于没有和弦的歌曲，将格式转化为list格式

        # 2.和弦数据以1拍为单位存储，但在训练时以2拍为单位。因此逐两拍生成和弦-和弦组合
        cc_pat_data, self.all_cc_pats, cc_pat_count = get_chord_chord_pattern(self.chord_data)  # 和弦-和弦组合
        self.cc_pat_num = len(self.all_cc_pats)
        cc_pattern_cls = BaseMusicPatterns()
        cc_pattern_cls.common_pattern_list = self.all_cc_pats
        cc_pattern_cls.pattern_number_list = cc_pat_count
        cc_pattern_cls.store('ChordChord')

        # 3.获取并保存主旋律与同时期和弦的状态转移矩阵
        # 3.1.获取和弦的状态转移矩阵
        tone_list = get_tone_list()  # 获取训练用曲的旋律列表
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.chord_data[song_it] and melody_pat_data[song_it]:
                self.freq_count(self.chord_data[song_it], core_note_pat_data[song_it], tone_list[song_it])
        # 3.2.保存和弦的状态转移矩阵
        np.save(os.path.join(PATH_PATTERNLOG, 'ChordTransferCount.npy'), self.transfer_count)
        np.save(os.path.join(PATH_PATTERNLOG, 'ChordTransferCountReal.npy'), self.real_transfer_count)

        # 4.生成输入输出数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.chord_data[song_it] and melody_pat_data[song_it]:
                self.get_model_io_data(cc_pat_data[song_it], melody_pat_data[song_it], continuous_bar_data[song_it])

        DiaryLog.warn('Generation of chord train data has finished!')

    def get_model_io_data(self, cc_pat_data, melody_pat_data, continuous_bar_data):
        """
        输入内容为当前时间的编码 过去两小节加一个步长的主旋律 过去两小节加一个步长的的和弦-和弦组合
        :param cc_pat_data: 一首歌的和弦-和弦组合数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :return:
        """
        melody1_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
        melody2_code_add_base = 4 + COMMON_MELODY_PAT_NUM + 2  # 主旋律第二拍数据编码增加的基
        chord_code_add_base = 4 + (COMMON_MELODY_PAT_NUM + 2) * 2  # 和弦数据编码增加的基数

        for step_it in range(-2 * TRAIN_CHORD_IO_BARS, len(cc_pat_data) - 2 * TRAIN_CHORD_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []

                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                beat_in_bar = (step_it % 2) * 2  # 小节内的第几拍

                # 2.input_data: 添加过去两小节加一个步长的主旋律 过去两小节加一个步长的的和弦-和弦组合
                for ahead_step_it in range(step_it, step_it + 2 * TRAIN_CHORD_IO_BARS + 1):  # 向前看10拍
                    # 2.1.添加过去两小节加一个步长的主旋律
                    if ahead_step_it >= 0:
                        ahead_bar_dx = ahead_step_it // 2
                        ahead_beat_in_bar = (ahead_step_it % 2) * 2
                        time_add = 0 if continuous_bar_data[ahead_bar_dx] == 0 else (1 - continuous_bar_data[ahead_bar_dx] % 2) * 2
                        input_time_data.append([time_add + ahead_beat_in_bar // 2, melody_pat_data[ahead_step_it * 2] + melody1_code_add_base, melody_pat_data[ahead_step_it * 2 + 1] + melody2_code_add_base])
                    else:
                        time_add = 0
                        ahead_beat_in_bar = (ahead_step_it % 2) * 2
                        input_time_data.append([time_add + ahead_beat_in_bar // 2, melody1_code_add_base, melody2_code_add_base])
                    # 2.2.添加过去两小节加一个步长的的和弦-和弦组合
                    if ahead_step_it >= 1:
                        input_time_data[-1].append(cc_pat_data[ahead_step_it - 1] + chord_code_add_base)
                    else:
                        input_time_data[-1].append(chord_code_add_base)
                for bar_it in range(cur_bar, cur_bar + TRAIN_CHORD_IO_BARS):
                    step_dx = (4 - beat_in_bar) // 2 + (bar_it - cur_bar) * 2  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it * 4: (bar_it * 4) + 4] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], melody1_code_add_base, melody2_code_add_base, chord_code_add_base] for t in range(step_dx)]

                # 3.output_data: 添加过去10拍的和弦-和弦组合 一共5个
                # 3.1.第一个小节
                if cur_bar < 0 or melody_pat_data[cur_bar * 4: (cur_bar + 1) * 4] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的和弦-和弦组合也置为空
                    output_time_data.extend([chord_code_add_base for t in range(2 - pat_step_in_bar)])
                else:
                    output_time_data.extend([t + chord_code_add_base for t in cc_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2]])
                # 3.2.中间小节
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_CHORD_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的和弦-和弦组合置为空
                        output_time_data.extend([chord_code_add_base, chord_code_add_base])
                    else:
                        output_time_data.extend([t + chord_code_add_base for t in cc_pat_data[bar_it * 2: bar_it * 2 + 2]])
                    if bar_it < 0 or melody_pat_data[bar_it * 4: (bar_it * 4) + 4] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [chord_code_add_base for t in range(len(output_time_data))]
                # 3.3.最后一个小节
                output_time_data.extend([t + chord_code_add_base for t in cc_pat_data[(cur_bar + TRAIN_CHORD_IO_BARS) * 2: (cur_bar + TRAIN_CHORD_IO_BARS) * 2 + pat_step_in_bar + 1]])  # 最后一个小节的和弦数据

                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的和弦-和弦组合不为空
                if melody_pat_data[(cur_bar + TRAIN_CHORD_IO_BARS) * 4: (cur_bar + TRAIN_CHORD_IO_BARS + 1) * 4] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_chord_data = cc_pat_data[(cur_bar + TRAIN_CHORD_IO_BARS) * 2: (cur_bar + TRAIN_CHORD_IO_BARS + 1) * 2]
                    if final_bar_chord_data != [0, 0]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except IndexError:
                pass

    def get_root_data(self, expect_root_base):
        """
        获取根音及和弦-根音组合（给其他的音轨训练用的）
        :param expect_root_base: 预期的根音均值
        :return: 根音列表 根音-和弦列表 根音-和弦对照表
        """
        root_data = [[] for t in range(TRAIN_FILE_NUMBERS)]
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.chord_data[song_it]:
                root_data[song_it] = get_root_data_1song(self.chord_data[song_it], expect_root_base)  # 根音和组合列表都是以拍为单位，而不是以小节为单位
        rc_pat_data, all_rc_pats, rc_pat_count = get_root_chord_pattern(self.chord_data, root_data)
        return root_data, rc_pat_data, all_rc_pats, rc_pat_count

    def freq_count(self, chord_data, core_note_pat_data, tone):
        """确定主旋律/调式与同时期和弦之间的转移频数"""
        # 不考虑空主旋律/不确定和弦/异常的主旋律
        for core_note_step_it in range(len(core_note_pat_data)):
            if core_note_pat_data[core_note_step_it] not in [0, COMMON_CORE_NOTE_PAT_NUM + 1]:
                if tone == DEF_TONE_MAJOR:
                    core_note_dx = core_note_pat_data[core_note_step_it]
                elif tone == DEF_TONE_MINOR:
                    core_note_dx = core_note_pat_data[core_note_step_it] + COMMON_CORE_NOTE_PAT_NUM
                else:
                    raise ValueError
                if core_note_step_it * 2 < len(chord_data):
                    if chord_data[core_note_step_it * 2] != 0:
                        self.transfer_count[core_note_dx][chord_data[core_note_step_it * 2]] += 1
                        self.real_transfer_count[core_note_dx][chord_data[core_note_step_it * 2]] += 1
                    if chord_data[core_note_step_it * 2 + 1] != 0:
                        self.transfer_count[core_note_dx][chord_data[core_note_step_it * 2 + 1]] += 1
                        self.real_transfer_count[core_note_dx][chord_data[core_note_step_it * 2 + 1]] += 1


class ChordTestData:

    def __init__(self):
        # 1.获取和弦的根音组合
        cc_pattern_cls = BaseMusicPatterns()
        cc_pattern_cls.restore('ChordChord')
        self.all_cc_pats = cc_pattern_cls.common_pattern_list
        self.cc_pat_num = len(self.all_cc_pats)

        # 2.获取主旋律与同时期和弦的状态转移矩阵
        self.transfer_count = np.load(os.path.join(PATH_PATTERNLOG, 'ChordTransferCount.npy'))
        self.real_transfer_count = np.load(os.path.join(PATH_PATTERNLOG, 'ChordTransferCountReal.npy'))
        assert self.transfer_count.shape[0] == COMMON_CORE_NOTE_PAT_NUM * 2 + 2
        assert self.transfer_count.shape[1] == len(CHORD_LIST) + 1
        assert self.real_transfer_count.shape[0] == COMMON_CORE_NOTE_PAT_NUM * 2 + 2
        assert self.real_transfer_count.shape[1] == len(CHORD_LIST) + 1

        DiaryLog.warn('Restoring of chord associated data has finished!')
