from settings import *
from interfaces.chord_parse import chord_rootnote
from interfaces.utils import get_dict_max_key
from interfaces.sql.sqlite import get_raw_song_data_from_dataset
from interfaces.music_patterns import BaseMusicPatterns
import numpy as np
from interfaces.sql.sqlite import get_tone_list


def get_root_data_1song(chord_data, expect_root_base):
    """
    根据一首歌的和弦获取其根音 根音以拍为单位
    :param expect_root_base: 预期的根音平均值
    :param chord_data: 一首歌的和弦
    :return: 这首歌的根音
    """
    root_data = []
    for bar_it in range(get_dict_max_key(chord_data) + 1):
        if chord_data[bar_it] == [0, 0, 0, 0]:  # 如果这个小节没有和弦 那么也同样没有根音
            root_data.extend([0, 0, 0, 0])
            continue
        for chord_it in range(4):
            if chord_it == 0:
                if bar_it == 0 or root_data[bar_it - 1] == 0:
                    root_data.append(chord_rootnote(chord_data[bar_it][chord_it], 0, expect_root_base))
                else:
                    root_data.append(chord_rootnote(chord_data[bar_it][chord_it], root_data[-1], expect_root_base))
            else:
                # print(chord_iterator, root_data[song_iterator][bar_iterator], chord_data[song_iterator][bar_iterator])
                root_data.append(chord_rootnote(chord_data[bar_it][chord_it], root_data[-1], expect_root_base))
    return root_data


def get_root_chord_pattern(chord_data, root_data):
    """
    获取并编码和弦和根音的组合列表。如[40,1]代表根音为40,和弦编号为1
    :param root_data: 所有歌曲的根音列表
    :param chord_data: 和弦列表
    :return: 所有歌曲的根音-和弦组合，及其对照表
    """
    rc_pattern_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 训练集中所有歌曲的根音－和弦组合
    rc_pattern_count = [[-1]]  # 根音－和弦组合的统计结果
    for song_it in range(TRAIN_FILE_NUMBERS):
        for bar_it in range(get_dict_max_key(chord_data[song_it]) + 1):
            rc_pattern_data[song_it].extend([0, 0, 0, 0])  # 先全部初始化为0
            for step_it in range(4):
                if chord_data[song_it][bar_it][step_it] != 0:  # 如果这个时间步长中和弦是0的话，就不用向下执行了
                    rc_pattern = [root_data[song_it][bar_it * 4 + step_it], chord_data[song_it][bar_it][step_it]]
                    try:
                        rc_pattern_index = rc_pattern_count.index(rc_pattern)  # 检查这个音符组合有没有被保存
                    except ValueError:
                        rc_pattern_count.append(rc_pattern)  # 添加这个根音－和弦 组合
                        rc_pattern_index = len(rc_pattern_count) - 1
                    rc_pattern_data[song_it][bar_it * 4 + step_it] = rc_pattern_index  # 将这个音符保存起来
    return rc_pattern_data, rc_pattern_count


def get_chord_chord_pattern(chord_data):
    """
    获取并编码和弦-和弦的组合列表。如[1,1]代表第一拍和第二拍的和弦编号都为1
    :param chord_data: 和弦列表
    :return: 所有歌曲的和弦-和弦组合，及其对照表
    """
    cc_pattern_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 训练集中所有歌曲的和弦－和弦组合
    cc_pattern_count = [[-1]]  # 和弦－和弦组合的统计结果
    for song_it in range(TRAIN_FILE_NUMBERS):
        for bar_it in range(get_dict_max_key(chord_data[song_it]) + 1):
            cc_pattern_data[song_it].extend([0, 0])  # 先全部初始化为0
            for step_dx in range(0, 4, 2):
                if chord_data[song_it][bar_it][step_dx] != 0 or chord_data[song_it][bar_it][step_dx + 1] != 0:  # 如果这两拍的和弦都是0的话，就不用向下执行了
                    if chord_data[song_it][bar_it][step_dx] != 0 and chord_data[song_it][bar_it][step_dx + 1] != 0:  # 两拍和弦都不是零的情况
                        cc_pattern = [chord_data[song_it][bar_it][step_dx], chord_data[song_it][bar_it][step_dx + 1]]
                    elif chord_data[song_it][bar_it][step_dx] == 0:  # 第一拍和弦为零 用两个第二拍和弦替代
                        cc_pattern = [chord_data[song_it][bar_it][step_dx + 1], chord_data[song_it][bar_it][step_dx + 1]]
                    else:  # 第二拍和弦为零 用两个第一拍和弦替代
                        cc_pattern = [chord_data[song_it][bar_it][step_dx], chord_data[song_it][bar_it][step_dx]]
                    try:
                        cc_pattern_index = cc_pattern_count.index(cc_pattern)  # 检查这个和弦组合有没有被保存
                    except ValueError:
                        cc_pattern_count.append(cc_pattern)  # 添加这个根音－和弦 组合
                        cc_pattern_index = len(cc_pattern_count) - 1
                    cc_pattern_data[song_it][bar_it * 2 + step_dx // 2] = cc_pattern_index  # 将这个音符保存起来
    return cc_pattern_data, cc_pattern_count


class ChordTrainData:

    def __init__(self, melody_pat_data, continuous_bar_data):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取歌的和弦数据
        self.chord_data = get_raw_song_data_from_dataset('chord', None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
        # 2.生成输入输出数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.chord_data[song_it] != {} and melody_pat_data[song_it] != {}:
                # 2.1.开头补上几个空小节 便于训练开头的数据
                for bar_it in range(-TRAIN_CHORD_IO_BARS, 0):
                    self.chord_data[song_it][bar_it] = [0 for t in range(4)]
                    melody_pat_data[song_it][bar_it] = [0 for t in range(4)]
                # 2.2.生成训练数据 输入内容是这两小节的主旋律和和弦 输出内容是这两拍的和弦
                self.get_model_io_data(self.chord_data[song_it], melody_pat_data[song_it], continuous_bar_data[song_it])

    def get_root_data(self, expect_root_base):
        """
        获取根音及和弦-根音组合
        :param expect_root_base: 预期的根音均值
        :return: 根音列表 根音-和弦列表 根音-和弦对照表
        """
        root_data = [{} for t in range(TRAIN_FILE_NUMBERS)]
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.chord_data[song_it] != {}:
                root_data[song_it] = get_root_data_1song(self.chord_data[song_it], expect_root_base)  # 根音和组合列表都是以拍为单位，而不是以小节为单位
        rc_pattern_data, rc_pattern_count = get_root_chord_pattern(self.chord_data, root_data)
        return root_data, rc_pattern_data, rc_pattern_count

    def get_model_io_data(self, chord_data, melody_pat_data, continuous_bar_data):
        """
        在完成数据的前期处理（读取/转换等）之后，接下来就是提取有效数据输入到model中了
        :param chord_data: 一首歌的和弦数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :return:
        """
        for key in chord_data:
            for time_in_bar in range(2):  # 和弦训练的步长为2拍
                try:
                    # 1.添加当前时间的编码（0-3）
                    time_add = (1 - continuous_bar_data[key + TRAIN_CHORD_IO_BARS] % 2) * 2
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 2.添加最近2小节多一个chord_generate_time_step的主旋律
                    input_time_data = input_time_data + melody_pat_data[key][time_in_bar * 2:]
                    output_time_data = output_time_data + melody_pat_data[key][time_in_bar * 2:]
                    for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS):
                        input_time_data = input_time_data + melody_pat_data[bar_it]
                        output_time_data = output_time_data + melody_pat_data[bar_it]
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = [input_time_data[0]] + [0 for t in range(len(input_time_data) - 1)]
                            output_time_data = [output_time_data[0]] + [0 for t in range(len(output_time_data) - 1)]
                    input_time_data = input_time_data + melody_pat_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]
                    output_time_data = output_time_data + melody_pat_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]
                    # 3.添加过去2小节的和弦
                    if melody_pat_data[key] == [0 for t in range(4)]:
                        input_time_data = input_time_data + [0 for t in range(4 - 2 * time_in_bar)]
                        output_time_data = output_time_data + [0 for t in range(2 - 2 * time_in_bar)]
                    else:
                        input_time_data = input_time_data + chord_data[key][2 * time_in_bar:]
                        output_time_data = output_time_data + chord_data[key][2 * (1 + time_in_bar):]
                    for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS):
                        input_time_data = input_time_data + chord_data[bar_it]
                        output_time_data = output_time_data + chord_data[bar_it]
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[:11] + [0 for t in range(len(input_time_data) - 11)]
                            output_time_data = output_time_data[:11] + [0 for t in range(len(output_time_data) - 11)]
                    input_time_data = input_time_data + chord_data[key + TRAIN_CHORD_IO_BARS][:2 * time_in_bar]
                    output_time_data = output_time_data + chord_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]
                    # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个步长的和弦不为空
                    if melody_pat_data[key + TRAIN_CHORD_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                        output_bar_data_temp = chord_data[key + TRAIN_CHORD_IO_BARS]
                        if output_bar_data_temp != [0 for t in range(4)]:
                            self.input_data.append(input_time_data)
                            self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass


class ChordTrainDataCheck(ChordTrainData):

    def __init__(self, melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data):

        super().__init__(melody_pat_data, continuous_bar_data)
        self.check_melody_data = []  # 用于check的molody组合数据(共16拍)
        self.check_raw_melody_data = []  # 用于check的molody原始数据(共16拍)
        self.check_chord_input_data = []  # 用于check的chord输入数据(共8拍)
        self.check_chord_output_data = []  # 用于check的chord输出数据(共8拍)
        self.time_add_data = []  # 用于check的时间编码数据

        self.transfer_count = np.zeros([COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2, len(CHORD_DICT) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦的转移矩阵 0是空主旋律 1-400是大调对应的主旋律 401-800是小调对应的主旋律 801对应的是罕见主旋律
        self.real_transfer_count = np.zeros([COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2, len(CHORD_DICT) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦的转移矩阵 与上个变量的区别是不添加e**(-3)
        self.transfer_count += np.e ** (-3)
        self.transfer = np.zeros([COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2, len(CHORD_DICT) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦 概率取对数后的转移矩阵 这个转移矩阵的数字精度必须是float32的
        # self.confidence_level = 0  # 连续四步预测的90%置信水平
        # 1.获取用于验证的数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.chord_data[song_it] != {} and melody_pat_data[song_it] != {}:
                self.get_check_io_data(self.chord_data[song_it], melody_pat_data[song_it], raw_melody_data[song_it], continuous_bar_data[song_it])
        # print(len(self.check_melody_data))
        # 2.获取训练用曲的旋律列表
        tone_list = get_tone_list()
        # 3.获取主旋律与同时期和弦的转移矩阵
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.chord_data[song_it] != {} and melody_pat_data[song_it] != {}:
                self.count(self.chord_data[song_it], core_note_pat_data[song_it], tone_list[song_it])

    def get_check_io_data(self, chord_data, melody_pat_data, raw_melody_data, continuous_bar_data):
        """生成一首歌校验所需的数据"""
        for key in chord_data:
            for time_in_bar in range(2):  # 和弦训练的步长为2拍
                try:
                    flag_drop = False  # 这组数据是否忽略
                    melody_input_time_data = []
                    raw_melody_time_data = []
                    chord_input_time_data = []
                    chord_output_time_data = []
                    # 1.添加当前时间的编码（0-3）
                    time_add = (1 - continuous_bar_data[key + TRAIN_CHORD_IO_BARS] % 2) * 2
                    # 2.添加最近4小节(TRAIN_CHORD_IO_BARS+2)的主旋律(包括原始的旋律和旋律组合)
                    melody_input_time_data = melody_input_time_data + melody_pat_data[key][time_in_bar * 2:]
                    raw_melody_time_data = raw_melody_time_data + raw_melody_data[key][time_in_bar * 16:]
                    for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS + 2):
                        melody_input_time_data = melody_input_time_data + melody_pat_data[bar_it]
                        raw_melody_time_data = raw_melody_time_data + raw_melody_data[bar_it]
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节
                            if bar_it >= key + TRAIN_CHORD_IO_BARS:  # 位于输入和弦与输出和弦分界处之后出现了空小节 则直接忽略这组数据
                                flag_drop = True
                                break
                            else:
                                melody_input_time_data = [0 for t in range(len(melody_input_time_data))]
                                raw_melody_time_data = [0 for t in range(len(raw_melody_time_data))]
                    if flag_drop is True:
                        continue
                    melody_input_time_data = melody_input_time_data + melody_pat_data[key + TRAIN_CHORD_IO_BARS + 2][:2 * time_in_bar]
                    raw_melody_time_data = raw_melody_time_data + raw_melody_data[key + TRAIN_CHORD_IO_BARS + 2][:16 * time_in_bar]
                    # 3.添加过去2小节的和弦
                    if melody_pat_data[key] == [0 for t in range(4)]:
                        chord_input_time_data = chord_input_time_data + [0 for t in range(4 - 2 * time_in_bar)]
                    else:
                        chord_input_time_data = chord_input_time_data + chord_data[key][2 * time_in_bar:]
                    for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS):
                        chord_input_time_data = chord_input_time_data + chord_data[bar_it]
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            chord_input_time_data = [0 for t in range(len(chord_input_time_data))]
                    chord_input_time_data = chord_input_time_data + chord_data[key + TRAIN_CHORD_IO_BARS][:2 * time_in_bar]
                    # 4.添加之后2小节的和弦
                    chord_output_time_data = chord_output_time_data + chord_data[key + TRAIN_CHORD_IO_BARS][2 * time_in_bar:]
                    chord_output_time_data = chord_output_time_data + chord_data[key + TRAIN_CHORD_IO_BARS + 1]
                    chord_output_time_data = chord_output_time_data + chord_data[key + TRAIN_CHORD_IO_BARS + 2][:2 * time_in_bar]
                    for bar_it in range(3):  # 检查该项数据是否可以用于训练。条件是最后三个小节的和弦均不为空
                        if chord_data[key + TRAIN_CHORD_IO_BARS + bar_it] == [0 for t in range(4)]:
                            flag_drop = True
                    if flag_drop is True:
                        continue
                    # 5.将这项数据添加进校验集中
                    self.check_melody_data.append(melody_input_time_data)
                    self.check_raw_melody_data.append(raw_melody_time_data)
                    self.check_chord_input_data.append(chord_input_time_data)
                    self.check_chord_output_data.append(chord_output_time_data)
                    self.time_add_data.append(time_add)
                except KeyError:
                    pass
                except IndexError:
                    pass

    def count(self, chord_data, core_note_pat_data, tone):
        """确定主旋律/调式与同时期和弦之间的转移频数"""
        # 不考虑空主旋律/不确定和弦/异常的主旋律
        for core_note_step_it in range(len(core_note_pat_data)):
            if core_note_pat_data[core_note_step_it] not in [0, COMMON_CORE_NOTE_PATTERN_NUMBER + 1]:
                bar_dx = core_note_step_it // 2
                step_in_bar = core_note_step_it - bar_dx * 2  # 这个core_note_step在小节中的位置及其小节数
                if tone == TONE_MAJOR:
                    core_note_dx = core_note_pat_data[core_note_step_it]
                elif tone == TONE_MINOR:
                    core_note_dx = core_note_pat_data[core_note_step_it] + COMMON_CORE_NOTE_PATTERN_NUMBER
                else:
                    raise ValueError
                if bar_dx in chord_data:
                    if chord_data[bar_dx][step_in_bar * 2] != 0:
                        self.transfer_count[core_note_dx][chord_data[bar_dx][step_in_bar * 2]] += 1
                        self.real_transfer_count[core_note_dx][chord_data[bar_dx][step_in_bar * 2]] += 1
                    if chord_data[bar_dx][step_in_bar * 2 + 1] != 0:
                        self.transfer_count[core_note_dx][chord_data[bar_dx][step_in_bar * 2 + 1]] += 1
                        self.real_transfer_count[core_note_dx][chord_data[bar_dx][step_in_bar * 2 + 1]] += 1

    def prob_convert(self):
        """将频率转化为概率 将频率归一化之后进行反softmax变换"""
        for core_note_pat_dx in range(0, COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2):
            self.transfer[core_note_pat_dx, :] = self.transfer_count[core_note_pat_dx, :] / sum(self.transfer_count[core_note_pat_dx, :])
            self.transfer[core_note_pat_dx, :] = np.log(self.transfer[core_note_pat_dx, :])


class ChordTrainData2(ChordTrainDataCheck):

    def __init__(self, melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data):
        # self.onehot_input_data = []  # 输入model的数据(one_hot的)
        # self.onehot_output_data = []  # 从model输出的数据(one_hot的)
        super().__init__(melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data)

    def get_model_io_data(self, chord_data, melody_pat_data, continuous_bar_data):
        """
        在完成数据的前期处理（读取/转换等）之后，接下来就是提取有效数据输入到model中了
        :param chord_data: 一首歌的和弦数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :return:
        """
        for key in chord_data:
            for time_in_bar in range(2):  # 和弦训练的步长为2拍
                try:
                    # 1.添加当前时间的编码（0-3）
                    time_add = (1 - continuous_bar_data[key + TRAIN_CHORD_IO_BARS] % 2) * 2
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 2.添加最近2小节多一个chord_generate_time_step的主旋律
                    input_time_data.extend([t + 4 for t in melody_pat_data[key][2 * (1 + time_in_bar):]])
                    for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS):
                        input_time_data.extend([t + 4 for t in melody_pat_data[bar_it]])
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = [input_time_data[0]] + [4 for t in range(len(input_time_data) - 1)]  # 这里0-3用作时间编码 所以这里从0变更为4
                    input_time_data.extend([t + 4 for t in melody_pat_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]])
                    # 3.添加过去2小节的和弦
                    code_add_base = 4 + COMMON_MELODY_PATTERN_NUMBER + 2  # 和弦数据编码增加的基数
                    if melody_pat_data[key] == [0 for t in range(4)]:
                        output_time_data.extend([code_add_base for t in range(2 - 2 * time_in_bar)])
                    else:
                        output_time_data.extend([t + code_add_base for t in chord_data[key][2 * (1 + time_in_bar):]])
                    for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS):
                        output_time_data.extend([t + code_add_base for t in chord_data[bar_it]])
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            output_time_data = [output_time_data[0]] + [code_add_base for t in range(len(output_time_data) - 1)]
                    output_time_data.extend([t + code_add_base for t in chord_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]])
                    # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个步长的和弦不为空
                    if melody_pat_data[key + TRAIN_CHORD_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                        output_bar_data_temp = chord_data[key + TRAIN_CHORD_IO_BARS]
                        if output_bar_data_temp != [0 for t in range(4)]:
                            self.input_data.append(input_time_data)
                            self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass


class ChordTrainData3(ChordTrainDataCheck):

    def __init__(self, melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data):
        super().__init__(melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data)

    def get_model_io_data(self, chord_data, melody_pat_data, continuous_bar_data):
        """
        在完成数据的前期处理（读取/转换等）之后，接下来就是提取有效数据输入到model中了
        :param chord_data: 一首歌的和弦数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :return:
        """
        for key in chord_data:
            for time_in_bar in range(2):  # 和弦训练的步长为2拍
                try:
                    # 1.添加当前时间的编码（0-3）
                    input_time_data = []
                    output_time_data = []
                    # 2.添加当前时间的编码最近2小节多一个chord_generate_time_step的主旋律 两者保存在一起
                    melody_code_add_base = 8
                    time_add = 0 if continuous_bar_data[key] == 0 else (1 - continuous_bar_data[key] % 2) * 4
                    for beat_it in range(2 * (1 + time_in_bar), 4):
                        input_time_data.append([time_add + beat_it, melody_pat_data[key][beat_it] + melody_code_add_base])
                    for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS):
                        time_add = 0 if continuous_bar_data[bar_it] == 0 else (1 - continuous_bar_data[bar_it] % 2) * 4
                        for beat_it in range(0, 4):
                            input_time_data.append([time_add + beat_it, melody_pat_data[bar_it][beat_it] + melody_code_add_base])
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = [[input_time_data[t][0], melody_code_add_base] for t in range(len(input_time_data))]  # 这里0-7用作时间编码 所以空的主旋律从0变更为8
                    time_add = 0 if continuous_bar_data[key + TRAIN_CHORD_IO_BARS] == 0 else (1 - continuous_bar_data[key + TRAIN_CHORD_IO_BARS] % 2) * 4
                    for beat_it in range(0, 2 * (1 + time_in_bar)):
                        input_time_data.append([time_add + beat_it, melody_pat_data[key + TRAIN_CHORD_IO_BARS][beat_it] + melody_code_add_base])
                    # 3.添加过去2小节的和弦
                    chord_code_add_base = 8 + COMMON_MELODY_PATTERN_NUMBER + 2  # 和弦数据编码增加的基数
                    if melody_pat_data[key] == [0 for t in range(4)]:
                        output_time_data.extend([chord_code_add_base for t in range(2 - 2 * time_in_bar)])
                    else:
                        output_time_data.extend([t + chord_code_add_base for t in chord_data[key][2 * (1 + time_in_bar):]])
                    for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS):
                        output_time_data.extend([t + chord_code_add_base for t in chord_data[bar_it]])
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            output_time_data = [chord_code_add_base for t in range(len(output_time_data))]
                    output_time_data.extend([t + chord_code_add_base for t in chord_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]])
                    # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个步长的和弦不为空
                    if melody_pat_data[key + TRAIN_CHORD_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                        output_bar_data_temp = chord_data[key + TRAIN_CHORD_IO_BARS]
                        if output_bar_data_temp != [0 for t in range(4)]:
                            self.input_data.append(input_time_data)
                            self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass


class ChordTrainData4(ChordTrainDataCheck):

    def __init__(self, melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data):
        super().__init__(melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data)
        cc_pat_data, self.cc_pat_count = get_chord_chord_pattern(self.chord_data)  # 和弦-和弦组合
        self.cc_pat_num = len(self.cc_pat_count)

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 2.生成输入输出数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.chord_data[song_it] != {} and melody_pat_data[song_it] != {}:
                self.get_model_io_data_4(cc_pat_data[song_it], melody_pat_data[song_it], continuous_bar_data[song_it])

    def get_model_io_data_4(self, cc_pat_data, melody_pat_data, continuous_bar_data):
        """
        输入内容为当前时间的编码 过去两小节加一个步长的主旋律 过去两小节加一个步长的的和弦-和弦组合
        :param cc_pat_data: 一首歌的和弦-和弦组合数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :return:
        """
        for step_it in range(-2 * TRAIN_CHORD_IO_BARS, len(cc_pat_data) - 2 * TRAIN_CHORD_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []
                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                time_in_bar = (step_it % 2) * 2  # 小节内的第几拍
                # 2.获取这两小节加两拍的主旋律和和弦,以及再前一个步长的bass组合
                melody1_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
                melody2_code_add_base = 4 + COMMON_MELODY_PATTERN_NUMBER + 2  # 主旋律第二拍数据编码增加的基
                chord_code_add_base = 4 + (COMMON_MELODY_PATTERN_NUMBER + 2) * 2  # 和弦数据编码增加的基数
                for forw_step_it in range(step_it, step_it + 2 * TRAIN_CHORD_IO_BARS + 1):  # 向前看10拍
                    if forw_step_it >= 0:
                        bar_dx = forw_step_it // 2
                        time_in_bar = (forw_step_it % 2) * 2
                        time_add = 0 if continuous_bar_data[bar_dx] == 0 else (1 - continuous_bar_data[bar_dx] % 2) * 2
                        input_time_data.append([time_add + time_in_bar // 2, melody_pat_data[bar_dx][time_in_bar] + melody1_code_add_base, melody_pat_data[bar_dx][time_in_bar + 1] + melody2_code_add_base])
                    else:
                        time_add = 0
                        time_in_bar = (forw_step_it % 2) * 2
                        input_time_data.append([time_add + time_in_bar // 2, melody1_code_add_base, melody2_code_add_base])
                    if forw_step_it - 1 >= 0:
                        input_time_data[-1].append(cc_pat_data[forw_step_it - 1] + chord_code_add_base)
                    else:
                        input_time_data[-1].append(chord_code_add_base)
                for bar_it in range(cur_bar, cur_bar + TRAIN_CHORD_IO_BARS):
                    step_dx = (4 - time_in_bar) // 2 + (bar_it - cur_bar) * 2  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], melody1_code_add_base, melody2_code_add_base, chord_code_add_base] for t in range(step_dx)]
                # 3.添加过去10拍的和弦-和弦组合 一共5个
                if cur_bar < 0 or melody_pat_data[cur_bar] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的和弦-和弦组合也置为空
                    output_time_data.extend([chord_code_add_base for t in range(2 - pat_step_in_bar)])
                else:
                    output_time_data.extend(t + chord_code_add_base for t in cc_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2])
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_CHORD_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的和弦-和弦组合置为空
                        output_time_data.extend([chord_code_add_base, chord_code_add_base])
                    else:
                        output_time_data.extend(t + chord_code_add_base for t in cc_pat_data[bar_it * 2: bar_it * 2 + 2])
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [chord_code_add_base for t in range(len(output_time_data))]
                output_time_data.extend(t + chord_code_add_base for t in cc_pat_data[(cur_bar + TRAIN_CHORD_IO_BARS) * 2: (cur_bar + TRAIN_CHORD_IO_BARS) * 2 + pat_step_in_bar + 1])  # 最后一个小节的和弦数据
                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的bass不为空
                if melody_pat_data[cur_bar + TRAIN_CHORD_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_chord_data = cc_pat_data[(cur_bar + TRAIN_CHORD_IO_BARS) * 2: (cur_bar + TRAIN_CHORD_IO_BARS + 1) * 2]
                    if final_bar_chord_data != [0, 0]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass
