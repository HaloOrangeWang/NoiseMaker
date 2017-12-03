from interfaces.sql.sqlite import GetRawSongDataFromDataset
from .piano_guitar import OneBarRelNoteList, RelNotePatternEncode, RootNote
from interfaces.functions import GetDictMaxKey
from settings import *
import numpy as np

# 弦乐的处理方法
# 节奏：固定为2拍 不岁主旋律的节奏变化而变化
# 旋律：以2拍为时间步长 如果2拍之后没有弦乐 则换成4拍之后 如果还没有那就是没有。如果两拍之内和弦发生了改变 也认为是没有


def CommonStringPatterns(raw_string_data, root_data, chord_data, note_time_step, pattern_time_step):
    common_pattern_dict = {}
    time_step_ratio = round(pattern_time_step / note_time_step)
    for string_part in raw_string_data:
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            if string_part[song_iterator] != {}:
                for key in string_part[song_iterator]:
                    try:
                        rel_note_list = OneBarRelNoteList(string_part[song_iterator][key], root_data[song_iterator][key], chord_data[song_iterator][key])
                        # print(song_iterator, key, rel_note_list)
                        pattern_list = [rel_note_list[time_step_ratio * t: time_step_ratio * (t + 1)] for t in range(round(4 / pattern_time_step))]
                        for pattern_iterator in pattern_list:
                            try:
                                common_pattern_dict[str(pattern_iterator)] += 1
                            except KeyError:
                                common_pattern_dict[str(pattern_iterator)] = 1
                    except KeyError:
                        pass
    common_pattern_list_temp = sorted(common_pattern_dict.items(), key=lambda asd: asd[1], reverse=True)  # 按照次数由高到底排序
    # print(len(common_pattern_list_temp))
    # sum1 = 0
    # sum2 = 0
    # for t in common_pattern_list_temp[1:]:
    #     sum1 += t[1]
    # for t in common_pattern_list_temp[1: (COMMON_STRING_PATTERN_NUMBER + 1)]:
    #     sum2 += t[1]
    # print(sum2 / sum1)
    common_pattern_list = []  # 最常见的number种组合
    pattern_number_list = []  # 这些组合出现的次数
    for pattern_tuple in common_pattern_list_temp[:(COMMON_STRING_PATTERN_NUMBER + 1)]:
        common_pattern_list.append(eval(pattern_tuple[0]))
        pattern_number_list.append(pattern_tuple[1])
    # print(common_pattern_list)
    # print(pattern_number_list)
    return common_pattern_list, pattern_number_list


class StringPatternEncode(RelNotePatternEncode):

    def handle_rare_pattern(self, bar_index, bar_pattern_list, rel_note_list, common_patterns):
        # 在常见的string列表里找不到某一个string组合的处理方法：
        # a.寻找符合以下条件的piano_guitar组合
        # a1.整半拍处的休止情况完全相同
        # a2.该组合的所有音符是待求组合所有音符的子集 或与待求组合的音符相差7的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为common_patterns+1
        for bar_string_pattern_iterator in range(len(bar_pattern_list)):
            if bar_pattern_list[bar_string_pattern_iterator] == -1:
                choose_pattern = 0  # 选取的pattern
                choose_pattern_diff_score = 75  # 两个旋律组合的差异程度
                for common_string_iterator in range(1, len(common_patterns)):
                    total_note_count = 0
                    diff_note_count = 0
                    # 1.1.检查该旋律组合是否符合要求
                    note_satisfactory = True
                    for group_iterator in range(len(common_patterns[common_string_iterator])):
                        # 1.1.1.检查整半拍的休止情况是否相同
                        if group_iterator % 2 == 0:
                            if bool(common_patterns[common_string_iterator][group_iterator]) ^ bool(rel_note_list[bar_string_pattern_iterator][group_iterator]):  # 一个为休止符 一个不为休止符
                                note_satisfactory = False
                                break
                        if common_patterns[common_string_iterator][group_iterator] == 0 and rel_note_list[bar_string_pattern_iterator][group_iterator] == 0:  # 如果这个时间步长是休止符的话，直接进入下个时间步长
                            continue
                        elif common_patterns[common_string_iterator][group_iterator] == 0 and rel_note_list[bar_string_pattern_iterator][group_iterator] != 0:  # pattern是休止而待求片段不是休止 计入不同后进入下个时间步长
                            total_note_count += len(rel_note_list[bar_string_pattern_iterator][group_iterator])
                            diff_note_count += len(rel_note_list[bar_string_pattern_iterator][group_iterator]) * 1.2
                            continue
                        elif common_patterns[common_string_iterator][group_iterator] != 0 and rel_note_list[bar_string_pattern_iterator][group_iterator] == 0:  # pattern不是休止而待求片段是休止 计入不同后进入下个时间步长
                            total_note_count += len(common_patterns[common_string_iterator][group_iterator])
                            diff_note_count += len(common_patterns[common_string_iterator][group_iterator]) * 1.2
                            continue
                        # 1.1.2.求出相对音高组合并对7取余数 按升降号分开
                        cur_pattern_note_list = common_patterns[common_string_iterator][group_iterator]  # 这个时间步长中常见组合的真实音符组合
                        cur_pattern_note_list_div7 = [[t[0] % 7, t[1]] for t in cur_pattern_note_list]  # 对7取余数
                        cur_pattern_sj_set = {t[1] for t in cur_pattern_note_list}  # 升降关系的集合
                        cur_pattern_note_dict_div7 = {t0: {t1 for [t1, t2] in cur_pattern_note_list_div7 if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开
                        cur_pattern_note_dict = {t0: {t1 for [t1, t2] in cur_pattern_note_list if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开

                        cur_step_note_list = rel_note_list[bar_string_pattern_iterator][group_iterator]  # 这个时间步长中待求组合的真实音符组合
                        cur_step_note_list_div7 = [[t[0] % 7, t[1]] for t in cur_step_note_list]  # 对7取余数
                        cur_step_sj_set = {t[1] for t in cur_step_note_list}  # 升降关系的集合
                        cur_step_note_dict_div7 = {t0: {t1 for [t1, t2] in cur_step_note_list_div7 if t2 == t0} for t0 in cur_step_sj_set}  # 按照升降号将音符分开
                        cur_step_note_dict = {t0: {t1 for [t1, t2] in cur_step_note_list if t2 == t0} for t0 in cur_step_sj_set}  # 按照升降号将音符分开
                        # 1.1.3.遍历升降号 如果组合音符列表是待求音符列表的子集的话则认为是可以替代的
                        for sj in cur_pattern_sj_set:
                            try:
                                if not cur_pattern_note_dict_div7[sj].issubset(cur_step_note_dict_div7[sj]):
                                    note_satisfactory = False
                                    break
                            except KeyError:
                                note_satisfactory = False
                                break
                        if not note_satisfactory:
                            break
                        # 1.1.4.计算该组合与待求组合之间的差异分
                        # print(bar_index, bar_pg_pattern_iterator, group_iterator, common_pg_iterator, cur_step_note_dict, cur_pattern_note_dict)
                        for sj in cur_step_note_dict:  # 遍历待求音符组合的所有升降号
                            total_note_count += len(cur_step_note_dict[sj])
                            if sj not in cur_pattern_note_dict:
                                diff_note_count += len(cur_step_note_dict[sj])
                                break
                            for rel_note in cur_step_note_dict[sj]:
                                if rel_note not in cur_pattern_note_dict[sj]:
                                    diff_note_count += 1
                            for pattern_note in cur_pattern_note_dict[sj]:
                                if pattern_note not in cur_step_note_dict[sj]:
                                        diff_note_count += 1.2  # 如果pattern中有而待求组合中没有，记为1.2分。（这里其实可以说是“宁缺毋滥”）
                    if not note_satisfactory:
                        continue
                    # 1.2.如果找到符合要求的组合 将其记录下来
                    pattern_diff_score = (100 * diff_note_count) // total_note_count
                    # print(bar_index, bar_pg_pattern_iterator, common_pg_iterator, pattern_diff_score)
                    if pattern_diff_score < choose_pattern_diff_score:
                        choose_pattern = common_string_iterator
                        choose_pattern_diff_score = pattern_diff_score
                # 2.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_PIANO_GUITAR_PATTERNS+1
                # print(bar_pg_pattern_iterator, raw_bar_pattern_list, choose_pattern, choose_pattern_diff_score)
                if choose_pattern != 0:
                    self.fi += 1
                    bar_pattern_list[bar_string_pattern_iterator] = choose_pattern
                else:
                    self.nf += 1
                    bar_pattern_list[bar_string_pattern_iterator] = COMMON_STRING_PATTERN_NUMBER + 1
        # print(bar_pattern_list, '\n')
        return bar_pattern_list


class StringTrainData:

    emission = []  # 观测矩阵
    transfer = []  # 状态转移矩阵
    pi = []  # 初始状态矩阵
    emission_count = []  # 观测计数矩阵
    transfer_count = []  # 状态转移计数矩阵
    pi_count = []  # 初始状态计数矩阵

    fi = 0
    nf = 0

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data):
        # 1.从数据集中读取歌的piano_guitar数据
        raw_string_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        string_part_iterator = 1
        while True:
            string_part_data = GetRawSongDataFromDataset('string' + str(string_part_iterator), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if string_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_string_data.append(string_part_data)
            # flatten_pg_data += pg_part_data
            string_part_iterator += 1
        self.train_song_list = [[True for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(string_part_iterator)]  # 用于训练的歌曲id列表
        # 2.获取根音组合及根音-和弦配对组合
        self.get_root_data(chord_data)  # 与piano_guitar那边不同的是：逐两拍生成根音-和弦组合
        self.get_root_chord_pattern_dict(chord_data)
        self.transfer_count = np.zeros([COMMON_STRING_PATTERN_NUMBER, COMMON_STRING_PATTERN_NUMBER])  # 统计数据时把0和common+1的数据排除在外
        self.emission_count = np.zeros([len(self.rc_pattern_dict) - 1, COMMON_STRING_PATTERN_NUMBER])  # “dict长度”行，pattern个数列。0和common+1被排除在外
        self.pi_count = np.zeros([COMMON_STRING_PATTERN_NUMBER])
        # print(root_data)
        # 3.获取最常见的string组合
        # self.common_pg_patterns, __ = CommonMusicPatterns(raw_pg_data, number=COMMON_PIANO_GUITAR_PATTERN_NUMBER, note_time_step=1 / 4, pattern_time_step=1)
        self.common_string_patterns, __ = CommonStringPatterns(raw_string_data, self.root_data, chord_data, 0.25, 2)
        for string_part_iterator in range(len(raw_string_data)):
            for song_iterator in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_string_data[string_part_iterator][song_iterator] != {} and melody_pattern_data[song_iterator] != {}:
                    # raw_pg_data[pg_part_iterator][song_iterator] = PianoGuitarPatternEncode(self.common_pg_patterns, raw_pg_data[pg_part_iterator][song_iterator], 0.25, 1).music_pattern_dict
                    raw_string_data[string_part_iterator][song_iterator] = StringPatternEncode(self.common_string_patterns, raw_string_data[string_part_iterator][song_iterator], chord_data[song_iterator], self.root_data[song_iterator], 0.25, 2).music_pattern_dict
                    self.count(raw_string_data[string_part_iterator][song_iterator], self.rc_pattern_list[song_iterator])
                    self.count_initial(raw_string_data[string_part_iterator][song_iterator], self.rc_pattern_list[song_iterator], continuous_bar_number_data[song_iterator])
        self.prob_convert()

    def get_root_data(self, chord_data):
        self.root_data = [{} for t in range(TRAIN_FILE_NUMBERS)]
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            for bar_iterator in range(GetDictMaxKey(chord_data[song_iterator]) + 1):
                if chord_data[song_iterator][bar_iterator] == [0, 0, 0, 0]:  # 如果这个小节没有和弦 那么也同样没有根音
                    self.root_data[song_iterator][bar_iterator] = [0, 0, 0, 0]
                for chord_iterator in range(4):
                    if chord_iterator == 0:
                        if bar_iterator == 0 or self.root_data[song_iterator][bar_iterator - 1] == 0:
                            self.root_data[song_iterator][bar_iterator] = [RootNote(chord_data[song_iterator][bar_iterator][chord_iterator], 0)]
                        else:
                            self.root_data[song_iterator][bar_iterator] = [RootNote(chord_data[song_iterator][bar_iterator][chord_iterator], self.root_data[song_iterator][bar_iterator - 1][3])]
                    else:
                        # print(chord_iterator, root_data[song_iterator][bar_iterator], chord_data[song_iterator][bar_iterator])
                        self.root_data[song_iterator][bar_iterator].append(RootNote(chord_data[song_iterator][bar_iterator][chord_iterator], self.root_data[song_iterator][bar_iterator][chord_iterator - 1]))

    def get_root_chord_pattern_dict(self, chord_data):
        """
        获取并编码和弦和根音的组合列表。如[40,1]代表根音为40,和弦编号为1
        这里与piano_guitar里的方法有一定不同之处 这里每2拍为一个时间步长，如果2拍内根音相同的话，取前一拍的根音和和弦 否则记为0
        :param chord_data: 和弦列表
        :return:
        """
        self.rc_pattern_list = [{} for t in range(TRAIN_FILE_NUMBERS)]
        self.rc_pattern_dict = [[-1]]
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            for bar_iterator in range(GetDictMaxKey(chord_data[song_iterator]) + 1):
                self.rc_pattern_list[song_iterator][bar_iterator] = [0, 0]  # 先全部初始化为0
                for chord_iterator in range(0, 4, 2):  # 时间步长为2拍
                    if chord_data[song_iterator][bar_iterator][chord_iterator] != 0:  # 如果这个时间步长中和弦是0的话，就不用向下执行了
                        if self.root_data[song_iterator][bar_iterator][chord_iterator] == self.root_data[song_iterator][bar_iterator][chord_iterator + 1]:  # 判断两拍的根音是否相同
                            rc_pattern = [self.root_data[song_iterator][bar_iterator][chord_iterator], chord_data[song_iterator][bar_iterator][chord_iterator]]
                            try:
                                rc_pattern_index = self.rc_pattern_dict.index(rc_pattern)  # 检查这个音符组合有没有被保存
                            except ValueError:
                                self.rc_pattern_dict.append(rc_pattern)  # 添加这个根音 和弦 组合
                                rc_pattern_index = len(self.rc_pattern_dict) - 1
                            self.rc_pattern_list[song_iterator][bar_iterator][chord_iterator // 2] = rc_pattern_index  # 将这个音符保存起来
        # print(self.rc_pattern_dict)
        # for t in range(TRAIN_FILE_NUMBERS):
        #     print(self.rc_pattern_list[t])

    def count(self, string_data, rc_pattern):
        for key in string_data:
            if key not in rc_pattern:  # 这一小节没有和弦 直接跳过
                continue
            for time_iterator in range(2):
                if key == 0 and time_iterator == 0:  # 不考虑第一小节的第一拍
                    continue
                if time_iterator == 1:
                    if (string_data[key][time_iterator] in [0, COMMON_STRING_PATTERN_NUMBER + 1]) or rc_pattern[key][time_iterator - 1] == 0:  # 当前拍 上一拍的伴奏编码不能是0或common+1 同时当前拍和弦根音编码不能是0
                        continue
                    if string_data[key][0] not in [0, COMMON_STRING_PATTERN_NUMBER + 1]:  # 如果前2拍有弦乐 选择前两拍 否则选择前四拍
                        self.transfer_count[string_data[key][0] - 1, string_data[key][1] - 1] += 1
                        self.emission_count[rc_pattern[key][1] - 1, string_data[key][1] - 1] += 1
                    elif key != 0 and string_data[key - 1][1] not in [0, COMMON_STRING_PATTERN_NUMBER + 1]:
                        self.transfer_count[string_data[key - 1][1] - 1, string_data[key][1] - 1] += 1
                        self.emission_count[rc_pattern[key][1] - 1, string_data[key][1] - 1] += 1
                else:
                    if (string_data[key][0] in [0, COMMON_STRING_PATTERN_NUMBER + 1]) or rc_pattern[key][0] == 0:
                        continue
                    if string_data[key - 1][1] not in [0, COMMON_STRING_PATTERN_NUMBER + 1]:
                        self.transfer_count[string_data[key - 1][1] - 1, string_data[key][0] - 1] += 1
                        self.emission_count[rc_pattern[key][0] - 1, string_data[key][0] - 1] += 1
                    elif string_data[key - 1][0] not in [0, COMMON_STRING_PATTERN_NUMBER + 1]:
                        self.transfer_count[string_data[key - 1][0] - 1, string_data[key][0] - 1] += 1
                        self.emission_count[rc_pattern[key][0] - 1, string_data[key][0] - 1] += 1

    def count_initial(self, string_data, rc_pattern, continuous_bar_number_data):
        for key in string_data:
            if key >= len(continuous_bar_number_data):  # 这个小节没有对应的主旋律 直接跳过
                continue
            if key == 0:
                if continuous_bar_number_data[0] != 0:
                    if string_data[0][0] not in [0, COMMON_STRING_PATTERN_NUMBER + 1] and rc_pattern[0][0] != 0:
                        self.pi_count[string_data[0][0] - 1] += 1
            else:
                if continuous_bar_number_data[key] != 0 and continuous_bar_number_data[key - 1] == 0:
                    if string_data[key][0] not in [0, COMMON_STRING_PATTERN_NUMBER + 1] and rc_pattern[key][0] != 0:
                        self.pi_count[string_data[key][0] - 1] += 1

    def prob_convert(self):
        self.transfer = np.zeros([COMMON_STRING_PATTERN_NUMBER, COMMON_STRING_PATTERN_NUMBER])
        self.emission = np.zeros([len(self.rc_pattern_dict) - 1, COMMON_STRING_PATTERN_NUMBER])
        self.pi = np.zeros([COMMON_STRING_PATTERN_NUMBER])
        # 1.计算转移矩阵
        for row_iterator, row in enumerate(self.transfer_count):
            row_sum = sum(row)
            if row_sum == 0:
                self.transfer[row_iterator] = [1 / COMMON_STRING_PATTERN_NUMBER for t in range(COMMON_STRING_PATTERN_NUMBER)]  # 如果数据集中这个状态没有接任何一个下一个状态，则概率均分
            else:
                self.transfer[row_iterator] = self.transfer_count[row_iterator] / row_sum
        # 2.计算观测矩阵
        for column_iterator in range(COMMON_STRING_PATTERN_NUMBER):
            column_sum = sum(self.emission_count[:, column_iterator])
            if column_sum == 0:
                self.emission[:, column_iterator] = [1 / (len(self.rc_pattern_dict) - 1) for t in range(len(self.rc_pattern_dict) - 1)]  # 为空列则概率均分
            else:
                self.emission[:, column_iterator] = self.emission_count[:, column_iterator] / column_sum
        # 3.计算初始化转移向量
        self.pi = self.pi_count / sum(self.pi_count)
