from interfaces.music_patterns import MusicPatternEncodeStep, CommonMusicPatterns
from interfaces.sql.sqlite import get_raw_song_data_from_dataset
from interfaces.utils import DiaryLog, flat_array
from datainputs.functions import one_song_rel_notelist_chord
from datainputs.piano_guitar import get_diff_value
from settings import *
import numpy as np


class StringPatternEncode(MusicPatternEncodeStep):

    def handle_rare_pattern(self, pattern_dx, raw_note_list, common_patterns):
        # 在常见的string列表里找不到某一个string组合的处理方法：
        # a.寻找符合以下条件的piano_guitar组合
        # a1.整半拍处的休止情况完全相同
        # a2.该组合的所有音符是待求组合所有音符的子集 或与待求组合的音符相差7的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为common_patterns+1
        if pattern_dx == -1:
            choose_pattern = 0  # 选取的pattern
            choose_pattern_diff_score = 75  # 两个旋律组合的差异程度
            for common_pat_it in range(1, len(common_patterns)):
                total_note_count = 0
                diff_note_count = 0
                # 1.1.检查该旋律组合是否符合要求
                note_satisfactory = True
                for note_group_it in range(len(common_patterns[common_pat_it])):
                    # 1.1.1.检查整半拍的休止情况是否相同
                    if note_group_it % 2 == 0:
                        if bool(common_patterns[common_pat_it][note_group_it]) ^ bool(raw_note_list[note_group_it]):  # 一个为休止符 一个不为休止符
                            note_satisfactory = False
                            break
                    if common_patterns[common_pat_it][note_group_it] == 0 and raw_note_list[note_group_it] == 0:  # 如果这个时间步长是休止符的话，直接进入下个时间步长
                        continue
                    elif common_patterns[common_pat_it][note_group_it] == 0 and raw_note_list[note_group_it] != 0:  # pattern是休止而待求片段不是休止 计入不同后进入下个时间步长
                        total_note_count += len(raw_note_list[note_group_it])
                        diff_note_count += len(raw_note_list[note_group_it]) * 1.2
                        continue
                    elif common_patterns[common_pat_it][note_group_it] != 0 and raw_note_list[note_group_it] == 0:  # pattern不是休止而待求片段是休止 计入不同后进入下个时间步长
                        total_note_count += len(common_patterns[common_pat_it][note_group_it])
                        diff_note_count += len(common_patterns[common_pat_it][note_group_it]) * 1.2
                        continue
                    # 1.1.2.求出相对音高组合并对7取余数 按升降号分开
                    cur_pattern_note_list = common_patterns[common_pat_it][note_group_it]  # 这个时间步长中常见组合的真实音符组合
                    cur_pattern_note_list_div7 = [[t[0] % 7, t[1]] for t in cur_pattern_note_list]  # 对7取余数
                    cur_pattern_sj_set = {t[1] for t in cur_pattern_note_list}  # 升降关系的集合
                    cur_pattern_note_dict_div7 = {t0: {t1 for [t1, t2] in cur_pattern_note_list_div7 if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开
                    cur_pattern_note_dict = {t0: {t1 for [t1, t2] in cur_pattern_note_list if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开

                    cur_step_note_list = raw_note_list[note_group_it]  # 这个时间步长中待求组合的真实音符组合
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
                    choose_pattern = common_pat_it
                    choose_pattern_diff_score = pattern_diff_score
            # 2.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_PIANO_GUITAR_PATTERNS+1
            # print(bar_pg_pattern_iterator, raw_bar_pattern_list, choose_pattern, choose_pattern_diff_score)
            if choose_pattern != 0:
                pattern_dx = choose_pattern
            else:
                pattern_dx = COMMON_STRING_PATTERN_NUMBER + 1
        # print(bar_pattern_list, '\n')
        return pattern_dx


class StringTrainData:

    def __init__(self, melody_pat_data, continuous_bar_data, chord_cls):
        """
        :param melody_pat_data: 主旋律组合的数据
        :param continuous_bar_data: 连续小节的计数数据
        :param chord_cls: 0.95开始这个参数变成和弦的类而不只是最后的和弦数据 因为需要用到和弦类的方法
        """
        raw_string_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        part_count = 1
        while True:
            string_part_data = get_raw_song_data_from_dataset('string' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if string_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_string_data.append(string_part_data)
            # flatten_pg_data += pg_part_data
            part_count += 1
        self.train_song_list = [[True for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]  # 用于训练的歌曲id列表
        # 2.获取根音组合及根音-和弦配对组合
        self.root_data, self.rc_pattern_data, self.rc_pattern_count = chord_cls.get_root_data(STRING_AVERAGE_ROOT)
        self.transfer_count = np.zeros([COMMON_STRING_PATTERN_NUMBER, COMMON_STRING_PATTERN_NUMBER])  # 统计数据时把0和common+1的数据排除在外
        self.emission_count = np.zeros([len(self.rc_pattern_count) - 1, COMMON_STRING_PATTERN_NUMBER])  # “dict长度”行，pattern个数列。0和common+1被排除在外
        self.pi_count = np.zeros([COMMON_STRING_PATTERN_NUMBER])
        # 3.将原数据转化成相对音高形式
        rel_note_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        for part_it in range(len(raw_string_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_string_data[part_it][song_it] != {} and melody_pat_data[song_it] != {}:
                    rel_note_list[part_it][song_it] = one_song_rel_notelist_chord(raw_string_data[part_it][song_it], self.root_data[song_it], chord_cls.chord_data[song_it])
        # 4.获取最常见的String组合
        common_pattern_cls = CommonMusicPatterns(COMMON_STRING_PATTERN_NUMBER)  # 这个类可以获取常见的主旋律组合
        if FLAG_IS_TRAINING is True:  # 训练模式
            common_pattern_cls.train(rel_note_list, 0.25, 2, multipart=True)
            common_pattern_cls.store('String')  # 存储在sqlite文件中
        else:
            common_pattern_cls.restore('String')  # 直接从sqlite文件中读取
        self.common_string_pats = common_pattern_cls.common_pattern_list  # 常见的旋律组合列表
        # 5.将其编码为组合并获取模型的输入输出数据
        # pg_pat_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        for part_it in range(len(raw_string_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_string_data[part_it][song_it] != {} and melody_pat_data[song_it] != {}:
                    string_pat_list = StringPatternEncode(self.common_string_pats, rel_note_list[part_it][song_it], 0.25, 2).music_pattern_ary
                    self.count(string_pat_list, self.rc_pattern_data[song_it])
                    self.count_initial(string_pat_list, self.rc_pattern_data[song_it], continuous_bar_data[song_it])
        self.prob_convert()

    def count(self, string_data, rc_pattern):
        """获取观测矩阵和状态转移矩阵的计数"""
        # for key in pg_data:
        #     if key not in rc_pattern:  # 这一小节没有和弦 直接跳过
        #         continue
        for step_it in range(len(string_data)):
            if step_it >= len(rc_pattern):  # 这一小节没有和弦 直接跳过
                continue
            if step_it == 0:  # 不考虑第一小节的第一拍
                continue
            if step_it != 0:
                if (string_data[step_it] in [0, COMMON_STRING_PATTERN_NUMBER + 1]) or (string_data[step_it - 1] in [0, COMMON_STRING_PATTERN_NUMBER + 1]) or rc_pattern[step_it - 1] == 0:  # 当前拍 上一拍的伴奏编码不能是0或common+1 同时当前拍和弦根音编码不能是0
                    continue
                self.transfer_count[string_data[step_it - 1] - 1, string_data[step_it] - 1] += 1
                self.emission_count[rc_pattern[step_it] - 1, string_data[step_it] - 1] += 1

    def count_initial(self, string_data, rc_pattern, continuous_bar_data):
        """获取初始状态矩阵的计数"""
        for bar_dx in range(0, len(string_data), 2):  # 遍历每个小节
            if (bar_dx // 2) >= len(continuous_bar_data):  # 这个小节没有对应的主旋律 直接跳过
                continue
            if bar_dx == 0:
                if continuous_bar_data[0] != 0:
                    if string_data[0] not in [0, COMMON_STRING_PATTERN_NUMBER + 1] and rc_pattern[0] != 0:
                        self.pi_count[string_data[0] - 1] += 1
            else:
                if continuous_bar_data[bar_dx // 2] != 0 and continuous_bar_data[(bar_dx // 2) - 1] == 0:
                    if string_data[bar_dx // 2] not in [0, COMMON_STRING_PATTERN_NUMBER + 1] and rc_pattern[bar_dx // 2] != 0:
                        self.pi_count[string_data[bar_dx // 2] - 1] += 1

    def prob_convert(self):
        self.transfer = np.zeros(self.transfer_count.shape)
        self.emission = np.zeros(self.emission_count.shape)
        self.pi = np.zeros([COMMON_STRING_PATTERN_NUMBER])
        # 1.计算转移矩阵
        for row_it, row in enumerate(self.transfer_count):
            row_sum = sum(row)
            if row_sum == 0:
                self.transfer[row_it] = [1 / COMMON_STRING_PATTERN_NUMBER for t in range(COMMON_STRING_PATTERN_NUMBER)]  # 如果数据集中这个状态没有接任何一个下一个状态，则概率均分
            else:
                self.transfer[row_it] = self.transfer_count[row_it] / row_sum
        # 2.计算观测矩阵
        for column_it in range(COMMON_STRING_PATTERN_NUMBER):
            column_sum = sum(self.emission_count[:, column_it])
            if column_sum == 0:
                self.emission[:, column_it] = [1 / self.emission.shape[0] for t in range(self.emission.shape[0])]  # 为空列则概率均分
            else:
                self.emission[:, column_it] = self.emission_count[:, column_it] / column_sum
        # 3.计算初始化转移向量
        self.pi = self.pi_count / sum(self.pi_count)


class StringTrainData2:

    def __init__(self, melody_pat_data, continuous_bar_data, corenote_data, corenote_pat_ary, chord_cls):
        """
        :param melody_pat_data: 主旋律组合的数据
        :param continuous_bar_data: 连续小节的计数数据
        :param chord_cls: 0.95开始这个参数变成和弦的类而不只是最后的和弦数据 因为需要用到和弦类的方法
        :param corenote_data: 主旋律骨干音的组合数据
        :param corenote_pat_ary: 主旋律骨干音组合的对照表
        """
        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取歌的string数据
        raw_string_data = []  # 四维数组 第一维是string的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        part_count = 1
        while True:
            string_part_data = get_raw_song_data_from_dataset('string' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if string_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_string_data.append(string_part_data)
            # flatten_pg_data += pg_part_data
            part_count += 1
        self.train_song_list = [[True for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]  # 用于训练的歌曲id列表
        # 2.获取根音组合及根音-和弦配对组合
        self.root_data, self.rc_pattern_data, self.rc_pattern_count = chord_cls.get_root_data(STRING_AVERAGE_ROOT)
        self.rc_pat_num = len(self.rc_pattern_count)
        # 3.将原数据转化成相对音高形式
        rel_note_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        for part_it in range(len(raw_string_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_string_data[part_it][song_it] != {} and melody_pat_data[song_it] != {}:
                    rel_note_list[part_it][song_it] = one_song_rel_notelist_chord(raw_string_data[part_it][song_it], self.root_data[song_it], chord_cls.chord_data[song_it])
        # 4.获取最常见的弦乐组合
        common_pattern_cls = CommonMusicPatterns(COMMON_STRING_PATTERN_NUMBER)  # 这个类可以获取常见的弦乐组合
        if FLAG_IS_TRAINING is True:  # 训练模式
            common_pattern_cls.train(rel_note_list, 0.25, 2, multipart=True)
            common_pattern_cls.store('String')  # 存储在sqlite文件中
        else:
            common_pattern_cls.restore('String')  # 直接从sqlite文件中读取
        self.common_string_pats = common_pattern_cls.common_pattern_list  # 常见的旋律组合列表
        # 5.将其编码为组合并获取模型的输入输出数据
        # pg_pat_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        self.string_pat_data = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        for part_it in range(len(raw_string_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_string_data[part_it][song_it] != {} and melody_pat_data[song_it] != {}:
                    self.string_pat_data[part_it][song_it] = StringPatternEncode(self.common_string_pats, rel_note_list[part_it][song_it], 0.25, 2).music_pattern_ary
                    self.get_model_io_data(self.string_pat_data[part_it][song_it], melody_pat_data[song_it], continuous_bar_data[song_it], corenote_data[song_it], self.rc_pattern_data[song_it])
        # 6.获取用于验证的数据
        string_prob_ary = []
        for part_it in range(len(raw_string_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if chord_cls.chord_data[song_it] != {} and self.string_pat_data[part_it][song_it] != {}:
                    string_prob_ary.extend(get_diff_value(flat_array(raw_string_data[part_it][song_it]), chord_cls.chord_data[song_it]))
        # 7.找出前90%所在位置
        string_prob_ary = sorted(string_prob_ary)
        prob_09_dx = int(len(string_prob_ary) * 0.9 + 1)
        self.ConfidenceLevel = string_prob_ary[prob_09_dx]

    def get_model_io_data(self, string_pat_data, melody_pat_data, continuous_bar_data, corenote_data, rc_pat_data):
        """
        获取模型的输入输出数据。输入数据为过去两小节加2拍的时间编码/主旋律骨干音组合/和弦根音组合，输出内容为两小节加2拍的弦乐组合
        :param corenote_data: 一首歌的主旋律骨干音数据
        :param string_pat_data: 一首歌的弦乐数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param rc_pat_data: 一首歌的和弦根音组合数据
        """
        for step_it in range(-2 * TRAIN_STRING_IO_BARS, len(string_pat_data) - 2 * TRAIN_STRING_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []
                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                time_in_bar = (step_it % 2) * 2  # 小节内的第几拍
                # 2.获取这两小节加两拍的主旋律和和弦,以及再前一个步长的piano_guitar组合
                corenote_code_add_base = 4  # 主旋律骨干音数据编码增加的基数
                rc1_code_add_base = 4 + (COMMON_CORE_NOTE_PATTERN_NUMBER + 2)  # 和弦第一拍数据编码增加的基数
                rc2_code_add_base = 4 + (COMMON_CORE_NOTE_PATTERN_NUMBER + 2) + self.rc_pat_num  # 和弦第二拍数据编码增加的基数
                string_code_add_base = 4 + (COMMON_CORE_NOTE_PATTERN_NUMBER + 2) + self.rc_pat_num * 2  # string数据编码增加的基数
                for forw_step_it in range(step_it, step_it + 2 * TRAIN_STRING_IO_BARS + 1):  # 向前看10拍
                    if forw_step_it >= 0:
                        bar_dx = forw_step_it // 2
                        beat_in_bar = (forw_step_it % 2) * 2  # 当前小节内的第几拍
                        time_add = 0 if continuous_bar_data[bar_dx] == 0 else (1 - continuous_bar_data[bar_dx] % 2) * 2
                        input_time_data.append([time_add + beat_in_bar // 2, corenote_data[forw_step_it] + corenote_code_add_base, rc_pat_data[forw_step_it * 2] + rc1_code_add_base, rc_pat_data[forw_step_it * 2 + 1] + rc2_code_add_base])
                    else:
                        time_add = 0
                        beat_in_bar = (forw_step_it % 2) * 2
                        input_time_data.append([time_add + beat_in_bar // 2, corenote_code_add_base, rc1_code_add_base, rc2_code_add_base])
                    if forw_step_it - 1 >= 0:
                        input_time_data[-1].append(string_pat_data[forw_step_it - 1] + string_code_add_base)
                    else:
                        input_time_data[-1].append(string_code_add_base)
                for bar_it in range(cur_bar, cur_bar + TRAIN_STRING_IO_BARS):
                    step_dx = (4 - time_in_bar) // 2 + (bar_it - cur_bar) * 2  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], corenote_code_add_base, rc1_code_add_base, rc2_code_add_base, string_code_add_base] for t in range(step_dx)]
                # 3.添加过去10拍的string 一共5个
                if cur_bar < 0 or melody_pat_data[cur_bar] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的string也置为空
                    output_time_data.extend([string_code_add_base for t in range(2 - pat_step_in_bar)])
                else:
                    output_time_data.extend([t + string_code_add_base for t in string_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2]])
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_STRING_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的string置为空
                        output_time_data.extend([string_code_add_base for t in range(2)])
                    else:
                        output_time_data.extend([t + string_code_add_base for t in string_pat_data[bar_it * 2: bar_it * 2 + 2]])
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [string_code_add_base for t in range(len(output_time_data))]
                output_time_data.extend([t + string_code_add_base for t in string_pat_data[(cur_bar + TRAIN_STRING_IO_BARS) * 2: (cur_bar + TRAIN_STRING_IO_BARS) * 2 + pat_step_in_bar + 1]])  # 最后一个小节的string数据
                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的string不为空,
                if melody_pat_data[cur_bar + TRAIN_STRING_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_string_data = string_pat_data[(cur_bar + TRAIN_STRING_IO_BARS) * 2: (cur_bar + TRAIN_STRING_IO_BARS + 1) * 2]
                    if not set(final_bar_string_data).issubset({0, COMMON_STRING_PATTERN_NUMBER + 1}):
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass


class StringTrainData3(StringTrainData2):

    def get_model_io_data(self, string_pat_data, melody_pat_data, continuous_bar_data, corenote_data, rc_pat_data):
        """
        获取模型的输入输出数据。输入数据为过去两小节加2拍的时间编码/主旋律骨干音组合/和弦根音组合，输出内容为两小节加2拍的弦乐组合
        :param corenote_data: 一首歌的主旋律骨干音数据
        :param string_pat_data: 一首歌的弦乐数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param rc_pat_data: 一首歌的和弦根音组合数据
        """
        for step_it in range(-2 * TRAIN_STRING_IO_BARS, len(string_pat_data) - 2 * TRAIN_STRING_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []
                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                time_in_bar = (step_it % 2) * 2  # 小节内的第几拍
                # 2.获取这两小节加两拍的主旋律和和弦,以及再前一个步长的piano_guitar组合
                rc1_code_add_base = 4  # 和弦第一拍数据编码增加的基数
                rc2_code_add_base = 4 + self.rc_pat_num  # 和弦第二拍数据编码增加的基数
                string_code_add_base = 4 + self.rc_pat_num * 2  # string数据编码增加的基数
                for forw_step_it in range(step_it, step_it + 2 * TRAIN_STRING_IO_BARS + 1):  # 向前看10拍
                    if forw_step_it >= 0:
                        bar_dx = forw_step_it // 2
                        beat_in_bar = (forw_step_it % 2) * 2  # 当前小节内的第几拍
                        time_add = 0 if continuous_bar_data[bar_dx] == 0 else (1 - continuous_bar_data[bar_dx] % 2) * 2
                        input_time_data.append([time_add + beat_in_bar // 2, rc_pat_data[forw_step_it * 2] + rc1_code_add_base, rc_pat_data[forw_step_it * 2 + 1] + rc2_code_add_base])
                    else:
                        time_add = 0
                        beat_in_bar = (forw_step_it % 2) * 2
                        input_time_data.append([time_add + beat_in_bar // 2, rc1_code_add_base, rc2_code_add_base])
                    if forw_step_it - 1 >= 0:
                        input_time_data[-1].append(string_pat_data[forw_step_it - 1] + string_code_add_base)
                    else:
                        input_time_data[-1].append(string_code_add_base)
                for bar_it in range(cur_bar, cur_bar + TRAIN_STRING_IO_BARS):
                    step_dx = (4 - time_in_bar) // 2 + (bar_it - cur_bar) * 2  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], rc1_code_add_base, rc2_code_add_base, string_code_add_base] for t in range(step_dx)]
                # 3.添加过去10拍的string 一共5个
                if cur_bar < 0 or melody_pat_data[cur_bar] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的string也置为空
                    output_time_data.extend([string_code_add_base for t in range(2 - pat_step_in_bar)])
                else:
                    output_time_data.extend([t + string_code_add_base for t in string_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2]])
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_STRING_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的string置为空
                        output_time_data.extend([string_code_add_base for t in range(2)])
                    else:
                        output_time_data.extend([t + string_code_add_base for t in string_pat_data[bar_it * 2: bar_it * 2 + 2]])
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [string_code_add_base for t in range(len(output_time_data))]
                output_time_data.extend([t + string_code_add_base for t in string_pat_data[(cur_bar + TRAIN_STRING_IO_BARS) * 2: (cur_bar + TRAIN_STRING_IO_BARS) * 2 + pat_step_in_bar + 1]])  # 最后一个小节的piano_guitar数据
                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的string不为空,
                if melody_pat_data[cur_bar + TRAIN_STRING_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_string_data = string_pat_data[(cur_bar + TRAIN_STRING_IO_BARS) * 2: (cur_bar + TRAIN_STRING_IO_BARS + 1) * 2]
                    if not set(final_bar_string_data).issubset({0, COMMON_STRING_PATTERN_NUMBER + 1}):
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass
