from settings import *
from interfaces.sql.sqlite import get_raw_song_data_from_dataset
from interfaces.utils import flat_array, get_nearest_number_multiple, DiaryLog
from interfaces.note_format import one_song_rel_notelist_chord
from interfaces.music_patterns import BaseMusicPatterns, CommonMusicPatterns, MusicPatternEncode
from validations.strings import StringConfidenceCheckConfig
from validations.functions import AccompanyConfidenceCheck


class StringPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, pat_step_dx, raw_note_list, common_patterns):
        # 在常见的string列表里找不到某一个string组合的处理方法：
        # a.寻找符合以下条件的string组合
        # a1.整半拍处的休止情况完全相同
        # a2.该组合的所有音符是待求组合所有音符的子集 或与待求组合的音符相差7的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为common_patterns+1
        choose_pat = 0  # 选取的pattern
        choose_pat_diff_score = 75  # 两个旋律组合的差异程度
        for pat_it in range(1, len(common_patterns)):
            total_note_count = 0
            diff_note_count = 0
            # 1.1.检查该旋律组合是否符合要求
            note_satisfactory = True
            for note_group_it in range(len(common_patterns[pat_it])):
                # 1.1.1.检查整半拍的休止情况是否相同
                if note_group_it % 2 == 0:
                    if bool(common_patterns[pat_it][note_group_it]) ^ bool(raw_note_list[note_group_it]):  # 一个为休止符 一个不为休止符
                        note_satisfactory = False
                        break
                if common_patterns[pat_it][note_group_it] == 0 and raw_note_list[note_group_it] == 0:  # 如果这个时间步长是休止符的话，直接进入下个时间步长
                    continue
                elif common_patterns[pat_it][note_group_it] == 0 and raw_note_list[note_group_it] != 0:  # pattern是休止而待求片段不是休止 计入不同后进入下个时间步长
                    total_note_count += len(raw_note_list[note_group_it])
                    diff_note_count += len(raw_note_list[note_group_it]) * 1.2
                    continue
                elif common_patterns[pat_it][note_group_it] != 0 and raw_note_list[note_group_it] == 0:  # pattern不是休止而待求片段是休止 计入不同后进入下个时间步长
                    total_note_count += len(common_patterns[pat_it][note_group_it])
                    diff_note_count += len(common_patterns[pat_it][note_group_it]) * 1.2
                    continue
                # 1.1.2.求出相对音高组合并对7取余数 按升降号分开
                cur_pattern_note_list = common_patterns[pat_it][note_group_it]  # 这个时间步长中常见组合的真实音符组合
                cur_pattern_note_list_div7 = [[t[0] % 7, t[1]] for t in cur_pattern_note_list]  # 对7取余数
                cur_pattern_updown_set = {t[1] for t in cur_pattern_note_list}  # 升降关系的集合
                cur_pattern_note_dic_div7 = {t0: {t1 for [t1, t2] in cur_pattern_note_list_div7 if t2 == t0} for t0 in cur_pattern_updown_set}  # 按照升降号将音符分开
                cur_pattern_note_dic = {t0: {t1 for [t1, t2] in cur_pattern_note_list if t2 == t0} for t0 in cur_pattern_updown_set}  # 按照升降号将音符分开

                cur_step_note_list = raw_note_list[note_group_it]  # 这个时间步长中待求组合的真实音符组合
                cur_step_note_list_div7 = [[t[0] % 7, t[1]] for t in cur_step_note_list]  # 对7取余数
                cur_step_updown_set = {t[1] for t in cur_step_note_list}  # 升降关系的集合
                cur_step_note_dic_div7 = {t0: {t1 for [t1, t2] in cur_step_note_list_div7 if t2 == t0} for t0 in cur_step_updown_set}  # 按照升降号将音符分开
                cur_step_note_dic = {t0: {t1 for [t1, t2] in cur_step_note_list if t2 == t0} for t0 in cur_step_updown_set}  # 按照升降号将音符分开
                # 1.1.3.遍历升降号 如果组合音符列表是待求音符列表的子集的话则认为是可以替代的
                for updown in cur_pattern_updown_set:
                    try:
                        if not cur_pattern_note_dic_div7[updown].issubset(cur_step_note_dic_div7[updown]):
                            note_satisfactory = False
                            break
                    except KeyError:
                        note_satisfactory = False
                        break
                if not note_satisfactory:
                    break
                # 1.1.4.计算该组合与待求组合之间的差异分
                for updown in cur_step_note_dic:  # 遍历待求音符组合的所有升降号
                    total_note_count += len(cur_step_note_dic[updown])
                    if updown not in cur_pattern_note_dic:
                        diff_note_count += len(cur_step_note_dic[updown])
                        break
                    for rel_note in cur_step_note_dic[updown]:
                        if rel_note not in cur_pattern_note_dic[updown]:
                            diff_note_count += 1
                    for pattern_note in cur_pattern_note_dic[updown]:
                        if pattern_note not in cur_step_note_dic[updown]:
                            diff_note_count += 1.2  # 如果pattern中有而待求组合中没有，记为1.2分。（这里其实可以说是“宁缺毋滥”）
            if not note_satisfactory:
                continue
            # 1.2.如果找到符合要求的组合 将其记录下来
            pattern_diff_score = (100 * diff_note_count) // total_note_count
            if pattern_diff_score < choose_pat_diff_score:
                choose_pat = pat_it
                choose_pat_diff_score = pattern_diff_score
        # 2.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_PIANO_GUITAR_PATTERNS+1
        if choose_pat != 0:
            pattern_dx = choose_pat
        else:
            pattern_dx = COMMON_STRING_PAT_NUM + 1
        return pattern_dx


class StringTrainData:

    def __init__(self, melody_pat_data, continuous_bar_data, corenote_pat_data, common_corenote_pats, chord_cls):
        """
        :param melody_pat_data: 主旋律组合的数据
        :param continuous_bar_data: 连续小节的计数数据
        :param corenote_pat_data: 主旋律骨干音的组合数据
        :param common_corenote_pats: 主旋律骨干音组合的对照表
        :param chord_cls: 0.95开始这个参数变成和弦的类而不只是最后的和弦数据 因为需要用到和弦类的方法
        """
        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取歌的string数据
        raw_string_data = []  # 四维数组 第一维是string的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        part_count = 1
        while True:
            string_part_data = get_raw_song_data_from_dataset('string' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if string_part_data == [dict() for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_string_data.append(string_part_data)
            part_count += 1
        del part_count
        for part_it in range(len(raw_string_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_string_data[part_it][song_it] != dict():
                    raw_string_data[part_it][song_it] = flat_array(raw_string_data[part_it][song_it])
                else:
                    raw_string_data[part_it][song_it] = []  # 对于没有和弦的歌曲，将格式转化为list格式

        # 2.获取根音组合及根音-和弦配对组合
        self.string_avr_root = get_nearest_number_multiple(STRING_AVR_NOTE, 12)
        self.root_data, self.rc_pat_data, self.all_rc_pats, rc_pat_count = chord_cls.get_root_data(self.string_avr_root)
        self.rc_pat_num = len(self.all_rc_pats)
        rc_pattern_cls = BaseMusicPatterns()
        rc_pattern_cls.common_pattern_list = self.all_rc_pats
        rc_pattern_cls.pattern_number_list = rc_pat_count
        rc_pattern_cls.store('StringRC')

        # 3.将原数据转化成相对音高形式
        rel_note_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(len(raw_string_data))]
        for part_it in range(len(raw_string_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_string_data[part_it][song_it] and melody_pat_data[song_it]:
                    rel_note_list[part_it][song_it] = one_song_rel_notelist_chord(raw_string_data[part_it][song_it], self.root_data[song_it], chord_cls.chord_data[song_it])

        # 4.获取最常见的弦乐组合
        common_pattern_cls = CommonMusicPatterns(COMMON_STRING_PAT_NUM)  # 这个类可以获取常见的弦乐组合
        common_pattern_cls.train(rel_note_list, 0.25, 2, multipart=True)
        common_pattern_cls.store('String')  # 存储在sqlite文件中
        self.common_string_pats = common_pattern_cls.common_pattern_list  # 常见的旋律组合列表

        # 5.获取用于验证的数据
        # 5.1.生成每首歌的piano_guitar的前后段差异，以及piano_guitar与同时期和弦的差异
        string_confidence_config = StringConfidenceCheckConfig()
        self.StringConfidence = AccompanyConfidenceCheck(string_confidence_config)
        for part_it in range(len(raw_string_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if chord_cls.chord_data[song_it] and raw_string_data[part_it][song_it]:
                    self.StringConfidence.train_1song(raw_data=raw_string_data[part_it][song_it], chord_data=chord_cls.chord_data[song_it])
        # 5.2.找出前90%所在位置
        self.StringConfidence.calc_confidence_level(0.9)
        self.StringConfidence.store('string')

        # 6.将其编码为组合并获取模型的输入输出数据
        for part_it in range(len(raw_string_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_string_data[part_it][song_it] and melody_pat_data[song_it]:
                    string_pat_data = StringPatternEncode(self.common_string_pats, rel_note_list[part_it][song_it], 0.25, 2).music_pattern_list
                    self.get_model_io_data(string_pat_data, melody_pat_data[song_it], continuous_bar_data[song_it], corenote_pat_data[song_it], self.rc_pat_data[song_it])

        DiaryLog.warn('Generation of string train data has finished!')

    def get_model_io_data(self, string_pat_data, melody_pat_data, continuous_bar_data, corenote_data, rc_pat_data):
        """
        获取模型的输入输出数据。输入数据为过去两小节加2拍的时间编码/主旋律骨干音组合/和弦根音组合，输出内容为两小节加2拍的弦乐组合
        :param corenote_data: 一首歌的主旋律骨干音数据
        :param string_pat_data: 一首歌的弦乐数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param rc_pat_data: 一首歌的和弦根音组合数据
        """
        corenote_code_add_base = 4  # 主旋律骨干音数据编码增加的基数
        rc1_code_add_base = 4 + (COMMON_CORE_NOTE_PAT_NUM + 2)  # 和弦第一拍数据编码增加的基数
        rc2_code_add_base = 4 + (COMMON_CORE_NOTE_PAT_NUM + 2) + self.rc_pat_num  # 和弦第二拍数据编码增加的基数
        string_code_add_base = 4 + (COMMON_CORE_NOTE_PAT_NUM + 2) + self.rc_pat_num * 2  # string数据编码增加的基数

        for step_it in range(-2 * TRAIN_STRING_IO_BARS, len(string_pat_data) - 2 * TRAIN_STRING_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []

                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                beat_in_bar = (step_it % 2) * 2  # 小节内的第几拍

                # 2.input_data: 获取这两小节加两拍的主旋律和和弦,以及再前一个步长的string组合
                for ahead_step_it in range(step_it, step_it + 2 * TRAIN_STRING_IO_BARS + 1):  # 向前看10拍
                    # 2.1.添加过去两小节加两拍的主旋律骨干音和和弦-根音组合
                    if ahead_step_it >= 0:
                        ahead_bar_dx = ahead_step_it // 2
                        ahead_beat_in_bar = (ahead_step_it % 2) * 2  # 当前小节内的第几拍
                        time_add = 0 if continuous_bar_data[ahead_bar_dx] == 0 else (1 - continuous_bar_data[ahead_bar_dx] % 2) * 2
                        input_time_data.append([time_add + beat_in_bar // 2, corenote_data[ahead_step_it] + corenote_code_add_base, rc_pat_data[ahead_step_it * 2] + rc1_code_add_base, rc_pat_data[ahead_step_it * 2 + 1] + rc2_code_add_base])
                    else:
                        time_add = 0
                        ahead_beat_in_bar = (ahead_step_it % 2) * 2
                        input_time_data.append([time_add + beat_in_bar // 2, corenote_code_add_base, rc1_code_add_base, rc2_code_add_base])
                    # 2.2.添加过去两小节加一个步长的string组合
                    if ahead_step_it - 1 >= 0:
                        input_time_data[-1].append(string_pat_data[ahead_step_it - 1] + string_code_add_base)
                    else:
                        input_time_data[-1].append(string_code_add_base)
                for bar_it in range(cur_bar, cur_bar + TRAIN_STRING_IO_BARS):
                    step_dx = (4 - beat_in_bar) // 2 + (bar_it - cur_bar) * 2  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it * 4: (bar_it * 4) + 4] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], corenote_code_add_base, rc1_code_add_base, rc2_code_add_base, string_code_add_base] for t in range(step_dx)]

                # 3.添加过去10拍的string 一共5个
                # 3.1.第一个小节
                if cur_bar < 0 or melody_pat_data[cur_bar * 4: (cur_bar + 1) * 4] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的string也置为空
                    output_time_data.extend([string_code_add_base for t in range(2 - pat_step_in_bar)])
                else:
                    output_time_data.extend([t + string_code_add_base for t in string_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2]])
                # 3.2.中间小节
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_STRING_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的string置为空
                        output_time_data.extend([string_code_add_base for t in range(2)])
                    else:
                        output_time_data.extend([t + string_code_add_base for t in string_pat_data[bar_it * 2: bar_it * 2 + 2]])
                    if bar_it < 0 or melody_pat_data[bar_it * 4: (bar_it * 4) + 4] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [string_code_add_base for t in range(len(output_time_data))]
                # 3.3.最后一个小节
                output_time_data.extend([t + string_code_add_base for t in string_pat_data[(cur_bar + TRAIN_STRING_IO_BARS) * 2: (cur_bar + TRAIN_STRING_IO_BARS) * 2 + pat_step_in_bar + 1]])  # 最后一个小节的string数据

                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的string不为空,
                if melody_pat_data[(cur_bar + TRAIN_STRING_IO_BARS) * 4: (cur_bar + TRAIN_STRING_IO_BARS + 1) * 4] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_string_data = string_pat_data[(cur_bar + TRAIN_STRING_IO_BARS) * 2: (cur_bar + TRAIN_STRING_IO_BARS + 1) * 2]
                    if not set(final_bar_string_data).issubset({0, COMMON_STRING_PAT_NUM + 1}):
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except IndexError:
                pass


class StringTestData:

    def __init__(self):
        # 1.从sqlite中读取common_string_pattern
        common_pattern_cls = CommonMusicPatterns(COMMON_STRING_PAT_NUM)  # 这个类可以获取常见的string组合
        common_pattern_cls.restore('String')  # 直接从sqlite文件中读取
        self.common_string_pats = common_pattern_cls.common_pattern_list  # 常见的string组合列表

        # 2.获取和弦的根音组合
        self.string_avr_root = get_nearest_number_multiple(STRING_AVR_NOTE, 12)
        rc_pattern_cls = BaseMusicPatterns()
        rc_pattern_cls.restore('StringRC')
        self.all_rc_pats = rc_pattern_cls.common_pattern_list
        self.rc_pat_num = len(self.all_rc_pats)

        # 3.从sqlite中读取每首歌的string的前后段差异，以及string与同时期和弦的差异的合理的阈值
        string_confidence_config = StringConfidenceCheckConfig()
        self.StringConfidence = AccompanyConfidenceCheck(string_confidence_config)
        self.StringConfidence.restore('string')

        DiaryLog.warn('Restoring of string associated data has finished!')
