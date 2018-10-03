from settings import *
import numpy as np
from interfaces.utils import get_dict_max_key, flat_array, DiaryLog
from interfaces.sql.sqlite import get_raw_song_data_from_dataset
from datainputs.functions import one_song_rel_notelist_chord, RhythmTrainData
from interfaces.music_patterns import MusicPatternEncodeStep, CommonMusicPatterns
from interfaces.chord_parse import notelist2chord
from interfaces.utils import DiaryLog
from interfaces.sql.sqlite import NoteDict
import copy


class PianoGuitarPatternEncode(MusicPatternEncodeStep):

    def handle_rare_pattern(self, pattern_dx, raw_note_list, common_patterns):
        # 在常见的piano_guitar列表里找不到某一个piano_guitar组合的处理方法：
        # a.寻找符合以下条件的piano_guitar组合
        # a1.四个位置的休止情况完全相同
        # a2.该组合的所有音符是待求组合所有音符的子集 或与待求组合的音符相差7的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小（与最高音相差一个八度以上的差异值记为1,相差不足一个八度的差异值记为2）
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
                    # 1.1.1.检查休止情况是否相同
                    if bool(common_patterns[common_pat_it][note_group_it]) ^ bool(raw_note_list[note_group_it]):  # 一个为休止符 一个不为休止符
                        note_satisfactory = False
                        break
                    if common_patterns[common_pat_it][note_group_it] == 0:  # 如果这个时间步长是休止符的话，直接进入下个时间步长
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
                pattern_dx = COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1
        # print(bar_pattern_list, '\n')
        return pattern_dx


class PGRhythmTrainData(RhythmTrainData):

    def get_raw_rhythm_data(self):
        raw_pg_rhythm_data = []  # 三维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是歌曲内容
        part_count = 1
        while True:
            pg_part_data = get_raw_song_data_from_dataset('piano_guitar' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if pg_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            for song_it in range(TRAIN_FILE_NUMBERS):
                if pg_part_data[song_it] != {}:  # 将每首歌的相关数据由以小节为单位转成以时间步长为单位
                    pg_part_data[song_it] = flat_array(pg_part_data[song_it])
            raw_pg_rhythm_data.append(pg_part_data)
            part_count += 1
        for part_it in range(len(raw_pg_rhythm_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_pg_rhythm_data[part_it][song_it] != {}:
                    # for bar_iterator in range(get_dict_max_key(raw_pg_rhythm_data[part_it][song_it]) + 1):
                    raw_pg_rhythm_data[part_it][song_it] = [1 if t != 0 else 0 for t in raw_pg_rhythm_data[part_it][song_it]]
        return raw_pg_rhythm_data


class PianoGuitarTrainData:

    def __init__(self, melody_pat_data, continuous_bar_data, chord_cls, keypress_pat_data, keypress_pat_ary):
        """
        :param melody_pat_data: 主旋律组合的数据
        :param continuous_bar_data: 连续小节的计数数据
        :param chord_cls: 0.95开始这个参数变成和弦的类而不只是最后的和弦数据 因为需要用到和弦类的方法
        :param keypress_pat_data: 主旋律按键的组合数据
        :param keypress_pat_ary: 主旋律组合的按键对照表
        """
        # 1.从数据集中读取歌的piano_guitar数据
        raw_pg_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        part_count = 1
        while True:
            pg_part_data = get_raw_song_data_from_dataset('piano_guitar' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if pg_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_pg_data.append(pg_part_data)
            # flatten_pg_data += pg_part_data
            part_count += 1
        self.train_song_list = [[True for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]  # 用于训练的歌曲id列表
        # 2.获取根音组合及根音-和弦配对组合
        self.root_data, self.rc_pattern_data, self.rc_pattern_count = chord_cls.get_root_data(PIANO_GUITAR_AVERAGE_ROOT)
        self.transfer_count = np.zeros([COMMON_PIANO_GUITAR_PATTERN_NUMBER, COMMON_PIANO_GUITAR_PATTERN_NUMBER])  # 统计数据时把0和common+1的数据排除在外
        self.emission_count = np.zeros([len(self.rc_pattern_count) - 1, COMMON_PIANO_GUITAR_PATTERN_NUMBER])  # “dict长度”行，pattern个数列。0和common+1被排除在外
        self.pi_count = np.zeros([COMMON_PIANO_GUITAR_PATTERN_NUMBER])
        # 3.将原数据转化成相对音高形式
        rel_note_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        for part_it in range(len(raw_pg_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_pg_data[part_it][song_it] != {} and melody_pat_data[song_it] != {}:
                    rel_note_list[part_it][song_it] = one_song_rel_notelist_chord(raw_pg_data[part_it][song_it], self.root_data[song_it], chord_cls.chord_data[song_it])
        # 4.获取最常见的PianoGuitar组合
        common_pattern_cls = CommonMusicPatterns(COMMON_PIANO_GUITAR_PATTERN_NUMBER)  # 这个类可以获取常见的主旋律组合
        if FLAG_IS_TRAINING is True:  # 训练模式
            common_pattern_cls.train(rel_note_list, 0.25, 1, multipart=True)
            common_pattern_cls.store('PianoGuitar')  # 存储在sqlite文件中
        else:
            common_pattern_cls.restore('PianoGuitar')  # 直接从sqlite文件中读取
        self.common_pg_pats = common_pattern_cls.common_pattern_list  # 常见的旋律组合列表
        # 5.将其编码为组合并获取模型的输入输出数据
        # pg_pat_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        for part_it in range(len(raw_pg_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_pg_data[part_it][song_it] != {} and melody_pat_data[song_it] != {}:
                    pg_pat_list = PianoGuitarPatternEncode(self.common_pg_pats, rel_note_list[part_it][song_it], 0.25, 1).music_pattern_ary
                    self.count(pg_pat_list, self.rc_pattern_data[song_it])
                    self.count_initial(pg_pat_list, self.rc_pattern_data[song_it], continuous_bar_data[song_it])
        self.prob_convert()
        # 6.获取节奏数据 并将节奏及音高进行整合
        self.pg_rhythm_data = PGRhythmTrainData(keypress_pat_data, keypress_pat_ary, continuous_bar_data, 0.25, 2)
        self.combine_rhythm_chord()

    def count(self, pg_data, rc_pattern):
        """获取观测矩阵和状态转移矩阵的计数"""
        # for key in pg_data:
        #     if key not in rc_pattern:  # 这一小节没有和弦 直接跳过
        #         continue
        for step_it in range(len(pg_data)):
            if step_it >= len(rc_pattern):  # 这一小节没有和弦 直接跳过
                continue
            if step_it == 0:  # 不考虑第一小节的第一拍
                continue
            if step_it != 0:
                if (pg_data[step_it] in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1]) or (pg_data[step_it - 1] in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1]) or rc_pattern[step_it - 1] == 0:  # 当前拍 上一拍的伴奏编码不能是0或common+1 同时当前拍和弦根音编码不能是0
                    continue
                self.transfer_count[pg_data[step_it - 1] - 1, pg_data[step_it] - 1] += 1
                self.emission_count[rc_pattern[step_it] - 1, pg_data[step_it] - 1] += 1

    def count_initial(self, pg_data, rc_pattern, continuous_bar_data):
        # TODO 这里可能有问题
        """获取初始状态矩阵的计数"""
        for bar_dx in range(0, len(pg_data), 4):  # 遍历每个小节
            if (bar_dx // 4) >= len(continuous_bar_data):  # 这个小节没有对应的主旋律 直接跳过
                continue
            if bar_dx == 0:
                if continuous_bar_data[0] != 0:
                    if pg_data[0] not in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1] and rc_pattern[0] != 0:
                        self.pi_count[pg_data[0] - 1] += 1
            else:
                if continuous_bar_data[bar_dx // 4] != 0 and continuous_bar_data[(bar_dx // 4) - 1] == 0:
                    if pg_data[bar_dx // 4] not in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1] and rc_pattern[bar_dx // 4] != 0:
                        self.pi_count[pg_data[bar_dx // 4] - 1] += 1

    def prob_convert(self):
        """将组合的变化规律由计数转变为概率"""
        self.transfer = np.zeros(self.transfer_count.shape)  # 概率矩阵与计数矩阵的大小相同
        self.emission = np.zeros(self.emission_count.shape)
        self.pi = np.zeros([COMMON_PIANO_GUITAR_PATTERN_NUMBER])
        # 1.计算转移矩阵
        for row_it, row in enumerate(self.transfer_count):
            row_sum = sum(row)
            if row_sum == 0:
                self.transfer[row_it] = [1 / COMMON_PIANO_GUITAR_PATTERN_NUMBER for t in range(COMMON_PIANO_GUITAR_PATTERN_NUMBER)]  # 如果数据集中这个状态没有接任何一个下一个状态，则概率均分
            else:
                self.transfer[row_it] = self.transfer_count[row_it] / row_sum
        # 2.计算观测矩阵
        for column_it in range(COMMON_PIANO_GUITAR_PATTERN_NUMBER):
            column_sum = sum(self.emission_count[:, column_it])
            if column_sum == 0:
                self.emission[:, column_it] = [1 / self.emission.shape[0] for t in range(self.emission.shape[0])]  # 为空列则概率均分
            else:
                self.emission[:, column_it] = self.emission_count[:, column_it] / column_sum
        # 3.计算初始化转移向量
        self.pi = self.pi_count / sum(self.pi_count)

    def combine_rhythm_chord(self):
        """将节奏 和弦和根音进行整合 获取最终的放射矩阵"""
        pg_pat_rhythm = [[1 if t0 != 0 else 0 for t0 in t1] for t1 in self.common_pg_pats[1:]]
        pg_pat_rhythm = [8 * t[0] + 4 * t[1] + 2 * t[2] + t[3] for t in pg_pat_rhythm]  # 每一个pattern对应的节奏（有音符为１　没音符为０）
        pg_rhythm_match = np.array([[int(pg_pat_rhythm[t0] == t1) for t0 in range(COMMON_PIANO_GUITAR_PATTERN_NUMBER)] for t1 in range(16)])
        # self.gg = pg_rhythm_match
        self.emission_final = np.zeros([16 * self.emission.shape[0], COMMON_PIANO_GUITAR_PATTERN_NUMBER])  # 矩阵 行为16×rc组合数量（因为每一个小节的节拍有2**4=16种可能） 列为这些PG组合
        for rhythm_it in range(pg_rhythm_match.shape[0]):
            for rc_it in range(self.emission.shape[0]):
                self.emission_final[rc_it * pg_rhythm_match.shape[0] + rhythm_it] = pg_rhythm_match[rhythm_it] * self.emission[rc_it]
                if sum(self.emission_final[rc_it * pg_rhythm_match.shape[0] + rhythm_it]) == 0:  # 如果没有到达它的状态，可能会引发bug 这里给一个定值
                    self.emission_final[rc_it * pg_rhythm_match.shape[0] + rhythm_it] = 0.0001


def get_diff_value(raw_pg_data, chord_data):
    # 计算连续八拍piano_guitar的偏离程度
    # 评定标准有三个:
    # A.后四拍的piano_guitar的平均音高与四拍前的相减 之后除七
    # B.后四拍的piano_guitar的按键与四拍前的相减 计算杰卡德距离
    # C.每两拍的piano_guitar与同时期和弦的包含关系 计算方法是 不包含量/全部（如果两拍内有两个和弦 取前一个作为这两拍的和弦）
    #   如果piano_guitar可以构成一个和弦 但与当前和弦不能匹配的 额外增加0.25个差异分
    # D.之后对上述量取平方和作为piano_guitar的误差
    # 舍弃包含了未知和弦的数据和连续四拍以上没有piano_guitar的数据
    diff_score_ary = []
    chord_diff_score_by_pat = [0, 0]  # 每个步长的piano_guitar与同时期和弦的差异分 如果同时期的和弦为未知和弦 则分数标记为-1
    chord_backup = 0  # 用于存储上两拍的和弦
    pg_backup = []  # 用于存储上两拍的最后一个piano_guitar音符
    # 在检验的时候 一个步长是两拍而不是一拍
    for pat_step_it in range(2, int(len(raw_pg_data) / 8) - 3):  # 前四拍不看 因为没有四拍前的数据
        note_dx = pat_step_it * 8  # piano_guitar的最小音符是16分,所以这里是乘八
        try:
            # 1.计算平均音高之间的差异
            note_ary = []  # 这四拍中有几个音符 它们的音高分别是多少
            note_ary_old = []
            note_count = 0  # 八拍之前的四拍中有几个音符 它们的音高分别是多少
            note_count_old = 0
            for note_it in range(note_dx + 16, note_dx + 32):
                if raw_pg_data[note_it] != 0:
                    note_ary.extend(NoteDict[raw_pg_data[note_it]])
                    note_count += len(NoteDict[raw_pg_data[note_it]])
                if raw_pg_data[note_it - 32] != 0:
                    note_ary_old.extend(NoteDict[raw_pg_data[note_it - 32]])
                    note_count_old += len(NoteDict[raw_pg_data[note_it - 32]])
            if note_count == 0 or note_count_old == 0:
                note_diff_score = 0
            else:
                avr_note = sum(note_ary) / note_count  # 四拍所有音符的平均音高
                avr_note_old = sum(note_ary_old) / note_count_old  # 八拍之前的四拍所有音符的平均音高
                note_diff_score = abs(avr_note - avr_note_old) / 7  # 音高的差异（如果是176543的话差异分刚好为1分）
            # 2.计算按键的差异
            note_same_count = 0  # 四拍中有几个相同按键位置
            note_diff_count = 0  # 四拍中有几个不同的按键位置
            for note_it in range(note_dx + 16, note_dx + 32):
                if bool(raw_pg_data[note_it]) ^ bool(raw_pg_data[note_it - 32]):
                    note_diff_count += 1
                elif raw_pg_data[note_it] != 0 and raw_pg_data[note_it - 32] != 0:
                    note_same_count += 1
            if note_same_count == 0 and note_diff_count == 0:
                keypress_diff_score = 0  # 按键的差异分
            else:
                keypress_diff_score = note_diff_count / (note_same_count + note_diff_count)
            # 3.计算与同时期和弦的差异分
            for forw_step_it in range(len(chord_diff_score_by_pat), pat_step_it + 4):
                bar_dx = forw_step_it // 2  # 当前小节以及这一步在小节中的位置
                step_in_bar = forw_step_it % 2
                chord_diff_score_1step = 0  # 与同期和弦的差异分
                note_diff_count = 0  # 和弦内不包含的piano_guitar音符数
                if chord_data[bar_dx][step_in_bar * 2] == 0 and chord_data[bar_dx][step_in_bar * 2 + 1] == 0:
                    chord_diff_score_by_pat.append(-1)  # 出现了未知和弦 差异分标记为-1
                    continue
                abs_notelist = []
                for note_it in range(forw_step_it * 8, forw_step_it * 8 + 8):
                    if raw_pg_data[note_it] != 0:
                        abs_notelist.extend(NoteDict[raw_pg_data[note_it]])
                        pg_backup = NoteDict[raw_pg_data[note_it]]
                pg_chord = notelist2chord(set(abs_notelist))  # 这个piano_guitar是否可以找出一个特定的和弦
                div12_note_list = []  # 把所有音符对12取余数的结果 用于分析和和弦之间的重复音关系
                for note in abs_notelist:
                    div12_note_list.append(note % 12)
                if pg_chord != 0:
                    if pg_chord != chord_data[bar_dx][step_in_bar * 2] and pg_chord != chord_data[bar_dx][step_in_bar * 2 + 1]:  # 这个piano_guitar组合与当前和弦不能匹配 额外增加0.5个差异分
                        chord_diff_score_1step += 0.25
                if chord_data[bar_dx][step_in_bar * 2] != 0:
                    chord_set = copy.deepcopy(CHORD_DICT[chord_data[bar_dx][step_in_bar * 2]])  # 和弦的音符列表(已对12取余数)
                    chord_dx = chord_data[bar_dx][step_in_bar * 2]
                else:
                    chord_set = copy.deepcopy(CHORD_DICT[chord_data[bar_dx][step_in_bar * 2 + 1]])
                    chord_dx = chord_data[bar_dx][step_in_bar * 2 + 1]
                if len(abs_notelist) == 0:  # 这两拍的piano_guitar没有音符。如果前两拍也没有则给值-1。如果和弦没变则赋前值 如果和弦变了则以前两拍最后一个音作为判断依据
                    if len(pg_backup) == 0:  # 前两拍也没有
                        chord_diff_score_by_pat.append(-1)
                        continue
                    elif chord_dx == chord_backup:
                        chord_diff_score_by_pat.append(chord_diff_score_by_pat[-1])  # 和弦没变 赋前值
                        continue
                    else:  # 用上两拍的最后一个bass组合来进行计算
                        abs_notelist = pg_backup
                        div12_note_list = [note % 12 for note in abs_notelist]
                chord_backup = chord_dx
                if 1 <= chord_dx <= 72 and chord_dx % 6 == 1:  # 大三和弦 chord_set增加大七度和小七度
                    chord_set.add((chord_dx // 6 + 10) % 12)
                    chord_set.add((chord_dx // 6 + 11) % 12)
                if 1 <= chord_dx <= 72 and chord_dx % 6 == 2:  # 小三和弦 chord_set增加小七度
                    chord_set.add((chord_dx // 6 + 10) % 12)
                for note in div12_note_list:
                    if note not in chord_set:
                        note_diff_count += 1
                chord_diff_score_1step += note_diff_count / len(abs_notelist)  # piano-guitar与同期和弦的差异分
                chord_diff_score_by_pat.append(chord_diff_score_1step)
            if -1 in chord_diff_score_by_pat[pat_step_it: pat_step_it + 4]:  # 中间有未知和弦 直接跳过
                continue
            if raw_pg_data[note_dx: note_dx + 16] == [0] * 16 or raw_pg_data[note_dx + 16: note_dx + 32] == [0] * 16:  # 连续四拍没有piano_guitar的情况
                continue
            chord_diff_score = chord_diff_score_by_pat[pat_step_it: pat_step_it + 4]  # 这四拍的和弦差异分
            total_diff_score = note_diff_score * note_diff_score + keypress_diff_score * keypress_diff_score + sum([t * t for t in chord_diff_score])
            diff_score_ary.append(total_diff_score)
        except IndexError:
            pass
        except KeyError:
            pass
    # print(len(chord_diff_score_by_pat))
    # print(len(raw_bass_data) // 16)
    # assert len(chord_diff_score_by_pat) == len(raw_bass_data) // 16
    return diff_score_ary


class PianoGuitarTrainData2:

    def __init__(self, melody_pat_data, continuous_bar_data, keypress_data, keypress_pat_ary, chord_cls):
        """
        :param melody_pat_data: 主旋律组合的数据
        :param continuous_bar_data: 连续小节的计数数据
        :param chord_cls: 0.95开始这个参数变成和弦的类而不只是最后的和弦数据 因为需要用到和弦类的方法
        :param keypress_data: 主旋律按键的组合数据
        :param keypress_pat_ary: 主旋律组合的按键对照表
        """
        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取歌的piano_guitar数据
        raw_pg_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        part_count = 1
        while True:
            pg_part_data = get_raw_song_data_from_dataset('piano_guitar' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if pg_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_pg_data.append(pg_part_data)
            # flatten_pg_data += pg_part_data
            part_count += 1
        self.train_song_list = [[True for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]  # 用于训练的歌曲id列表
        # 2.获取根音组合及根音-和弦配对组合
        self.root_data, self.rc_pattern_data, self.rc_pattern_count = chord_cls.get_root_data(PIANO_GUITAR_AVERAGE_ROOT)
        self.keypress_pat_num = len(keypress_pat_ary)  # 一共有多少种按键组合数据（步长为2拍）
        self.rc_pat_num = len(self.rc_pattern_count)
        # 3.将原数据转化成相对音高形式
        rel_note_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        for part_it in range(len(raw_pg_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_pg_data[part_it][song_it] != {} and melody_pat_data[song_it] != {}:
                    rel_note_list[part_it][song_it] = one_song_rel_notelist_chord(raw_pg_data[part_it][song_it], self.root_data[song_it], chord_cls.chord_data[song_it])
        # 4.获取最常见的PianoGuitar组合
        common_pattern_cls = CommonMusicPatterns(COMMON_PIANO_GUITAR_PATTERN_NUMBER)  # 这个类可以获取常见的piano_guitar组合
        if FLAG_IS_TRAINING is True:  # 训练模式
            common_pattern_cls.train(rel_note_list, 0.25, 1, multipart=True)
            common_pattern_cls.store('PianoGuitar')  # 存储在sqlite文件中
        else:
            common_pattern_cls.restore('PianoGuitar')  # 直接从sqlite文件中读取
        self.common_pg_pats = common_pattern_cls.common_pattern_list  # 常见的旋律组合列表
        # 5.将其编码为组合并获取模型的输入输出数据
        # pg_pat_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        self.pg_pat_data = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(part_count)]
        for part_it in range(len(raw_pg_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_pg_data[part_it][song_it] != {} and melody_pat_data[song_it] != {}:
                    self.pg_pat_data[part_it][song_it] = PianoGuitarPatternEncode(self.common_pg_pats, rel_note_list[part_it][song_it], 0.25, 1).music_pattern_ary
                    self.get_model_io_data(self.pg_pat_data[part_it][song_it], melody_pat_data[song_it], continuous_bar_data[song_it], keypress_data[song_it], self.rc_pattern_data[song_it])
        # 6.获取用于验证的数据
        pg_prob_ary = []
        for part_it in range(len(raw_pg_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if chord_cls.chord_data[song_it] != {} and self.pg_pat_data[part_it][song_it] != {}:
                    pg_prob_ary.extend(get_diff_value(flat_array(raw_pg_data[part_it][song_it]), chord_cls.chord_data[song_it]))
        # 7.找出前90%所在位置
        pg_prob_ary = sorted(pg_prob_ary)
        prob_09_dx = int(len(pg_prob_ary) * 0.9 + 1)
        self.ConfidenceLevel = pg_prob_ary[prob_09_dx]

    def get_model_io_data(self, pg_pat_data, melody_pat_data, continuous_bar_data, keypress_data, rc_pat_data):
        """
        获取模型的输入输出数据。输入数据为过去两小节加一拍的时间编码/主旋律按键组合/和弦根音组合，输出内容为两小节加一拍的piano_guitar组合
        :param keypress_data: 一首歌的主旋律按键数据
        :param pg_pat_data: 一首歌的piano_guitar数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param rc_pat_data: 一首歌的和弦根音组合数据
        """
        for step_it in range(-4 * TRAIN_PIANO_GUITAR_IO_BARS, len(pg_pat_data) - 4 * TRAIN_PIANO_GUITAR_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []
                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 4  # 第几小节
                pat_step_in_bar = step_it % 4
                time_in_bar = step_it % 4  # 小节内的第几拍
                # 2.获取这两小节加两拍的主旋律和和弦,以及再前一个步长的piano_guitar组合
                keypress_code_add_base = 8  # 主旋律第一拍数据编码增加的基数
                rc_code_add_base = 8 + self.keypress_pat_num  # 和弦第一拍数据编码增加的基数
                pg_code_add_base = 8 + self.keypress_pat_num + self.rc_pat_num  # piano_guitar数据编码增加的基数
                for forw_step_it in range(step_it, step_it + 4 * TRAIN_PIANO_GUITAR_IO_BARS + 1):  # 向前看9拍
                    if forw_step_it >= 0:
                        bar_dx = forw_step_it // 4
                        beat_in_bar = forw_step_it % 4  # 当前小节内的第几拍
                        time_add = 0 if continuous_bar_data[bar_dx] == 0 else (1 - continuous_bar_data[bar_dx] % 2) * 4
                        input_time_data.append([time_add + beat_in_bar, keypress_data[forw_step_it // 2] + keypress_code_add_base, rc_pat_data[forw_step_it] + rc_code_add_base])
                    else:
                        time_add = 0
                        beat_in_bar = forw_step_it % 4
                        input_time_data.append([time_add + beat_in_bar, keypress_code_add_base, rc_code_add_base])
                    if forw_step_it - 1 >= 0:
                        input_time_data[-1].append(pg_pat_data[forw_step_it - 1] + pg_code_add_base)
                    else:
                        input_time_data[-1].append(pg_code_add_base)
                for bar_it in range(cur_bar, cur_bar + TRAIN_PIANO_GUITAR_IO_BARS):
                    step_dx = (4 - time_in_bar) + (bar_it - cur_bar) * 4  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], keypress_code_add_base, rc_code_add_base, pg_code_add_base] for t in range(step_dx)]
                # 3.添加过去9拍的piano_guitar 一共9个
                if cur_bar < 0 or melody_pat_data[cur_bar] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的piano_guitar也置为空
                    output_time_data.extend([pg_code_add_base for t in range(4 - pat_step_in_bar)])
                else:
                    output_time_data.extend([t + pg_code_add_base for t in pg_pat_data[cur_bar * 4 + pat_step_in_bar: cur_bar * 4 + 4]])
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_PIANO_GUITAR_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的piano_guitar置为空
                        output_time_data.extend([pg_code_add_base for t in range(4)])
                    else:
                        output_time_data.extend([t + pg_code_add_base for t in pg_pat_data[bar_it * 4: bar_it * 4 + 4]])
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [pg_code_add_base for t in range(len(output_time_data))]
                output_time_data.extend([t + pg_code_add_base for t in pg_pat_data[(cur_bar + TRAIN_PIANO_GUITAR_IO_BARS) * 4: (cur_bar + TRAIN_PIANO_GUITAR_IO_BARS) * 4 + pat_step_in_bar + 1]])  # 最后一个小节的piano_guitar数据
                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的piano_guitar不为空,
                if melody_pat_data[cur_bar + TRAIN_PIANO_GUITAR_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_pg_data = pg_pat_data[(cur_bar + TRAIN_PIANO_GUITAR_IO_BARS) * 4: (cur_bar + TRAIN_PIANO_GUITAR_IO_BARS + 1) * 4]
                    if not set(final_bar_pg_data).issubset({0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1}):
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass
