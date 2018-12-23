from settings import *
from interfaces.sql.sqlite import get_raw_song_data_from_dataset
from interfaces.utils import flat_array, get_nearest_number_multiple, DiaryLog
from interfaces.note_format import one_song_rel_notelist_chord
from interfaces.music_patterns import BaseMusicPatterns, CommonMusicPatterns, MusicPatternEncode
from validations.piano_guitar import PgConfidenceCheckConfig
from validations.functions import AccompanyConfidenceCheck


class PianoGuitarPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, pat_step_dx, raw_note_list, common_patterns):
        # 在常见的piano_guitar列表里找不到某一个piano_guitar组合的处理方法：
        # a.寻找符合以下条件的piano_guitar组合
        # a1.四个位置的休止情况完全相同
        # a2.该组合的所有音符是待求组合所有音符的子集 或与待求组合的音符相差7的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小（与最高音相差一个八度以上的差异值记为1,相差不足一个八度的差异值记为2）
        # b.记为common_patterns+1
        choose_pat = 0  # 选取的pattern
        choose_pat_diff_score = 75  # 两个旋律组合的差异程度
        for pat_it in range(1, len(common_patterns)):
            total_note_count = 0
            diff_note_count = 0
            # 1.1.检查该旋律组合是否符合要求
            note_satisfactory = True
            for note_group_it in range(len(common_patterns[pat_it])):
                # 1.1.1.检查休止情况是否相同
                if bool(common_patterns[pat_it][note_group_it]) ^ bool(raw_note_list[note_group_it]):  # 一个为休止符 一个不为休止符
                    note_satisfactory = False
                    break
                if common_patterns[pat_it][note_group_it] == 0:  # 如果这个时间步长是休止符的话，直接进入下个时间步长
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
            pattern_dx = COMMON_PG_PAT_NUM + 1
        return pattern_dx


class PianoGuitarTrainData:

    def __init__(self, melody_pat_data, continuous_bar_data, keypress_pat_data, all_keypress_pats, chord_cls):
        """
        :param melody_pat_data: 主旋律组合的数据
        :param continuous_bar_data: 连续小节的计数数据
        :param keypress_pat_data: 主旋律按键的组合数据
        :param all_keypress_pats: 主旋律组合的按键对照表
        :param chord_cls: 0.95开始这个参数变成和弦的类而不只是最后的和弦数据 因为需要用到和弦类的方法
        """
        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取歌的piano_guitar数据，并变更为以音符步长为单位的列表
        raw_pg_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        part_count = 1
        while True:
            pg_part_data = get_raw_song_data_from_dataset('piano_guitar' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if pg_part_data == [dict() for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_pg_data.append(pg_part_data)
            part_count += 1
        del part_count
        for part_it in range(len(raw_pg_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_pg_data[part_it][song_it] != dict():
                    raw_pg_data[part_it][song_it] = flat_array(raw_pg_data[part_it][song_it])
                else:
                    raw_pg_data[part_it][song_it] = []  # 对于没有和弦的歌曲，将格式转化为list格式

        # 2.获取根音组合及根音-和弦配对组合
        self.pg_avr_root = get_nearest_number_multiple(PG_AVR_NOTE, 12)
        self.root_data, self.rc_pat_data, self.all_rc_pats, rc_pat_count = chord_cls.get_root_data(self.pg_avr_root)
        self.keypress_pat_num = len(all_keypress_pats)  # 一共有多少种按键组合数据（步长为2拍）
        self.rc_pat_num = len(self.all_rc_pats)
        rc_pattern_cls = BaseMusicPatterns()
        rc_pattern_cls.common_pattern_list = self.all_rc_pats
        rc_pattern_cls.pattern_number_list = rc_pat_count
        rc_pattern_cls.store('PGRC')

        # 3.将原数据转化成相对音高形式
        rel_note_list = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(len(raw_pg_data))]
        for part_it in range(len(raw_pg_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_pg_data[part_it][song_it] and melody_pat_data[song_it]:
                    rel_note_list[part_it][song_it] = one_song_rel_notelist_chord(raw_pg_data[part_it][song_it], self.root_data[song_it], chord_cls.chord_data[song_it])

        # 4.获取最常见的PianoGuitar组合
        common_pattern_cls = CommonMusicPatterns(COMMON_PG_PAT_NUM)  # 这个类可以获取常见的piano_guitar组合
        common_pattern_cls.train(rel_note_list, 0.25, 1, multipart=True)
        common_pattern_cls.store('PianoGuitar')  # 存储在sqlite文件中
        self.common_pg_pats = common_pattern_cls.common_pattern_list  # 常见的旋律组合列表

        # 5.获取用于验证的数据
        # 5.1.生成每首歌的piano_guitar的前后段差异，以及piano_guitar与同时期和弦的差异
        pg_confidence_config = PgConfidenceCheckConfig()
        self.PgConfidence = AccompanyConfidenceCheck(pg_confidence_config)
        for part_it in range(len(raw_pg_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if chord_cls.chord_data[song_it] and raw_pg_data[part_it][song_it]:
                    self.PgConfidence.train_1song(raw_data=raw_pg_data[part_it][song_it], chord_data=chord_cls.chord_data[song_it])
        # 5.2.找出前90%所在位置
        self.PgConfidence.calc_confidence_level(0.9)
        self.PgConfidence.store('piano_guitar')

        # 6.获取模型的输入输出数据
        for part_it in range(len(raw_pg_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_pg_data[part_it][song_it] and melody_pat_data[song_it]:
                    pg_pat_data = PianoGuitarPatternEncode(self.common_pg_pats, rel_note_list[part_it][song_it], 0.25, 1).music_pattern_list
                    self.get_model_io_data(pg_pat_data, melody_pat_data[song_it], continuous_bar_data[song_it], keypress_pat_data[song_it], self.rc_pat_data[song_it])

        DiaryLog.warn('Generation of piano guitar train data has finished!')

    def get_model_io_data(self, pg_pat_data, melody_pat_data, continuous_bar_data, keypress_data, rc_pat_data):
        """
        获取模型的输入输出数据。输入数据为过去两小节加一拍的时间编码/主旋律按键组合/和弦根音组合，输出内容为两小节加一拍的piano_guitar组合
        :param pg_pat_data: 一首歌的piano_guitar数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param keypress_data: 一首歌的主旋律按键数据
        :param rc_pat_data: 一首歌的和弦根音组合数据
        """
        keypress_code_add_base = 8  # 主旋律第一拍数据编码增加的基数
        rc_code_add_base = 8 + self.keypress_pat_num  # 和弦第一拍数据编码增加的基数
        pg_code_add_base = 8 + self.keypress_pat_num + self.rc_pat_num  # piano_guitar数据编码增加的基数

        for step_it in range(-4 * TRAIN_PG_IO_BARS, len(pg_pat_data) - 4 * TRAIN_PG_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []

                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 4  # 第几小节
                beat_in_bar = pat_step_in_bar = step_it % 4

                # 2.input_data: 获取这两小节加一拍的主旋律和和弦-根音组合,以及再前一个步长的piano_guitar组合
                for ahead_step_it in range(step_it, step_it + 4 * TRAIN_PG_IO_BARS + 1):  # 向前看9拍
                    # 2.1.添加过去两小节加一个步长的主旋律和和弦-根音组合
                    if ahead_step_it >= 0:
                        ahead_bar_dx = ahead_step_it // 4
                        ahead_beat_in_bar = ahead_step_it % 4  # 当前小节内的第几拍
                        time_add = 0 if continuous_bar_data[ahead_bar_dx] == 0 else (1 - continuous_bar_data[ahead_bar_dx] % 2) * 4
                        input_time_data.append([time_add + ahead_beat_in_bar, keypress_data[ahead_step_it // 2] + keypress_code_add_base, rc_pat_data[ahead_step_it] + rc_code_add_base])
                    else:
                        time_add = 0
                        ahead_beat_in_bar = ahead_step_it % 4
                        input_time_data.append([time_add + ahead_beat_in_bar, keypress_code_add_base, rc_code_add_base])
                    # 2.2.添加过去两小节加一个步长的piano-guitar组合
                    if ahead_step_it - 1 >= 0:
                        input_time_data[-1].append(pg_pat_data[ahead_step_it - 1] + pg_code_add_base)
                    else:
                        input_time_data[-1].append(pg_code_add_base)
                for bar_it in range(cur_bar, cur_bar + TRAIN_PG_IO_BARS):
                    step_dx = (4 - beat_in_bar) + (bar_it - cur_bar) * 4  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it * 4: (bar_it * 4) + 4] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], keypress_code_add_base, rc_code_add_base, pg_code_add_base] for t in range(step_dx)]

                # 3.output_data: 添加过去9拍的piano_guitar 一共9个
                # 3.1.第一个小节
                if cur_bar < 0 or melody_pat_data[cur_bar * 4: (cur_bar + 1) * 4] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的piano_guitar也置为空
                    output_time_data.extend([pg_code_add_base for t in range(4 - pat_step_in_bar)])
                else:
                    output_time_data.extend([t + pg_code_add_base for t in pg_pat_data[cur_bar * 4 + pat_step_in_bar: cur_bar * 4 + 4]])
                # 3.2.中间小节
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_PG_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的piano_guitar置为空
                        output_time_data.extend([pg_code_add_base for t in range(4)])
                    else:
                        output_time_data.extend([t + pg_code_add_base for t in pg_pat_data[bar_it * 4: bar_it * 4 + 4]])
                    if bar_it < 0 or melody_pat_data[bar_it * 4: (bar_it * 4) + 4] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [pg_code_add_base for t in range(len(output_time_data))]
                # 3.3.最后一个小节
                output_time_data.extend([t + pg_code_add_base for t in pg_pat_data[(cur_bar + TRAIN_PG_IO_BARS) * 4: (cur_bar + TRAIN_PG_IO_BARS) * 4 + pat_step_in_bar + 1]])  # 最后一个小节的piano_guitar数据

                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的piano_guitar不为空
                if melody_pat_data[(cur_bar + TRAIN_PG_IO_BARS) * 4: (cur_bar + TRAIN_PG_IO_BARS + 1) * 4] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_pg_data = pg_pat_data[(cur_bar + TRAIN_PG_IO_BARS) * 4: (cur_bar + TRAIN_PG_IO_BARS + 1) * 4]
                    if not set(final_bar_pg_data).issubset({0, COMMON_PG_PAT_NUM + 1}):
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except IndexError:
                pass


class PianoGuitarTestData:

    def __init__(self, all_keypress_pats):
        # 1.从sqlite中读取common_piano_guitar_pattern
        common_pattern_cls = CommonMusicPatterns(COMMON_PG_PAT_NUM)  # 这个类可以获取常见的piano_guitar组合
        common_pattern_cls.restore('PianoGuitar')  # 直接从sqlite文件中读取
        self.common_pg_pats = common_pattern_cls.common_pattern_list  # 常见的bass组合列表

        # 2.获取和弦的根音组合
        self.pg_avr_root = get_nearest_number_multiple(PG_AVR_NOTE, 12)
        rc_pattern_cls = BaseMusicPatterns()
        rc_pattern_cls.restore('PGRC')
        self.all_rc_pats = rc_pattern_cls.common_pattern_list
        self.rc_pat_num = len(self.all_rc_pats)
        self.keypress_pat_num = len(all_keypress_pats)  # 一共有多少种按键组合数据（步长为2拍）

        # 3.从sqlite中读取每首歌的piano_guitar的前后段差异，以及piano_guitar与同时期和弦的差异的合理的阈值
        pg_confidence_config = PgConfidenceCheckConfig()
        self.PgConfidence = AccompanyConfidenceCheck(pg_confidence_config)
        self.PgConfidence.restore('piano_guitar')

        DiaryLog.warn('Restoring of piano_guitar associated data has finished!')
