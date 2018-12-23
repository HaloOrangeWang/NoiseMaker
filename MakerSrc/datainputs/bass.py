from settings import *
from interfaces.music_patterns import BaseMusicPatterns, CommonMusicPatterns, MusicPatternEncode
from interfaces.sql.sqlite import get_raw_song_data_from_dataset
from interfaces.utils import flat_array, get_nearest_number_multiple, DiaryLog
from interfaces.note_format import one_song_rel_notelist_chord
from validations.bass import BassConfidenceCheckConfig
from validations.functions import AccompanyConfidenceCheck


class BassPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, pat_step_dx, raw_note_list, common_patterns):
        # 在常见的bass列表里找不到某一个bass组合的处理方法：
        # a.寻找符合以下条件的bass组合
        # a1.首音不为休止符
        # a2.该旋律组合的整拍音与待求旋律组合的整拍音相同
        # a3.该旋律组合中半拍音符与待求旋律组合对应位置的音符休止情况全部相同
        # a4.满足上述三个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为0
        choose_pat = 0  # 选取的pattern
        choose_pat_like_score = 0  # 暂时选择的pattern与待求音符组合的相似程度
        for pat_it in range(1, len(common_patterns)):
            # 1.检查首音是否为休止符
            if common_patterns[pat_it][0] == 0:
                continue
            # 2.检查两个旋律组合的整拍音是否相同
            if (common_patterns[pat_it][0] != raw_note_list[0]) or (common_patterns[pat_it][8] != raw_note_list[8]):
                continue
            # 3.检查该旋律组合中所有半拍音符与待求旋律组合对应位置的音符的休止情况是否全部相同
            note_all_same = True
            for note_it in range(0, len(common_patterns[pat_it]), 4):
                # TODO 这个if条件是不是写错了
                if common_patterns[pat_it][note_it] != 0 and common_patterns[pat_it][note_it] != raw_note_list[note_it]:
                    note_all_same = False
                    break
            if not note_all_same:
                continue
            # 4.求该旋律组合与待求旋律组合的差别
            pattern_like_score = 9  # 初始的旋律组合相似度为9分 每发现一个不同音符 按权重扣分。
            note_diff_list = [10, 2, 3, 3, 6, 3, 4, 3, 10, 2, 3, 3, 6, 3, 4, 3]  # 音符差别的权重列表
            for note_it in range(len(common_patterns[pat_it])):
                if common_patterns[pat_it][note_it] != raw_note_list[note_it]:
                    if common_patterns[pat_it][note_it] != 0 and raw_note_list[note_it] == 0:  # 当该旋律组合有一个音而待求旋律组合中相同位置为休止符时，双倍扣分
                        pattern_like_score -= 2 * note_diff_list[note_it]
                    else:
                        pattern_like_score -= note_diff_list[note_it]
            # 5.如果这个旋律组合的差别是目前最小的 则保存它
            if pattern_like_score > choose_pat_like_score:
                choose_pat_like_score = pattern_like_score
                choose_pat = pat_it
        # 6.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_BASS_PATTERNS+1
        if choose_pat_like_score > 0:
            pattern_dx = choose_pat
        else:
            pattern_dx = len(common_patterns)
        return pattern_dx


class BassTrainData:

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

        # 1.从数据集中读取bass信息，并变更为以音符步长为单位的列表
        raw_bass_data = get_raw_song_data_from_dataset('bass', None)
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_bass_data[song_it] != dict():
                raw_bass_data[song_it] = flat_array(raw_bass_data[song_it])
            else:
                raw_bass_data[song_it] = []  # 对于没有bass的歌曲，将格式转化为list格式

        # 2.获取和弦的根音组合 预期的根音平均值为36（即比中央C低2个八度）
        self.bass_avr_root = get_nearest_number_multiple(BASS_AVR_NOTE, 12)
        self.root_data, self.rc_pat_data, self.all_rc_pats, rc_pat_count = chord_cls.get_root_data(self.bass_avr_root)
        self.keypress_pat_num = len(all_keypress_pats)  # 一共有多少种按键组合数据（步长为2拍）
        self.rc_pat_num = len(self.all_rc_pats)
        rc_pattern_cls = BaseMusicPatterns()
        rc_pattern_cls.common_pattern_list = self.all_rc_pats
        rc_pattern_cls.pattern_number_list = rc_pat_count
        rc_pattern_cls.store('BassRC')

        # 3.将原bass数据转化成相对音高形式
        rel_note_list = [[] for t in range(TRAIN_FILE_NUMBERS)]
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_bass_data[song_it]:
                rel_note_list[song_it] = one_song_rel_notelist_chord(raw_bass_data[song_it], self.root_data[song_it], chord_cls.chord_data[song_it], note_time_step=0.125)

        # 4.获取最常见的Bass组合
        common_pattern_cls = CommonMusicPatterns(COMMON_BASS_PAT_NUM)  # 这个类可以获取常见的bass组合
        common_pattern_cls.train(rel_note_list, 0.125, 2, multipart=False)
        common_pattern_cls.store('Bass')  # 存储在sqlite文件中
        self.common_bass_pats = common_pattern_cls.common_pattern_list  # 常见的bass组合列表

        # 5.获取用于验证的数据
        # 5.1.生成每首歌的bass的前后段差异，以及bass与同时期和弦的差异
        bass_confidence_config = BassConfidenceCheckConfig()
        self.BassConfidence = AccompanyConfidenceCheck(bass_confidence_config)
        for song_it in range(TRAIN_FILE_NUMBERS):
            if chord_cls.chord_data[song_it] and raw_bass_data[song_it]:
                self.BassConfidence.train_1song(raw_data=raw_bass_data[song_it], chord_data=chord_cls.chord_data[song_it])
        # 5.2.找出前90%所在位置
        self.BassConfidence.calc_confidence_level(0.9)
        self.BassConfidence.store('bass')

        # 6.获取模型的输入输出数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if rel_note_list[song_it] and melody_pat_data[song_it]:
                bass_pat_data = BassPatternEncode(self.common_bass_pats, rel_note_list[song_it], 0.125, 2).music_pattern_list
                self.get_model_io_data(bass_pat_data, melody_pat_data[song_it], continuous_bar_data[song_it], keypress_pat_data[song_it], self.rc_pat_data[song_it])

        DiaryLog.warn('Generation of bass train data has finished!')

    def get_model_io_data(self, bass_pat_data, melody_pat_data, continuous_bar_data, keypress_data, rc_pat_data):
        """
        获取模型的输入输出数据。输入数据为过去两小节加两拍的时间编码/主旋律按键组合/和弦根音组合/过去三小节的bass，输出内容为两小节加两拍的bass组合
        :param bass_pat_data: 一首歌的bass数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param keypress_data: 一首歌的主旋律按键数据
        :param rc_pat_data: 一首歌的和弦根音组合数据
        """
        keypress_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
        rc1_code_add_base = 4 + self.keypress_pat_num  # 和弦第一拍数据编码增加的基数
        rc2_code_add_base = 4 + self.keypress_pat_num + self.rc_pat_num  # 和弦第二拍数据编码增加的基数
        bass_code_add_base = 4 + self.keypress_pat_num + self.rc_pat_num * 2  # bass数据编码增加的基数

        for step_it in range(-2 * TRAIN_BASS_IO_BARS, len(bass_pat_data) - 2 * TRAIN_BASS_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []

                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                beat_in_bar = (step_it % 2) * 2  # 小节内的第几拍

                # 2.input_data: 获取这两小节加两拍的主旋律和和弦,以及再前一个步长的bass组合
                for ahead_step_it in range(step_it, step_it + 2 * TRAIN_BASS_IO_BARS + 1):  # 向前看10拍
                    # 2.1.添加过去两小节加一个步长的主旋律和和弦
                    if ahead_step_it >= 0:
                        ahead_bar_dx = ahead_step_it // 2
                        ahead_beat_in_bar = (ahead_step_it % 2) * 2
                        time_add = 0 if continuous_bar_data[ahead_bar_dx] == 0 else (1 - continuous_bar_data[ahead_bar_dx] % 2) * 2
                        input_time_data.append([time_add + ahead_beat_in_bar // 2, keypress_data[ahead_step_it] + keypress_code_add_base, rc_pat_data[ahead_step_it * 2] + rc1_code_add_base, rc_pat_data[ahead_step_it * 2 + 1] + rc2_code_add_base])
                    else:
                        time_add = 0
                        ahead_beat_in_bar = (ahead_step_it % 2) * 2
                        input_time_data.append([time_add + ahead_beat_in_bar // 2, keypress_code_add_base, rc1_code_add_base, rc2_code_add_base])
                    # 2.2.添加过去两小节加一个步长的bass组合
                    if ahead_step_it - 1 >= 0:
                        input_time_data[-1].append(bass_pat_data[ahead_step_it - 1] + bass_code_add_base)
                    else:
                        input_time_data[-1].append(bass_code_add_base)
                for bar_it in range(cur_bar, cur_bar + TRAIN_BASS_IO_BARS):
                    step_dx = (4 - beat_in_bar) // 2 + (bar_it - cur_bar) * 2  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it * 4: (bar_it * 4) + 4] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], keypress_code_add_base, rc1_code_add_base, rc2_code_add_base, bass_code_add_base] for t in range(step_dx)]

                # 3.output_data: 添加过去10拍的bass 一共5个
                # 3.1.第一个小节
                if cur_bar < 0 or melody_pat_data[cur_bar * 4: (cur_bar + 1) * 4] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的bass也置为空
                    output_time_data.extend([bass_code_add_base for t in range(2 - pat_step_in_bar)])
                else:
                    output_time_data.extend([t + bass_code_add_base for t in bass_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2]])
                # 3.2.中间小节
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的bass置为空
                        output_time_data.extend([bass_code_add_base, bass_code_add_base])
                    else:
                        output_time_data.extend([t + bass_code_add_base for t in bass_pat_data[bar_it * 2: bar_it * 2 + 2]])
                    if bar_it < 0 or melody_pat_data[bar_it * 4: (bar_it * 4) + 4] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [bass_code_add_base for t in range(len(output_time_data))]
                # 3.3.最后一个小节
                output_time_data.extend([t + bass_code_add_base for t in bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS) * 2 + pat_step_in_bar + 1]])  # 最后一个小节的bass数据

                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的bass不为空
                if melody_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 4: (cur_bar + TRAIN_BASS_IO_BARS + 1) * 4] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_bass_data = bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS + 1) * 2]
                    if not set(final_bar_bass_data).issubset({0, COMMON_BASS_PAT_NUM + 1}):
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except IndexError:
                pass


class BassTestData:

    def __init__(self, all_keypress_pats):
        # 1.从sqlite中读取common_bass_pattern
        common_pattern_cls = CommonMusicPatterns(COMMON_BASS_PAT_NUM)  # 这个类可以获取常见的bass组合
        common_pattern_cls.restore('Bass')  # 直接从sqlite文件中读取
        self.common_bass_pats = common_pattern_cls.common_pattern_list  # 常见的bass组合列表

        # 2.获取和弦的根音组合
        self.bass_avr_root = get_nearest_number_multiple(BASS_AVR_NOTE, 12)
        rc_pattern_cls = BaseMusicPatterns()
        rc_pattern_cls.restore('BassRC')
        self.all_rc_pats = rc_pattern_cls.common_pattern_list
        self.rc_pat_num = len(self.all_rc_pats)
        self.keypress_pat_num = len(all_keypress_pats)  # 一共有多少种按键组合数据（步长为2拍）

        # 3.从sqlite中读取每首歌的bass的前后段差异，以及bass与同时期和弦的差异的合理的阈值
        bass_confidence_config = BassConfidenceCheckConfig()
        self.BassConfidence = AccompanyConfidenceCheck(bass_confidence_config)
        self.BassConfidence.restore('bass')

        DiaryLog.warn('Restoring of bass associated data has finished!')
