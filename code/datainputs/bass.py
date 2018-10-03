from interfaces.sql.sqlite import get_raw_song_data_from_dataset, NoteDict
from datainputs.functions import one_song_rel_notelist_chord
from settings import *
from interfaces.music_patterns import MusicPatternEncodeStep, CommonMusicPatterns
from interfaces.utils import flat_array
from interfaces.chord_parse import chord_rootnote, notelist2chord
import numpy as np
import copy

# 从0.94.02开始 bass的训练方式变成相对音高形式


class BassPatternEncode(MusicPatternEncodeStep):

    def handle_rare_pattern(self, pattern_dx, raw_note_list, common_patterns):
        # 在常见的bass列表里找不到某一个bass组合的处理方法：
        # a.寻找符合以下条件的bass组合
        # a1.首音不为休止符
        # a2.该旋律组合的整拍音与待求旋律组合的整拍音相同
        # a3.该旋律组合中半拍音符与待求旋律组合对应位置的音符休止情况全部相同
        # a4.满足上述三个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为0
        if pattern_dx == -1:
            choose_pat = 0  # 选取的pattern
            choose_pat_like_score = 0  # 暂时选择的pattern与待求音符组合的相似程度
            for common_pat_it in range(1, len(common_patterns)):
                # 1.检查首音是否为休止符
                if common_patterns[common_pat_it][0] == 0:
                    continue
                # 2.检查两个旋律组合的整拍音是否相同
                if (common_patterns[common_pat_it][0] != raw_note_list[0]) or (common_patterns[common_pat_it][8] != raw_note_list[8]):
                    continue
                # 3.检查该旋律组合中所有半拍音符与待求旋律组合对应位置的音符的休止情况是否全部相同
                note_all_same = True
                for note_it in range(0, len(common_patterns[common_pat_it]), 4):
                    if common_patterns[common_pat_it][note_it] != 0 and common_patterns[common_pat_it][note_it] != raw_note_list[note_it]:
                        note_all_same = False
                        break
                if not note_all_same:
                    continue
                # 4.求该旋律组合与待求旋律组合的差别
                pattern_like_score = 9  # 初始的旋律组合相似度为9分 每发现一个不同音符 按权重扣分。
                note_diff_list = [10, 2, 3, 3, 6, 3, 4, 3, 10, 2, 3, 3, 6, 3, 4, 3]  # 音符差别的权重列表
                for note_it in range(len(common_patterns[common_pat_it])):
                    if common_patterns[common_pat_it][note_it] != raw_note_list[note_it]:
                        if common_patterns[common_pat_it][note_it] != 0 and raw_note_list[note_it] == 0:  # 当该旋律组合有一个音而待求旋律组合中相同位置为休止符时，双倍扣分
                            pattern_like_score -= 2 * note_diff_list[note_it]
                        else:
                            pattern_like_score -= note_diff_list[note_it]
                # 5.如果这个旋律组合的差别是目前最小的 则保存它
                # print(common_melody_iterator, pattern_like_score)
                if pattern_like_score > choose_pat_like_score:
                    choose_pat_like_score = pattern_like_score
                    choose_pat = common_pat_it
            # 6.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_BASS_PATTERNS+1
            if choose_pat_like_score > 0:
                pattern_dx = choose_pat
            else:
                pattern_dx = len(common_patterns)
        return pattern_dx


class BassTrainData:

    def __init__(self, melody_pat_data, continuous_bar_data, chord_cls):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取bass信息
        raw_bass_data = get_raw_song_data_from_dataset('bass', None)
        self.raw_bass_data = copy.deepcopy(raw_bass_data)
        # 2.获取和弦的根音组合 预期的根音平均值为36（即比中央C低2个八度）
        self.root_data, self.rc_pattern_data, self.rc_pattern_count = chord_cls.get_root_data(BASS_AVERAGE_ROOT)
        # 3.将原数据转化成相对音高形式 并转换成以曲为单位的形式
        rel_note_list = [[] for t in range(TRAIN_FILE_NUMBERS)]
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_bass_data[song_it] != {}:
                rel_note_list[song_it] = one_song_rel_notelist_chord(raw_bass_data[song_it], self.root_data[song_it], chord_cls.chord_data[song_it], unit='bar', note_time_step=0.125)
        # 4.获取最常见的Bass组合
        common_pattern_cls = CommonMusicPatterns(COMMON_BASS_PATTERN_NUMBER)  # 这个类可以获取常见的主旋律组合
        if FLAG_IS_TRAINING is True:  # 训练模式
            common_pattern_cls.train(rel_note_list, 0.125, 2, multipart=False)
            common_pattern_cls.store('Bass')  # 存储在sqlite文件中
        else:
            common_pattern_cls.restore('Bass')  # 直接从sqlite文件中读取
        self.common_bass_pats = common_pattern_cls.common_pattern_list  # 常见的bass组合列表
        # print(common_pattern_cls.sum1, common_pattern_cls.sum2)
        # 5.获取模型的输入输出数据
        self.bass_pat_data = [{} for t in range(TRAIN_FILE_NUMBERS)]  # bass数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if rel_note_list[song_it] != [] and melody_pat_data[song_it] != {}:
                self.bass_pat_data[song_it] = BassPatternEncode(self.common_bass_pats, rel_note_list[song_it], 0.125, 2).music_pattern_ary
                self.get_model_io_data(self.bass_pat_data[song_it], melody_pat_data[song_it], continuous_bar_data[song_it], chord_cls.chord_data[song_it])

    def get_model_io_data(self, bass_pat_data, melody_pat_data, continuous_bar_data, chord_data):
        for step_it in range(-2 * TRAIN_BASS_IO_BARS, len(bass_pat_data) - 2 * TRAIN_BASS_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                # 1.添加当前时间的编码（0-3）
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                time_in_bar = (step_it % 2) * 2  # 小节内的第几拍
                time_add = (1 - continuous_bar_data[cur_bar + TRAIN_BASS_IO_BARS] % 2) * 2
                input_time_data = [pat_step_in_bar + time_add]
                output_time_data = [pat_step_in_bar + time_add]
                # 2.添加过去10拍的主旋律
                input_time_data.extend(melody_pat_data[cur_bar][time_in_bar:])
                output_time_data.extend(melody_pat_data[cur_bar][time_in_bar:])
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
                    input_time_data.extend(melody_pat_data[bar_it])
                    output_time_data.extend(melody_pat_data[bar_it])
                    if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        input_time_data = [input_time_data[0]] + [0 for t in range(len(input_time_data) - 1)]
                        output_time_data = [output_time_data[0]] + [0 for t in range(len(output_time_data) - 1)]
                input_time_data.extend(melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS][:2 + time_in_bar])
                output_time_data.extend(melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS][:2 + time_in_bar])
                # 3.添加过去10拍的和弦
                if melody_pat_data[cur_bar] == [0 for t in range(4)]:
                    input_time_data.extend([0 for t in range(4 - time_in_bar)])
                    output_time_data.extend([0 for t in range(4 - time_in_bar)])
                else:
                    input_time_data.extend(chord_data[cur_bar][time_in_bar:])
                    output_time_data.extend(chord_data[cur_bar][time_in_bar:])
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
                    input_time_data.extend(chord_data[bar_it])
                    output_time_data.extend(chord_data[bar_it])
                    if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        input_time_data = input_time_data[:11] + [0 for t in range(len(input_time_data) - 11)]  # 1的时间编码+10的主旋律 所以是11
                        output_time_data = output_time_data[:11] + [0 for t in range(len(output_time_data) - 11)]
                input_time_data.extend(chord_data[cur_bar + TRAIN_BASS_IO_BARS][:2 + time_in_bar])
                output_time_data.extend(chord_data[cur_bar + TRAIN_BASS_IO_BARS][:2 + time_in_bar])
                # 4.添加过去8拍的bass 一共4个 输出比输入错后一个step
                if cur_bar < 0 or melody_pat_data[cur_bar] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的bass也置为空
                    input_time_data.extend([0 for t in range(2 - pat_step_in_bar)])
                    output_time_data.extend([0 for t in range(1 - pat_step_in_bar)])
                else:
                    input_time_data.extend(bass_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2])
                    output_time_data.extend(bass_pat_data[cur_bar * 2 + 1 + pat_step_in_bar: cur_bar * 2 + 2])
                for bar_iterator in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
                    if bar_iterator < 0:  # 当处于负拍时 这个小节对应的bass置为空
                        input_time_data.extend([0, 0])
                        output_time_data.extend([0, 0])
                    else:
                        input_time_data.extend(bass_pat_data[bar_iterator * 2: bar_iterator * 2 + 2])
                        output_time_data.extend(bass_pat_data[bar_iterator * 2: bar_iterator * 2 + 2])
                    if melody_pat_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        input_time_data = input_time_data[:21] + [0 for t in range(len(input_time_data) - 21)]  # 1的时间编码+10的主旋律+10拍的和弦 所以是21
                        output_time_data = output_time_data[:21] + [0 for t in range(len(output_time_data) - 21)]
                input_time_data.extend(bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS) * 2 + pat_step_in_bar])  # 最后一个小节的bass数据
                output_time_data.extend(bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS) * 2 + 1 + pat_step_in_bar])
                # 5.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的bass不为空
                if melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_bass_data = bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS + 1) * 2]
                    if final_bar_bass_data != [0, 0]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass


def get_diff_value(raw_bass_data, chord_data):
    # 计算连续八拍bass的偏离程度
    # 评定标准有三个:
    # A.第三、四个步长的bass的平均音高与四拍前的相减 之后除七
    # B.第三、四个步长的bass的按键与四拍前的相减 计算杰卡德距离
    # C.每一个步长的bass与同时期和弦的包含关系 计算方法是 不包含量/全部（如果两拍内有两个和弦 取前一个作为这两拍的和弦）
    #   如果bass可以构成一个和弦 但与当前和弦不能匹配的 额外增加0.25个差异分
    # D.之后对上述量取平方和作为bass的误差
    # 舍弃包含了未知和弦的数据和连续四拍以上没有bass的数据
    diff_score_ary = []
    chord_diff_score_by_pat = [0, 0]  # 每一个步长的bass与同时期和弦的差异分 如果同时期的和弦为未知和弦 则分数标记为-1
    chord_backup = 0  # 用于存储上两拍的和弦
    bass_backup = []  # 用于存储上两拍的最后一个bass音符
    for pat_step_it in range(2, int(len(raw_bass_data) / 16) - 3):  # 前两个步长不看 因为没有两个步长前的数据
        note_dx = pat_step_it * 16
        try:
            # 1.计算平均音高之间的差异
            note_ary = []  # 这四拍中有几个音符 它们的音高分别是多少
            note_ary_old = []
            note_count = 0  # 八拍之前的四拍中有几个音符 它们的音高分别是多少
            note_count_old = 0
            for note_it in range(note_dx + 32, note_dx + 64):
                if raw_bass_data[note_it] != 0:
                    note_ary.extend(NoteDict[raw_bass_data[note_it]])
                    note_count += len(NoteDict[raw_bass_data[note_it]])
                if raw_bass_data[note_it - 64] != 0:
                    note_ary_old.extend(NoteDict[raw_bass_data[note_it - 64]])
                    note_count_old += len(NoteDict[raw_bass_data[note_it - 64]])
            if note_count == 0 or note_count_old == 0:
                note_diff_score = 0
            else:
                avr_note = sum(note_ary) / note_count  # 四拍所有音符的平均音高
                avr_note_old = sum(note_ary_old) / note_count_old  # 八拍之前的四拍所有音符的平均音高
                note_diff_score = abs(avr_note - avr_note_old) / 7  # 音高的差异（如果是176543的话差异分刚好为1分）
            # 2.计算按键的差异
            note_same_count = 0  # 四拍中有几个相同按键位置
            note_diff_count = 0  # 四拍中有几个不同的按键位置
            for note_it in range(note_dx + 32, note_dx + 64):
                if bool(raw_bass_data[note_it]) ^ bool(raw_bass_data[note_it - 64]):
                    note_diff_count += 1
                elif raw_bass_data[note_it] != 0 and raw_bass_data[note_it - 64] != 0:
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
                note_diff_count = 0  # 和弦内不包含的bass音符数
                if chord_data[bar_dx][step_in_bar * 2] == 0 and chord_data[bar_dx][step_in_bar * 2 + 1] == 0:
                    chord_diff_score_by_pat.append(-1)  # 出现了未知和弦 差异分标记为-1
                    continue
                abs_notelist = []
                for note_it in range(forw_step_it * 16, forw_step_it * 16 + 16):
                    if raw_bass_data[note_it] != 0:
                        abs_notelist.extend(NoteDict[raw_bass_data[note_it]])
                        bass_backup = NoteDict[raw_bass_data[note_it]]
                bass_chord = notelist2chord(set(abs_notelist))  # 这个ｂａｓｓ是否可以找出一个特定的和弦
                div12_note_list = []  # 把所有音符对12取余数的结果 用于分析和和弦之间的重复音关系
                for note in abs_notelist:
                    div12_note_list.append(note % 12)
                if bass_chord != 0:
                    if bass_chord != chord_data[bar_dx][step_in_bar * 2] and bass_chord != chord_data[bar_dx][step_in_bar * 2 + 1]:  # 这个ｂａｓｓ组合与当前和弦不能匹配 额外增加0.5个差异分
                        chord_diff_score_1step += 0.25
                if chord_data[bar_dx][step_in_bar * 2] != 0:
                    chord_set = copy.deepcopy(CHORD_DICT[chord_data[bar_dx][step_in_bar * 2]])  # 和弦的音符列表(已对１２取余数)
                    chord_dx = chord_data[bar_dx][step_in_bar * 2]
                else:
                    chord_set = copy.deepcopy(CHORD_DICT[chord_data[bar_dx][step_in_bar * 2 + 1]])
                    chord_dx = chord_data[bar_dx][step_in_bar * 2 + 1]
                if len(abs_notelist) == 0:  # 这两拍的bass没有音符。如果前两拍也没有则给值-1。如果和弦没变则赋前值 如果和弦变了则以前两拍最后一个音作为判断依据
                    if len(bass_backup) == 0:  # 前两拍也没有
                        chord_diff_score_by_pat.append(-1)
                        continue
                    elif chord_dx == chord_backup:
                        chord_diff_score_by_pat.append(chord_diff_score_by_pat[-1])  # 和弦没变 赋前值
                        continue
                    else:  # 用上两拍的最后一个bass组合来进行计算
                        abs_notelist = bass_backup
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
                chord_diff_score_1step += note_diff_count / len(abs_notelist)  # * 2  # bass与同期和弦的差异分
                chord_diff_score_by_pat.append(chord_diff_score_1step)
            if -1 in chord_diff_score_by_pat[pat_step_it: pat_step_it + 4]:  # 中间有未知和弦 直接跳过
                continue
            if raw_bass_data[note_dx: note_dx + 32] == [0] * 32 or raw_bass_data[note_dx + 32: note_dx + 64] == [0] * 32:  # 连续四拍没有bass的情况
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


class BassTrainDataCheck(BassTrainData):

    def __init__(self, melody_pat_data, continuous_bar_data, chord_cls):

        super().__init__(melody_pat_data, continuous_bar_data, chord_cls)
        self.check_melody_data = []  # 用于check的主旋律组合数据(共16拍)
        self.check_chord_data = []  # 用于check的和弦组合数据(共16拍)
        self.check_bass_input_data = []  # 用于check的bass输入数据(共8拍)
        self.check_bass_output_data = []  # 用于check的bass输出数据(共8拍)
        self.time_add_data = []  # 用于check的时间编码数据
        self.chord_data = chord_cls.chord_data
        bass_prob_ary = []
        # 1.获取用于验证的数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if chord_cls.chord_data[song_it] != {} and self.bass_pat_data[song_it] != {}:
                self.get_check_io_data(self.bass_pat_data[song_it], melody_pat_data[song_it], continuous_bar_data[song_it], chord_cls.chord_data[song_it])
                bass_prob_ary.extend(get_diff_value(flat_array(self.raw_bass_data[song_it]), self.chord_data[song_it]))
        # 2.找出前90%所在位置
        bass_prob_ary = sorted(bass_prob_ary)
        prob_09_dx = int(len(bass_prob_ary) * 0.9 + 1)
        self.ConfidenceLevel = bass_prob_ary[prob_09_dx]

    def get_check_io_data(self, bass_pat_data, melody_pat_data, continuous_bar_data, chord_data):
        """生成一首歌bass校验所需的数据"""
        for step_it in range(len(bass_pat_data)):  # 这里bass_pat_data是以步长为单位的,而不是以小节为单位的
            try:
                flag_drop = False  # 这组数据是否忽略
                melody_input_time_data = []
                chord_input_time_data = []
                bass_input_time_data = []
                bass_output_time_data = []
                # 1.获取当前小节号添加当前时间的编码（0-3）
                cur_bar = step_it // 2
                pat_step_in_bar = step_it % 2
                time_add = (1 - continuous_bar_data[cur_bar + TRAIN_BASS_IO_BARS] % 2) * 2 + pat_step_in_bar
                # 2.添加最近4小节(TRAIN_BASS_IO_BARS+2)的主旋律
                melody_input_time_data = melody_input_time_data + melody_pat_data[cur_bar][pat_step_in_bar * 2:]
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS + 2):
                    melody_input_time_data = melody_input_time_data + melody_pat_data[bar_it]
                    if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节
                        if bar_it >= cur_bar + TRAIN_BASS_IO_BARS:  # 位于输入和弦与输出和弦分界处之后出现了空小节 则直接忽略这组数据
                            flag_drop = True
                            break
                        else:
                            melody_input_time_data = [0 for t in range(len(melody_input_time_data))]
                if flag_drop is True:
                    continue
                melody_input_time_data = melody_input_time_data + melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS + 2][:2 * pat_step_in_bar]
                # 3.添加最近4小节(TRAIN_BASS_IO_BARS+2)的和弦
                chord_input_time_data = chord_input_time_data + chord_data[cur_bar][pat_step_in_bar * 2:]
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS + 2):
                    chord_input_time_data = chord_input_time_data + chord_data[bar_it]
                    if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节
                        chord_input_time_data = [0 for t in range(len(chord_input_time_data))]
                chord_input_time_data = chord_input_time_data + chord_data[cur_bar + TRAIN_BASS_IO_BARS + 2][:2 * pat_step_in_bar]
                # 4.添加过去2小节的bass
                if melody_pat_data[cur_bar] == [0 for t in range(4)]:
                    bass_input_time_data = bass_input_time_data + [0 for t in range(2 - pat_step_in_bar)]
                else:
                    bass_input_time_data = bass_input_time_data + bass_pat_data[cur_bar * 2 + pat_step_in_bar: (cur_bar + 1) * 2]
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
                    bass_input_time_data = bass_input_time_data + bass_pat_data[bar_it * 2: bar_it * 2 + 2]
                    if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        bass_input_time_data = [0 for t in range(len(bass_input_time_data))]
                bass_input_time_data = bass_input_time_data + bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS) * 2 + pat_step_in_bar]
                # 5.添加之后2小节的bass
                bass_output_time_data = bass_output_time_data + bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2 + pat_step_in_bar: (cur_bar + TRAIN_BASS_IO_BARS + 2) * 2 + pat_step_in_bar]
                if len(bass_output_time_data) < 4:  # 如果bass序列没那么长（到头了） 则舍弃这组序列
                    flag_drop = True
                for bar_it in range(3):  # 检查该项数据是否可以用于训练。条件是最后三个小节的bass均不为空
                    if bass_pat_data[cur_bar + TRAIN_BASS_IO_BARS + bar_it] == [0 for t in range(2)]:
                        flag_drop = True
                if flag_drop is True:
                    continue
                # 5.将这项数据添加进校验集中
                self.check_melody_data.append(melody_input_time_data)
                self.check_chord_data.append(chord_input_time_data)
                self.check_bass_input_data.append(bass_input_time_data)
                self.check_bass_output_data.append(bass_output_time_data)
                self.time_add_data.append(time_add)
            except KeyError:
                pass
            except IndexError:
                pass


class BassTrainData2(BassTrainDataCheck):

    def __init__(self, melody_pat_data, continuous_bar_data, chord_cls):
        super().__init__(melody_pat_data, continuous_bar_data, chord_cls)

    def get_model_io_data(self, bass_pat_data, melody_pat_data, continuous_bar_data, chord_data):
        """
        获取模型的输入输出数据。输入数据为过去两小节的时间编码/主旋律/和弦，输出内容为两小节的bass组合
        :param bass_pat_data: 一首歌的bass数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param chord_data: 一首歌的和弦数据
        :return:
        """
        for step_it in range(-2 * TRAIN_BASS_IO_BARS, len(bass_pat_data) - 2 * TRAIN_BASS_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []
                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                time_in_bar = (step_it % 2) * 2  # 小节内的第几拍
                # 2.获取这两小节加两拍的主旋律和和弦
                melody1_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
                melody2_code_add_base = 4 + COMMON_MELODY_PATTERN_NUMBER + 2  # 主旋律第二拍数据编码增加的基数
                chord1_code_add_base = 4 + (COMMON_MELODY_PATTERN_NUMBER + 2) * 2  # 和弦第一拍数据编码增加的基数
                chord2_code_add_base = 4 + (COMMON_MELODY_PATTERN_NUMBER + 2) * 2 + len(CHORD_DICT)  # 和弦第二拍数据编码增加的基数
                time_add = 0 if continuous_bar_data[cur_bar] == 0 else (1 - continuous_bar_data[cur_bar] % 2) * 2
                for beat_it in range(time_in_bar, 4, 2):
                    input_time_data.append([time_add + beat_it // 2, melody_pat_data[cur_bar][beat_it] + melody1_code_add_base, melody_pat_data[cur_bar][beat_it + 1] + melody2_code_add_base, chord_data[cur_bar][beat_it] + chord1_code_add_base, chord_data[cur_bar][beat_it + 1] + chord2_code_add_base])
                if melody_pat_data[cur_bar] == [0 for t in range(4)]:
                    input_time_data = [[input_time_data[t][0], melody1_code_add_base, melody2_code_add_base, chord1_code_add_base, chord2_code_add_base] for t in range(len(input_time_data))]
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
                    time_add = 0 if continuous_bar_data[bar_it] == 0 else (1 - continuous_bar_data[bar_it] % 2) * 2
                    for beat_it in range(0, 4, 2):
                        input_time_data.append([time_add + beat_it // 2, melody_pat_data[bar_it][beat_it] + melody1_code_add_base, melody_pat_data[bar_it][beat_it + 1] + melody2_code_add_base, chord_data[bar_it][beat_it] + chord1_code_add_base, chord_data[bar_it][beat_it + 1] + chord2_code_add_base])
                    if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        input_time_data = [[input_time_data[t][0], melody1_code_add_base, melody2_code_add_base, chord1_code_add_base, chord2_code_add_base] for t in range(len(input_time_data))]
                time_add = 0 if continuous_bar_data[cur_bar + TRAIN_BASS_IO_BARS] == 0 else (1 - continuous_bar_data[cur_bar + TRAIN_BASS_IO_BARS] % 2) * 2
                for beat_it in range(0, time_in_bar + 2, 2):
                    input_time_data.append([time_add + beat_it // 2, melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS][beat_it] + melody1_code_add_base, melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS][beat_it + 1] + melody2_code_add_base, chord_data[cur_bar + TRAIN_BASS_IO_BARS][beat_it] + chord1_code_add_base, chord_data[cur_bar + TRAIN_BASS_IO_BARS][beat_it + 1] + chord2_code_add_base])
                # 3.添加过去10拍的bass 一共5个
                bass_code_add_base = 4 + (COMMON_MELODY_PATTERN_NUMBER + 2) * 2 + (len(CHORD_DICT)) * 2  # bass数据编码增加的基数
                if cur_bar < 0 or melody_pat_data[cur_bar] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的bass也置为空
                    output_time_data.extend([bass_code_add_base for t in range(2 - pat_step_in_bar)])
                else:
                    output_time_data.extend(t + bass_code_add_base for t in bass_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2])
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的bass置为空
                        output_time_data.extend([bass_code_add_base, bass_code_add_base])
                    else:
                        output_time_data.extend(t + bass_code_add_base for t in bass_pat_data[bar_it * 2: bar_it * 2 + 2])
                    if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [bass_code_add_base for t in range(len(output_time_data))]
                output_time_data.extend(t + bass_code_add_base for t in bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS) * 2 + pat_step_in_bar + 1])  # 最后一个小节的bass数据
                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的bass不为空
                if melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_bass_data = bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS + 1) * 2]
                    if final_bar_bass_data != [0, 0]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass


class BassTrainData3:

    def __init__(self, melody_pat_data, continuous_bar_data, keypress_data, keypress_pat_ary, chord_cls):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取bass信息
        raw_bass_data = get_raw_song_data_from_dataset('bass', None)
        self.raw_bass_data = copy.deepcopy(raw_bass_data)
        # 2.获取和弦的根音组合 预期的根音平均值为36（即比中央C低2个八度）
        self.root_data, self.rc_pattern_data, self.rc_pattern_count = chord_cls.get_root_data(BASS_AVERAGE_ROOT)
        self.keypress_pat_num = len(keypress_pat_ary)  # 一共有多少种按键组合数据（步长为2拍）
        self.rc_pat_num = len(self.rc_pattern_count)
        # 3.将原数据转化成相对音高形式 并转换成以曲为单位的形式
        rel_note_list = [[] for t in range(TRAIN_FILE_NUMBERS)]
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_bass_data[song_it] != {}:
                rel_note_list[song_it] = one_song_rel_notelist_chord(raw_bass_data[song_it], self.root_data[song_it], chord_cls.chord_data[song_it], unit='bar', note_time_step=0.125)
        # 4.获取最常见的Bass组合
        common_pattern_cls = CommonMusicPatterns(COMMON_BASS_PATTERN_NUMBER)  # 这个类可以获取常见的主旋律组合
        if FLAG_IS_TRAINING is True:  # 训练模式
            common_pattern_cls.train(rel_note_list, 0.125, 2, multipart=False)
            common_pattern_cls.store('Bass')  # 存储在sqlite文件中
        else:
            common_pattern_cls.restore('Bass')  # 直接从sqlite文件中读取
        self.common_bass_pats = common_pattern_cls.common_pattern_list  # 常见的bass组合列表
        # print(common_pattern_cls.sum1, common_pattern_cls.sum2)
        # 5.获取模型的输入输出数据
        self.bass_pat_data = [{} for t in range(TRAIN_FILE_NUMBERS)]  # bass数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if rel_note_list[song_it] != [] and melody_pat_data[song_it] != {}:
                self.bass_pat_data[song_it] = BassPatternEncode(self.common_bass_pats, rel_note_list[song_it], 0.125, 2).music_pattern_ary
                self.get_model_io_data(self.bass_pat_data[song_it], melody_pat_data[song_it], continuous_bar_data[song_it], keypress_data[song_it], self.rc_pattern_data[song_it])
        # 6.获取用于验证的数据
        bass_prob_ary = []
        for song_it in range(TRAIN_FILE_NUMBERS):
            if chord_cls.chord_data[song_it] != {} and self.bass_pat_data[song_it] != {}:
                bass_prob_ary.extend(get_diff_value(flat_array(self.raw_bass_data[song_it]), chord_cls.chord_data[song_it]))
        # 7.找出前90%所在位置
        bass_prob_ary = sorted(bass_prob_ary)
        prob_09_dx = int(len(bass_prob_ary) * 0.9 + 1)
        self.ConfidenceLevel = bass_prob_ary[prob_09_dx]

    def get_model_io_data(self, bass_pat_data, melody_pat_data, continuous_bar_data, keypress_data, rc_pat_data):
        """
        获取模型的输入输出数据。输入数据为过去两小节加两拍的时间编码/主旋律按键组合/和弦根音组合，输出内容为两小节加两拍的bass组合
        :param keypress_data: 一首歌的主旋律按键数据
        :param bass_pat_data: 一首歌的bass数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param rc_pat_data: 一首歌的和弦根音组合数据
        """
        for step_it in range(-2 * TRAIN_BASS_IO_BARS, len(bass_pat_data) - 2 * TRAIN_BASS_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []
                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                time_in_bar = (step_it % 2) * 2  # 小节内的第几拍
                # 2.获取这两小节加两拍的主旋律和和弦
                keypress_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
                rc1_code_add_base = 4 + self.keypress_pat_num  # 和弦第一拍数据编码增加的基数
                rc2_code_add_base = 4 + self.keypress_pat_num + self.rc_pat_num  # 和弦第二拍数据编码增加的基数
                for forw_step_it in range(step_it, step_it + 2 * TRAIN_BASS_IO_BARS + 1):  # 向前看10拍
                    if forw_step_it >= 0:
                        bar_dx = forw_step_it // 2
                        time_in_bar = (forw_step_it % 2) * 2
                        time_add = 0 if continuous_bar_data[bar_dx] == 0 else (1 - continuous_bar_data[bar_dx] % 2) * 2
                        input_time_data.append([time_add + time_in_bar // 2, keypress_data[forw_step_it] + keypress_code_add_base, rc_pat_data[forw_step_it * 2] + rc1_code_add_base, rc_pat_data[forw_step_it * 2 + 1] + rc2_code_add_base])
                    else:
                        time_add = 0
                        time_in_bar = (forw_step_it % 2) * 2
                        input_time_data.append([time_add + time_in_bar // 2, keypress_code_add_base, rc1_code_add_base, rc2_code_add_base])
                for bar_it in range(cur_bar, cur_bar + TRAIN_BASS_IO_BARS - 1):
                    step_dx = (4 - time_in_bar) // 2 + (bar_it - cur_bar) * 2  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], keypress_code_add_base, rc1_code_add_base, rc2_code_add_base] for t in range(step_dx)]
                # 3.添加过去10拍的bass 一共5个
                bass_code_add_base = 4 + self.keypress_pat_num + self.rc_pat_num * 2  # bass数据编码增加的基数
                if cur_bar < 0 or melody_pat_data[cur_bar] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的bass也置为空
                    output_time_data.extend([bass_code_add_base for t in range(2 - pat_step_in_bar)])
                else:
                    output_time_data.extend(t + bass_code_add_base for t in bass_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2])
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的bass置为空
                        output_time_data.extend([bass_code_add_base, bass_code_add_base])
                    else:
                        output_time_data.extend(t + bass_code_add_base for t in bass_pat_data[bar_it * 2: bar_it * 2 + 2])
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [bass_code_add_base for t in range(len(output_time_data))]
                output_time_data.extend(t + bass_code_add_base for t in bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS) * 2 + pat_step_in_bar + 1])  # 最后一个小节的bass数据
                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的bass不为空
                if melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_bass_data = bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS + 1) * 2]
                    if final_bar_bass_data != [0, 0]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass


class BassTrainData4(BassTrainData3):

    def get_model_io_data(self, bass_pat_data, melody_pat_data, continuous_bar_data, keypress_data, rc_pat_data):
        """
        获取模型的输入输出数据。输入数据为过去两小节加两拍的时间编码/主旋律按键组合/和弦根音组合/过去三小节的bass，输出内容为两小节加两拍的bass组合
        :param keypress_data: 一首歌的主旋律按键数据
        :param bass_pat_data: 一首歌的bass数据
        :param melody_pat_data: 一首歌的主旋律组合数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param rc_pat_data: 一首歌的和弦根音组合数据
        """
        for step_it in range(-2 * TRAIN_BASS_IO_BARS, len(bass_pat_data) - 2 * TRAIN_BASS_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                input_time_data = []
                output_time_data = []
                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                time_in_bar = (step_it % 2) * 2  # 小节内的第几拍
                # 2.获取这两小节加两拍的主旋律和和弦,以及再前一个步长的bass组合
                keypress_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
                rc1_code_add_base = 4 + self.keypress_pat_num  # 和弦第一拍数据编码增加的基数
                rc2_code_add_base = 4 + self.keypress_pat_num + self.rc_pat_num  # 和弦第二拍数据编码增加的基数
                bass_code_add_base = 4 + self.keypress_pat_num + self.rc_pat_num * 2  # bass数据编码增加的基数
                for forw_step_it in range(step_it, step_it + 2 * TRAIN_BASS_IO_BARS + 1):  # 向前看10拍
                    if forw_step_it >= 0:
                        bar_dx = forw_step_it // 2
                        time_in_bar = (forw_step_it % 2) * 2
                        time_add = 0 if continuous_bar_data[bar_dx] == 0 else (1 - continuous_bar_data[bar_dx] % 2) * 2
                        input_time_data.append([time_add + time_in_bar // 2, keypress_data[forw_step_it] + keypress_code_add_base, rc_pat_data[forw_step_it * 2] + rc1_code_add_base, rc_pat_data[forw_step_it * 2 + 1] + rc2_code_add_base])
                    else:
                        time_add = 0
                        time_in_bar = (forw_step_it % 2) * 2
                        input_time_data.append([time_add + time_in_bar // 2, keypress_code_add_base, rc1_code_add_base, rc2_code_add_base])
                    if forw_step_it - 1 >= 0:
                        input_time_data[-1].append(bass_pat_data[forw_step_it - 1] + bass_code_add_base)
                    else:
                        input_time_data[-1].append(bass_code_add_base)
                for bar_it in range(cur_bar, cur_bar + TRAIN_BASS_IO_BARS):
                    step_dx = (4 - time_in_bar) // 2 + (bar_it - cur_bar) * 2  # 组合数据的第几拍位于这个小节内
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:
                        input_time_data[:step_dx] = [[input_time_data[t][0], keypress_code_add_base, rc1_code_add_base, rc2_code_add_base, bass_code_add_base] for t in range(step_dx)]
                # 3.添加过去10拍的bass 一共5个
                if cur_bar < 0 or melody_pat_data[cur_bar] == [0 for t in range(4)]:  # 当处于负拍 或 这个小节没有主旋律 那么这个小节对应的bass也置为空
                    output_time_data.extend([bass_code_add_base for t in range(2 - pat_step_in_bar)])
                else:
                    output_time_data.extend(t + bass_code_add_base for t in bass_pat_data[cur_bar * 2 + pat_step_in_bar: cur_bar * 2 + 2])
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
                    if bar_it < 0:  # 当处于负拍时 这个小节对应的bass置为空
                        output_time_data.extend([bass_code_add_base, bass_code_add_base])
                    else:
                        output_time_data.extend(t + bass_code_add_base for t in bass_pat_data[bar_it * 2: bar_it * 2 + 2])
                    if bar_it < 0 or melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        output_time_data = [bass_code_add_base for t in range(len(output_time_data))]
                output_time_data.extend(t + bass_code_add_base for t in bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS) * 2 + pat_step_in_bar + 1])  # 最后一个小节的bass数据
                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的bass不为空
                if melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    final_bar_bass_data = bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS + 1) * 2]
                    if final_bar_bass_data != [0, 0]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except KeyError:
                pass
            except IndexError:
                pass
