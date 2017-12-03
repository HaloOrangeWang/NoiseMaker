from interfaces.functions import CommonMusicPatterns, MusicPatternEncode, GetDictMaxKey
from interfaces.chord_parse import RootNote
from interfaces.sql.sqlite import GetRawSongDataFromDataset, NoteDict
import copy
from settings import *
import numpy as np

# 方案1: 使用LSTM训练piano_guitar
# 方案2: 使用HMM生成piano_guitar,时间步长为1拍
# 方案3: 使用HMM生成piano_guitar，时间步长为2拍
# 方案4: 使用双HMM生成piano_guitar，时间步长为1拍
# 方案5: 使用双HMM生成piano_guitar，时间步长为2拍


class PianoGuitarPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, bar_index, bar_pattern_list, raw_bar_pattern_list, common_patterns):
        # print(bar_pattern_list)
        for bar_pg_pattern_iterator in range(len(bar_pattern_list)):
            if bar_pattern_list[bar_pg_pattern_iterator] == -1:
                # 在常见的piano_guitar列表里找不到某一个piano_guitar组合的处理方法：
                # a.寻找符合以下条件的piano_guitar组合
                # a1.四个位置的最高音完全相同
                # a2.该组合的所有音符是待求组合所有音符的子集
                # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小（与最高音相差一个八度以上的差异值记为1,相差不足一个八度的差异值记为2）
                # b.寻找符合以下条件的piano_guitar组合
                # b1.四个位置的休止情况完全相同
                # b2.音符经转置之后可以实现重合
                # b3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小（与最高音相差一个八度以上的差异值记为1,相差不足一个八度的差异值记为2）
                # c.记为common_patterns+1
                # 1.根据上述a方案寻找piano_guitar组合
                choose_pattern = 0  # 选取的pattern
                choose_pattern_diff_score = 5  # 两个旋律组合的差异程度
                for common_pg_iterator in range(1, len(common_patterns)):
                    pattern_diff_score = 0  # 初始的旋律组合不像似度为0分 每发现一个不同音符 按权重加分
                    # 1.1.检查该旋律组合是否符合要求
                    note_satisfactory = True
                    for note_iterator in range(len(common_patterns[common_pg_iterator])):
                        # 1.1.1.检查休止情况是否相同
                        if bool(common_patterns[common_pg_iterator][note_iterator]) ^ bool(raw_bar_pattern_list[bar_pg_pattern_iterator][note_iterator]):  # 一个为休止符 一个不为休止符
                            note_satisfactory = False
                            break
                        common_pattern_note_set = set(NoteDict[common_patterns[common_pg_iterator][note_iterator]])  # 这个时间步长中常见组合的真实音符组合
                        bar_pattern_note_set = set(NoteDict[raw_bar_pattern_list[bar_pg_pattern_iterator][note_iterator]])  # 这个时间步长中待求组合的真实音符组合
                        if common_patterns[common_pg_iterator][note_iterator] != 0:
                            # 1.1.2.检查该旋律组合中所有最高音与待求旋律组合对应位置的最高音是否全部相同
                            if max(common_pattern_note_set) != max(bar_pattern_note_set):  # 两者的最高音不一样
                                note_satisfactory = False
                                break
                            # print(common_pg_iterator, note_iterator, common_pattern_note_set, bar_pattern_note_set)
                            # 1.1.3.检查该组合的所有音符是不是待求组合所有音符的子集
                            if not common_pattern_note_set.issubset(bar_pattern_note_set):  # 该组合的所有音符不是待求组合所有音符的子集
                                note_satisfactory = False
                                break
                            # 1.1.4.计算该组合与待求组合之间的差异分
                            for note in bar_pattern_note_set:
                                if note not in common_pattern_note_set:  # 在待求组合中有这个音符而该组合中没有这个音符。增加差异分
                                    if max(bar_pattern_note_set) - note >= 12:
                                        pattern_diff_score += 1
                                    else:
                                        pattern_diff_score += 2
                    if not note_satisfactory:
                        continue
                    # 1.2.如果找到符合要求的组合 将其记录下来
                    # print(pattern_diff_score)
                    if pattern_diff_score < choose_pattern_diff_score:
                        choose_pattern = common_pg_iterator
                        choose_pattern_diff_score = pattern_diff_score
                # 2.根据上述b方案寻找piano_guitar组合
                if choose_pattern == 0:
                    for common_pg_iterator in range(1, len(common_patterns)):
                        diff_note_count = 0  # 初始的旋律组合不像似度为0分 每发现一个不同音符 按权重加分
                        total_note_count = 0
                        # 2.1.检查该旋律组合是否符合要求
                        note_satisfactory = True
                        for note_iterator in range(len(common_patterns[common_pg_iterator])):
                            # 2.1.1.检查休止情况是否相同
                            if bool(common_patterns[common_pg_iterator][note_iterator]) ^ bool(raw_bar_pattern_list[bar_pg_pattern_iterator][note_iterator]):  # 一个为休止符 一个不为休止符
                                note_satisfactory = False
                                break
                            common_pattern_note_set = set(NoteDict[common_patterns[common_pg_iterator][note_iterator]])  # 这个时间步长中常见组合的真实音符组合
                            bar_pattern_note_set = set(NoteDict[raw_bar_pattern_list[bar_pg_pattern_iterator][note_iterator]])  # 这个时间步长中待求组合的真实音符组合
                            if common_patterns[common_pg_iterator][note_iterator] != 0:
                                # 2.1.2.检查转位之后是否相同
                                if set(map(lambda x: (x % 12), common_pattern_note_set)) != set(map(lambda x: (x % 12), bar_pattern_note_set)):
                                    note_satisfactory = False
                                    break
                                # 2.1.3.计算该组合与待求组合之间的差异分
                                total_note_count += len(bar_pattern_note_set)
                                for note in bar_pattern_note_set:
                                    if note not in common_pattern_note_set:  # 在待求组合中有这个音符而该组合中没有这个音符。增加差异分
                                        diff_note_count += 1
                                for note in common_pattern_note_set:
                                    if note not in bar_pattern_note_set:  # 在待求组合中没有这个音符而该组合中有这个音符。增加差异分
                                        diff_note_count += 1
                        if not note_satisfactory:
                            continue
                        # 2.2.如果找到符合要求的组合 将其记录下来
                        # print(pattern_diff_score)
                        pattern_diff_score = (7 * diff_note_count) // total_note_count
                        if pattern_diff_score < choose_pattern_diff_score:
                            choose_pattern = common_pg_iterator
                            choose_pattern_diff_score = pattern_diff_score
                    # print(bar_pg_pattern_iterator, raw_bar_pattern_list, choose_pattern, choose_pattern_diff_score)
                # print(bar_pattern_list)
                # 3.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_PIANO_GUITAR_PATTERNS+1
                # print(bar_pg_pattern_iterator, raw_bar_pattern_list, choose_pattern, choose_pattern_diff_score)
                if choose_pattern != 0:
                    bar_pattern_list[bar_pg_pattern_iterator] = choose_pattern
                else:
                    bar_pattern_list[bar_pg_pattern_iterator] = len(common_patterns)
        # print(bar_pattern_list, '\n')
        return bar_pattern_list


def GetRelNoteList(note_list, root, chord):
    """
    转换成相对音高列表。音高用音符的音高和根音的差值代替
    :param note_list:
    :param root: 根音
    :param chord:
    :return:
    """
    # TODO 136和135一样真的好吗
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


def OneBarRelNoteList(note_list, root_data, chord_data):
    rel_note_list = []  # 转化为相对音高形式的音符列表
    for note_iterator, note in enumerate(note_list):
        if note == 0:
            rel_note_list.append(0)
        else:
            rel_note_group = GetRelNoteList(NoteDict[note], root_data[note_iterator // 4], chord_data[note_iterator // 4])
            rel_note_list.append(rel_note_group)
    return rel_note_list


def CommonPianoGuitarPatterns(raw_pg_data, root_data, chord_data, note_time_step, pattern_time_step):
    common_pattern_dict = {}
    time_step_ratio = round(pattern_time_step / note_time_step)
    for pg_part in raw_pg_data:
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            if pg_part[song_iterator] != {}:
                for key in pg_part[song_iterator]:
                    try:
                        rel_note_list = OneBarRelNoteList(pg_part[song_iterator][key], root_data[song_iterator][key], chord_data[song_iterator][key])
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
    # for t in common_pattern_list_temp[1: (COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1)]:
    #     sum2 += t[1]
    # print(sum2 / sum1)
    common_pattern_list = []  # 最常见的number种组合
    pattern_number_list = []  # 这些组合出现的次数
    for pattern_tuple in common_pattern_list_temp[:(COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1)]:
        common_pattern_list.append(eval(pattern_tuple[0]))
        pattern_number_list.append(pattern_tuple[1])
    # print(common_pattern_list)
    # print(pattern_number_list)
    return common_pattern_list, pattern_number_list


class RelNotePatternEncode(object):

    fi = 0
    nf = 0

    def __init__(self, common_patterns, music_data_dict, chord_data, root_data, note_time_step, pattern_time_step):
        self.music_pattern_dict = copy.deepcopy(music_data_dict)  # 按照最常见的音符组合编码之后的组合列表（dict形式）
        time_step_ratio = round(pattern_time_step / note_time_step)
        music_data_list = sorted(music_data_dict.items(), key=lambda asd: asd[0], reverse=False)  # 把ｄｉｃｔ形式的音符列表转化成list形式
        for bar_data in music_data_list:
            # print(bar_iterator[0])
            bar_index = bar_data[0]  # 这个小节是这首歌的第几小节
            if bar_index not in chord_data:  # 如果这一小节没有和弦 则相对音高列表置为全零
                self.music_pattern_dict[bar_data[0]] = [0 for t in range(time_step_ratio)]
                continue
            rel_note_list = OneBarRelNoteList(bar_data[1], root_data[bar_index], chord_data[bar_index])
            rel_note_list = [rel_note_list[time_step_ratio * t: time_step_ratio * (t + 1)] for t in range(round(4 / pattern_time_step))]  # 将音符列表按照pattern_time_step进行分割 使其变成二维数组
            bar_pattern_list = self.handle_common_patterns(rel_note_list, common_patterns)
            bar_pattern_list = self.handle_rare_pattern(bar_index, bar_pattern_list, rel_note_list, common_patterns)
            self.music_pattern_dict[bar_data[0]] = bar_pattern_list  # 将编码后的pattern list保存在新的pattern dict中

    def handle_common_patterns(self, rel_note_list, common_patterns):
        bar_pattern_list = []
        for bar_pattern_iterator in range(len(rel_note_list)):
            try:
                pattern_code = common_patterns.index(rel_note_list[bar_pattern_iterator])  # 在常见的组合列表中找到这个音符组合
                bar_pattern_list.append(pattern_code)
            except ValueError:  # 找不到
                bar_pattern_list.append(-1)
        return bar_pattern_list

    def handle_rare_pattern(self, bar_index, bar_pattern_list, rel_note_list, common_patterns):
        # 在常见的piano_guitar列表里找不到某一个piano_guitar组合的处理方法：
        # a.寻找符合以下条件的piano_guitar组合
        # a1.四个位置的休止情况完全相同
        # a2.该组合的所有音符是待求组合所有音符的子集 或与待求组合的音符相差7的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小（与最高音相差一个八度以上的差异值记为1,相差不足一个八度的差异值记为2）
        # b.记为common_patterns+1
        for bar_pg_pattern_iterator in range(len(bar_pattern_list)):
            if bar_pattern_list[bar_pg_pattern_iterator] == -1:
                choose_pattern = 0  # 选取的pattern
                choose_pattern_diff_score = 75  # 两个旋律组合的差异程度
                for common_pg_iterator in range(1, len(common_patterns)):
                    total_note_count = 0
                    diff_note_count = 0
                    # 1.1.检查该旋律组合是否符合要求
                    note_satisfactory = True
                    for group_iterator in range(len(common_patterns[common_pg_iterator])):
                        # 1.1.1.检查休止情况是否相同
                        if bool(common_patterns[common_pg_iterator][group_iterator]) ^ bool(rel_note_list[bar_pg_pattern_iterator][group_iterator]):  # 一个为休止符 一个不为休止符
                            note_satisfactory = False
                            break
                        if common_patterns[common_pg_iterator][group_iterator] == 0:  # 如果这个时间步长是休止符的话，直接进入下个时间步长
                            continue
                        # 1.1.2.求出相对音高组合并对7取余数 按升降号分开
                        cur_pattern_note_list = common_patterns[common_pg_iterator][group_iterator]  # 这个时间步长中常见组合的真实音符组合
                        cur_pattern_note_list_div7 = [[t[0] % 7, t[1]] for t in cur_pattern_note_list]  # 对7取余数
                        cur_pattern_sj_set = {t[1] for t in cur_pattern_note_list}  # 升降关系的集合
                        cur_pattern_note_dict_div7 = {t0: {t1 for [t1, t2] in cur_pattern_note_list_div7 if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开
                        cur_pattern_note_dict = {t0: {t1 for [t1, t2] in cur_pattern_note_list if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开

                        cur_step_note_list = rel_note_list[bar_pg_pattern_iterator][group_iterator]  # 这个时间步长中待求组合的真实音符组合
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
                        choose_pattern = common_pg_iterator
                        choose_pattern_diff_score = pattern_diff_score
                # 2.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_PIANO_GUITAR_PATTERNS+1
                # print(bar_pg_pattern_iterator, raw_bar_pattern_list, choose_pattern, choose_pattern_diff_score)
                if choose_pattern != 0:
                    self.fi += 1
                    bar_pattern_list[bar_pg_pattern_iterator] = choose_pattern
                else:
                    self.nf += 1
                    bar_pattern_list[bar_pg_pattern_iterator] = COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1
        # print(bar_pattern_list, '\n')
        return bar_pattern_list


class PianoGuitarTrainData:

    input_data = []  # 输入model的数据
    output_data = []  # 从model输出的数据

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data):
        # 1.从数据集中读取歌的piano_guitar数据
        raw_pg_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        flatten_pg_data = []  # 三维数组 将上面的第1/2维合二为一 将同一首歌不同的piano_guitar音轨视作不同的歌
        pg_iterator = 1
        while True:
            pg_part_data = GetRawSongDataFromDataset('piano_guitar' + str(pg_iterator), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if pg_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_pg_data.append(pg_part_data)
            flatten_pg_data += pg_part_data
            pg_iterator += 1
        self.train_song_list = [[True for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(pg_iterator)]  # 用于训练的歌曲id列表
        # 2.获取最常见的piano_guitar组合
        self.common_pg_patterns, __ = CommonMusicPatterns(flatten_pg_data, number=COMMON_PIANO_GUITAR_PATTERN_NUMBER, note_time_step=1/4, pattern_time_step=1)
        # 3.对数据进行编码并去除未知组合的比例过高的piano_guitar
        raw_pg_data = self.pg_pattern_encode(raw_pg_data, melody_pattern_data)
        # 4.生成输入输出数据
        for pg_part_iterator in range(len(raw_pg_data)):
            for song_iterator in range(TRAIN_FILE_NUMBERS):
                if raw_pg_data[pg_part_iterator][song_iterator] != {} and melody_pattern_data[song_iterator] != {} and self.train_song_list[pg_part_iterator][song_iterator]:
                    self.get_model_io_data(raw_pg_data[pg_part_iterator][song_iterator], melody_pattern_data[song_iterator], continuous_bar_number_data[song_iterator], chord_data[song_iterator])
        # print(len(self.input_data), len(self.output_data))
        # print('\n\n\n\n\n')
        # for t in self.input_data:
        #     print(t)
        # print('\n\n\n')
        # for t in self.output_data:
        #     print(t)

    def pg_pattern_encode(self, raw_pg_data, melody_pattern_data):
        for pg_part_iterator in range(len(raw_pg_data)):
            for song_iterator in range(TRAIN_FILE_NUMBERS):
                if raw_pg_data[pg_part_iterator][song_iterator] != {} and melody_pattern_data[song_iterator] != {}:
                    # 1.开头补上几个空小节 便于训练开头的数据
                    for bar_iterator in range(-TRAIN_CHORD_IO_BARS, 0):
                        raw_pg_data[pg_part_iterator][song_iterator][bar_iterator] = [0 for t in range(16)]
                        # melody_pattern_data[song_iterator][bar_iterator] = [0 for t in range(4)]
                        # chord_data[song_iterator][bar_iterator] = [0 for t in range(4)]
                    # 2.将一首歌的piano_guitar编码为常见的piano_guitar组合。如果该piano_guitar组合不常见，则记为common_pg_patterns+1
                    raw_pg_data[pg_part_iterator][song_iterator] = PianoGuitarPatternEncode(self.common_pg_patterns, raw_pg_data[pg_part_iterator][song_iterator], 0.25, 1).music_pattern_dict
                    # 3.如果一段音乐中不常见的piano_guitar比例过高 则这段不予收录 门限是30%
                    common_pattern_number = 0
                    rare_pattern_number = 0
                    for key in raw_pg_data[pg_part_iterator][song_iterator]:
                        for pg_pattern in raw_pg_data[pg_part_iterator][song_iterator][key]:
                            if pg_pattern == COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1:
                                rare_pattern_number += 1
                            elif pg_pattern != 0:
                                common_pattern_number += 1
                    # print('%.2f\t%d\t%d' % (rare_pattern_number / (common_pattern_number + rare_pattern_number), pg_part_iterator, song_iterator))
                    if rare_pattern_number / (common_pattern_number + rare_pattern_number) > 0.3:
                        self.train_song_list[pg_part_iterator][song_iterator] = False
                else:
                    self.train_song_list[pg_part_iterator][song_iterator] = False
        return raw_pg_data

    def get_model_io_data(self, pg_pattern_data, melody_pattern_data, continuous_bar_number_data, chord_data):
        """
        模型的训练数据包括：当前时间的编码（1-8） 过去9拍的主旋律 过去9拍的和弦 过去8拍的piano_guitar 输出数据为最后一拍的piano_guitar 共计长度为27
        :param pg_pattern_data: 一首歌的piano_guitar数据
        :param melody_pattern_data: 一首歌的主旋律组合的列表
        :param continuous_bar_number_data: 一首歌主旋律连续不为空的小节列表
        :param chord_data: 一首歌的和弦列表
        :return:
        """
        # TODO: 501出现的比例过高则不予收录
        for key in pg_pattern_data:
            for time_in_bar in range(round(4)):  # piano_guitar训练的步长为1拍
                try:
                    # 1.添加当前时间的编码（0-7）
                    time_add = (1 - continuous_bar_number_data[key + TRAIN_PIANO_GUITAR_IO_BARS] % 2) * 4
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 2.添加过去9拍的主旋律
                    input_time_data = input_time_data + melody_pattern_data[key][time_in_bar:]
                    output_time_data = output_time_data + melody_pattern_data[key][time_in_bar:]
                    for bar_iterator in range(key + 1, key + TRAIN_PIANO_GUITAR_IO_BARS):
                        input_time_data = input_time_data + melody_pattern_data[bar_iterator]
                        output_time_data = output_time_data + melody_pattern_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = [input_time_data[0]] + [0 for t in range(len(input_time_data) - 1)]
                            output_time_data = [output_time_data[0]] + [0 for t in range(len(output_time_data) - 1)]
                    input_time_data = input_time_data + melody_pattern_data[key + TRAIN_PIANO_GUITAR_IO_BARS][:1 + time_in_bar]
                    output_time_data = output_time_data + melody_pattern_data[key + TRAIN_PIANO_GUITAR_IO_BARS][:1 + time_in_bar]
                    # 3.添加过去9拍的和弦
                    if melody_pattern_data[key] == [0 for t in range(4)]:
                        input_time_data = input_time_data + [0 for t in range(4 - time_in_bar)]
                        output_time_data = output_time_data + [0 for t in range(4 - time_in_bar)]
                    else:
                        input_time_data = input_time_data + chord_data[key][time_in_bar:]
                        output_time_data = output_time_data + chord_data[key][time_in_bar:]
                    for bar_iterator in range(key + 1, key + TRAIN_PIANO_GUITAR_IO_BARS):
                        input_time_data = input_time_data + chord_data[bar_iterator]
                        output_time_data = output_time_data + chord_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[:10] + [0 for t in range(len(input_time_data) - 10)]  # 1的时间编码+9的主旋律 所以是10
                            output_time_data = output_time_data[:10] + [0 for t in range(len(output_time_data) - 10)]
                    input_time_data = input_time_data + chord_data[key + TRAIN_PIANO_GUITAR_IO_BARS][:1 + time_in_bar]
                    output_time_data = output_time_data + chord_data[key + TRAIN_PIANO_GUITAR_IO_BARS][:1 + time_in_bar]
                    # 4.添加过去8拍的piano_guitar
                    if melody_pattern_data[key] == [0 for t in range(4)]:  # 如果某一个小节没有主旋律 那么这个小节对应的piano_guitar也置为空
                        input_time_data = input_time_data + [0 for t in range(4 - time_in_bar)]
                        output_time_data = output_time_data + [0 for t in range(3 - time_in_bar)]
                    else:
                        input_time_data = input_time_data + pg_pattern_data[key][time_in_bar:]
                        output_time_data = output_time_data + pg_pattern_data[key][1 + time_in_bar:]
                    for bar_iterator in range(key + 1, key + TRAIN_PIANO_GUITAR_IO_BARS):
                        input_time_data = input_time_data + pg_pattern_data[bar_iterator]
                        output_time_data = output_time_data + pg_pattern_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[:19] + [0 for t in range(len(input_time_data) - 19)]  # 1的时间编码+9的主旋律+9拍的和弦 所以是19
                            output_time_data = output_time_data[:19] + [0 for t in range(len(output_time_data) - 19)]
                    input_time_data = input_time_data + pg_pattern_data[key + TRAIN_PIANO_GUITAR_IO_BARS][:time_in_bar]
                    output_time_data = output_time_data + pg_pattern_data[key + TRAIN_PIANO_GUITAR_IO_BARS][:1 + time_in_bar]
                    # 5.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的piano_guitar不为空
                    if melody_pattern_data[key + TRAIN_PIANO_GUITAR_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                        output_bar_data_temp = pg_pattern_data[key + TRAIN_PIANO_GUITAR_IO_BARS]
                        if output_bar_data_temp != [0 for t in range(4)]:
                            self.input_data.append(input_time_data)
                            self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass


# TODO:四种方案 时间步长：1-2拍 数据生成方法：统计方法/Baum-Wales
class PianoGuitarTrainData_2:

    emission = []  # 观测矩阵
    transfer = []  # 状态转移矩阵
    pi = []  # 初始状态矩阵
    emission_count = []  # 观测计数矩阵
    transfer_count = []  # 状态转移计数矩阵
    pi_count = []  # 初始状态计数矩阵

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data):
        # 1.从数据集中读取歌的piano_guitar数据
        raw_pg_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        pg_iterator = 1
        while True:
            pg_part_data = GetRawSongDataFromDataset('piano_guitar' + str(pg_iterator), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if pg_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_pg_data.append(pg_part_data)
            # flatten_pg_data += pg_part_data
            pg_iterator += 1
        self.train_song_list = [[True for t0 in range(TRAIN_FILE_NUMBERS)] for t in range(pg_iterator)]  # 用于训练的歌曲id列表
        # 2.获取根音组合及根音-和弦配对组合
        self.get_root_data(chord_data)
        self.get_root_chord_pattern_dict(chord_data)
        self.transfer_count = np.zeros([COMMON_PIANO_GUITAR_PATTERN_NUMBER, COMMON_PIANO_GUITAR_PATTERN_NUMBER])  # 统计数据时把0和common+1的数据排除在外
        self.emission_count = np.zeros([len(self.rc_pattern_dict) - 1, COMMON_PIANO_GUITAR_PATTERN_NUMBER])  # “dict长度”行，pattern个数列。0和common+1被排除在外
        self.pi_count = np.zeros([COMMON_PIANO_GUITAR_PATTERN_NUMBER])
        # print(root_data)
        # 3.获取最常见的piano_guitar组合
        # self.common_pg_patterns, __ = CommonMusicPatterns(raw_pg_data, number=COMMON_PIANO_GUITAR_PATTERN_NUMBER, note_time_step=1 / 4, pattern_time_step=1)
        self.common_pg_patterns, __ = CommonPianoGuitarPatterns(raw_pg_data, self.root_data, chord_data, 0.25, 1)
        for pg_part_iterator in range(len(raw_pg_data)):
            for song_iterator in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_pg_data[pg_part_iterator][song_iterator] != {} and melody_pattern_data[song_iterator] != {}:
                    # raw_pg_data[pg_part_iterator][song_iterator] = PianoGuitarPatternEncode(self.common_pg_patterns, raw_pg_data[pg_part_iterator][song_iterator], 0.25, 1).music_pattern_dict
                    raw_pg_data[pg_part_iterator][song_iterator] = RelNotePatternEncode(self.common_pg_patterns, raw_pg_data[pg_part_iterator][song_iterator], chord_data[song_iterator], self.root_data[song_iterator], 0.25, 1).music_pattern_dict
                    self.count(raw_pg_data[pg_part_iterator][song_iterator], self.rc_pattern_list[song_iterator])
                    self.count_initial(raw_pg_data[pg_part_iterator][song_iterator], self.rc_pattern_list[song_iterator], continuous_bar_number_data[song_iterator])
        # print(list(self.transfer_count[0: 3]))
        # print(list(self.emission_count[:, 0: 3]))
        # print(list(self.pi_count))
        self.prob_convert()
        # print(list(self.transfer[0: 3]))
        # print(list(self.emission[:, 0: 3]))
        # print(list(self.pi))

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
        :param chord_data: 和弦列表
        :return:
        """
        self.rc_pattern_list = [{} for t in range(TRAIN_FILE_NUMBERS)]
        self.rc_pattern_dict = [[-1]]
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            for bar_iterator in range(GetDictMaxKey(chord_data[song_iterator]) + 1):
                self.rc_pattern_list[song_iterator][bar_iterator] = [0, 0, 0, 0]  # 先全部初始化为0
                for chord_iterator in range(4):
                    if chord_data[song_iterator][bar_iterator][chord_iterator] != 0:  # 如果这个时间步长中和弦是0的话，就不用向下执行了
                        rc_pattern = [self.root_data[song_iterator][bar_iterator][chord_iterator], chord_data[song_iterator][bar_iterator][chord_iterator]]
                        try:
                            rc_pattern_index = self.rc_pattern_dict.index(rc_pattern)  # 检查这个音符组合有没有被保存
                        except ValueError:
                            self.rc_pattern_dict.append(rc_pattern)  # 添加这个根音 和弦 组合
                            rc_pattern_index = len(self.rc_pattern_dict) - 1
                        self.rc_pattern_list[song_iterator][bar_iterator][chord_iterator] = rc_pattern_index  # 将这个音符保存起来
        # print(self.rc_pattern_dict)
        # for t in range(TRAIN_FILE_NUMBERS):
        #     print(self.rc_pattern_list[t])

    def count(self, pg_data, rc_pattern):
        for key in pg_data:
            if key not in rc_pattern:  # 这一小节没有和弦 直接跳过
                continue
            for time_iterator in range(4):
                if key == 0 and time_iterator == 0:  # 不考虑第一小节的第一拍
                    continue
                if time_iterator != 0:
                    if (pg_data[key][time_iterator] in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1]) or (pg_data[key][time_iterator - 1] in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1]) or rc_pattern[key][time_iterator - 1] == 0:  # 当前拍 上一拍的伴奏编码不能是0或common+1 同时当前拍和弦根音编码不能是0
                        continue
                    self.transfer_count[pg_data[key][time_iterator - 1] - 1, pg_data[key][time_iterator] - 1] += 1
                    self.emission_count[rc_pattern[key][time_iterator] - 1, pg_data[key][time_iterator] - 1] += 1
                else:
                    if (pg_data[key][0] in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1]) or (pg_data[key - 1][3] in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1]) or rc_pattern[key][0] == 0:
                        continue
                    self.transfer_count[pg_data[key - 1][3] - 1, pg_data[key][0] - 1] += 1
                    self.emission_count[rc_pattern[key][0] - 1, pg_data[key][0] - 1] += 1

    def count_initial(self, pg_data, rc_pattern, continuous_bar_number_data):
        for key in pg_data:
            if key >= len(continuous_bar_number_data):  # 这个小节没有对应的主旋律 直接跳过
                continue
            if key == 0:
                if continuous_bar_number_data[0] != 0:
                    if pg_data[0][0] not in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1] and rc_pattern[0][0] != 0:
                        self.pi_count[pg_data[0][0] - 1] += 1
            else:
                if continuous_bar_number_data[key] != 0 and continuous_bar_number_data[key - 1] == 0:
                    if pg_data[key][0] not in [0, COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1] and rc_pattern[key][0] != 0:
                        self.pi_count[pg_data[key][0] - 1] += 1

    def prob_convert(self):
        self.transfer = np.zeros([COMMON_PIANO_GUITAR_PATTERN_NUMBER, COMMON_PIANO_GUITAR_PATTERN_NUMBER])
        self.emission = np.zeros([len(self.rc_pattern_dict) - 1, COMMON_PIANO_GUITAR_PATTERN_NUMBER])
        self.pi = np.zeros([COMMON_PIANO_GUITAR_PATTERN_NUMBER])
        # 1.计算转移矩阵
        for row_iterator, row in enumerate(self.transfer_count):
            row_sum = sum(row)
            if row_sum == 0:
                self.transfer[row_iterator] = [1 / COMMON_PIANO_GUITAR_PATTERN_NUMBER for t in range(COMMON_PIANO_GUITAR_PATTERN_NUMBER)]  # 如果数据集中这个状态没有接任何一个下一个状态，则概率均分
            else:
                self.transfer[row_iterator] = self.transfer_count[row_iterator] / row_sum
        # 2.计算观测矩阵
        for column_iterator in range(COMMON_PIANO_GUITAR_PATTERN_NUMBER):
            column_sum = sum(self.emission_count[:, column_iterator])
            if column_sum == 0:
                self.emission[:, column_iterator] = [1 / (len(self.rc_pattern_dict) - 1) for t in range(len(self.rc_pattern_dict) - 1)]  # 为空列则概率均分
            else:
                self.emission[:, column_iterator] = self.emission_count[:, column_iterator] / column_sum
        # 3.计算初始化转移向量
        self.pi = self.pi_count / sum(self.pi_count)


def RhythmPatterns(raw_music_data, time_step, pattern_time_step):
    rhythm_pattern_dict = [[0 for t in range(8)]]
    rhythm_pattern_data = [[[] for t0 in range(TRAIN_FILE_NUMBERS)] for t1 in range(len(raw_music_data))]  # 三维数组 第一维是歌曲列表 第二维是小节编号 第三维是按键的组合（步长是2拍）
    rhythm_pattern_count = [0]
    for part_iterator, rhythm_part in enumerate(raw_music_data):
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            if rhythm_part[song_iterator] != {}:
                for key in range(GetDictMaxKey(rhythm_part[song_iterator]) + 1):
                    rhythm_pattern_data[part_iterator][song_iterator].append([0, 0])
                    bar_rhythm_data = [1 if t != 0 else 0 for t in rhythm_part[song_iterator][key]]
                    raw_pattern_list = [bar_rhythm_data[8 * t: 8 * (t + 1)] for t in range(2)]
                    for bar_pattern_iterator, raw_pattern in enumerate(raw_pattern_list):
                        if raw_pattern not in rhythm_pattern_dict:
                            rhythm_pattern_data[part_iterator][song_iterator][key][bar_pattern_iterator] = len(rhythm_pattern_dict)
                            rhythm_pattern_dict.append(raw_pattern)
                            rhythm_pattern_count.append(1)
                        else:
                            rhythm_pattern_data[part_iterator][song_iterator][key][bar_pattern_iterator] = rhythm_pattern_dict.index(raw_pattern)
                            rhythm_pattern_count[rhythm_pattern_dict.index(raw_pattern)] += 1
    return rhythm_pattern_data, rhythm_pattern_dict, rhythm_pattern_count


class CmpRhythmTrainData:

    emission = []  # 观测矩阵
    transfer = []  # 状态转移矩阵
    pi = []  # 初始状态矩阵
    emission_count = []  # 观测计数矩阵
    transfer_count = []  # 状态转移计数矩阵
    pi_count = []  # 初始状态计数矩阵

    def __init__(self, keypress_pattern_data, keypress_pattern_dict, continuous_bar_number_data, time_step, pattern_time_step):
        self.rhythm_data = self.get_raw_rhythm_data()
        # self.melody_rhythm_data = self.get_keypress_data()
        # 3.获取最常见的piano_guitar组合
        # self.common_pg_patterns, __ = CommonMusicPatterns(raw_pg_data, number=COMMON_PIANO_GUITAR_PATTERN_NUMBER, note_time_step=1 / 4, pattern_time_step=1)
        self.rhythm_patterns, self.rhythm_pattern_dict, self.rhythm_pattern_count = RhythmPatterns(self.rhythm_data, time_step, pattern_time_step)
        # self.melody_rhythm_patterns, __ = RhythmPatterns(self.rhythm_data, time_step, pattern_time_step)
        self.transfer_count = np.zeros([len(self.rhythm_pattern_count) - 1, len(self.rhythm_pattern_count) - 1])
        self.emission_count = np.zeros([len(keypress_pattern_dict) - 1, len(self.rhythm_pattern_count) - 1])
        self.pi_count = np.zeros([len(self.rhythm_pattern_count) - 1])
        for pg_part_iterator in range(len(self.rhythm_data)):
            for song_iterator in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if self.rhythm_data[pg_part_iterator][song_iterator] != {} and keypress_pattern_data[song_iterator] != []:
                    # self.rhythm_data[pg_part_iterator][song_iterator] = RhythmPatternEncode(self.rhythm_patterns, self.rhythm_data[pg_part_iterator][song_iterator], keypress_pattern_data[song_iterator], time_step, pattern_time_step).music_pattern_dict
                    self.count(self.rhythm_patterns[pg_part_iterator][song_iterator], keypress_pattern_data[song_iterator], song_iterator)
                    self.count_initial(self.rhythm_patterns[pg_part_iterator][song_iterator], keypress_pattern_data[song_iterator], continuous_bar_number_data[song_iterator])
        self.prob_convert(len(self.rhythm_pattern_count) - 1, len(keypress_pattern_dict) - 1)

    def get_raw_rhythm_data(self):
        pass

    def count(self, rhythm_data, keypress_data, g):
        for bar_iterator in range(len(rhythm_data)):
            if bar_iterator >= len(keypress_data):  # 这一小节没有主旋律 直接跳过
                continue
            for time_iterator in range(2):
                if bar_iterator == 0 and time_iterator == 0:  # 不考虑第一小节的第一拍
                    continue
                if time_iterator != 0:
                    if (rhythm_data[bar_iterator][time_iterator] == 0) or (rhythm_data[bar_iterator][time_iterator - 1] == 0) or (keypress_data[bar_iterator][time_iterator] == 0):  # 当前拍 上一拍的节奏编码不能是0或common+1 同时当前拍主旋律按键编码不能是0
                        continue
                    self.transfer_count[rhythm_data[bar_iterator][time_iterator - 1] - 1, rhythm_data[bar_iterator][time_iterator] - 1] += 1
                    self.emission_count[keypress_data[bar_iterator][time_iterator] - 1, rhythm_data[bar_iterator][time_iterator] - 1] += 1
                else:
                    if (rhythm_data[bar_iterator][0] == 0) or (rhythm_data[bar_iterator - 1][1] == 0) or keypress_data[bar_iterator][0] == 0:
                        continue
                    self.transfer_count[rhythm_data[bar_iterator - 1][1] - 1, rhythm_data[bar_iterator][0] - 1] += 1
                    self.emission_count[keypress_data[bar_iterator][0] - 1, rhythm_data[bar_iterator][0] - 1] += 1

    def count_initial(self, rhythm_data, keypress_data, continuous_bar_number_data):
        for bar_iterator in range(len(rhythm_data)):
            if bar_iterator >= len(continuous_bar_number_data):  # 这个小节没有对应的主旋律 直接跳过
                continue
            if bar_iterator == 0:
                if continuous_bar_number_data[0] != 0:
                    if rhythm_data[0][0] != 0:
                        self.pi_count[rhythm_data[0][0] - 1] += 1
            else:
                if continuous_bar_number_data[bar_iterator] != 0 and continuous_bar_number_data[bar_iterator - 1] == 0:
                    if rhythm_data[bar_iterator][0] != 0:
                        self.pi_count[rhythm_data[bar_iterator][0] - 1] += 1

    def prob_convert(self, rhythm_pattern_number, keypress_pattern_number):
        self.transfer = np.zeros([rhythm_pattern_number, rhythm_pattern_number])
        self.emission = np.zeros([keypress_pattern_number, rhythm_pattern_number])
        self.pi = np.zeros([rhythm_pattern_number])
        # 1.计算转移矩阵
        for row_iterator, row in enumerate(self.transfer_count):
            row_sum = sum(row)
            if row_sum == 0:
                self.transfer[row_iterator] = [1 / rhythm_pattern_number for t in range(rhythm_pattern_number)]  # 如果数据集中这个状态没有接任何一个下一个状态，则概率均分
            else:
                self.transfer[row_iterator] = self.transfer_count[row_iterator] / row_sum
        # 2.计算观测矩阵
        for column_iterator in range(rhythm_pattern_number):
            column_sum = sum(self.emission_count[:, column_iterator])
            if column_sum == 0:
                self.emission[:, column_iterator] = [1 / keypress_pattern_number for t in range(keypress_pattern_number)]  # 为空列则概率均分
            else:
                self.emission[:, column_iterator] = self.emission_count[:, column_iterator] / column_sum
        # 3.计算初始化转移向量
        self.pi = self.pi_count / sum(self.pi_count)


class PGRhythmTrainData(CmpRhythmTrainData):

    def get_raw_rhythm_data(self):
        raw_pg_rhythm_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        pg_iterator = 1
        while True:
            pg_part_data = GetRawSongDataFromDataset('piano_guitar' + str(pg_iterator), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if pg_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_pg_rhythm_data.append(pg_part_data)
            pg_iterator += 1
        for pg_part_iterator in range(len(raw_pg_rhythm_data)):
            for song_iterator in range(TRAIN_FILE_NUMBERS):
                if raw_pg_rhythm_data[pg_part_iterator][song_iterator] != {}:
                    for bar_iterator in range(GetDictMaxKey(raw_pg_rhythm_data[pg_part_iterator][song_iterator]) + 1):
                        raw_pg_rhythm_data[pg_part_iterator][song_iterator][bar_iterator] = [1 if t != 0 else 0 for t in raw_pg_rhythm_data[pg_part_iterator][song_iterator][bar_iterator]]
        return raw_pg_rhythm_data


class PianoGuitarTrainData_3(PianoGuitarTrainData_2):

    emission_final = []

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data, keypress_pattern_data, keypress_pattern_dict):
        super().__init__(melody_pattern_data, continuous_bar_number_data, chord_data)
        self.pg_rhythm_data = PGRhythmTrainData(keypress_pattern_data, keypress_pattern_dict, continuous_bar_number_data, 0.25, 2)

    def combine_rck(self):
        """
        对节奏 根音和和弦进行整合
        :return:
        """
        pg_pattern_rhythm = [[1 if t0 != 0 else 0 for t0 in t1] for t1 in self.common_pg_patterns[1:]]
        pg_rhythm_number = len(self.pg_rhythm_data.rhythm_pattern_count) - 1
        pg_rhythm_match = np.zeros([2 * pg_rhythm_number, COMMON_PIANO_GUITAR_PATTERN_NUMBER])  # 一个矩阵 行为2×节拍字典 列为这些PG组合
        for row in range(2 * pg_rhythm_number):
            for column in range(len(pg_pattern_rhythm)):
                if pg_pattern_rhythm[column] == self.pg_rhythm_data.rhythm_pattern_dict[row % pg_rhythm_number + 1][4 * (row // pg_rhythm_number): 4 * ((row // pg_rhythm_number) + 1)]:
                    pg_rhythm_match[row, column] = 1
        # TODO: 向量化？
        self.emission_final = np.zeros([pg_rhythm_match.shape[0] * self.emission.shape[0], pg_rhythm_match.shape[1]])  # 矩阵 行为2×节拍字典×rc组合数量 列为这些PG组合
        for rhythm_it in range(pg_rhythm_match.shape[0]):
            for rc_it in range(self.emission.shape[0]):
                self.emission_final[rc_it * pg_rhythm_match.shape[0] + rhythm_it] = pg_rhythm_match[rhythm_it] * self.emission[rc_it]
                if sum(self.emission_final[rc_it * pg_rhythm_match.shape[0] + rhythm_it]) == 0:  # 如果没有到达它的状态，可能会引发bug 这里给一个定值
                    self.emission_final[rc_it * pg_rhythm_match.shape[0] + rhythm_it] = 0.0001
        # 这里不归一化

    def combine_rck_2(self):
        pg_pattern_rhythm = [[1 if t0 != 0 else 0 for t0 in t1] for t1 in self.common_pg_patterns[1:]]
        pg_pattern_rhythm = [8 * t[0] + 4 * t[1] + 2 * t[2] + t[3] for t in pg_pattern_rhythm]
        pg_rhythm_match = np.array([[int(pg_pattern_rhythm[t0] == t1) for t0 in range(COMMON_PIANO_GUITAR_PATTERN_NUMBER)] for t1 in range(16)])
        self.gg = pg_rhythm_match
        self.emission_final = np.zeros([16 * self.emission.shape[0], COMMON_PIANO_GUITAR_PATTERN_NUMBER])  # 矩阵 行为2×节拍字典×rc组合数量 列为这些PG组合
        for rhythm_it in range(pg_rhythm_match.shape[0]):
            for rc_it in range(self.emission.shape[0]):
                self.emission_final[rc_it * pg_rhythm_match.shape[0] + rhythm_it] = pg_rhythm_match[rhythm_it] * self.emission[rc_it]
                if sum(self.emission_final[rc_it * pg_rhythm_match.shape[0] + rhythm_it]) == 0:  # 如果没有到达它的状态，可能会引发bug 这里给一个定值
                    self.emission_final[rc_it * pg_rhythm_match.shape[0] + rhythm_it] = 0.0001
