import numpy as np
from settings import *
from interfaces.functions import GetFirstIndexBigger, GetFirstIndexSmaller, GetDictMaxKey
from interfaces.sql.sqlite import NoteDict, GetRawSongDataFromDataset, GetSongToneList
import copy
# from .strings import StringPatternEncode


def JudgeImitation(input_melody_list, input_comp_list, speed_ratio_dict):
    """
    判断伴奏是否对主旋律存在某种模仿关系 以1拍为时间步长进行存储
    存储方式为[时间差别 音高差别 速度差别]。其中时间差别不能超过8拍（64），速度差别必须是1（一半） 2（相同） 3（两倍）之间选择
    为０则不模仿　为１则与前一拍的情况相同
    只有一个音符的情况下不能视作模仿
    :param speed_ratio_dict:
    :param input_melody_list: 主旋律列表 相对音高形式
    :param input_comp_list: 伴奏列表 相对音高形式
    :return:
    """
    # 1.首先将melody_list和comp_list中的零值都去掉 加上位置 并转成np.array形式
    melody_list = np.array([[t[0][0], t[0][1], i] for (i, t) in enumerate(input_melody_list) if t != 0])
    comp_list = np.array([[t[0][0], t[0][1], i] for (i, t) in enumerate(input_comp_list) if t != 0])
    imitation_comp_list = np.array([0 for t in range(len(input_comp_list) // 8)], dtype=object)
    # A.一整拍以内的所有音符必须都要符合此模仿关系
    # B.被模仿的主旋律音符相邻两个音符间隔不能超过1拍
    # C.相隔距离不能超过模仿长度的2倍+2拍
    for beat_iterator in range(0, len(input_comp_list), 8):
        beat_comp = np.array(input_comp_list[beat_iterator: beat_iterator + 8], dtype=object)
        if imitation_comp_list[beat_iterator // 8] != 0:
            continue
        if len(beat_comp[beat_comp != 0]) == 0:
            continue
        first_index, __ = GetFirstIndexBigger(melody_list[:, 2], beat_iterator - 56)
        last_index, __ = GetFirstIndexSmaller(melody_list[:, 2], beat_iterator + 1)
        part_comp_list = comp_list[comp_list[:, 2] >= beat_iterator]  # 这一拍之后的伴奏列表
        longest = 1  # 最长的模仿序列长度
        longest_message = {'note': 0, 'time': 0, 'speed': 0}  # 最长的模仿序列在主旋律列表中的位置
        for startnote_iterator in range(first_index, last_index + 1):
            try:
                note_diff = part_comp_list[0, 0] - melody_list[startnote_iterator, 0]  # 音高差别
                time_diff = part_comp_list[0, 2] - melody_list[startnote_iterator, 2]  # 时间差别
                speed_ratio = round((part_comp_list[1, 2] - part_comp_list[0, 2]) / (melody_list[startnote_iterator + 1, 2] - melody_list[startnote_iterator, 2]), 1)  # 速度的比
                imitation_length = 1  # 模仿序列的长度
                note_iterator = startnote_iterator + 1
                while note_iterator <= len(melody_list) - 1:  # 判断模仿序列的长度
                    if (part_comp_list[imitation_length, 0] - melody_list[note_iterator, 0] != note_diff) or (part_comp_list[imitation_length, 2] - melody_list[note_iterator, 2] != time_diff):
                        break
                    if melody_list[note_iterator, 2] - melody_list[note_iterator - 1, 2] > 8:  # 被模仿的主旋律音符相邻两个音符间隔不能超过1拍
                        break
                    if round((part_comp_list[imitation_length, 2] - part_comp_list[imitation_length - 1, 2]) / (melody_list[note_iterator, 2] - melody_list[note_iterator - 1, 2]), 1) != speed_ratio:
                        break
                    imitation_length += 1
                    note_iterator += 1
                if imitation_length >= 2 and imitation_length >= longest:  # 模仿序列的长度在２以上且大于等于当前的最长序列 则认为是
                    speed_ratio = speed_ratio_dict.get(speed_ratio, 0)
                    if speed_ratio == 0:
                        continue
                    if time_diff >= 16 + 2 * (part_comp_list[imitation_length - 1, 2] - part_comp_list[0, 2]):  # 相隔距离不能超过模仿长度的2倍+2拍
                        continue
                    step_lack = 8 - part_comp_list[imitation_length - 1, 2] + beat_iterator
                    if step_lack >= 2 and input_comp_list[part_comp_list[imitation_length - 1, 2]: beat_iterator + 8] != [0 for t in range(step_lack)]:  # 一整拍以内的所有音符必须都要符合此模仿关系
                        continue
                    longest = imitation_length
                    longest_message = {'note': note_diff, 'time': time_diff, 'speed': speed_ratio}
            except IndexError:
                pass
        if longest >= 2:
            imitation_comp_list[beat_iterator // 8] = [longest_message['note'], longest_message['time'], longest_message['speed']]
            imitation_end_beat = part_comp_list[longest - 1, 2] // 8  # 模仿序列结束于哪拍
            imitation_comp_list[beat_iterator // 8 + 1: imitation_end_beat + 1] = 1
    return imitation_comp_list


def GetRelNoteList(note_list, tone, root):
    """
    获取相对音高的音符列表
    :param note_list: 原始音符列表
    :param tone: 节奏（０为大调　１为小调）
    :param root: 根音（０为６０　１为５７）
    :return: 相对音高的音符列表
    """
    # 1.获取相对音高字典
    if tone == TONE_MAJOR:
        rel_note_dic = [[0, 0], [1, -1], [1, 0], [2, -1], [2, 0], [3, 0], [3, 1], [4, 0], [5, -1], [5, 0], [6, -1], [6, 0]]
    elif tone == TONE_MINOR:
        rel_note_dic = [[0, 0], [1, -1], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [5, 0], [5, 1], [6, 0], [6, 1]]
    # 2.转化为相对音高
    rel_note_list = []
    for note in sorted(note_list, reverse=True):  # 先从大到小排序
        rel_note_list.append([7 * ((note - root) // 12) + rel_note_dic[(note - root) % 12][0], rel_note_dic[(note - root) % 12][1]])
    return rel_note_list


def OneSongRelNoteList(note_list, tone, root, use_note_dict=False):
    """
    获取一首歌的相对音高的音符列表
    :param note_list: 这首歌的原始音高列表
    :param tone: 节奏（０为大调　１为小调）
    :param root: 根音（０为72　１为69）
    :return: 这首歌的相对音高的音符列表
    """
    rel_note_list = []  # 转化为相对音高形式的音符列表
    for note_iterator, note in enumerate(note_list):
        if note == 0:
            rel_note_list.append(0)
        else:
            if use_note_dict is True:
                rel_note_group = GetRelNoteList(NoteDict[note], tone, root)
            else:
                rel_note_group = GetRelNoteList([note], tone, root)
            rel_note_list.append(rel_note_group)
    return rel_note_list


def MelodyCoreNote(melody_data):
    """
    寻找主旋律的核心音符
    :param melody_data: 主旋律 展开形式 相对音高
    :return: 核心音列表
    """
    melody_core_list = []
    for time_step_iterator in range(0, len(melody_data)):
        if melody_data[time_step_iterator] == 0:  # 如果这个时间步长没有音符 回溯前四拍 都没有则置为0
            if melody_data[time_step_iterator - time_step_iterator % 8] != 0:  # 先观察这一拍的第一个时间步长有没有音符 如果有则取之
                melody_core_list.append(melody_data[time_step_iterator - time_step_iterator % 8])
            else:  # 如果这个时间步长没有音符 这一拍的第一个时间步长也没有音符 回溯前四拍 都没有则置为0
                find_note = False
                for note_iterator in range(time_step_iterator, max(-1, time_step_iterator - 33), -1):
                    if melody_data[note_iterator] != 0:
                        melody_core_list.append(melody_data[note_iterator])
                        find_note = True
                        break
                if find_note is False:
                    melody_core_list.append(0)
        else:  # 不是这一拍的第一个时间步长 那么先观察这一拍的第一个时间步长有没有音符 如果有则取之 如果没有则使用这个时间步长的
            if melody_data[time_step_iterator - time_step_iterator % 8] != 0:
                melody_core_list.append(melody_data[time_step_iterator - time_step_iterator % 8])
            else:
                melody_core_list.append(melody_data[time_step_iterator])
    return melody_core_list


def CommonFillPatterns(flatten_fill_data, note_time_step, pattern_time_step):
    common_pattern_dict = {}
    time_step_ratio = round(pattern_time_step / note_time_step)
    for fill_part in flatten_fill_data:
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            if fill_part[song_iterator] != {}:
                beat_number = len(fill_part[song_iterator]) // 8
                pattern_list = [fill_part[song_iterator][time_step_ratio * t: time_step_ratio * (t + 1)] for t in range(round(beat_number / pattern_time_step))]
                for pattern_iterator in pattern_list:
                    try:
                        common_pattern_dict[str(pattern_iterator)] += 1
                    except KeyError:
                        common_pattern_dict[str(pattern_iterator)] = 1
    common_pattern_list_temp = sorted(common_pattern_dict.items(), key=lambda asd: asd[1], reverse=True)  # 按照次数由高到底排序
    # print(len(common_pattern_list_temp))
    # sum1 = 0
    # sum2 = 0
    # for t in common_pattern_list_temp[1:]:
    #     sum1 += t[1]
    # for t in common_pattern_list_temp[1: (COMMON_FILL_PATTERN_NUMBER + 1)]:
    #     sum2 += t[1]
    # print(sum2 / sum1)
    common_pattern_list = []  # 最常见的number种组合
    pattern_number_list = []  # 这些组合出现的次数
    for pattern_tuple in common_pattern_list_temp[:(COMMON_FILL_PATTERN_NUMBER + 1)]:
        common_pattern_list.append(eval(pattern_tuple[0]))
        pattern_number_list.append(pattern_tuple[1])
    # print(common_pattern_list)
    # print(pattern_number_list)
    return common_pattern_list, pattern_number_list


class FillPatternEncode:

    def __init__(self, common_patterns, song_fill_list, note_time_step, pattern_time_step):
        self.music_pattern_list = []  # 按照最常见的音符组合编码之后的组合列表（dict形式）
        time_step_ratio = round(pattern_time_step / note_time_step)
        # print(bar_iterator[0])
        beat_number = len(song_fill_list) // 8  # 这首歌有多少拍
        song_note_list = [song_fill_list[time_step_ratio * t: time_step_ratio * (t + 1)] for t in range(round(beat_number / pattern_time_step))]  # 将音符列表按照pattern_time_step进行分割 使其变成二维数组
        song_pattern_list = self.handle_common_patterns(song_note_list, common_patterns)
        song_pattern_list = self.handle_rare_pattern(song_pattern_list, song_note_list, common_patterns)
        self.music_pattern_list = song_pattern_list  # 将编码后的pattern list保存在新的pattern dict中

    def handle_common_patterns(self, raw_song_note_list, common_patterns):
        song_pattern_list = []
        for song_pattern_iterator in range(len(raw_song_note_list)):
            try:
                pattern_code = common_patterns.index(raw_song_note_list[song_pattern_iterator])  # 在常见的组合列表中找到这个音符组合
                song_pattern_list.append(pattern_code)
            except ValueError:  # 找不到
                song_pattern_list.append(-1)
        return song_pattern_list

    def handle_rare_pattern(self, song_pattern_list, song_note_list, common_patterns):
        # 在常见的加花列表里找不到某一个加花组合的处理方法：
        # a.寻找符合以下条件的加花组合
        # a1.整半拍处的休止情况完全相同 且没有模仿关系
        # a2.该组合的所有音符是待求组合所有音符的子集 或与待求组合的音符相差7的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为common_patterns+1
        for fill_pattern_iterator in range(len(song_pattern_list)):
            if song_pattern_list[fill_pattern_iterator] == -1:
                choose_pattern = 0  # 选取的pattern
                choose_pattern_diff_score = 75  # 两个旋律组合的差异程度
                for common_fill_iterator in range(1, len(common_patterns)):
                    total_note_count = 0
                    diff_note_count = 0
                    # 1.1.检查该旋律组合是否符合要求
                    if song_note_list[fill_pattern_iterator].count(1) > 0 or common_patterns[common_fill_iterator].count(1) > 0:  # 有模仿关系 则不近似
                        continue
                    note_satisfactory = True
                    for group_iterator in range(len(common_patterns[common_fill_iterator])):
                        # 1.1.1.检查有无模仿关系 且整半拍的休止情况是否相同
                        if group_iterator % 2 == 0:
                            if bool(common_patterns[common_fill_iterator][group_iterator]) ^ bool(song_note_list[fill_pattern_iterator][group_iterator]):  # 一个为休止符 一个不为休止符
                                note_satisfactory = False
                                break
                        if common_patterns[common_fill_iterator][group_iterator] == 0 and song_note_list[fill_pattern_iterator][group_iterator] == 0:  # 如果这个时间步长是休止符的话，直接进入下个时间步长
                            continue
                        elif common_patterns[common_fill_iterator][group_iterator] == 0 and song_note_list[fill_pattern_iterator][group_iterator] != 0:  # pattern是休止而待求片段不是休止 计入不同后进入下个时间步长
                            total_note_count += len(song_note_list[fill_pattern_iterator][group_iterator]) - 1
                            diff_note_count += (len(song_note_list[fill_pattern_iterator][group_iterator]) - 1) * 1.2
                            continue
                        elif common_patterns[common_fill_iterator][group_iterator] != 0 and song_note_list[fill_pattern_iterator][group_iterator] == 0:  # pattern不是休止而待求片段是休止 计入不同后进入下个时间步长
                            total_note_count += len(common_patterns[common_fill_iterator][group_iterator]) - 1
                            diff_note_count += (len(common_patterns[common_fill_iterator][group_iterator]) - 1) * 1.2
                            continue
                        # 1.1.2.求出相对音高组合并对7取余数 按升降号分开
                        cur_pattern_note_list = common_patterns[common_fill_iterator][group_iterator][1:]  # 这个时间步长中常见组合的真实音符组合
                        cur_pattern_note_list_div7 = [[t[0] % 7, t[1]] for t in cur_pattern_note_list]  # 对7取余数
                        cur_pattern_sj_set = {t[1] for t in cur_pattern_note_list}  # 升降关系的集合
                        cur_pattern_note_dict_div7 = {t0: {t1 for [t1, t2] in cur_pattern_note_list_div7 if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开
                        cur_pattern_note_dict = {t0: {t1 for [t1, t2] in cur_pattern_note_list if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开

                        cur_step_note_list = song_note_list[fill_pattern_iterator][group_iterator][1:]  # 这个时间步长中待求组合的真实音符组合
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
                        choose_pattern = common_fill_iterator
                        choose_pattern_diff_score = pattern_diff_score
                # 2.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_PIANO_GUITAR_PATTERNS+1
                # print(bar_pg_pattern_iterator, raw_bar_pattern_list, choose_pattern, choose_pattern_diff_score)
                if choose_pattern != 0:
                    song_pattern_list[fill_pattern_iterator] = choose_pattern
                else:
                    song_pattern_list[fill_pattern_iterator] = COMMON_FILL_PATTERN_NUMBER + 1
        # print(bar_pattern_list, '\n')
        return song_pattern_list


class FillTrainData:

    input_data = []  # 输入model的数据
    output_data = []  # 从model输出的数据

    fi = 0
    nf = 0
    speed_ratio_dict = {0.5: 1, 1: 2}

    def __init__(self, melody_data, melody_pattern_data, continuous_bar_number_data):
        # 1.从数据集中读取歌的加花数据
        raw_fill_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        fill_part_iterator = 1
        song_tone_list = GetSongToneList()
        flatten_melody_pattern_data = [[] for t in range(TRAIN_FILE_NUMBERS)]
        for song_iterator in range(len(melody_pattern_data)):
            for key in range(GetDictMaxKey(melody_pattern_data[song_iterator]) + 1):
                flatten_melody_pattern_data[song_iterator] += melody_pattern_data[song_iterator][key]
        while True:
            fill_part_data = GetRawSongDataFromDataset('fill' + str(fill_part_iterator), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if fill_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_fill_data.append(fill_part_data)
            # flatten_pg_data += pg_part_data
            fill_part_iterator += 1
        for fill_part_iterator, fill_part in enumerate(raw_fill_data):  # 将绝对音高转变为相对音高
            for song_iterator in range(TRAIN_FILE_NUMBERS):
                if fill_part[song_iterator] != {} and song_tone_list[song_iterator] is not None:
                    raw_fill_data[fill_part_iterator][song_iterator] = self.get_rel_fill_data(fill_part[song_iterator], melody_data[song_iterator], song_tone_list[song_iterator])  # 将加花数据变更为相对音高数据
        self.common_fill_patterns, __ = CommonFillPatterns(raw_fill_data, 0.125, 1)
        for fill_part_iterator in range(len(raw_fill_data)):
            for song_iterator in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_fill_data[fill_part_iterator][song_iterator] != {} and melody_data[song_iterator] != {}:
                    # raw_pg_data[pg_part_iterator][song_iterator] = PianoGuitarPatternEncode(self.common_pg_patterns, raw_pg_data[pg_part_iterator][song_iterator], 0.25, 1).music_pattern_dict
                    raw_fill_data[fill_part_iterator][song_iterator] = FillPatternEncode(self.common_fill_patterns, raw_fill_data[fill_part_iterator][song_iterator], 0.125, 1).music_pattern_list
                    self.get_model_io_data(raw_fill_data[fill_part_iterator][song_iterator], flatten_melody_pattern_data[song_iterator], continuous_bar_number_data[song_iterator])
                # if fill_part_iterator == 0 and song_iterator == 6:
                #     for t in self.input_data:
                #         print(t)
                #     print('\n\n\n')
                #     for t in self.output_data:
                #         print(t)
        print(len(self.input_data), len(self.output_data))

    def get_rel_fill_data(self, fill_data, melody_data, tone):
        # 1.首先将fill_Data和melody_data展成一维数组的形式
        flatten_fill_data = []
        flatten_melody_data = []
        for key in range(GetDictMaxKey(fill_data) + 1):
            flatten_fill_data += fill_data[key]
        for key in range(GetDictMaxKey(melody_data) + 1):
            flatten_melody_data += melody_data[key]
        # 2.将fill_data和melody_data都转化成相对音高形式
        if tone == TONE_MAJOR:
            flatten_fill_data = OneSongRelNoteList(flatten_fill_data, TONE_MAJOR, 72, use_note_dict=True)
            flatten_melody_data = OneSongRelNoteList(flatten_melody_data, TONE_MAJOR, 72)
        elif tone == TONE_MINOR:
            flatten_fill_data = OneSongRelNoteList(flatten_fill_data, TONE_MINOR, 69, use_note_dict=True)
            flatten_melody_data = OneSongRelNoteList(flatten_melody_data, TONE_MINOR, 69)
        # 3.寻找模仿结构
        imitate_fill_list = JudgeImitation(flatten_melody_data, flatten_fill_data, self.speed_ratio_dict)
        # 4.寻找这首歌的骨干音列表
        core_notelist = MelodyCoreNote(flatten_melody_data)
        # 5.得到相对音高列表
        rel_fill_data = []
        for beat_iterator in range(0, len(flatten_fill_data), 8):
            if imitate_fill_list[beat_iterator // 8] not in [0, 1]:
                rel_fill_data += [[1, imitate_fill_list[beat_iterator // 8]]] + [1 for t in range(7)]
            elif imitate_fill_list[beat_iterator // 8] == 1:
                rel_fill_data += [1 for t in range(8)]
            elif beat_iterator >= len(flatten_melody_data):  # 这一拍没有主旋律
                rel_fill_data += [0, 0, 0, 0, 0, 0, 0, 0]
            else:  # 这一拍没有模仿结构
                for note_iterator in range(beat_iterator, beat_iterator + 8):
                    if flatten_fill_data[note_iterator] == 0 or core_notelist[note_iterator] == 0:  # 这一拍没有加花或主旋律
                        rel_fill_data.append(0)
                    else:
                        rel_fill_data.append([0] + [[t[0] - core_notelist[note_iterator][0][0], t[1] - core_notelist[note_iterator][0][1]] for t in flatten_fill_data[note_iterator]])  # 添加加花数据和主旋律数据的音高差值
        return rel_fill_data

    def get_model_io_data(self, fill_data, melody_pattern_data, continuous_bar_number_data):
        # 模型输入内容为当前时间的编码，前四拍+当拍+后两拍的主旋律，前三拍的加花，以及上一次的四拍加花和四拍主旋律 总计19
        if len(melody_pattern_data) <= 6:
            return
        for fill_iterator in range(len(fill_data)):
            if fill_iterator >= len(melody_pattern_data) + 4:
                break
            if fill_data[fill_iterator] != 0 and melody_pattern_data[fill_iterator] != 0:
                time_in_bar = fill_iterator % 4
                time_add = (1 - continuous_bar_number_data[fill_iterator // 4] % 2) * 4
                input_time_data = [time_in_bar + time_add] + [0 for t in range(8)]
                output_time_data = [time_in_bar + time_add] + [0 for t in range(8)]

                if fill_iterator <= 3:
                    input_time_data = input_time_data + [0 for t in range(4 - fill_iterator)] + melody_pattern_data[:(fill_iterator + 3)]
                    output_time_data = output_time_data + [0 for t in range(4 - fill_iterator)] + melody_pattern_data[:(fill_iterator + 3)]
                elif fill_iterator >= len(melody_pattern_data) - 2:
                    input_time_data = input_time_data + melody_pattern_data[fill_iterator - 4:] + [0 for t in range(fill_iterator - len(melody_pattern_data) + 3)]
                    output_time_data = output_time_data + melody_pattern_data[fill_iterator - 4:] + [0 for t in range(fill_iterator - len(melody_pattern_data) + 3)]
                else:
                    input_time_data = input_time_data + melody_pattern_data[fill_iterator - 4: fill_iterator + 3]
                    output_time_data = output_time_data + melody_pattern_data[fill_iterator - 4: fill_iterator + 3]
                if fill_iterator <= 2:
                    input_time_data = input_time_data + [0 for t in range(3 - fill_iterator)] + fill_data[:fill_iterator]
                    output_time_data = output_time_data + [0 for t in range(2 - fill_iterator)] + fill_data[:(fill_iterator + 1)]
                else:
                    input_time_data = input_time_data + fill_data[fill_iterator - 3: fill_iterator]
                    output_time_data = output_time_data + fill_data[fill_iterator - 2: fill_iterator + 1]

                for lookback_iterator in range(fill_iterator - 4, -1, -1):
                    if fill_data[lookback_iterator] != 0 and melody_pattern_data[lookback_iterator] != 0:
                        if lookback_iterator <= 2:
                            input_time_data[4 - lookback_iterator: 5] = melody_pattern_data[:lookback_iterator + 1]
                            input_time_data[8 - lookback_iterator: 9] = fill_data[:lookback_iterator + 1]
                            output_time_data[4 - lookback_iterator: 5] = melody_pattern_data[:lookback_iterator + 1]
                            output_time_data[8 - lookback_iterator: 9] = fill_data[:lookback_iterator + 1]
                        else:
                            input_time_data[1: 5] = melody_pattern_data[lookback_iterator - 3: lookback_iterator + 1]
                            input_time_data[5: 9] = fill_data[lookback_iterator - 3: lookback_iterator + 1]
                            output_time_data[1: 5] = melody_pattern_data[lookback_iterator - 3: lookback_iterator + 1]
                            output_time_data[5: 9] = fill_data[lookback_iterator - 3: lookback_iterator + 1]
                        break
                self.input_data.append(input_time_data)
                self.output_data.append(output_time_data)
