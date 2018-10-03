from interfaces.chord_parse import notelist2chord
from settings import *
import numpy as np
import tensorflow as tf
import copy
import math


def melody_check(melody_list):
    """
    检查生成的一小节音乐是否有异常内容 从按键的位置和音高来判别是否符合要求
    :param melody_list: 旋律列表
    :return: 返回是否符合要求
    """
    # 1.检查是否存在错位情况
    for note_it in range(0, len(melody_list), 4):
        if melody_list[note_it] == 0 and melody_list[note_it + 1] != 0:
            return False
    for note_it in range(0, len(melody_list) - 4, 4):
        if melody_list[note_it] == 0 and melody_list[note_it + 3] != 0 and melody_list[note_it + 4] == 0:
            return False
    for note_it in range(0, len(melody_list), 8):
        if melody_list[note_it] == 0 and melody_list[note_it + 2] != 0:
            return False
    for note_it in range(0, len(melody_list) - 8, 8):
        if melody_list[note_it] == 0 and melody_list[note_it + 6] != 0 and melody_list[note_it + 8] == 0:
            return False
        if melody_list[note_it] == 0 and melody_list[note_it + 4] != 0 and melody_list[note_it + 8] == 0 and melody_list[note_it + 12] == 0:
            return False
    # 2.检查相邻两个音符的的音高之差是否太大
    # 相邻两个32分音符之间不超过小三度 相邻两个16分音符之间不超过纯四度 相邻两个八分音符间不超过纯五度 相邻两个四分音符间不超过大六度 相邻两个二分音符间不超过八度 其余无限制
    # if FLAG_IS_DEBUG:
    #     max_scale_diff_list = [7, 7, 7, 9, 9, 9, 9, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    # else:
    #     max_scale_diff_list = [3, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 12]
    # for note_it in range(len(melody_list)):
    #     if melody_list[note_it] != 0:  # 只有当不是休止符时才检查音高
    #         last_note_dx = -1  # 上一个音符的位置
    #         last_note_scale = 0  # 上一个音符的音高
    #         for t in range(note_it - 1, 0, -1):  # 找到上一个音符的位置和音高
    #             if melody_list[t] != 0:
    #                 last_note_dx = t
    #                 last_note_scale = melody_list[t]
    #                 break
    #         if last_note_dx != -1:
    #             note_time_diff = note_it - last_note_dx  # 音符时间间隔
    #             scale_diff = abs(melody_list[note_it] - last_note_scale)  # 音符音高间隔
    #             if note_time_diff <= 16 and scale_diff > max_scale_diff_list[note_time_diff - 1]:  # 相邻两个音符音高差别过大
    #                 # print(note_iterator, note_dict[melody_list[note_iterator]][-1], note_time_diff, scale_diff)
    #                 return False
    return True


def melody_confidence_check(melody_output):
    last_note = -1  # 上一个音符的音高
    last_note_step = -1  # 上一个音符所在的位置
    note_count = 0  # 两小节一共多少音符
    shift_score = 0
    for step_it in range(0, 64):
        if melody_output[step_it] != 0:  # 计算变化分
            if last_note > 0:
                step_diff = step_it - last_note_step
                shift_score += (melody_output[step_it] - last_note) * (melody_output[step_it] - last_note) / (step_diff * step_diff)
            last_note = melody_output[step_it]
            last_note_step = step_it
            note_count += 1
    if note_count == 1:  # 只有一个音符的情况下，音高差异分为0分
        return 0
    elif note_count > 1:
        return shift_score / (note_count - 1)


def dfnote_confidence_check(melody_output):

    middle_step_dx = int(len(melody_output) / 2)
    note_count_1half = [0] * 12  # 将各个音符与它们持续时间的对数乘积
    note_count_2half = [0] * 12
    keypress_count_1half = [0] * 32
    keypress_count_2half = [0] * 32
    last_note_step = -1  # 上一个音符所在的位置
    last_note = -1
    for step_it in range(0, len(melody_output)):
        if melody_output[step_it] == 0:  # 这一拍没音符 直接跳过
            continue
        if last_note_step < middle_step_dx:
            if last_note_step != -1:
                keypress_count_1half[last_note_step % 32] += math.log2(step_it - last_note_step) + 1
                note_count_1half[last_note % 12] += math.log2(step_it - last_note_step) + 1
        else:
            if last_note_step != -1:
                keypress_count_2half[last_note_step % 32] += math.log2(step_it - last_note_step) + 1
                note_count_2half[last_note % 12] += math.log2(step_it - last_note_step) + 1
        last_note_step = step_it
        last_note = melody_output[step_it]
    note_diff = sum([abs(note_count_2half[t] - note_count_1half[t]) for t in range(12)]) / (sum(note_count_1half) + sum(note_count_2half))
    keypress_diff = sum([abs(keypress_count_2half[t] - keypress_count_1half[t]) for t in range(32)]) / (sum(keypress_count_1half) + sum(keypress_count_2half))
    return note_diff + keypress_diff * 2


def melody_cluster_check(session, melody_profile_cls, melody_list, right_cluster, max_allow_diff=0):
    """
    检查两小节的音乐cluster是否符合要求
    :param session: session
    :param melody_profile_cls: melody profile计算的相关类
    :param melody_list: 两小节的音符列表
    :param right_cluster: 这两小节合适的cluster
    :param max_allow_diff: 允许这两小节的melody_cluster与合适的cluster有多大的差别
    :return:
    """
    # 1.对melody_list的数据进行一些处理
    melody_cluster_check_input = {int(t / 32): melody_list[-64:][t:t + 32] for t in range(0, len(melody_list[-64:]), 32)}
    # 2.检查cluster是否正确
    melody_cluster = melody_profile_cls.get_melody_profile_by_song(session, melody_cluster_check_input)[0]
    if abs(melody_cluster - right_cluster) <= max_allow_diff:
        return True
    return False


def chord_check(chord_list, melody_list):
    """
    检查两小节的和弦是否存在和弦与同期旋律完全不同的情况 如果存在次数超过1次 则认为是和弦不合格
    :param melody_list: 同时期的主旋律列表（非pattern）
    :param chord_list: 两小节和弦列表
    :return:
    """
    # 还要考虑这段时间没有主旋律/那个音符持续时间过短等情况
    abnormal_chord_num = 0
    last_time_step_abnormal = False  # 上一个时间区段是否是异常 如果是异常 则这个值为True 否则为False
    for chord_pat_step_it in range(0, len(chord_list), 2):  # 时间步长为2拍
        melody_set = set()  # 这段时间内的主旋律列表
        for note_it in range(round(chord_pat_step_it * 8), round((chord_pat_step_it + 2) * 8)):
            if melody_list[note_it] != 0:
                try:
                    if note_it % 8 == 0 or (melody_list[note_it + 1] == 0 and melody_list[note_it + 2] == 0):  # 时间过短的音符不要
                        melody_set.add(melody_list[note_it] % 12)
                except IndexError:
                    pass
        if len(melody_set) != 0 and len(CHORD_DICT[chord_list[chord_pat_step_it]] | melody_set) == len(CHORD_DICT[chord_list[chord_pat_step_it]]) + len(melody_set):
            abnormal_chord_num += 1
            last_time_step_abnormal = True
        elif len(melody_set) == 0 and last_time_step_abnormal is True:  # 如果这段时间内没有主旋律 则看上个时间区段内是否和弦不符合要求 如果上个时间区段内和弦不符合要求 则这个时间区段的和弦也同样不符合要求
            abnormal_chord_num += 1
        else:
            last_time_step_abnormal = False
    if abnormal_chord_num > 1:
        return False
    return True


def melody_end_check(melody_list, tone_restrict=TONE_MAJOR):
    """
    检查一首歌是否以dol/la为结尾 且最后一个音持续时间必须为1/2/4拍
    :param melody_list: 旋律列表
    :param tone_restrict: 这首歌是大调还是小调
    :return: 歌曲的结尾是否符合要求
    """
    last_note_loc_ary = [-32, -28, -24, -20, -16, -12, -8]  # [-32, -16, -8]
    check_result = False
    if tone_restrict == TONE_MAJOR:
        end_note = 0  # 最后一个音符 大调为0（dol） 小调为9（la）
    elif tone_restrict == TONE_MINOR:
        end_note = 9
    else:
        raise ValueError
    for note_it in range(-32, 0):
        if note_it in last_note_loc_ary and melody_list[note_it] != 0 and melody_list[note_it] % 12 == end_note:
            check_result = True
        elif melody_list[note_it] != 0:
            check_result = False
    return check_result


def section_end_check(melody_list, tone_restrict=TONE_MAJOR, last_note_min_len=16):
    """
    检查个乐段是否以dol/mi/sol/la为结尾 且最后一个音持续时间必须至少为1.5拍
    :param last_note_min_len: 这个乐段的最后一个音符至少要有多长
    :param melody_list: 旋律列表
    :param tone_restrict: 这首歌是大调还是小调
    :return: 乐段的结尾是否符合要求
    """
    check_result = False
    if tone_restrict == TONE_MAJOR:
        best_end_note = 0
        end_note_ary = [0, 2, 4, 7]  # 最后一个音符 大调为0,4,7（dol,mi,sol） 小调为4,7,9（mi,sol,la）
    elif tone_restrict == TONE_MINOR:
        best_end_note = 9
        end_note_ary = [4, 7, 9]
    else:
        raise ValueError
    for note_it in range(-32, 0):
        if note_it <= -12 and note_it % 4 == 0 and melody_list[note_it] != 0 and melody_list[note_it] % 12 == best_end_note:
            check_result = True
        elif note_it <= -20 and note_it % 4 == 0 and melody_list[note_it] != 0 and melody_list[note_it] % 12 in end_note_ary:
            check_result = True
        elif melody_list[note_it] != 0:
            check_result = False
    return check_result


def section_begin_check(melody_list, tone_restrict=TONE_MAJOR, note_count=32):
    steps_in_chord = 0
    flag_in_chord = True
    if tone_restrict == TONE_MAJOR:
        chord_note_ary = [0, 4, 7]  # 弦内音符 大调为0,4,7（dol,mi,sol） 小调为4,7,9（mi,sol,la）
    elif tone_restrict == TONE_MINOR:
        chord_note_ary = [4, 7, 9]
    else:
        raise ValueError
    for note_it in range(note_count):
        if melody_list[note_it] == 0:
            steps_in_chord += int(flag_in_chord)
        elif melody_list[note_it] % 12 in chord_note_ary:
            steps_in_chord += 1
            flag_in_chord = True
        else:
            flag_in_chord = False
    return steps_in_chord


def melody_similarity_check(melody_list, melody_dataset_data):
    """
    检查一段音乐是否与训练集中的某一首歌雷同 如果所生成的有连续3小节的相同 则认为是存在雷同
    :param melody_dataset_data: 数据集中的主旋律数据
    :param melody_list:
    :return: 如有雷同返回False 否则返回True
    """
    assert len(melody_list) == 96  # 待监测的音符列表长度必须为3小节
    # 1.读取数据集中的主旋律数据
    # melody_dataset_data = GetRawSongDataFromDataset('main', None)  # 没有旋律限制的主旋律数据
    # 2.检查是否有连续三小节的完全相同
    if melody_list[-96: -64] != [0 for t in range(32)] and melody_list[-64: -32] != [0 for t in range(32)] and melody_list[-32:] != [0 for t in range(32)]:  # 这三个小节都不是空小节的情况下才进行检查 如果中间有空小节 则返回True
        for song_it in range(TRAIN_FILE_NUMBERS):
            if melody_dataset_data[song_it] != {}:
                for bar in melody_dataset_data[song_it]:
                    try:
                        if melody_list[-96: -64] == melody_dataset_data[song_it][bar] and melody_list[-64: -32] == melody_dataset_data[song_it][bar + 1] and melody_list[-32:] == melody_dataset_data[song_it][bar + 2]:
                            return False
                    except KeyError:
                        pass
    return True


def intro_end_check(intro_list, tone_restrict=TONE_MAJOR):
    """
    前奏结束阶段的检查。
    :param intro_list: 前奏的列表
    :param tone_restrict: 调式限制
    :return:
    """
    flag_have_long_note = False  # 前奏的末尾阶段是否有持续时间较长的音符
    long_note = -1  # 持续时间较长的音符的音高
    note_dx = -1
    note_len = 0  # 音符的长度
    for note_it in range(len(intro_list) - 1, len(intro_list) - 33, -1):
        if intro_list[note_it] == 0:
            note_len += 1
        elif note_it - len(intro_list) in [-1, -2, -3, -5, -6, -7] and intro_list[note_it] != 0:  # 前奏的最后一拍只能有整拍和半拍的位置才能有音符
            return False
        else:
            if note_len >= 7 and note_len == len(intro_list) - note_it - 1 and ((tone_restrict == TONE_MAJOR and intro_list[note_it] % 12 == 0) or (tone_restrict == TONE_MINOR and intro_list[note_it] % 12 == 9)):
                flag_have_long_note = True
                long_note = intro_list[note_it]
                note_dx = note_it
                break
            elif note_len >= 15 and ((tone_restrict == TONE_MAJOR and intro_list[note_it] % 12 in [0, 4, 7]) or (tone_restrict == TONE_MINOR and intro_list[note_it] % 12 in [0, 4, 9])):
                flag_have_long_note = True
                long_note = intro_list[note_it]
                note_dx = note_it
                break
            note_len = 0
    if flag_have_long_note is False:
        return False  # 没有持续时间较长的音符 则直接返回验证失败
    if (tone_restrict == TONE_MAJOR and long_note % 12 != 0) or (tone_restrict == TONE_MINOR and long_note % 12 != 9):  # 持续时间较长的音符不是dou/la，还需要看前面一个小节，是否是弦内的
        steps_in_chord = section_begin_check(intro_list[note_dx - 32: note_dx], tone_restrict)
        if steps_in_chord < 16:
            return False
        if note_dx + note_len + 1 < len(intro_list):  # 如果持续时间较长的音符后面还有音符，则后面的音符也必须满足至少一般为弦内音
            steps_in_chord = section_begin_check(intro_list[note_dx + note_len + 1:], tone_restrict, note_count=len(intro_list) - note_dx - note_len - 1)
            if steps_in_chord < 0.5 * (len(intro_list) - note_dx - note_len - 1):
                return False
    return True


def intro_connect_check(intro_list, melody_list):
    """
    获取前奏的最后阶段和主旋律的第一个音符的连接情况得分
    :param intro_list: 前奏
    :param melody_list: 主旋律的开始部分
    :return: 连接情况得分
    """
    last_note = -1  # 上一个音符的音高
    last_note_step = -1  # 上一个音符所在的位置
    note_count = 0  # 两小节一共多少音符
    shift_score = 0
    for step_it in range(len(intro_list) - 32, len(intro_list)):
        if intro_list[step_it] != 0:  # 计算变化分
            if last_note > 0:
                step_diff = step_it - last_note_step
                shift_score += (intro_list[step_it] - last_note) * (intro_list[step_it] - last_note) / (step_diff * step_diff)
            last_note = intro_list[step_it]
            last_note_step = step_it
            note_count += 1
    if last_note > 0:  # 添加前奏最后一个音符和主旋律第一个音符的音高差异分
        for step_it in range(len(melody_list)):  # 找出主旋律第一个音符所在的位置及其音高
            if melody_list[step_it] != 0:
                melody_first_note = melody_list[step_it]
                melody_first_note_step = step_it
                break
        if note_count > 0:
            step_diff = melody_first_note_step + len(intro_list) - last_note_step
            shift_score += (melody_first_note - last_note) * (melody_first_note - last_note) / (step_diff * step_diff)
            note_count += 1
    if note_count <= 1:  # 只有一个音符的情况下，音高差异分为0分
        return 0
    elif note_count > 1:
        return shift_score / (note_count - 1)


def bass_check(bass_list, chord_list):
    """
    检查两小节的bass是否存在和弦与同期和弦完全不同的情况 如果存在次数超过1次 则认为是bass不合格
    :param bass_list: bass列表
    :param chord_list: 同时期的和弦列表
    :return:
    """
    abnormal_bass_number = 0  # 不合要求的bass组合的数量
    # 逐两拍进行检验
    for beat_it in range(0, len(chord_list), 1):  # 时间步长为1拍
        if 0 < chord_list[beat_it] < len(CHORD_DICT):  # 防止出现和弦为空的情况
            bass_set = set()  # 这段时间内的bass音符列表（除以12的余数）
            # 1.生成这段时间bass的音符set 只记录在整拍或半拍上的，且持续时间要大于等于半拍的音符
            for step_it in range(beat_it * 8, (beat_it + 1) * 8):
                if bass_list[step_it] != 0:  # 这段时间的bass不为空
                    try:
                        if step_it % 4 == 0 and bass_list[step_it + 1] == 0 and bass_list[step_it + 2] == 0 and bass_list[step_it + 3] == 0:  # 必须得是在整拍或半拍上的音符 且持续时间要大于等于半拍才算数
                            # current_time_note_list = NoteDict[bass_list[current_18_beat]]  # 找到这段时间真实的bass音符列表
                            for note in bass_list[step_it]:
                                bass_set.add(note % 12)  # 把这些音符的音高都保存起来
                    except IndexError:
                        pass
            # 2.如果bass音符（主要的）与同时间区段内的和弦互不相同的话 则认为bass不合格
            if len(bass_set) != 0 and len(CHORD_DICT[chord_list[beat_it]] | bass_set) == len(CHORD_DICT[chord_list[beat_it]]) + len(bass_set):
                abnormal_bass_number += 1
    # print(abnormal_bass_number)
    if abnormal_bass_number > 2:
        return False
    return True


def pg_chord_check(pg_list, chord_list):
    """
    检查两小节的piano guitar是否存在和弦与同期和弦完全不同的情况 如果存在次数超过1次 则认为是piano_guitar不合格
    :param pg_list: piano_guitar列表
    :param chord_list: 同时期的和弦列表
    :return:
    """
    abnormal_pg_number = 0  # 不合要求的piano_guitar组合的数量
    # 逐两拍进行检验
    for beat_it in range(0, len(chord_list), 1):  # 时间步长为1拍
        if 0 < chord_list[beat_it] < len(CHORD_DICT):  # 防止出现和弦为空的情况
            pg_set = set()  # 这段时间内的piano_guitar音符列表（除以12的余数）
            # 1.生成这段时间bass的音符set 只记录在整拍或半拍上的，且持续时间要大于等于半拍的音符
            for step_it in range(beat_it * 4, (beat_it + 1) * 4):
                if pg_list[step_it] != 0:  # 这段时间的piano_guitar不为空
                    try:
                        if step_it % 2 == 0 and pg_list[step_it + 1] == 0:  # 必须得是在整拍或半拍上的音符 且持续时间要大于等于半拍才算数
                            # current_time_note_list = NoteDict[bass_list[current_18_beat]]  # 找到这段时间真实的bass音符列表
                            for note in pg_list[step_it]:
                                pg_set.add(note % 12)  # 把这些音符的音高都保存起来
                    except IndexError:
                        pass
            # 2.如果piano_guitar音符（主要的）与同时间区段内的和弦互不相同的话 则认为piano_guitar不合格
            if len(pg_set) != 0 and len(CHORD_DICT[chord_list[beat_it]] | pg_set) == len(CHORD_DICT[chord_list[beat_it]]) + len(pg_set):
                abnormal_pg_number += 1
    # print(abnormal_bass_number)
    if abnormal_pg_number > 2:
        return False
    return True


def string_chord_check(string_list, chord_list):
    """
    检查两小节的string是否存在和弦与同期和弦完全不同的情况 如果存在次数超过1次 则认为是string不合格
    :param string_list: string列表
    :param chord_list: 同时期的和弦列表
    :return:
    """
    abnormal_string_number = 0  # 不合要求的string组合的数量
    # 逐两拍进行检验
    for beat_it in range(0, len(chord_list), 2):  # 时间步长为2拍
        if 0 < chord_list[beat_it] < len(CHORD_DICT):  # 防止出现和弦为空的情况
            string_set = set()  # 这段时间内的string音符列表（除以12的余数）
            # 1.生成这段时间string的音符set 只记录在整拍或半拍上的，且持续时间要大于等于半拍的音符
            for step_it in range(beat_it * 4, (beat_it + 2) * 4):
                if string_list[step_it] != 0:  # 这段时间的string不为空
                    try:
                        if step_it % 2 == 0 and string_list[step_it + 1] == 0:  # 必须得是在整拍或半拍上的音符 且持续时间要大于等于半拍才算数
                            # current_time_note_list = NoteDict[bass_list[current_18_beat]]  # 找到这段时间真实的bass音符列表
                            for note in string_list[step_it]:
                                string_set.add(note % 12)  # 把这些音符的音高都保存起来
                    except IndexError:
                        pass
            # 2.如果string音符（主要的）与同时间区段内的和弦互不相同的话 则认为string不合格
            if len(string_set) != 0 and len(CHORD_DICT[chord_list[beat_it]] | string_set) == len(CHORD_DICT[chord_list[beat_it]]) + len(string_set):
                abnormal_string_number += 1
    # print(abnormal_bass_number)
    if abnormal_string_number > 1:
        return False
    return True


def bass_end_check(bass_output, tone_restrict=TONE_MAJOR):
    """
    检查bass的收束是否都是弦外音
    :param tone_restrict: 调式限制
    :param bass_output: bass的输出
    :return: 是否符合要求
    """
    for note_it in range(len(bass_output) - 1, -1, -1):
        if bass_output[note_it] != 0:
            note_set = set([t % 12 for t in bass_output[note_it]])
            if tone_restrict == TONE_MAJOR:
                if len(note_set | {0, 4, 7}) == len(note_set) + len({0, 4, 7}):  # 全部都是弦外音
                    return False
                return True
            elif tone_restrict == TONE_MINOR:
                if len(note_set | {0, 4, 9}) == len(note_set) + len({0, 4, 9}):  # 全部都是弦外音
                    return False
                return True
            else:
                raise ValueError
    return True


def piano_guitar_end_check(pg_output, tone_restrict=TONE_MAJOR):
    """
    检查piano_guitar的收束是否都是弦外音
    :param tone_restrict: 调式限制
    :param pg_output: bass的输出
    :return: 是否符合要求
    """
    for note_it in range(len(pg_output) - 1, -1, -1):
        if pg_output[note_it] != 0:
            note_set = set([t % 12 for t in pg_output[note_it]])
            if tone_restrict == TONE_MAJOR:
                if not note_set.issubset({0, 4, 7}):  # 全部都是弦外音
                    return False
                return True
            elif tone_restrict == TONE_MINOR:
                if not note_set.issubset({0, 4, 9}):  # 全部都是弦外音
                    return False
                return True
            else:
                raise ValueError
    return True


def bass_confidence_check(bass_output, chord_data):
    """
    根据bass的损失估值来判断bass是否符合要求
    :param bass_output: bass的数据（十二拍 绝对音高）
    :param chord_data: 和弦的数据（八拍）
    :return:
    """
    # 1.计算平均音高之间的差异
    note_ary = []  # 这四拍中有几个音符 它们的音高分别是多少
    note_ary_old = []
    note_count = 0  # 八拍之前的四拍中有几个音符 它们的音高分别是多少
    note_count_old = 0
    for note_it in range(64, 96):
        if bass_output[note_it] != 0:
            note_ary.extend(bass_output[note_it])
            note_count += len(bass_output[note_it])
        if bass_output[note_it - 64] != 0:
            note_ary_old.extend(bass_output[note_it - 64])
            note_count_old += len(bass_output[note_it - 64])
    if note_count == 0 or note_count_old == 0:
        note_diff_score = 0
    else:
        avr_note = sum(note_ary) / note_count  # 四拍所有音符的平均音高
        avr_note_old = sum(note_ary_old) / note_count_old  # 八拍之前的四拍所有音符的平均音高
        note_diff_score = abs(avr_note - avr_note_old) / 7  # 音高的差异（如果是176543的话差异分刚好为1分）
    # 2.计算按键的差异
    note_same_count = 0  # 四拍中有几个相同按键位置
    note_diff_count = 0  # 四拍中有几个不同的按键位置
    for note_it in range(64, 96):
        if bool(bass_output[note_it]) ^ bool(bass_output[note_it - 64]):
            note_diff_count += 1
        elif bass_output[note_it] != 0 and bass_output[note_it - 64] != 0:
            note_same_count += 1
    if note_same_count == 0 and note_diff_count == 0:
        keypress_diff_score = 0  # 按键的差异分
    else:
        keypress_diff_score = note_diff_count / (note_same_count + note_diff_count)
    # 3.计算与同时期和弦的差异分
    chord_backup = 0  # 用于存储上两拍的和弦
    bass_backup = []  # 用于存储上两拍的最后一个bass音符
    chord_diff_score_by_pat = [0] * 4
    for step_it in range(2, 6):
        chord_step_dx = (step_it - 2) * 2  # 这个bass的位置对应和弦的第几个下标
        chord_diff_score_1step = 0  # 与同期和弦的差异分
        note_diff_count = 0  # 和弦内不包含的bass音符数
        if chord_data[chord_step_dx] == 0 and chord_data[chord_step_dx + 1] == 0:
            chord_diff_score_by_pat[step_it - 2] = -1  # 出现了未知和弦 差异分标记为-1
            continue
        abs_notelist = []
        for note_it in range(step_it * 16, step_it * 16 + 16):
            if bass_output[note_it] != 0:
                abs_notelist.extend(bass_output[note_it])
                bass_backup = bass_output[note_it]
        bass_chord = notelist2chord(set(abs_notelist))  # 这个ｂａｓｓ是否可以找出一个特定的和弦
        div12_note_list = []  # 把所有音符对12取余数的结果 用于分析和和弦之间的重复音关系
        for note in abs_notelist:
            div12_note_list.append(note % 12)
        if bass_chord != 0:
            if bass_chord != chord_data[chord_step_dx] and bass_chord != chord_data[chord_step_dx + 1]:  # 这个bass组合与当前和弦不能匹配 额外增加0.5个差异分
                chord_diff_score_1step += 0.25
        if chord_data[chord_step_dx] != 0:
            chord_set = copy.deepcopy(CHORD_DICT[chord_data[chord_step_dx]])  # 和弦的音符列表(已对１２取余数)
            chord_dx = chord_data[chord_step_dx]
        else:
            chord_set = copy.deepcopy(CHORD_DICT[chord_data[chord_step_dx + 1]])
            chord_dx = chord_data[chord_step_dx + 1]
        if len(abs_notelist) == 0:  # 这两拍的bass没有音符。如果前两拍也没有则给值-1。如果和弦没变则赋前值 如果和弦变了则以前两拍最后一个音作为判断依据
            if len(bass_backup) == 0:  # 前两拍也没有
                chord_diff_score_by_pat[step_it - 2] = -1
                continue
            elif chord_dx == chord_backup:
                chord_diff_score_by_pat[step_it - 2] = chord_diff_score_by_pat[step_it - 3]  # 和弦没变 赋前值
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
        chord_diff_score_1step += note_diff_count / len(abs_notelist)  # bass与同期和弦的差异分
        chord_diff_score_by_pat[step_it - 2] = chord_diff_score_1step
        if -1 in chord_diff_score_by_pat:  # 把-1替换成均值 如果全是-1则替换为全零
            score_sum = 0
            score_count = 0
            for score in chord_diff_score_by_pat:
                if score != -1:
                    score_sum += score
                    score_count += 1
            avr_score = 0 if score_count == 0 else score_sum / score_count
            for score_it in range(len(chord_diff_score_by_pat)):
                if chord_diff_score_by_pat[score_it] == -1:
                    chord_diff_score_by_pat[score_it] = avr_score
    total_diff_score = note_diff_score * note_diff_score + keypress_diff_score * keypress_diff_score + sum([t * t for t in chord_diff_score_by_pat])
    return total_diff_score


def pg_confidence_check(pg_output, chord_data):
    """
    根据piano_guitar的损失估值来判断piano_guitar是否符合要求
    :param pg_output: piano_guitar的数据（十二拍 绝对音高）
    :param chord_data: 和弦的数据（八拍）
    :return:
    """
    # 1.计算平均音高之间的差异
    note_ary = []  # 这四拍中有几个音符 它们的音高分别是多少
    note_ary_old = []
    note_count = 0  # 八拍之前的四拍中有几个音符 它们的音高分别是多少
    note_count_old = 0
    for note_it in range(32, 48):
        if pg_output[note_it] != 0:
            note_ary.extend(pg_output[note_it])
            note_count += len(pg_output[note_it])
        if pg_output[note_it - 32] != 0:
            note_ary_old.extend(pg_output[note_it - 32])
            note_count_old += len(pg_output[note_it - 32])
    if note_count == 0 or note_count_old == 0:
        note_diff_score = 0
    else:
        avr_note = sum(note_ary) / note_count  # 四拍所有音符的平均音高
        avr_note_old = sum(note_ary_old) / note_count_old  # 八拍之前的四拍所有音符的平均音高
        note_diff_score = abs(avr_note - avr_note_old) / 7  # 音高的差异（如果是176543的话差异分刚好为1分）
    # 2.计算按键的差异
    note_same_count = 0  # 四拍中有几个相同按键位置
    note_diff_count = 0  # 四拍中有几个不同的按键位置
    for note_it in range(32, 48):
        if bool(pg_output[note_it]) ^ bool(pg_output[note_it - 32]):
            note_diff_count += 1
        elif pg_output[note_it] != 0 and pg_output[note_it - 32] != 0:
            note_same_count += 1
    if note_same_count == 0 and note_diff_count == 0:
        keypress_diff_score = 0  # 按键的差异分
    else:
        keypress_diff_score = note_diff_count / (note_same_count + note_diff_count)
    # 3.计算与同时期和弦的差异分
    chord_backup = 0  # 用于存储上两拍的和弦
    pg_backup = []  # 用于存储上两拍的最后一个piano_guitar音符
    chord_diff_score_by_pat = [0] * 4
    for step_it in range(2, 6):
        chord_step_dx = (step_it - 2) * 2  # 这个piano_guitar的位置对应和弦的第几个下标
        chord_diff_score_1step = 0  # 与同期和弦的差异分
        note_diff_count = 0  # 和弦内不包含的piano_guitar音符数
        if chord_data[chord_step_dx] == 0 and chord_data[chord_step_dx + 1] == 0:
            chord_diff_score_by_pat[step_it - 2] = -1  # 出现了未知和弦 差异分标记为-1
            continue
        abs_notelist = []
        for note_it in range(step_it * 8, step_it * 8 + 8):
            if pg_output[note_it] != 0:
                abs_notelist.extend(pg_output[note_it])
                pg_backup = pg_output[note_it]
        pg_chord = notelist2chord(set(abs_notelist))  # 这个piano_guitar是否可以找出一个特定的和弦
        div12_note_list = []  # 把所有音符对12取余数的结果 用于分析和和弦之间的重复音关系
        for note in abs_notelist:
            div12_note_list.append(note % 12)
        if pg_chord != 0:
            if pg_chord != chord_data[chord_step_dx] and pg_chord != chord_data[chord_step_dx + 1]:  # 这个piano_guitar组合与当前和弦不能匹配 额外增加0.25个差异分
                chord_diff_score_1step += 0.25
        if chord_data[chord_step_dx] != 0:
            chord_set = copy.deepcopy(CHORD_DICT[chord_data[chord_step_dx]])  # 和弦的音符列表(已对12取余数)
            chord_dx = chord_data[chord_step_dx]
        else:
            chord_set = copy.deepcopy(CHORD_DICT[chord_data[chord_step_dx + 1]])
            chord_dx = chord_data[chord_step_dx + 1]
        if len(abs_notelist) == 0:  # 这两拍的piano_guitar没有音符。如果前两拍也没有则给值-1。如果和弦没变则赋前值 如果和弦变了则以前两拍最后一个音作为判断依据
            if len(pg_backup) == 0:  # 前两拍也没有
                chord_diff_score_by_pat[step_it - 2] = -1
                continue
            elif chord_dx == chord_backup:
                chord_diff_score_by_pat[step_it - 2] = chord_diff_score_by_pat[step_it - 3]  # 和弦没变 赋前值
                continue
            else:  # 用上两拍的最后一个piano_guitar组合来进行计算
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
        chord_diff_score_1step += note_diff_count / len(abs_notelist)  # piano_guitar与同期和弦的差异分
        chord_diff_score_by_pat[step_it - 2] = chord_diff_score_1step
        if -1 in chord_diff_score_by_pat:  # 把-1替换成均值 如果全是-1则替换为全零
            score_sum = 0
            score_count = 0
            for score in chord_diff_score_by_pat:
                if score != -1:
                    score_sum += score
                    score_count += 1
            avr_score = 0 if score_count == 0 else score_sum / score_count
            for score_it in range(len(chord_diff_score_by_pat)):
                if chord_diff_score_by_pat[score_it] == -1:
                    chord_diff_score_by_pat[score_it] = avr_score
    total_diff_score = note_diff_score * note_diff_score + keypress_diff_score * keypress_diff_score + sum([t * t for t in chord_diff_score_by_pat])
    return total_diff_score


class GetChordConfidenceLevel:

    def __init__(self, transfer_count, real_transfer_count):
        """
        求出几个连续根音组合对应和弦的0.9置信区间
        :param transfer_count: 主旋律与同期和弦的统计（转移）关系（在真实的计数上加了一个定值 避免出现log(0)的情况）
        :param real_transfer_count: 主旋律与同期和弦的真实统计（转移）关系
        """
        self.transfer_prob = np.zeros([COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2, len(CHORD_DICT) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦 概率取对数后的转移矩阵 这个转移矩阵的数字精度必须是float32的
        self.transfer_mat = np.zeros([COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2, len(CHORD_DICT) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦。这个变量是上个变量进行了反softmax变换得到的
        self.transfer_prob_real = np.zeros([COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2, len(CHORD_DICT) + 1], dtype=np.float32)  # 真实的主旋律/调式与同时期和弦 概率取对数后的转移矩阵 这个转移矩阵的数字精度必须是float32的
        self.real_transfer_count = real_transfer_count
        # 1.将频率转化为概率
        for core_note_pat_dx in range(0, COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2):
            self.transfer_prob[core_note_pat_dx, :] = transfer_count[core_note_pat_dx, :] / sum(transfer_count[core_note_pat_dx, :])
            if sum(real_transfer_count[core_note_pat_dx, :]) != 0:  # 如果计数全部为零的话，就把那一列直接记零
                self.transfer_prob_real[core_note_pat_dx, :] = real_transfer_count[core_note_pat_dx, :] / sum(real_transfer_count[core_note_pat_dx, :])
            self.transfer_mat[core_note_pat_dx, :] = np.log(self.transfer_prob[core_note_pat_dx, :])
        # 2.定义交叉熵的计算方法
        self.core_note_pat = tf.placeholder(tf.int32, [])
        self.cross_entropy_lost = self.loss_func(self.core_note_pat)  # 各个和弦对应的损失函数
        # 3.定义1个和弦的交叉熵的计算方法
        self.core_note_pat_1chord = tf.placeholder(tf.int32, [])
        self.chord_dx = tf.placeholder(tf.int32, [])
        self.cross_entropy_lost_1chord = self.loss_func_1chord(self.core_note_pat_1chord, self.chord_dx)

    def loss_func(self, core_note_pat):
        """定义一个主旋律下,各个和弦对应的损失函数的计算方法"""
        transfer_vec = tf.gather(self.transfer_mat, core_note_pat)  # 这个主旋律下出现各个和弦的概率
        transfer_vec = tf.tile(tf.expand_dims(transfer_vec, 0), [len(CHORD_DICT) + 1, 1])  # 把它复刻若干遍 便于进行交叉熵的计算
        chord_one_hot = tf.constant(np.eye(len(CHORD_DICT) + 1))  # 所有和弦的独热编码
        cross_entropy_lost = tf.nn.softmax_cross_entropy_with_logits(labels=chord_one_hot, logits=transfer_vec)  # 这个主旋律下每个和弦对应的交叉熵损失函数
        # softmax_cross_entropy_with_logits首先将logits进行softmax变换，然后用labels*ln(1/logits)来计算
        # 如当labels为[1,0,0,0] logits为ln([0.5,0.25,0.125,0.125])时，交叉熵为1*ln(2)=0.693
        return cross_entropy_lost

    def loss_func_1chord(self, core_note_pat, chord_dx):
        """定义一个主旋律下，一个和弦对应的损失函数的计算方法"""
        transfer_vec = tf.gather(self.transfer_mat, core_note_pat)  # 这个主旋律下出现各个和弦的概率
        chord_one_hot = tf.one_hot(chord_dx, depth=len(CHORD_DICT) + 1)  # 这个和弦的独热编码
        cross_entropy_lost = tf.nn.softmax_cross_entropy_with_logits(labels=chord_one_hot, logits=transfer_vec)  # 这个主旋律下每个和弦对应的交叉熵损失函数
        return cross_entropy_lost

    def get_loss09(self, session, core_note_pat_ary):
        """计算一个主旋律进行下的损失函数0.9置信区间"""
        section_prob = {}  # 区间概率。存储方式为值:概率 如{1.5:0.03,2.3:0.05,...}
        for pat_step_dx in range(len(core_note_pat_ary)):
            # st = time.time()
            if core_note_pat_ary[pat_step_dx] not in [0, COMMON_CORE_NOTE_PATTERN_NUMBER + 1]:  # 只在这个根音不为空 不为罕见根音组合时才计算
                # 3.1.计算当前步骤的每个和弦及其损失函数的关系
                lost_each_chord = session.run(self.cross_entropy_lost, feed_dict={self.core_note_pat: core_note_pat_ary[pat_step_dx]})  # 计算这个主旋律下 每个和弦对应的交叉熵损失函数
                for loss_it in range(len(lost_each_chord)):
                    lost_each_chord[loss_it] = round(lost_each_chord[loss_it], 2)
                step_trans_vector = self.transfer_prob_real[core_note_pat_ary[pat_step_dx], :]  # 这个主旋律对应的各个和弦的比利
                step_section_prob = dict()
                for loss_it in range(len(lost_each_chord)):  # 写成{交叉熵损失: 对应比利}的结构
                    if step_trans_vector[loss_it] != 0:
                        if lost_each_chord[loss_it] not in step_section_prob:
                            step_section_prob[lost_each_chord[loss_it]] = step_trans_vector[loss_it]
                        else:
                            step_section_prob[lost_each_chord[loss_it]] += step_trans_vector[loss_it]
                # 3.2.计算几步的损失函数总和
                if section_prob == {}:  # 这段校验的第一步
                    section_prob = step_section_prob
                else:  # 不是这段校验的第一步 公式是{过去的损失+当前步的损失: 过去的概率×当前步的概率}
                    section_prob_backup = {}
                    for loss_old in section_prob:
                        for loss_step in step_section_prob:
                            prob = round(loss_old + loss_step, 2)
                            if prob in section_prob_backup:
                                section_prob_backup[prob] += section_prob[loss_old] * step_section_prob[loss_step]
                            else:
                                section_prob_backup[prob] = section_prob[loss_old] * step_section_prob[loss_step]
                    section_prob = section_prob_backup
                # print(section_prob)
            # ed = time.time()
            # print('time used:', ed - st)
        # 4.获取交叉熵误差的0.9置信区间
        accumulate_prob = 0
        sorted_section_prob = sorted(section_prob.items(), key=lambda asd: asd[0], reverse=False)
        # print(section_prob)
        for prob_tuple in sorted_section_prob:
            accumulate_prob += prob_tuple[1]
            if accumulate_prob >= 0.9:
                loss09 = prob_tuple[0]
                self.loss09 = loss09
                return
        self.loss09 = np.inf

    @staticmethod
    def chord_check_1step(chord_dx, melody_list, last_time_step_level):
        """
        检查两拍的和弦是否存在和弦与同期旋律完全不同的情况 如果存在次数超过1次 则认为是和弦不合格
        :param last_time_step_level: 上一拍的和弦和谐程度判定等级
        :param melody_list: 同时期的主旋律列表（非pattern）
        :param chord_dx: 这两拍的和弦
        :return: 分为三个等级 2为和弦符合预期 1为和弦与同期主旋律有重叠音但不是特别和谐 0为和弦与同期主旋律完全不重叠
        """
        melody_set = set()  # 这段时间内的主旋律列表
        for note_it in range(len(melody_list)):
            if melody_list[note_it] != 0:
                try:
                    if note_it % 8 == 0 or (melody_list[note_it + 1] == 0 and melody_list[note_it + 2] == 0):  # 时间过短的音符不要
                        melody_set.add(melody_list[note_it] % 12)
                except IndexError:
                    pass
        if len(melody_set) != 0 and len(CHORD_DICT[chord_dx] | melody_set) == len(CHORD_DICT[chord_dx]) + len(melody_set):
            return 0
        elif len(melody_set) == 0:  # 如果这段时间内没有主旋律 则看上个时间区段内是否和弦不符合要求 如果上个时间区段内和弦不符合要求 则这个时间区段的和弦也同样不符合要求
            return last_time_step_level
        elif len(melody_set) == 1:  # 主旋律的长度为1 则只要有重叠音就返回2
            return 2
        else:
            # 主旋律与同期和弦有两个重叠音 或一个重叠音一个七音 则返回2 其他情况返回1
            # TODO chord_dx // 6 + 10是不是应该再对１２取余数
            if len(CHORD_DICT[chord_dx]) + len(melody_set) - len(CHORD_DICT[chord_dx] | melody_set) >= 2:
                return 2
            elif 1 <= chord_dx <= 72 and chord_dx % 6 == 1 and ((chord_dx // 6 + 10) % 12 in melody_set or (chord_dx // 6 + 11) % 12 in melody_set):  # 大三和弦 且其大七度或小七度在主旋律中
                return 2
            elif 1 <= chord_dx <= 72 and chord_dx % 6 == 2 and (chord_dx // 6 + 10) % 12 in melody_set:  # 大三和弦 且其小七度在主旋律中
                return 2
            else:
                return 1

    def check_chord_ary(self, session, melody_list, core_note_pat_ary, chord_ary):
        """计算这个和弦对应的交叉熵损失函数 如果不在0.9置信区间内 则认为是不正确的"""
        lost_sum = 0  # 四拍的损失函数的总和
        last_time_step_level = 2
        # 2.计算四个主旋律对应的和弦损失函数之和
        for pat_it in range(len(core_note_pat_ary)):
            flag_use_loss = True  # 计算交叉熵损失函数还是用乐理来替代
            # 2.1.这两拍的主旋律是否可以使用交叉熵损失的方法来计算 判定条件为：主旋律不为空/不为罕见根音组合，该根音组合在训练集中至少出现100次，至少有3种输出和弦出现了十次
            if core_note_pat_ary[pat_it] in [0, COMMON_CORE_NOTE_PATTERN_NUMBER + 1]:  # 只在这个根音不为空 不为罕见根音组合时才计算
                flag_use_loss = False
            if flag_use_loss is True:
                real_count_ary = self.real_transfer_count[core_note_pat_ary[pat_it], :]
                if real_count_ary.sum() < 100:  # 该根音组合在训练集中应该至少出现100次
                    flag_use_loss = False
                if len(real_count_ary[real_count_ary >= 10]) < 3:  # 该根音与至少3种输出和弦在训练集中出现了十次
                    flag_use_loss = False
            # 2.2.如果符合计算交叉熵的标准则用交叉熵方法计算 否则用乐理知识估算
            if flag_use_loss is True:
                lost_1step = session.run(self.cross_entropy_lost_1chord, feed_dict={self.core_note_pat_1chord: core_note_pat_ary[pat_it], self.chord_dx: chord_ary[pat_it]})  # 每一拍的交叉熵损失函数
                lost_sum += lost_1step
                last_time_step_level = self.chord_check_1step(chord_ary[pat_it], melody_list[pat_it * 16: (pat_it + 1) * 16], last_time_step_level)  # 更新这一拍的和弦是否符合乐理的情况
            else:
                last_time_step_level = self.chord_check_1step(chord_ary[pat_it], melody_list[pat_it * 16: (pat_it + 1) * 16], last_time_step_level)
                if last_time_step_level == 0:
                    lost_sum += self.loss09 * 0.7  # 如果不符合乐理 我们认为它等价与0.9置信区间的70%的交叉熵损失
                elif last_time_step_level == 1:
                    lost_sum += self.loss09 * 0.35  # 如果不完全和谐 我们认为它等价与0.9置信区间的35%的交叉熵损失
        # 3.检查这个损失函数值是否在0.9置信区间内 如果在则返回true 否则返回false
        if lost_sum <= self.loss09:
            return True, lost_sum
        else:
            return False, lost_sum
