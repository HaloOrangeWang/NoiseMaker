from settings import *
from interfaces.sql.sqlite import NoteDict
from interfaces.utils import get_first_index_bigger, get_last_index_smaller
import copy
import numpy as np


def get_rel_notelist_melody(note_list, tone, root):
    """
    以某个音为基准　获取相对音高的音符列表(主旋律和fill通常使用这个)
    :param note_list: 原始音符列表
    :param tone: 节奏（0为大调 1为小调）
    :param root: 根音（通常情况下 大调为60 小调为57）
    :return: 相对音高的音符列表
    """
    # 1.获取相对音高对照表
    if tone == DEF_TONE_MAJOR:
        rel_note_ary = REL_NOTE_COMPARE_DIC['Major']
    elif tone == DEF_TONE_MINOR:
        rel_note_ary = REL_NOTE_COMPARE_DIC['Minor']
    else:
        raise ValueError
    # 2.转化为相对音高
    rel_note_list = []
    for note in sorted(note_list, reverse=True):  # 先从大到小排序
        rel_note_list.append([7 * ((note - root) // 12) + rel_note_ary[(note - root) % 12][0], rel_note_ary[(note - root) % 12][1]])
    return rel_note_list


def one_song_rel_notelist_melody(note_list, tone, root, use_note_dict=False):
    """
    以某个音为基准 获取一首歌的相对音高的音符列表
    :param use_note_dict: 原note_list中的音是note_dict中的音还是绝对音高
    :param note_list: 这首歌的原始音高列表
    :param tone: 节奏（0为大调　1为小调）
    :param root: 根音（大调为72　小调为69）
    :return: 这首歌的相对音高的音符列表
    """
    rel_note_list = []  # 转化为相对音高形式的音符列表
    for note in note_list:
        if note == 0:
            rel_note_list.append(0)
        else:
            if use_note_dict is True:
                rel_note_group = get_rel_notelist_melody(NoteDict.nd[note], tone, root)
            else:
                rel_note_group = get_rel_notelist_melody([note], tone, root)
            rel_note_list.append(rel_note_group)
    return rel_note_list


def get_rel_notelist_chord(note_list, root, chord):
    """
    以同时期和弦的根音为基准 转换成相对音高列表。音高用音符的音高和根音的差值代替
    如和弦为Am，根音为57，则[57, 60, 64]会变成[[0, 0], [2, 0], [4, 0]]
    :param note_list: 音符列表
    :param root: 和弦的根音
    :param chord: 同时期的和弦
    :return: 相对音高列表
    """
    root_list = [[0, 0], [1, -1], [1, 0], [2, -1], [2, 0], [3, 0], [3, 1], [4, 0], [5, -1], [5, 0], [6, -1], [6, 0]]
    standard_namelist = [0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6]  # 音名列表
    aug_namelist = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]
    dim_namelist = [0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 6]
    standard_rel_list = [0, 2, 4, 5, 7, 9, 11]
    rel_note_list = []
    # 1.确定根音的音名
    root_name = root_list[root % 12]
    # 2.判断该和弦是否为增减和弦
    if 1 <= chord <= 72 and (chord - 1) % 6 == 2:
        namelist = aug_namelist
    elif 1 <= chord <= 72 and (chord - 1) % 6 == 3:
        namelist = dim_namelist
    else:
        namelist = standard_namelist
    # 3.根据和弦和当前音的音高确定音名列表
    for note in sorted(note_list):
        namediff = 7 * ((note - root) // 12) + namelist[(note - root) % 12]
        notename = [namediff, (note % 12) - standard_rel_list[(namediff + root_name[0]) % 7]]
        notename[1] -= 12 * round(notename[1] / 12)
        rel_note_list.append(notename)
    return rel_note_list


def one_song_rel_notelist_chord(raw_note_ary, root_data, chord_data, note_time_step=0.25):
    """
    以和弦根音为基准 获取一首歌的相对音高的音符列表
    :param note_time_step: 音符的时间步长
    :param raw_note_ary: 原始的音符列表
    :param root_data: 和弦的根音
    :param chord_data: 和弦数据
    :return:
    """
    # :param unit: 以小节为单元还是以拍为单元
    time_step_ratio = int(1 / note_time_step)  # 和弦的时间步长与音符的时间步长的比值
    # 1.准备工作: 将原始的数据展成以拍为单位的音符列表
    rel_note_list = []  # 转化为相对音高形式的音符列表
    # if unit == 'bar':
    #     note_list = flat_array(raw_note_ary)  # 把以小节为单位的列表降一维 变成以音符步长为单位的列表
    note_list = copy.deepcopy(raw_note_ary)
    # 2.遍历所有音符 获相对音高列表
    for note_it, note in enumerate(note_list):
        if note == 0:
            rel_note_list.append(0)
        elif note_it // time_step_ratio >= len(root_data):  # 这一拍没有和弦 则相对音高为0
            rel_note_list.append(0)
        elif root_data[note_it // time_step_ratio] == 0:  # 这一拍没有根音 则相对音高为0
            rel_note_list.append(0)
        else:
            rel_note_group = get_rel_notelist_chord(NoteDict.nd[note], root_data[note_it // time_step_ratio], chord_data[note_it // time_step_ratio])
            rel_note_list.append(rel_note_group)
    return rel_note_list


def get_abs_notelist_melody(cur_step, rel_note, melody_core_note_list, tone, root_note):
    """
    根据与主旋律骨干音相对音高求出绝对音高
    :param cur_step: 第几个时间步长
    :param rel_note: 相对音高列表
    :param melody_core_note_list: 主旋律骨干音列表
    :param tone: 调式
    :param root_note: 根音
    :return: 绝对音高列表
    """
    if tone == DEF_TONE_MAJOR:
        rel_list = [0, 2, 4, 5, 7, 9, 11]
    elif tone == DEF_TONE_MINOR:
        rel_list = [0, 2, 3, 5, 7, 8, 10]
    else:
        raise ValueError
    rel_root_notelist = [[t[0] + melody_core_note_list[cur_step][0][0], t[1] + melody_core_note_list[cur_step][0][1]] for t in rel_note]  # 相对根音的音高
    abs_note_list = [12 * (t[0] // 7) + rel_list[t[0] % 7] + t[1] + root_note for t in rel_root_notelist]  # 得到绝对音高
    return abs_note_list


def get_abs_notelist_chord(rel_note_list, root_note):
    """
    根据与根音的相对音高求出绝对音高
    :param rel_note_list: 相对音高列表
    :param root_note: 根音的绝对音高
    :return: 绝对音高列表
    """
    rootdict = [[0, 0], [1, -1], [1, 0], [2, -1], [2, 0], [3, 0], [3, 1], [4, 0], [5, -1], [5, 0], [6, -1], [6, 0]]
    rel_list = [0, 2, 4, 5, 7, 9, 11]
    output_notelist = []
    # 1.找到root_note的音名和音域
    rootname = rootdict[root_note % 12][0]
    rootbase = root_note - root_note % 12
    # 2.求出rel_note_list所有音符的音名
    for rel_note in rel_note_list:
        note = rel_note[0] + rootname  # 获取音名
        note = rel_list[note % 7] + 12 * (note // 7)  # 获取相对rootbase的音高
        note = note + rootbase + rel_note[1]  # 计算绝对音高
        output_notelist.append(note)
    return output_notelist


def judge_imitation(input_melody_list, input_comp_list, speed_ratio_dict):
    """
    判断伴奏是否对主旋律存在某种模仿关系 以1拍为时间步长进行存储
    存储方式为[时间差别 音高差别 速度差别]。其中时间差别不能超过8拍（64），速度差别必须是1（一半） 2（相同） 3（两倍）之间选择
    为０则不模仿　为１则与前一拍的情况相同
    只有一个音符的情况下不能视作模仿
    :param speed_ratio_dict: 主旋律速度和伴奏速度比例的对照表
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
    for step_it in range(0, len(input_comp_list), 8):
        beat_comp = np.array(input_comp_list[step_it: step_it + 8], dtype=object)
        if imitation_comp_list[step_it // 8] != 0:
            continue
        if len(beat_comp[beat_comp != 0]) == 0:
            continue
        imit_range_first_dx, __ = get_first_index_bigger(melody_list[:, 2], step_it - 56)  # 在主旋律中模仿的区间 前７拍之内模仿
        imit_range_last_dx, __ = get_last_index_smaller(melody_list[:, 2], step_it + 1)
        part_comp_list = comp_list[comp_list[:, 2] >= step_it]  # 这一拍之后的伴奏列表
        longest = 1  # 最长的模仿序列长度
        longest_message = {'note': 0, 'time': 0, 'speed': 0}  # 最长的模仿序列在主旋律列表中的位置
        for startnote_it in range(imit_range_first_dx, imit_range_last_dx + 1):
            try:
                note_diff = part_comp_list[0, 0] - melody_list[startnote_it, 0]  # 音高差别
                step_diff = part_comp_list[0, 2] - melody_list[startnote_it, 2]  # 时间差别
                speed_ratio = round((part_comp_list[1, 2] - part_comp_list[0, 2]) / (melody_list[startnote_it + 1, 2] - melody_list[startnote_it, 2]), 1)  # 速度的比
                imitation_length = 1  # 模仿序列的长度
                melody_note_it = startnote_it + 1
                while melody_note_it <= len(melody_list) - 1:  # 判断模仿序列的长度
                    if (part_comp_list[imitation_length, 0] - melody_list[melody_note_it, 0] != note_diff) or (part_comp_list[imitation_length, 2] - melody_list[melody_note_it, 2] != step_diff):
                        break
                    if melody_list[melody_note_it, 2] - melody_list[melody_note_it - 1, 2] > 8:  # 被模仿的主旋律音符相邻两个音符间隔不能超过1拍
                        break
                    if round((part_comp_list[imitation_length, 2] - part_comp_list[imitation_length - 1, 2]) / (melody_list[melody_note_it, 2] - melody_list[melody_note_it - 1, 2]), 1) != speed_ratio:
                        break
                    imitation_length += 1
                    melody_note_it += 1
                if imitation_length >= 2 and imitation_length >= longest:  # 模仿序列的长度在２以上且大于等于当前的最长序列 则认为是
                    speed_ratio = speed_ratio_dict.get(speed_ratio, 0)
                    if speed_ratio == 0:
                        continue
                    if step_diff >= 16 + 2 * (part_comp_list[imitation_length - 1, 2] - part_comp_list[0, 2]):  # 相隔距离不能超过模仿长度的2倍+2拍
                        continue
                    step_lack = 8 - part_comp_list[imitation_length - 1, 2] + step_it
                    if step_lack >= 2 and input_comp_list[part_comp_list[imitation_length - 1, 2]: step_it + 8] != [0 for t in range(step_lack)]:  # 一整拍以内的所有音符必须都要符合此模仿关系
                        continue
                    longest = imitation_length
                    longest_message = {'note': note_diff, 'time': step_diff, 'speed': speed_ratio}
            except IndexError:
                pass
        if longest >= 2:
            imitation_comp_list[step_it // 8] = [longest_message['note'], longest_message['time'], longest_message['speed']]
            imitation_end_beat = part_comp_list[longest - 1, 2] // 8  # 模仿序列结束于哪拍
            imitation_comp_list[step_it // 8 + 1: imitation_end_beat + 1] = 1
    return imitation_comp_list
