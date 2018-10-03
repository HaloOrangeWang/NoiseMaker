from settings import *


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
    if tone == TONE_MAJOR:
        rel_list = [0, 2, 4, 5, 7, 9, 11]
    elif tone == TONE_MINOR:
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
