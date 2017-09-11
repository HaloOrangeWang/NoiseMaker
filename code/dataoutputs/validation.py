from settings import *
from datainputs.melody import GetMelodyProfileBySong


def MelodyCheck(melody_list):
    """
    检查生成的一小节音乐是否有异常内容 从按键的位置和音高来判别是否符合要求
    :param melody_list: 旋律列表
    :return: 返回是否符合要求
    """
    assert MELODY_TIME_STEP == 1 / 8  # 这个验证方法只针对时间步长为八分之一拍的情况
    # 1.检查是否存在错位情况
    for note_iterator in range(0, len(melody_list), 4):
        if melody_list[note_iterator] == 0 and melody_list[note_iterator + 1] != 0:
            return False
    for note_iterator in range(0, len(melody_list) - 4, 4):
        if melody_list[note_iterator] == 0 and melody_list[note_iterator + 3] != 0 and melody_list[note_iterator + 4] == 0:
            return False
    for note_iterator in range(0, len(melody_list), 8):
        if melody_list[note_iterator] == 0 and melody_list[note_iterator + 2] != 0:
            return False
    for note_iterator in range(0, len(melody_list) - 8, 8):
        if melody_list[note_iterator] == 0 and melody_list[note_iterator + 6] != 0 and melody_list[note_iterator + 8] == 0:
            return False
        if melody_list[note_iterator] == 0 and melody_list[note_iterator + 4] != 0 and melody_list[note_iterator + 8] == 0 and melody_list[note_iterator + 12] == 0:
            return False
    # 2.检查相邻两个音符的的音高之差是否太大
    # 相邻两个32分音符之间不超过小三度 相邻两个16分音符之间不超过纯四度 相邻两个八分音符间不超过纯五度 相邻两个四分音符间不超过大六度 相邻两个二分音符间不超过八度 其余无限制
    if FLAG_IS_DEBUG:
        max_scale_diff_list = [7, 7, 7, 9, 9, 9, 9, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    else:
        max_scale_diff_list = [3, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 12]
    for note_iterator in range(len(melody_list)):
        if melody_list[note_iterator] != 0:  # 只有当不是休止符时才检查音高
            last_note_index = -1  # 上一个音符的位置
            last_note_scale = 0  # 上一个音符的音高
            for t in range(note_iterator-1, 0, -1):  # 找到上一个音符的位置和音高
                if melody_list[t] != 0:
                    last_note_index = t
                    last_note_scale = melody_list[t]
                    break
            if last_note_index != -1:
                note_time_diff = note_iterator - last_note_index  # 音符时间间隔
                scale_diff = abs(melody_list[note_iterator] - last_note_scale)  # 音符音高间隔
                if note_time_diff <= 16 and scale_diff > max_scale_diff_list[note_time_diff - 1]:  # 相邻两个音符音高差别过大
                    # print(note_iterator, note_dict[melody_list[note_iterator]][-1], note_time_diff, scale_diff)
                    return False
    return True


def MelodyClusterCheck(kmeans_session, kmeans_model, cluster_center_points, melody_list, right_cluster, max_allow_diff=0, train_pattern=False):
    # 1.对melody_list的数据进行一些处理
    if train_pattern:
        melody_cluster_input = melody_list
    else:
        melody_cluster_input = [melody_list[t] + MELODY_LOW_NOTE - 1 if melody_list[t] else 0 for t in range(len(melody_list))]
    melody_cluster_check_input = {int(t * MELODY_TIME_STEP / 4): melody_cluster_input[-int(8 / MELODY_TIME_STEP):][t:t + int(4 / MELODY_TIME_STEP)] for t in range(0, len(melody_cluster_input[-int(8 / MELODY_TIME_STEP):]), int(4 / MELODY_TIME_STEP))}
    # 2.检查cluster是否正确
    melody_cluster = GetMelodyProfileBySong(kmeans_session, kmeans_model, cluster_center_points, melody_cluster_check_input)[0]
    if abs(melody_cluster - right_cluster) <= max_allow_diff:
        return True
    return False


def ChordCheck(chord_list, melody_list):
    """
    检查两小节的和弦是否存在和弦与同期旋律完全不同的情况 如果存在次数超过1次 则认为是和弦不合格
    :param chord_list:
    :return:
    """
    # 还要考虑这段时间没有主旋律/那个音符持续时间过短等情况
    abnormal_chord_number = 0
    for chord_time_iterator in range(0, len(chord_list), CHORD_GENERATE_TIME_STEP):  # 时间步长为2拍
        melody_set = set()  # 这段时间内的主旋律列表
        assert MELODY_TIME_STEP == 1 / 8
        for note_iterator in range(round(chord_time_iterator / MELODY_TIME_STEP), round((chord_time_iterator + CHORD_GENERATE_TIME_STEP) / MELODY_TIME_STEP)):
            if melody_list[note_iterator] != 0:
                try:
                    if note_iterator % 8 == 0 or (melody_list[note_iterator + 1] == 0 and melody_list[note_iterator + 2] == 0):  # 时间过短的音符不要
                        melody_set.add(melody_list[note_iterator] % 12)
                except IndexError:
                    pass
        if len(melody_set) != 0 and len(CHORD_DICT[chord_list[chord_time_iterator]] | melody_set) == len(CHORD_DICT[chord_list[chord_time_iterator]]) + len(melody_set):
            abnormal_chord_number += 1
    if abnormal_chord_number > 1:
        return False
    return True
