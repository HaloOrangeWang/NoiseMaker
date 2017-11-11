from interfaces.sql.sqlite import GetRawSongDataFromDataset, NoteDict
from interfaces.chord_parse import NoteList2Chord, ChordTo3
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
    last_time_step_abnormal = False  # 上一个时间区段是否是异常 如果是异常 则这个值为True 否则为False
    for chord_time_iterator in range(0, len(chord_list), 2):  # 时间步长为2拍
        melody_set = set()  # 这段时间内的主旋律列表
        # assert MELODY_TIME_STEP == 1 / 8
        for note_iterator in range(round(chord_time_iterator * 8), round((chord_time_iterator + 2) * 8)):
            if melody_list[note_iterator] != 0:
                try:
                    if note_iterator % 8 == 0 or (melody_list[note_iterator + 1] == 0 and melody_list[note_iterator + 2] == 0):  # 时间过短的音符不要
                        melody_set.add(melody_list[note_iterator] % 12)
                except IndexError:
                    pass
        if len(melody_set) != 0 and len(CHORD_DICT[chord_list[chord_time_iterator]] | melody_set) == len(CHORD_DICT[chord_list[chord_time_iterator]]) + len(melody_set):
            abnormal_chord_number += 1
            last_time_step_abnormal = True
        elif len(melody_set) == 0 and last_time_step_abnormal is True:  # 如果这段时间内没有主旋律 则看上个时间区段内是否和弦不符合要求 如果上个时间区段内和弦不符合要求 则这个时间区段的和弦也同样不符合要求
            abnormal_chord_number += 1
        else:
            last_time_step_abnormal = False
    if abnormal_chord_number > 1:
        return False
    return True


def MelodyEndCheck(melody_list, tone_restrict=TONE_MAJOR):
    """
    检查一首歌是否以dol/la为结尾 且最后一个音持续时间必须为1/2/4拍
    :param melody_list: 旋律列表
    :param tone_restrict: 这首歌是大调还是小调
    :return: 歌曲的结尾是否符合要求
    """
    check_result = False
    if tone_restrict == TONE_MAJOR:
        end_note = 0  # 最后一个音符 大调为0（dol） 小调为9（la）
    elif tone_restrict == TONE_MINOR:
        end_note = 9
    for note_iterator in range(-32, 0):
        if note_iterator in [-32, -16, -8] and melody_list[note_iterator] != 0 and melody_list[note_iterator] % 12 == end_note:
            check_result = True
        elif melody_list[note_iterator] != 0:
            check_result = False
    return check_result


def MelodySimilarityCheck(melody_list):
    """
    检查一段音乐是否与训练集中的某一首歌雷同 如果所生成的有连续3小节的相同 则认为是存在雷同
    :param melody_list:
    :return: 如有雷同返回False 否则返回True
    """
    assert len(melody_list) == 96  # 待监测的音符列表长度必须为3小节
    # 1.读取数据集中的主旋律数据
    melody_dataset_data = GetRawSongDataFromDataset('main', None)  # 没有旋律限制的主旋律数据
    # 2.检查是否有连续三小节的完全相同
    if melody_list[-96: -64] != [0 for t in range(32)] and melody_list[-64: -32] != [0 for t in range(32)] and melody_list[-32:] != [0 for t in range(32)]:  # 这三个小节都不是空小节的情况下才进行检查 如果中间有空小节 则返回True
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            if melody_dataset_data[song_iterator] != {}:
                for key in melody_dataset_data[song_iterator]:
                    try:
                        if melody_list[-96: -64] == melody_dataset_data[song_iterator][key] and melody_list[-64: -32] == melody_dataset_data[song_iterator][key + 1] and melody_list[-32:] == melody_dataset_data[song_iterator][key + 2]:
                            return False
                    except KeyError:
                        pass
    return True


def BassCheck(bass_list, chord_list):
    """
    检查两小节的bass是否存在和弦与同期和弦完全不同的情况 如果存在次数超过1次 则认为是bass不合格
    :param bass_list: bass列表
    :param chord_list: 同时期的和弦列表
    :return:
    """
    abnormal_bass_number = 0  # 不合要求的bass组合的数量
    # 逐两拍进行检验
    for chord_time_iterator in range(0, len(chord_list), 1):  # 时间步长为1拍
        bass_set = set()  # 这段时间内的bass音符列表（除以12的余数）
        # 1.生成这段时间bass的音符set
        for current_18_beat in range(round(chord_time_iterator * 8), round((chord_time_iterator + 1) * 8)):
            if bass_list[current_18_beat] != 0:  # 这段时间的bass不为空
                try:
                    if current_18_beat % 4 == 0 and bass_list[current_18_beat + 1] == 0 and bass_list[current_18_beat + 2] == 0 and bass_list[current_18_beat + 3] == 0:  # 必须得是在整拍或半拍上的音符 且持续时间要大于等于半拍才算数
                        current_time_note_list = NoteDict[bass_list[current_18_beat]]  # 找到这段时间真实的bass音符列表
                        for note in current_time_note_list:
                            bass_set.add(note % 12)  # 把这些音符的音高都保存起来
                except IndexError:
                    pass
        # 2.如果bass音符（主要的）与同时间区段内的和弦互不相同的话 则认为bass不合格
        if len(bass_set) != 0 and len(CHORD_DICT[chord_list[chord_time_iterator]] | bass_set) == len(CHORD_DICT[chord_list[chord_time_iterator]]) + len(bass_set):
            abnormal_bass_number += 1
    # print(abnormal_bass_number)
    if abnormal_bass_number > 2:
        return False
    return True


def BassEndCheck(bass_output):
    """
    检查bass的收束是否都是弦外音
    :param bass_output: bass的输出
    :return: 是否符合要求
    """
    for note_it in range(len(bass_output) - 1, -1, -1):
        if bass_output[note_it] != 0:
            note_set = set([t % 12 for t in NoteDict[bass_output[note_it]]])
            if len(note_set | {0, 4, 7}) == len(note_set) + len({0, 4, 7}):  # 全部都是弦外音
                return False
            return True
    return True


def PianoGuitarCheck(pg_list, chord_list, tone):
    """
    检查两小节的piano_guitar是否符合要求。
    逐2拍检查piano_guitar要表达的和弦是不是chord_list对应位置的和弦。如果和弦差异达到2次，则认为piano_guitar不合格
    a.如果出现了七和弦 将其转换为对应的三和弦进行检验
    b.如果在一个位置上piano_guitar没有内容，那么根据其上一拍的情况来判断这一拍是否符合要求
    :param pg_list:
    :param chord_list:
    :return:
    """
    last_step_valid = True
    last_step_chord = 0
    invalid_pg_number = 0
    for notecode_iterator in range(0, len(pg_list), 8):
        # 1.逐两拍获取音符列表
        note_set = set()  # 两拍的音符列表
        for note in pg_list[notecode_iterator: notecode_iterator + 8]:
            if note != 0:
                note_set = note_set | set(NoteDict[note])
        # 2.检查这两拍的piano_guitar是否符合要求
        if len(note_set) == 0:
            if not last_step_valid:
                invalid_pg_number += 1
        else:
            current_step_chord = NoteList2Chord(note_set, saved_chord=last_step_chord, tone=tone)
            if current_step_chord == 0 and len(note_set) == 1:  # 这段时间只有1个音符的特殊情况
                if note_set.pop() % 12 in CHORD_DICT[chord_list[notecode_iterator // 4]]:
                    current_step_chord = chord_list[notecode_iterator // 4]
            last_step_chord = current_step_chord
            if current_step_chord != 0:
                current_step_chord = ChordTo3(current_step_chord)
            required_chord = ChordTo3(chord_list[notecode_iterator // 4])
            if current_step_chord != required_chord:
                invalid_pg_number += 1
                last_step_valid = False
            else:
                last_step_valid = True
            # print(notecode_iterator, note_set, current_step_chord, required_chord)
    # if invalid_pg_number >= 2:
    #     return False
    # return True
    return invalid_pg_number
