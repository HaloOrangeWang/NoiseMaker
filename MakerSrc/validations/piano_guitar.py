from settings import *


class PgConfidenceCheckConfig:
    note_time = 0.25
    pitch_diff_ratio = 1 / 7
    keypress_diff_ratio = 1
    chord_diff_score = 0.25
    chord_note_diff_score = 1


def pg_chord_check(pg_list, chord_list):
    """
    检查两小节的piano guitar是否存在和弦与同期和弦完全不同的情况 如果存在次数超过1次 则认为是piano_guitar不合格
    :param pg_list: piano_guitar列表
    :param chord_list: 同时期的和弦列表
    :return:
    """
    abnm_pg_number = 0  # 不合要求的piano_guitar组合的数量
    # 逐两拍进行检验
    for beat_it in range(0, len(chord_list)):  # 时间步长为1拍
        if 0 < chord_list[beat_it] < len(CHORD_LIST):  # 防止出现和弦为空的情况
            pg_set = set()  # 这段时间内的piano_guitar音符列表（除以12的余数）
            # 1.生成这段时间bass的音符set 只记录在整拍或半拍上的，且持续时间要大于等于半拍的音符
            for step_it in range(beat_it * 4, (beat_it + 1) * 4):
                if pg_list[step_it] != 0:  # 这段时间的piano_guitar不为空
                    try:
                        if step_it % 2 == 0 and pg_list[step_it + 1] == 0:  # 必须得是在整拍或半拍上的音符 且持续时间要大于等于半拍才算数
                            for note in pg_list[step_it]:
                                pg_set.add(note % 12)  # 把这些音符的音高都保存起来
                    except IndexError:
                        pass
            # 2.如果piano_guitar音符（主要的）与同时间区段内的和弦互不相同的话 则认为piano_guitar不合格
            if len(pg_set) != 0 and len(CHORD_LIST[chord_list[beat_it]] | pg_set) == len(CHORD_LIST[chord_list[beat_it]]) + len(pg_set):
                abnm_pg_number += 1
    if abnm_pg_number > 2:
        return False
    return True


def pg_end_check(pg_output, tone_restrict=DEF_TONE_MAJOR):
    """
    检查piano_guitar的收束是否包含弦外音
    :param tone_restrict: 调式限制
    :param pg_output: bass的输出
    :return: 是否符合要求
    """
    for note_it in range(len(pg_output) - 1, -1, -1):
        if pg_output[note_it] != 0:
            note_set = set([t % 12 for t in pg_output[note_it]])
            if tone_restrict == DEF_TONE_MAJOR:
                if not note_set.issubset({0, 4, 7}):  # 包含弦外音
                    return False
                return True
            elif tone_restrict == DEF_TONE_MINOR:
                if not note_set.issubset({0, 4, 9}):  # 包含弦外音
                    return False
                return True
            else:
                raise ValueError
    return True
