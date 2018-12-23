from settings import *


class BassConfidenceCheckConfig:
    note_time = 0.125
    pitch_diff_ratio = 1 / 7
    keypress_diff_ratio = 1
    chord_diff_score = 0.25
    chord_note_diff_score = 1


def bass_check(bass_list, chord_list):
    """
    检查两小节的bass是否存在和弦与同期和弦完全不同的情况 如果存在次数超过1次 则认为是bass不合格
    :param bass_list: bass列表
    :param chord_list: 同时期的和弦列表
    :return:
    """
    abnm_bass_number = 0  # 不合要求的bass组合的数量
    # 逐两拍进行检验
    for beat_it in range(0, len(chord_list)):  # 时间步长为1拍
        if 0 < chord_list[beat_it] < len(CHORD_LIST):  # 防止出现和弦为空的情况
            bass_set = set()  # 这段时间内的bass音符列表（除以12的余数）
            # 1.生成这段时间bass的音符set 只记录在整拍或半拍上的，且持续时间要大于等于半拍的音符
            for step_it in range(beat_it * 8, (beat_it + 1) * 8):
                if bass_list[step_it] != 0:  # 这段时间的bass不为空
                    try:
                        if step_it % 4 == 0 and bass_list[step_it + 1] == 0 and bass_list[step_it + 2] == 0 and bass_list[step_it + 3] == 0:  # 必须得是在整拍或半拍上的音符 且持续时间要大于等于半拍才算数
                            for note in bass_list[step_it]:
                                bass_set.add(note % 12)  # 把这些音符的音高都保存起来
                    except IndexError:
                        pass
            # 2.如果bass音符（主要的）与同时间区段内的和弦互不相同的话 则认为bass不合格
            if len(bass_set) != 0 and len(CHORD_LIST[chord_list[beat_it]] | bass_set) == len(CHORD_LIST[chord_list[beat_it]]) + len(bass_set):
                abnm_bass_number += 1
    if abnm_bass_number > 2:
        return False
    return True


def bass_end_check(bass_output, tone_restrict=DEF_TONE_MAJOR):
    """
    检查bass的收束是否都是弦外音
    :param tone_restrict: 调式限制
    :param bass_output: bass的输出
    :return: 是否符合要求
    """
    for note_it in range(len(bass_output) - 1, -1, -1):
        if bass_output[note_it] != 0:
            note_set = set([t % 12 for t in bass_output[note_it]])
            if tone_restrict == DEF_TONE_MAJOR:
                if len(note_set | {0, 4, 7}) == len(note_set) + len({0, 4, 7}):  # 全部都是弦外音
                    return False
                return True
            elif tone_restrict == DEF_TONE_MINOR:
                if len(note_set | {0, 4, 9}) == len(note_set) + len({0, 4, 9}):  # 全部都是弦外音
                    return False
                return True
            else:
                raise ValueError
    return True
