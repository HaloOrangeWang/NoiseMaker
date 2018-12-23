from settings import *


class StringConfidenceCheckConfig:
    note_time = 0.25
    pitch_diff_ratio = 1 / 7
    keypress_diff_ratio = 1
    chord_diff_score = 0.25
    chord_note_diff_score = 1


def string_chord_check(string_list, chord_list):
    """
    检查两小节的string是否存在和弦与同期和弦完全不同的情况 如果存在次数超过1次 则认为是string不合格
    :param string_list: string列表
    :param chord_list: 同时期的和弦列表
    :return:
    """
    abnm_string_number = 0  # 不合要求的string组合的数量
    flag_last_step_abnm = False  # 上一个时间区段是否是异常 如果是异常 则这个值为True 否则为False
    # 逐两拍进行检验
    for beat_it in range(0, len(chord_list), 2):  # 时间步长为2拍
        if 0 < chord_list[beat_it] < len(CHORD_LIST):  # 防止出现和弦为空的情况
            string_set = set()  # 这段时间内的string音符列表（除以12的余数）
            # 1.生成这段时间string的音符set 只记录在整拍或半拍上的，且持续时间要大于等于半拍的音符
            for step_it in range(beat_it * 4, (beat_it + 2) * 4):
                if string_list[step_it] != 0:  # 这段时间的string不为空
                    try:
                        if step_it % 2 == 0 and string_list[step_it + 1] == 0:  # 必须得是在整拍或半拍上的音符 且持续时间要大于等于半拍才算数
                            for note in string_list[step_it]:
                                string_set.add(note % 12)  # 把这些音符的音高都保存起来
                    except IndexError:
                        pass
            # 2.如果string音符（主要的）与同时间区段内的和弦互不相同的话 则认为string不合格
            if len(string_set) != 0 and len(CHORD_LIST[chord_list[beat_it]] | string_set) == len(CHORD_LIST[chord_list[beat_it]]) + len(string_set):
                abnm_string_number += 1
                flag_last_step_abnm = True
            elif len(string_set) == 0 and flag_last_step_abnm is True:  # 如果这段时间内没有主旋律 则看上个时间区段内是否和弦不符合要求 如果上个时间区段内和弦不符合要求 则这个时间区段的和弦也同样不符合要求
                abnm_string_number += 1
            else:
                flag_last_step_abnm = False
    if abnm_string_number > 1:
        return False
    return True


def string_end_check(string_output, tone_restrict=DEF_TONE_MAJOR):
    """
    检查string_guitar的收束是否包含弦外音
    :param tone_restrict: 调式限制
    :param string_output: bass的输出
    :return: 是否符合要求
    """
    for note_it in range(len(string_output) - 1, -1, -1):
        if string_output[note_it] != 0:
            note_set = set([t % 12 for t in string_output[note_it]])
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
