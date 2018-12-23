from settings import *
from validations.functions import BaseConfidenceLevelCheck
from validations.melody import section_begin_check


class IntroShiftConfidenceCheck(BaseConfidenceLevelCheck):

    def train_1song(self, **kwargs):
        """
        获取一段前奏/间奏的最后一个小节，与主旋律的连接处音高变化得分
        kwargs['raw_intro_data']: 一首歌的前奏/间奏数据
        kwargs['raw_melody_data']: 一首歌的主旋律数据
        kwargs['continuous_bar_data']: 一首歌的连续的小节数据
        """
        raw_melody_data = kwargs['raw_melody_data']
        raw_intro_data = kwargs['raw_intro_data']
        continuous_bar_data = kwargs['continuous_bar_data']

        score_ary = []
        for bar_it in range(len(continuous_bar_data)):
            if bar_it != 0 and continuous_bar_data[bar_it] != 0 and continuous_bar_data[bar_it - 1] == 0:  # 主旋律开始的小节,说明前奏/间奏在这一小节结束了
                melody_start_step_dx = -1  # 主旋律准确开始的步长
                for step_it in range(32):
                    if raw_melody_data[bar_it * 32 + step_it] != 0 and (step_it == 0 or raw_melody_data[bar_it * 32 + step_it - 1] == 0):
                        melody_start_step_dx = step_it + bar_it * 32
                        break

                last_note_pitch = -1  # 上一个音符的音高
                last_note_step = -1  # 上一个音符所在的位置
                note_count = 0  # 一共多少音符
                shift_score = 0
                for step_it in range(melody_start_step_dx - 32, melody_start_step_dx):
                    try:
                        if raw_intro_data[step_it] != 0:  # 计算变化分
                            if last_note_pitch > 0:
                                step_diff = step_it - last_note_step
                                shift_score += (raw_intro_data[step_it] - last_note_pitch) * (raw_intro_data[step_it] - last_note_pitch) / (step_diff * step_diff)
                            last_note_pitch = raw_intro_data[step_it]
                            last_note_step = step_it
                            note_count += 1
                    except IndexError:
                        pass
                if last_note_pitch > 0:  # 添加前奏最后一个音符和主旋律第一个音符的音高差异分
                    step_diff = melody_start_step_dx - last_note_step
                    shift_score += (raw_melody_data[melody_start_step_dx] - last_note_pitch) * (raw_melody_data[melody_start_step_dx] - last_note_pitch) / (step_diff * step_diff)
                    note_count += 1
                if note_count == 1:  # 只有一个音符的情况下，音高差异分为0分（没有音符的情况就不存储了）
                    score_ary.append(0)
                elif note_count > 1:  # 多余一个音符的情况下，将所有的音高变化/时间差取平方和作为音高变化得分
                    score_ary.append(shift_score / (note_count - 1))
        self.evaluating_score_list.extend(score_ary)

    def evaluate(self, **kwargs):
        intro_list = kwargs['intro_list']
        melody_list = kwargs['melody_list']

        last_note_pitch = -1  # 上一个音符的音高
        last_note_step = -1  # 上一个音符所在的位置
        note_count = 0  # 两小节一共多少音符
        shift_score = 0
        for step_it in range(len(intro_list) - 32, len(intro_list)):
            if intro_list[step_it] != 0:  # 计算变化分
                if last_note_pitch > 0:
                    step_diff = step_it - last_note_step
                    shift_score += (intro_list[step_it] - last_note_pitch) * (intro_list[step_it] - last_note_pitch) / (step_diff * step_diff)
                last_note_pitch = intro_list[step_it]
                last_note_step = step_it
                note_count += 1
        if last_note_pitch > 0:  # 添加前奏最后一个音符和主旋律第一个音符的音高差异分
            for step_it in range(len(melody_list)):  # 找出主旋律第一个音符所在的位置及其音高
                if melody_list[step_it] != 0:
                    melody_start_pitch = melody_list[step_it]
                    melody_start_step = step_it
                    break
            if note_count > 0:
                step_diff = melody_start_step + len(intro_list) - last_note_step
                shift_score += (melody_start_pitch - last_note_pitch) * (melody_start_pitch - last_note_pitch) / (step_diff * step_diff)
                note_count += 1
        if note_count <= 1:  # 只有一个音符的情况下，音高差异分为0分
            return 0
        elif note_count > 1:
            return shift_score / (note_count - 1)


def intro_end_check(intro_list, tone_restrict=DEF_TONE_MAJOR):
    """
    前奏结束阶段的检查。检查通过的条件包括
    1.前奏的最后一拍只能有整拍和半拍的位置才能有音符
    2.最后一小节有持续时间较长的弦内音（持续两拍以上，若最后一个音为c或a，则只持续一拍以上就可以）
    3.长音符的前一个小节有至少半数时长为弦内音
    4.如果长音符的后面还有音符，则后面的音符也必须满足至少一半为弦内音
    :param intro_list: 前奏的列表
    :param tone_restrict: 调式限制
    :return:
    """
    flag_have_long_note = False  # 前奏的末尾阶段是否有持续时间较长的音符
    long_note_pitch = -1  # 持续时间较长的音符的音高
    note_dx = -1
    note_len = 0  # 音符的长度
    for note_it in range(len(intro_list) - 1, len(intro_list) - 33, -1):
        if intro_list[note_it] == 0:
            note_len += 1
        elif note_it - len(intro_list) in [-1, -2, -3, -5, -6, -7] and intro_list[note_it] != 0:  # 前奏的最后一拍只能有整拍和半拍的位置才能有音符
            return False
        else:
            if note_len >= 7 and note_len == len(intro_list) - note_it - 1 and ((tone_restrict == DEF_TONE_MAJOR and intro_list[note_it] % 12 == 0) or (tone_restrict == DEF_TONE_MINOR and intro_list[note_it] % 12 == 9)):
                flag_have_long_note = True
                long_note_pitch = intro_list[note_it]
                note_dx = note_it
                break
            elif note_len >= 15 and ((tone_restrict == DEF_TONE_MAJOR and intro_list[note_it] % 12 in [0, 4, 7]) or (tone_restrict == DEF_TONE_MINOR and intro_list[note_it] % 12 in [0, 4, 9])):
                flag_have_long_note = True
                long_note_pitch = intro_list[note_it]
                note_dx = note_it
                break
            note_len = 0
    if flag_have_long_note is False:
        return False  # 没有持续时间较长的音符 则直接返回验证失败
    if (tone_restrict == DEF_TONE_MAJOR and long_note_pitch % 12 != 0) or (tone_restrict == DEF_TONE_MINOR and long_note_pitch % 12 != 9):  # 持续时间较长的音符不是dou/la，还需要看前面一个小节，是否是弦内的
        steps_in_chord = section_begin_check(intro_list[note_dx - 32: note_dx], tone_restrict)
        if steps_in_chord < 16:
            return False
        if note_dx + note_len + 1 < len(intro_list):  # 如果持续时间较长的音符后面还有音符，则后面的音符也必须满足至少一般为弦内音
            steps_in_chord = section_begin_check(intro_list[note_dx + note_len + 1:], tone_restrict, note_count=len(intro_list) - note_dx - note_len - 1)
            if steps_in_chord < 0.5 * (len(intro_list) - note_dx - note_len - 1):
                return False
    return True
