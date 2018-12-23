from settings import *
from validations.functions import BaseConfidenceLevelCheck
import copy
import math


def keypress_check(melody_note_list):
    """
    检查生成的一小节音乐是否有异常内容 从按键的位置和音高来判别是否符合要求
    :param melody_note_list: 旋律列表
    :return: 返回是否符合要求
    """
    for note_it in range(0, len(melody_note_list), 4):
        if melody_note_list[note_it] == 0 and melody_note_list[note_it + 1] != 0:
            return False
    for note_it in range(0, len(melody_note_list) - 4, 4):
        if melody_note_list[note_it] == 0 and melody_note_list[note_it + 3] != 0 and melody_note_list[note_it + 4] == 0:
            return False
    for note_it in range(0, len(melody_note_list), 8):
        if melody_note_list[note_it] == 0 and melody_note_list[note_it + 2] != 0:
            return False
    for note_it in range(0, len(melody_note_list) - 8, 8):
        if melody_note_list[note_it] == 0 and melody_note_list[note_it + 6] != 0 and melody_note_list[note_it + 8] == 0:
            return False
        if melody_note_list[note_it] == 0 and melody_note_list[note_it + 4] != 0 and melody_note_list[note_it + 8] == 0 and melody_note_list[note_it + 12] == 0:
            return False
    return True


def section_begin_check(melody_note_list, tone_restrict=DEF_TONE_MAJOR, note_count=32):
    """
    一个小节的音符有多少个是合适的，将这些合适的音符的时长求和
    :param melody_note_list: 主旋律音符列表
    :param tone_restrict: 调式信息
    :param note_count: 一共有多少个音符
    :return:
    """
    steps_in_chord = 0
    flag_in_chord = True
    if tone_restrict == DEF_TONE_MAJOR:
        chord_note_ary = [0, 4, 7]  # 弦内音符 大调为0,4,7（dol,mi,sol） 小调为4,7,9（mi,sol,la）
    elif tone_restrict == DEF_TONE_MINOR:
        chord_note_ary = [4, 7, 9]
    else:
        raise ValueError
    for note_it in range(note_count):
        if melody_note_list[note_it] == 0:
            steps_in_chord += int(flag_in_chord)
        elif melody_note_list[note_it] % 12 in chord_note_ary:
            steps_in_chord += 1
            flag_in_chord = True
        else:
            flag_in_chord = False
    return steps_in_chord


def section_end_check(melody_list, tone_restrict=DEF_TONE_MAJOR, last_note_min_len=20, last_note_min_len_special=12):
    """
    检查个乐段是否以dol/mi/sol/la为结尾 且最后一个音持续时间必须至少要有一定长度
    :param last_note_min_len: 这个乐段的最后一个音符至少要有多长
    :param last_note_min_len_special: 在特殊情况下（最后一个音符刚好为dol或la，时长的要求可以降低）
    :param melody_list: 旋律列表
    :param tone_restrict: 这首歌是大调还是小调
    :return: 乐段的结尾是否符合要求
    """
    check_result = False
    if tone_restrict == DEF_TONE_MAJOR:
        best_end_note = 0
        end_note_ary = [0, 2, 4, 7]  # 最后一个音符 大调为0,4,7（dol,mi,sol） 小调为4,7,9（mi,sol,la）
    elif tone_restrict == DEF_TONE_MINOR:
        best_end_note = 9
        end_note_ary = [4, 7, 9]
    else:
        raise ValueError
    for note_it in range(-32, 0):
        if note_it <= -last_note_min_len_special and note_it % 4 == 0 and melody_list[note_it] != 0 and melody_list[note_it] % 12 == best_end_note:
            check_result = True
        elif note_it <= -last_note_min_len and note_it % 4 == 0 and melody_list[note_it] != 0 and melody_list[note_it] % 12 in end_note_ary:
            check_result = True
        elif melody_list[note_it] != 0:
            check_result = False
    return check_result


def melody_end_check(melody_list, tone_restrict=DEF_TONE_MAJOR):
    """
    检查一首歌是否以dol/la为结尾 且最后一个音持续时间必须为1-4之间的0.5整数倍拍
    :param melody_list: 旋律列表
    :param tone_restrict: 这首歌是大调还是小调
    :return: 歌曲的结尾是否符合要求
    """
    last_note_loc_ary = [-32, -28, -24, -20, -16, -12, -8]
    check_result = False
    if tone_restrict == DEF_TONE_MAJOR:
        end_note = 0  # 最后一个音符 大调为0（dol） 小调为9（la）
    elif tone_restrict == DEF_TONE_MINOR:
        end_note = 9
    else:
        raise ValueError
    for note_it in range(-32, 0):
        if note_it in last_note_loc_ary and melody_list[note_it] != 0 and melody_list[note_it] % 12 == end_note:
            check_result = True
        elif melody_list[note_it] != 0:
            check_result = False
    return check_result


class ShiftConfidenceCheck(BaseConfidenceLevelCheck):

    def train_1song(self, **kwargs):
        """
        获取主旋律连续两小节（不跨越乐段）的音高变化得分 分数为（音高变化/时间差）的平方和
        kwargs['section_data']: 这首歌的乐段数据
        kwargs['raw_melody_data']: 一首歌的主旋律数据
        """
        raw_melody_data = kwargs['raw_melody_data']
        sec_data = kwargs['section_data']
        section_data = copy.deepcopy(sec_data)

        score_ary = []
        if section_data:  # 训练集中的部分歌没有乐段
            section_data.sort()  # 按照小节先后顺序排序
        for bar_it in range(0, len(raw_melody_data) // 32 - 1):
            shift_score = 0
            sec_dx = -1  # 记录这一拍属于这首歌的第几个乐段
            if section_data:  # 有乐段的情况下 当这个小节和下一小节均不为空且不跨越乐段时进行收录
                for sec_it in range(len(section_data)):
                    if section_data[sec_it][0] * 4 + section_data[sec_it][1] > bar_it * 4:  # 将区段的起始拍和当前拍进行比较 如果起始拍在当前拍之前则说明是属于这个区段
                        sec_dx = sec_it - 1
                        break
                if sec_dx == -1:
                    sec_dx = len(section_data) - 1  # 属于这首歌的最后一个区段
                if section_data[sec_dx][2] == DEF_SEC_EMPTY:  # 这个乐段是间奏 不进行分数选择
                    continue
                if sec_dx != len(section_data) - 1:
                    if section_data[sec_dx + 1][0] * 4 + section_data[sec_dx + 1][1] < (bar_it + 2) * 4:
                        continue  # 出现跨越乐段的情况，不收录
            else:
                if raw_melody_data[bar_it * 32: (bar_it + 1) * 32] == [0] * 32 or raw_melody_data[(bar_it + 1) * 32: (bar_it + 2) * 32] == [0] * 32:
                    continue  # 没有乐段的情况下 这一小节和下一小节均不能为空
            last_note = -1  # 上一个音符的音高
            last_note_step = -1  # 上一个音符所在的位置
            note_count = 0  # 两小节一共多少音符
            # for cal_bar_it in range(bar_it, bar_it + 2):
            for step_it in range(64):
                if raw_melody_data[bar_it * 32 + step_it] != 0:  # 计算变化分
                    if last_note > 0:
                        step_diff = step_it - last_note_step
                        shift_score += (raw_melody_data[bar_it * 32 + step_it] - last_note) * (raw_melody_data[bar_it * 32 + step_it] - last_note) / (step_diff * step_diff)
                    last_note = raw_melody_data[bar_it * 32 + step_it]
                    last_note_step = step_it
                    note_count += 1
            if note_count == 1:  # 只有一个音符的情况下，音高差异分为0分
                score_ary.append(0)
            elif note_count > 1:
                score_ary.append(shift_score / (note_count - 1))
        self.evaluating_score_list.extend(score_ary)

    def evaluate(self, **kwargs):
        melody_note_list = kwargs['melody_note_list']

        last_note_pitch = -1  # 上一个音符的音高
        last_note_step = -1  # 上一个音符所在的位置
        note_count = 0  # 两小节一共多少音符
        shift_score = 0
        for step_it in range(0, 64):
            if melody_note_list[step_it] != 0:  # 计算变化分
                if last_note_pitch > 0:
                    step_diff = step_it - last_note_step
                    shift_score += (melody_note_list[step_it] - last_note_pitch) * (melody_note_list[step_it] - last_note_pitch) / (step_diff * step_diff)
                last_note_pitch = melody_note_list[step_it]
                last_note_step = step_it
                note_count += 1
        if note_count == 1:  # 只有一个音符的情况下，音高差异分为0分
            return 0
        elif note_count > 1:
            return shift_score / (note_count - 1)
        else:
            raise ValueError


class DiffNoteConfidenceCheck(BaseConfidenceLevelCheck):
    """检查一个乐段的前半部分和后半部分之间的音符差异。一个段落内，前后部分的差异不能过大"""

    def train_1song(self, **kwargs):
        """
        获取主旋律连续两小节（不跨越乐段）的音高变化得分 分数为（音高变化/时间差）的平方和
        kwargs['section_data']: 这首歌的乐段数据
        kwargs['raw_melody_data']: 一首歌的主旋律数据
        """
        raw_melody_data = kwargs['raw_melody_data']
        sec_data = kwargs['section_data']
        section_data = copy.deepcopy(sec_data)

        score_ary = []
        if not section_data:  # 如果这首歌没有乐段信息，则直接返回空值
            return
        section_data.sort()  # 按照小节先后顺序排序
        for sec_it in range(len(section_data)):
            if section_data[sec_it][2] == DEF_SEC_EMPTY:
                continue  # 这个乐段是间奏 不进行评价
            start_step_dx = min(int(section_data[sec_it][0] * 32), len(raw_melody_data))
            if sec_it == len(section_data) - 1:
                end_step_dx = len(raw_melody_data)
            else:
                end_step_dx = min(int(section_data[sec_it + 1][0] * 32), len(raw_melody_data))
            if end_step_dx - start_step_dx < 4 * 32:
                continue  # 至少要达到4小节才有继续的意义
            middle_step_dx = int((start_step_dx + end_step_dx) / 2)
            note_count_1half = [0 for t in range(12)]  # 将各个音符与它们持续时间的对数乘积
            note_count_2half = [0 for t in range(12)]
            keypress_count_1half = [0 for t in range(32)]
            keypress_count_2half = [0 for t in range(32)]
            last_note_step = -1  # 上一个音符所在的位置
            last_note = -1
            for step_it in range(start_step_dx, end_step_dx + 1):
                if step_it == end_step_dx:  # 到达了最后一拍
                    keypress_count_2half[last_note_step % 32] += math.log2(end_step_dx - last_note_step) + 1
                    note_count_2half[last_note % 12] += math.log2(end_step_dx - last_note_step) + 1
                    break
                if raw_melody_data[step_it] == 0:  # 这一拍没音符 直接跳过
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
                last_note = raw_melody_data[last_note_step]
            note_diff = sum([abs(note_count_2half[t] - note_count_1half[t]) for t in range(12)]) / (sum(note_count_1half) + sum(note_count_2half))
            keypress_diff = sum([abs(keypress_count_2half[t] - keypress_count_1half[t]) for t in range(32)]) / (sum(keypress_count_1half) + sum(keypress_count_2half))
            score_ary.append(note_diff + keypress_diff * 2)
        self.evaluating_score_list.extend(score_ary)

    def evaluate(self, **kwargs):
        melody_note_list = kwargs['melody_note_list']

        middle_step_dx = int(len(melody_note_list) / 2)
        note_count_1half = [0 for t in range(12)]  # 将各个音符与它们持续时间的对数乘积
        note_count_2half = [0 for t in range(12)]
        keypress_count_1half = [0 for t in range(32)]
        keypress_count_2half = [0 for t in range(32)]
        last_note_step = -1  # 上一个音符所在的位置
        last_note = -1
        for step_it in range(0, len(melody_note_list)):
            if melody_note_list[step_it] == 0:  # 这一拍没音符 直接跳过
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
            last_note = melody_note_list[step_it]

        keypress_count_2half[last_note_step % 32] += math.log2(len(melody_note_list) - last_note_step) + 1
        note_count_2half[last_note % 12] += math.log2(len(melody_note_list) - last_note_step) + 1
        note_diff = sum([abs(note_count_2half[t] - note_count_1half[t]) for t in range(12)]) / (sum(note_count_1half) + sum(note_count_2half))
        keypress_diff = sum([abs(keypress_count_2half[t] - keypress_count_1half[t]) for t in range(32)]) / (sum(keypress_count_1half) + sum(keypress_count_2half))
        return note_diff + keypress_diff * 2
