from settings import *
from interfaces.sql.sqlite import NoteDict
import copy
import random


def music_promote(stream_input):
    stream_output = dict()
    # 1.获取前奏
    stream_output['intro'] = get_intro(stream_input['melody'])
    intro_bars = len(stream_output['intro']) // 32  # 前奏占了几个小节
    split = [4 * intro_bars + len(stream_input['melody']) // 8, 4 * intro_bars + len(stream_input['melody']) // 4 + 8]  # 间奏和尾声开始的位置 单位是拍
    if stream_input['melody'][-8:] != [0 for t in range(8)]:
        split = [t + 2 for t in split]
    # 2.旋律和和弦各重复两遍空出两小节前奏 空出两小节间奏 空出一小节尾声
    stream_output['melody'] = [0 for t in range(intro_bars * 32)] + stream_input['melody'] + [0 for t in range(64)] + stream_input['melody'] + [0 for t in range(32)]
    stream_output['bass'] = [0 for t in range(intro_bars * 32)] + stream_input['bass'] + [0 for t in range(64)] + stream_input['bass'] + [0 for t in range(32)]
    stream_output['chord'] = [0 for t in range(intro_bars * 4)] + stream_input['chord'] + [0 for t in range(8)] + stream_input['chord'] + [0 for t in range(4)]
    stream_output['fill'] = [0 for t in range(intro_bars * 32)] + stream_input['fill'] + [0 for t in range(64)] + stream_input['fill'] + [0 for t in range(32)]
    stream_output['piano_guitar'] = [0 for t in range(intro_bars * 16)] + stream_input['piano_guitar'] + [0 for t in range(32)] + stream_input['piano_guitar'] + [0 for t in range(16)]
    stream_output['string'] = [0 for t in range(intro_bars * 16)] + stream_input['string'] + [0 for t in range(32)] + stream_input['string'] + [0 for t in range(16)]
    # 2.加上鼓 从0.91.03开始鼓点也加入到训练内容中
    stream_output['drum'] = stream_input['drum'][0: 32] * intro_bars + stream_input['drum'] + stream_input['drum'][0: 32] * 2 + stream_input['drum'] + stream_input['drum'][-32:]
    return stream_output, split


def music_promote_2(stream_input):
    """
    对音符的输出进行调整。首先将前奏在主旋律的前面复制一份，然后把主旋律生成的
    :param stream_input:
    :return:
    """
    stream_output = dict()
    # 1.获取前奏
    melody_bar_num = len(stream_input['melody']) // 32  # 主旋律占了多少小节
    intro_bar_num = len(stream_input['intro']) // 32  # 前奏占了几个小节
    # 2.各条音轨的内容均为前奏+旋律+前奏+旋律
    stream_output['melody'] = [0 for t in range(intro_bar_num * 32)] + stream_input['melody'] + [0 for t in range(intro_bar_num * 32)] + stream_input['melody']
    stream_output['intro'] = stream_input['intro'] + [0 for t in range(melody_bar_num * 32)] + stream_input['intro'] + [0 for t in range(melody_bar_num * 32)]
    stream_output['bass'] = stream_input['bass'][melody_bar_num * 32:] + stream_input['bass'][:32 * melody_bar_num] + stream_input['bass'][melody_bar_num * 32:] + stream_input['bass'][:32 * melody_bar_num]
    stream_output['chord'] = stream_input['chord'][melody_bar_num * 4:] + stream_input['chord'][:4 * melody_bar_num] + stream_input['chord'][melody_bar_num * 4:] + stream_input['chord'][:4 * melody_bar_num]
    stream_output['fill'] = [0 for t in range(intro_bar_num * 32)] + stream_input['fill'] + [0 for t in range(intro_bar_num * 32)] + stream_input['fill']
    stream_output['piano_guitar'] = stream_input['piano_guitar'][melody_bar_num * 16:] + stream_input['piano_guitar'][:16 * melody_bar_num] + stream_input['piano_guitar'][melody_bar_num * 16:] + stream_input['piano_guitar'][:16 * melody_bar_num]
    stream_output['string'] = stream_input['string'][melody_bar_num * 16:] + stream_input['string'][:16 * melody_bar_num] + stream_input['string'][melody_bar_num * 16:] + stream_input['string'][:16 * melody_bar_num]
    stream_output['drum'] = stream_input['drum'][melody_bar_num * 32:] + stream_input['drum'][:32 * melody_bar_num] + stream_input['drum'][melody_bar_num * 32:] + stream_input['drum'][:32 * melody_bar_num]
    return stream_output


def music_promote_3(stream_input, generate_type=-1):
    """
    按照生成的方式来生成一段音乐
    :param stream_input: 输入的各条音轨的信息
    :param generate_type: 生成的方式。0,1为正常方式 2,3为去掉部分鼓点和bass 4,5为去掉部分bass 6,7为去掉部分鼓点和piano guitar。奇数表示在结尾部分增加一个crash
    :return:
    """
    if generate_type < 0:
        generate_type = random.randint(0, 7)
    stream_output = dict()
    # 1.获取前奏
    melody_bar_num = len(stream_input['melody']) // 32  # 主旋律占了多少小节
    intro_bar_num = len(stream_input['intro']) // 32  # 前奏占了几个小节
    # 2.各条音轨的内容均为前奏+旋律+前奏+旋律
    stream_output['melody'] = [0 for t in range(intro_bar_num * 32)] + stream_input['melody'] + [0 for t in range(intro_bar_num * 32)] + stream_input['melody']
    stream_output['intro'] = stream_input['intro'] + [0 for t in range(melody_bar_num * 32)] + stream_input['intro'] + [0 for t in range(melody_bar_num * 32)]
    if len(stream_input['section']) >= 4 and generate_type // 2 == 1:  # 至少有三个乐段且生成的类型为1 则前奏没有bass
        stream_output['bass'] = [0] * (32 * intro_bar_num) + stream_input['bass'][:32 * melody_bar_num] + stream_input['bass'][melody_bar_num * 32:] + stream_input['bass'][:32 * melody_bar_num]
    elif len(stream_input['section']) >= 4 and generate_type // 2 == 2:  # 至少有三个乐段且生成的类型为1 则前奏和第一个乐段没有bass
        stream_output['bass'] = [0] * (32 * (intro_bar_num + stream_input['section'][1][0])) + stream_input['bass'][32 * stream_input['section'][1][0]: 32 * melody_bar_num] + stream_input['bass'][melody_bar_num * 32:] + stream_input['bass'][:32 * melody_bar_num]
    else:
        stream_output['bass'] = stream_input['bass'][melody_bar_num * 32:] + stream_input['bass'][:32 * melody_bar_num] + stream_input['bass'][melody_bar_num * 32:] + stream_input['bass'][:32 * melody_bar_num]
    stream_output['chord'] = stream_input['chord'][melody_bar_num * 4:] + stream_input['chord'][:4 * melody_bar_num] + stream_input['chord'][melody_bar_num * 4:] + stream_input['chord'][:4 * melody_bar_num]
    stream_output['fill'] = [0 for t in range(intro_bar_num * 32)] + stream_input['fill'] + [0 for t in range(intro_bar_num * 32)] + stream_input['fill']
    if len(stream_input['section']) >= 4 and generate_type // 2 == 3:  # 至少有三个乐段且生成的类型为3 则前两个乐段没有piano guitar
        stream_output['piano_guitar'] = stream_input['piano_guitar'][melody_bar_num * 16:] + [0] * (16 * stream_input['section'][2][0]) + stream_input['piano_guitar'][16 * stream_input['section'][2][0]: 16 * melody_bar_num] + stream_input['piano_guitar'][melody_bar_num * 16:] + stream_input['piano_guitar'][:16 * melody_bar_num]
    else:
        stream_output['piano_guitar'] = stream_input['piano_guitar'][melody_bar_num * 16:] + stream_input['piano_guitar'][:16 * melody_bar_num] + stream_input['piano_guitar'][melody_bar_num * 16:] + stream_input['piano_guitar'][:16 * melody_bar_num]
    stream_output['string'] = stream_input['string'][melody_bar_num * 16:] + stream_input['string'][:16 * melody_bar_num] + stream_input['string'][melody_bar_num * 16:] + stream_input['string'][:16 * melody_bar_num]
    if len(stream_input['section']) >= 4 and generate_type // 2 == 1:  # 至少有三个乐段且生成的类型为1 则前奏和第一个乐段没有鼓点
        stream_output['drum'] = [0] * (32 * (intro_bar_num + stream_input['section'][1][0])) + stream_input['drum'][32 * stream_input['section'][1][0]: 32 * melody_bar_num] + stream_input['drum'][melody_bar_num * 32:] + stream_input['drum'][:32 * melody_bar_num]
    elif len(stream_input['section']) >= 4 and generate_type // 2 == 3:  # 至少有三个乐段且生成的类型为3 则第一个乐段没有鼓点
        stream_output['drum'] = stream_input['drum'][melody_bar_num * 32:] + [0] * (32 * stream_input['section'][1][0]) + stream_input['drum'][32 * stream_input['section'][1][0]: 32 * melody_bar_num] + stream_input['drum'][melody_bar_num * 32:] + stream_input['drum'][:32 * melody_bar_num]
    else:
        stream_output['drum'] = stream_input['drum'][melody_bar_num * 32:] + stream_input['drum'][:32 * melody_bar_num] + stream_input['drum'][melody_bar_num * 32:] + stream_input['drum'][:32 * melody_bar_num]
    # 3.可能会在末尾处增加一个crash
    if generate_type % 2 == 1:
        for key in NoteDict:
            if NoteDict[key] == [49]:
                crash1_key = key
            elif NoteDict[key] == [57]:
                crash2_key = key
        stream_output['drum'].extend([[crash1_key, crash2_key][random.randint(0, 1)]] + [0] * 7)
        stream_output['melody'].extend([0] * 8)
        stream_output['intro'].extend([0] * 8)
        stream_output['bass'].extend([0] * 8)
        stream_output['chord'].extend([0])
        stream_output['fill'].extend([0] * 8)
        stream_output['piano_guitar'].extend([0] * 4)
        stream_output['string'].extend([0] * 4)
    return stream_output


def get_intro(melody_output):
    # 1.获取主旋律后四个小节的平均节拍
    last_4b_sum = 0
    for note in melody_output[-128:]:
        if note != 0:
            last_4b_sum += 1
    average_length = 128 / last_4b_sum
    # 2.找出倒数2/3小节中最后一个高过平均节拍的音符 之后的记为前奏
    note_front = 0
    note_behind = 0
    flag_find = False
    for note_it in range(len(melody_output) - 32, len(melody_output) - 96, -1):
        if melody_output[note_it] != 0:
            if note_behind == 0:
                note_behind = note_it
            else:
                note_front = note_behind
                note_behind = note_it
                if note_behind - note_front > average_length:
                    flag_find = True
                    break
    if flag_find is False:
        note_front = len(melody_output) - 64
    intro = [0 for t in range(note_front % 32)] + melody_output[note_front:]
    # 3.如果主旋律的最后一拍只有一个音符的话，前奏多加一拍
    if melody_output[-8:] != [0 for t in range(8)]:
        intro += [0 for t in range(32)]
    return intro


def melodylist2pianoroll(melody_list, velocity=100, length_ratio=0.9, split=None):
    # 音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(len(melody_list)):
        if melody_list[step_it] != 0:
            # 1.求这个音符
            actual_note = melody_list[step_it]  # 这个音符的实际音高
            # 2.求这个音符的持续时间
            note_length = 0.125
            for forward_step_it in range(len(melody_list[step_it+1:])):
                if melody_list[step_it+1:][forward_step_it] == 0 and (split is None or (forward_step_it + step_it + 1) * 0.125 not in split):
                    note_length += 0.125
                else:
                    break
            note_length *= length_ratio
            # 3.保存这个音符
            piano_roll_list.append([step_it * 0.125, actual_note, velocity, note_length])
    return piano_roll_list


def chordlist2pianoroll(chord_list, velocity=100, length_ratio=0.9):
    # 1.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(0, len(chord_list), 2):
        if chord_list[step_it] != 0:
            # 1.1.求这个和弦对应的音符和根音
            dict_content = CHORD_DICT[chord_list[step_it]]
            dict_content = list(dict_content)
            root_note = (chord_list[step_it] - 1) // 6  # 根音
            # 1.2.将和弦的音高转位到53-64之间(F4-E5)，根音转位到41-52之间（F3-E4）
            for note_dict_iterator in range(len(dict_content)):
                if dict_content[note_dict_iterator] <= 4:
                    dict_content[note_dict_iterator] += 60
                else:
                    dict_content[note_dict_iterator] += 48
            if 0 <= root_note <= 4:
                root_note += 48
            elif root_note >= 5:
                root_note += 36
            # 1.3.保存这些音符
            piano_roll_list.append([step_it, root_note, velocity, length_ratio])
            for note_dict_it in dict_content:
                piano_roll_list.append([step_it + 1, note_dict_it, velocity, length_ratio])
    return piano_roll_list


def drumlist2pianoroll(note_list, velocity=100, length_ratio=0.9, split=None):
    # 1.获取音符词典
    # TODO drum的时值应当为一个定值
    # for key in note_dict:
    #     print(key, note_dict[key])
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(len(note_list)):
        if note_list[step_it] != 0:
            # 2.1.求这个音符的持续时间
            note_length = 0.125
            for forward_step_it in range(len(note_list[step_it+1:])):
                if note_list[step_it+1:][forward_step_it] == 0 and (split is None or (forward_step_it + step_it + 1) * 0.125 not in split):
                    note_length += 0.125
                else:
                    break
            note_length *= length_ratio
            # 2.2.保存这个音符
            dict_content = NoteDict[note_list[step_it]]
            for note_dict_it in dict_content:
                piano_roll_list.append([step_it * 0.125, note_dict_it, velocity, note_length])
    return piano_roll_list


def basslist2pianoroll(note_list, velocity=100, length_ratio=0.9, scale_adjust=0, split=None):
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(len(note_list)):
        if note_list[step_it] != 0:
            # 1.求这个音符的持续时间
            note_length = 0.125
            for forward_step_it in range(len(note_list[step_it+1:])):
                if note_list[step_it+1:][forward_step_it] == 0 and (split is None or (forward_step_it + step_it + 1) * 0.125 not in split):
                    note_length += 0.125
                else:
                    break
            note_length *= length_ratio
            # 2.保存这个音符
            for note_dict_it in note_list[step_it]:
                piano_roll_list.append([step_it * 0.125, note_dict_it + scale_adjust, velocity, note_length])
    return piano_roll_list


def pglist2pianoroll(note_list, velocity=100, length_ratio=0.9, scale_adjust=0, split=None):
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(len(note_list)):
        if note_list[step_it] != 0:
            # 1.求这个音符的持续时间
            note_length = 0.25
            for forward_step_it in range(len(note_list[step_it+1:])):
                if note_list[step_it+1:][forward_step_it] == 0 and (split is None or (forward_step_it + step_it + 1) * 0.25 not in split):
                    note_length += 0.25
                else:
                    break
            note_length *= length_ratio
            # 2.根据音符的音高和来调整velocity。每多一个音符，velocity就下降3。同时调整幅度限制在-5～+5之间
            # TODO 下一版本中 3 -5 5 写到settings中
            vel_adj = int(-((len(note_list[step_it]) - 3) * 3))
            if vel_adj < -5:
                vel_adj = -5
            elif vel_adj > 5:
                vel_adj = 5
            # 2.保存这个音符
            for note_dict_it in note_list[step_it]:
                piano_roll_list.append([step_it * 0.25, note_dict_it + scale_adjust, velocity + vel_adj, note_length])
    return piano_roll_list


def stringlist2pianoroll(note_list, velocity=100, length_ratio=0.9, scale_adjust=0, split=None):
    new_note_list = copy.deepcopy(note_list)
    last_note_group = 0
    # 1.去除4拍内相同的音符 延长一个音符组合的持续时间
    for step_it in range(len(new_note_list)):
        if new_note_list[step_it] != 0:
            if new_note_list[step_it] == new_note_list[last_note_group] and step_it - last_note_group <= 16:
                new_note_list[step_it] = 0
            else:
                last_note_group = step_it
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(len(new_note_list)):
        if new_note_list[step_it] != 0:
            # 1.求这个音符的持续时间
            note_length = 0.25
            for forward_step_it in range(len(new_note_list[step_it+1:])):
                if new_note_list[step_it+1:][forward_step_it] == 0 and (split is None or (forward_step_it + step_it + 1) * 0.25 not in split):
                    note_length += 0.25
                else:
                    break
            note_length *= length_ratio
            # 2.根据音符的音高和来调整velocity。每多一个音符，音符每高一个八度，velocity就下降5。同时调整幅度限制在-15～+10之间
            avr_pitch = sum(note_list[step_it]) / len(note_list[step_it])
            # TODO 下一版本中 3 50 -15 10 写到settings中
            vel_adj = int(-((len(note_list[step_it]) - 3) * 5 + (avr_pitch - 50) * 5 / 12))
            if vel_adj < -15:
                vel_adj = -15
            elif vel_adj > 10:
                vel_adj = 10
            # 3.保存这个音符
            for note_dict_it in new_note_list[step_it]:
                piano_roll_list.append([step_it * 0.25, note_dict_it + scale_adjust, velocity + vel_adj, note_length])
    return piano_roll_list


def filllist2pianoroll(note_list, velocity=100, note_length=0.5, scale_adjust=0):
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(len(note_list)):
        if note_list[step_it] != 0:
            # 1.根据音符的音高和来调整velocity。每多一个音符，音符每高一个八度，velocity就下降5。同时调整幅度限制在-14～+7之间
            avr_pitch = sum(note_list[step_it]) / len(note_list[step_it])
            # TODO 下一版本中 1 72 -14 7 写到settings中
            vel_adj = int(-((len(note_list[step_it]) - 1) * 5 + (avr_pitch - 72) * 5 / 12))
            if vel_adj < -14:
                vel_adj = -14
            elif vel_adj > 7:
                vel_adj = 7
            # 2.保存这个音符
            for note_dict_it in note_list[step_it]:
                piano_roll_list.append([step_it * 0.125, note_dict_it + scale_adjust, velocity + vel_adj, note_length])
    return piano_roll_list
