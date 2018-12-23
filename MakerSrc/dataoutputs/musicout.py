from interfaces.sql.sqlite import NoteDict
from settings import *
import random
import copy


def music_promote(stream_in, generate_type=-1):
    """
    按照生成的方式来生成一段音乐
    :param stream_in: 输入的各条音轨的信息
    :param generate_type: 生成的方式。0,1为正常方式 2,3为去掉部分鼓点和bass 4,5为去掉部分bass 6,7为去掉部分鼓点和piano guitar。奇数表示在结尾部分增加一个crash
    :return:
    """
    if generate_type < 0:
        generate_type = random.randint(0, 7)
    stream_out = dict()
    # 1.获取前奏
    melody_bar_num = len(stream_in['melody']) // 32  # 主旋律占了多少小节
    intro_bar_num = len(stream_in['intro']) // 32  # 前奏占了几个小节
    # 2.各条音轨的内容均为前奏+旋律+前奏+旋律
    stream_out['melody'] = [0 for t in range(intro_bar_num * 32)] + stream_in['melody'] + [0 for t in range(intro_bar_num * 32)] + stream_in['melody']
    stream_out['intro'] = stream_in['intro'] + [0 for t in range(melody_bar_num * 32)] + stream_in['intro'] + [0 for t in range(melody_bar_num * 32)]
    if len(stream_in['section']) >= 4 and generate_type // 2 == 1:  # 至少有三个乐段且生成的类型为1 则前奏没有bass
        stream_out['bass'] = [0] * (32 * intro_bar_num) + stream_in['bass'][:32 * melody_bar_num] + stream_in['bass'][melody_bar_num * 32:] + stream_in['bass'][:32 * melody_bar_num]
    elif len(stream_in['section']) >= 4 and generate_type // 2 == 2:  # 至少有三个乐段且生成的类型为1 则前奏和第一个乐段没有bass
        stream_out['bass'] = [0] * (32 * (intro_bar_num + stream_in['section'][1][0])) + stream_in['bass'][32 * stream_in['section'][1][0]: 32 * melody_bar_num] + stream_in['bass'][melody_bar_num * 32:] + stream_in['bass'][:32 * melody_bar_num]
    else:
        stream_out['bass'] = stream_in['bass'][melody_bar_num * 32:] + stream_in['bass'][:32 * melody_bar_num] + stream_in['bass'][melody_bar_num * 32:] + stream_in['bass'][:32 * melody_bar_num]
    stream_out['chord'] = stream_in['chord'][melody_bar_num * 4:] + stream_in['chord'][:4 * melody_bar_num] + stream_in['chord'][melody_bar_num * 4:] + stream_in['chord'][:4 * melody_bar_num]
    stream_out['fill'] = [0 for t in range(intro_bar_num * 32)] + stream_in['fill'] + [0 for t in range(intro_bar_num * 32)] + stream_in['fill']
    if len(stream_in['section']) >= 4 and generate_type // 2 == 3:  # 至少有三个乐段且生成的类型为3 则前两个乐段没有piano guitar
        stream_out['pg'] = stream_in['pg'][melody_bar_num * 16:] + [0] * (16 * stream_in['section'][2][0]) + stream_in['pg'][16 * stream_in['section'][2][0]: 16 * melody_bar_num] + stream_in['pg'][melody_bar_num * 16:] + stream_in['pg'][:16 * melody_bar_num]
    else:
        stream_out['pg'] = stream_in['pg'][melody_bar_num * 16:] + stream_in['pg'][:16 * melody_bar_num] + stream_in['pg'][melody_bar_num * 16:] + stream_in['pg'][:16 * melody_bar_num]
    stream_out['string'] = stream_in['string'][melody_bar_num * 16:] + stream_in['string'][:16 * melody_bar_num] + stream_in['string'][melody_bar_num * 16:] + stream_in['string'][:16 * melody_bar_num]
    if len(stream_in['section']) >= 4 and generate_type // 2 == 1:  # 至少有三个乐段且生成的类型为1 则前奏和第一个乐段没有鼓点
        stream_out['drum'] = [0] * (32 * (intro_bar_num + stream_in['section'][1][0])) + stream_in['drum'][32 * stream_in['section'][1][0]: 32 * melody_bar_num] + stream_in['drum'][melody_bar_num * 32:] + stream_in['drum'][:32 * melody_bar_num]
    elif len(stream_in['section']) >= 4 and generate_type // 2 == 3:  # 至少有三个乐段且生成的类型为3 则第一个乐段没有鼓点
        stream_out['drum'] = stream_in['drum'][melody_bar_num * 32:] + [0] * (32 * stream_in['section'][1][0]) + stream_in['drum'][32 * stream_in['section'][1][0]: 32 * melody_bar_num] + stream_in['drum'][melody_bar_num * 32:] + stream_in['drum'][:32 * melody_bar_num]
    else:
        stream_out['drum'] = stream_in['drum'][melody_bar_num * 32:] + stream_in['drum'][:32 * melody_bar_num] + stream_in['drum'][melody_bar_num * 32:] + stream_in['drum'][:32 * melody_bar_num]
    # 3.可能会在末尾处增加一个crash
    if generate_type % 2 == 1:
        for key in NoteDict.nd:
            if NoteDict.nd[key] == [49]:
                crash1_key = key
            elif NoteDict.nd[key] == [57]:
                crash2_key = key
        stream_out['drum'].extend([[crash1_key, crash2_key][random.randint(0, 1)]] + [0] * 7)
        stream_out['melody'].extend([0] * 8)
        stream_out['intro'].extend([0] * 8)
        stream_out['bass'].extend([0] * 8)
        stream_out['chord'].extend([0])
        stream_out['fill'].extend([0] * 8)
        stream_out['pg'].extend([0] * 4)
        stream_out['string'].extend([0] * 4)
    return stream_out


def get_pitch_adj_value(note_list, expect_avr_note):
    """
    根据预期的输出音高均值，获取应该调整的音高数量
    :param note_list: 音符列表
    :param expect_avr_note: 预期的音高均值
    :return 应该调整的音高数量
    """
    sum_pitch = 0  # 这个列表中所有音符的音高总和
    note_count = 0
    for note_group in note_list:
        if note_group != 0:
            sum_pitch += sum(note_group)
            note_count += len(note_group)
    avr_pitch = sum_pitch / note_count  # 音符的平均音高
    pitch_diff = avr_pitch - expect_avr_note
    adjust_pitch = 12 * round(pitch_diff / 12)  # 要调整的数量
    return adjust_pitch


def melodylist2pianoroll(melody_list, velocity=100, length_ratio=0.9, split=None):
    # 音符列表解码，转成piano_roll
    pianoroll_list = []
    for step_it in range(len(melody_list)):
        if melody_list[step_it] != 0:
            # 1.求这个音符
            note_pitch = melody_list[step_it]  # 这个音符的实际音高
            # 2.求这个音符的持续时间
            note_length = 0.125
            for forward_step_it in range(len(melody_list[step_it+1:])):
                if melody_list[step_it+1:][forward_step_it] == 0 and (split is None or (forward_step_it + step_it + 1) * 0.125 not in split):
                    note_length += 0.125
                else:
                    break
            note_length *= length_ratio
            # 3.保存这个音符
            pianoroll_list.append([step_it * 0.125, note_pitch, velocity, note_length])
    return pianoroll_list


def chordlist2pianoroll(chord_list, velocity=100, length_ratio=0.9):
    # 1.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(0, len(chord_list), 2):
        if chord_list[step_it] != 0:
            # 1.1.求这个和弦对应的音符和根音
            note_list = copy.deepcopy(CHORD_LIST[chord_list[step_it]])
            note_list = list(note_list)
            root_pitch = (chord_list[step_it] - 1) // 6  # 根音
            # 1.2.将和弦的音高转位到53-64之间(F4-E5)，根音转位到41-52之间（F3-E4）
            for note_it in range(len(note_list)):
                if note_list[note_it] <= 4:
                    note_list[note_it] += 60
                else:
                    note_list[note_it] += 48
            if 0 <= root_pitch <= 4:
                root_pitch += 48
            elif root_pitch >= 5:
                root_pitch += 36
            # 1.3.保存这些音符
            piano_roll_list.append([step_it, root_pitch, velocity, length_ratio])
            for note_dict_it in note_list:
                piano_roll_list.append([step_it + 1, note_dict_it, velocity, length_ratio])
    return piano_roll_list


def drumlist2pianoroll(note_list, velocity=100, length_ratio=0.9, split=None):
    # 1.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(len(note_list)):
        if note_list[step_it] != 0:
            # 1.1.求这个音符的持续时间
            note_length = 0.125
            for forward_step_it in range(len(note_list[step_it+1:])):
                if note_list[step_it+1:][forward_step_it] == 0 and (split is None or (forward_step_it + step_it + 1) * 0.125 not in split):
                    note_length += 0.125
                else:
                    break
            note_length *= length_ratio
            # 1.2.保存这个音符
            dict_content = NoteDict.nd[note_list[step_it]]
            for note_dict_it in dict_content:
                piano_roll_list.append([step_it * 0.125, note_dict_it, velocity, note_length])
    return piano_roll_list


def basslist2pianoroll(note_list, velocity=100, length_ratio=0.9, split=None):
    # 1.获取应该调整的音高数量
    adj_pitch = get_pitch_adj_value(note_list, BASS_AVR_NOTE_OUT)
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
            for note_dict_it in note_list[step_it]:
                piano_roll_list.append([step_it * 0.125, note_dict_it - adj_pitch, velocity, note_length])
    return piano_roll_list


def pglist2pianoroll(note_list, velocity=100, length_ratio=0.9, split=None):
    # 1.获取应该调整的音高数量
    adj_pitch = get_pitch_adj_value(note_list, PG_AVR_NOTE_OUT)
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(len(note_list)):
        if note_list[step_it] != 0:
            # 2.1.求这个音符的持续时间
            note_length = 0.25
            for forward_step_it in range(len(note_list[step_it+1:])):
                if note_list[step_it+1:][forward_step_it] == 0 and (split is None or (forward_step_it + step_it + 1) * 0.25 not in split):
                    note_length += 0.25
                else:
                    break
            note_length *= length_ratio
            # 2.2.根据音符的音高和来调整velocity。每多一个音符，velocity就下降3。同时调整幅度限制在-5～+5之间
            vel_adj = int(-((len(note_list[step_it]) - 3) * 3))
            # TODO 下一版本中 3 -5 5 写到settings中
            if vel_adj < -5:
                vel_adj = -5
            elif vel_adj > 5:
                vel_adj = 5
            # 2.3.保存这个音符
            for note_dict_it in note_list[step_it]:
                piano_roll_list.append([step_it * 0.25, note_dict_it - adj_pitch, velocity + vel_adj, note_length])
    return piano_roll_list


def stringlist2pianoroll(note_list, velocity=100, length_ratio=0.9, split=None):
    new_note_list = copy.deepcopy(note_list)
    last_note_group = 0
    # 1.去除4拍内相同的音符 延长一个音符组合的持续时间
    for step_it in range(len(new_note_list)):
        if new_note_list[step_it] != 0:
            if new_note_list[step_it] == new_note_list[last_note_group] and step_it - last_note_group <= 16:
                new_note_list[step_it] = 0
            else:
                last_note_group = step_it
    # 2.获取应该调整的音高数量
    adj_pitch = get_pitch_adj_value(note_list, STRING_AVR_NOTE_OUT)
    # 3.音符列表解码，转成piano_roll
    piano_roll_list = []
    for step_it in range(len(new_note_list)):
        if new_note_list[step_it] != 0:
            # 3.1.求这个音符的持续时间
            note_length = 0.25
            for forward_step_it in range(len(new_note_list[step_it+1:])):
                if new_note_list[step_it+1:][forward_step_it] == 0 and (split is None or (forward_step_it + step_it + 1) * 0.25 not in split):
                    note_length += 0.25
                else:
                    break
            note_length *= length_ratio
            # 3.2.根据音符的音高和来调整velocity。每多一个音符，音符每高一个八度，velocity就下降5。同时调整幅度限制在-15～+10之间
            avr_pitch = sum(note_list[step_it]) / len(note_list[step_it])
            # TODO 下一版本中 3 50 -15 10 写到settings中
            vel_adj = int(-((len(note_list[step_it]) - 3) * 5 + (avr_pitch - STRING_AVR_NOTE_OUT) * 5 / 12))
            if vel_adj < -15:
                vel_adj = -15
            elif vel_adj > 10:
                vel_adj = 10
            # 3.3.保存这个音符
            for note_dict_it in new_note_list[step_it]:
                piano_roll_list.append([step_it * 0.25, note_dict_it - adj_pitch, velocity + vel_adj, note_length])
    return piano_roll_list


def filllist2pianoroll(note_list, velocity=100, note_length=0.5):
    # 1.音符列表解码，转成piano_roll（fill不根据平均音高进行调整）
    piano_roll_list = []
    for step_it in range(len(note_list)):
        if note_list[step_it] != 0:
            # 1.1.根据音符的音高和来调整velocity。每多一个音符，音符每高一个八度，velocity就下降5。同时调整幅度限制在-14～+7之间
            avr_pitch = sum(note_list[step_it]) / len(note_list[step_it])
            # TODO 下一版本中 1 72 -14 7 写到settings中
            vel_adj = int(-((len(note_list[step_it]) - 1) * 5 + (avr_pitch - FILL_AVR_NOTE_OUT) * 5 / 12))
            if vel_adj < -14:
                vel_adj = -14
            elif vel_adj > 7:
                vel_adj = 7
            # 1.2.保存这个音符
            for note_dict_it in note_list[step_it]:
                piano_roll_list.append([step_it * 0.125, note_dict_it, velocity + vel_adj, note_length])
    return piano_roll_list
