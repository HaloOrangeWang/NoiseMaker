from settings import *
from interfaces.sql.sqlite import ReadNoteDict
import copy


def MusicPromote(melody_output, chord_output, drum_output, bass_output, fill_output, pg_output, string_output):
    # 1.获取前奏
    intro = GetIntro(melody_output)
    intro_bars = len(intro) // 32  # 前奏占了几个小节
    split = [4 * intro_bars + len(melody_output) // 8, 4 * intro_bars + len(melody_output) // 4 + 8]  # 间奏和尾声开始的位置 单位是拍
    if melody_output[-8:] != [0 for t in range(8)]:
        split = [t + 2 for t in split]
    # 2.旋律和和弦各重复两遍空出两小节前奏 空出两小节间奏 空出一小节尾声
    melody_output = [0 for t in range(intro_bars * 32)] + melody_output + [0 for t in range(64)] + melody_output + [0 for t in range(32)]
    bass_output = [0 for t in range(intro_bars * 32)] + bass_output + [0 for t in range(64)] + bass_output + [0 for t in range(32)]
    chord_output = [0 for t in range(intro_bars * 4)] + chord_output + [0 for t in range(8)] + chord_output + [0 for t in range(4)]
    fill_output = [0 for t in range(intro_bars * 32)] + fill_output + [0 for t in range(64)] + fill_output + [0 for t in range(32)]
    pg_output = [0 for t in range(intro_bars * 16)] + pg_output + [0 for t in range(32)] + pg_output + [0 for t in range(16)]
    string_output = [0 for t in range(intro_bars * 16)] + string_output + [0 for t in range(32)] + string_output + [0 for t in range(16)]
    # 2.加上鼓 从0.91.03开始鼓点也加入到训练内容中
    drum_output = drum_output[0: 32] * intro_bars + drum_output + drum_output[0: 32] * 2 + drum_output + drum_output[-32:]
    return intro, split, melody_output, chord_output, drum_output, bass_output, fill_output, pg_output, string_output


def GetIntro(melody_output):
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
    for note_iterator in range(len(melody_output) - 32, len(melody_output) - 96, -1):
        if melody_output[note_iterator] != 0:
            if note_behind == 0:
                note_behind = note_iterator
            else:
                note_front = note_behind
                note_behind = note_iterator
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


def GetAbsNoteList(rel_note_list, root_note):
    rootdict = [[0, 0], [1, -1], [1, 0], [2, -1], [2, 0], [3, 0], [3, 1], [4, 0], [5, -1], [5, 0], [6, -1], [6, 0]]
    rel_list = [0, 2, 4, 5, 7, 9, 11]
    output_notelist = []
    # 1.找到root_note的音名和音域
    rootname = rootdict[root_note % 12][0]
    rootbase = root_note - root_note % 12
    # 2.求出rel_note_list所有音符的音名
    for rel_note in rel_note_list:
        note = rel_note[0] + rootname  # 获取音名
        note = rel_list[note % 7] + 12 * (note // 7)  # 获取相对ｒｏｏｔｂａｓｅ的音高
        note = note + rootbase - rel_note[1]  # 计算绝对音高
        output_notelist.append(note)
    return output_notelist


def NoteList2PianoRoll(note_list, time_step, velocity=100, length_ratio=0.9, split=None):
    # 1.获取音符词典
    note_dict = ReadNoteDict(1)
    # for key in note_dict:
    #     print(key, note_dict[key])
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for note_iterator in range(len(note_list)):
        if note_list[note_iterator] != 0:
            # 2.1.求这个音符的持续时间
            note_length = time_step
            for t in range(len(note_list[note_iterator+1:])):
                if note_list[note_iterator+1:][t] == 0 and (split is None or (t + note_iterator + 1) * time_step not in split):
                    note_length += time_step
                else:
                    break
            note_length *= length_ratio
            # 2.2.保存这个音符
            dict_content = note_dict[note_list[note_iterator]]
            for note_dict_iterator in dict_content:
                piano_roll_list.append([note_iterator * time_step, note_dict_iterator, velocity, note_length])
    return piano_roll_list


def MelodyList2PianoRoll(melody_list, velocity=100, length_ratio=0.9, split=None):
    # 音符列表解码，转成piano_roll
    piano_roll_list = []
    for note_iterator in range(len(melody_list)):
        if melody_list[note_iterator] != 0:
            # 1.求这个音符
            actual_note = melody_list[note_iterator]  # + MELODY_LOW_NOTE - 1  # 这个音符的实际音高
            # 2.求这个音符的持续时间
            note_length = 0.125
            for t in range(len(melody_list[note_iterator+1:])):
                if melody_list[note_iterator+1:][t] == 0 and (split is None or (t + note_iterator + 1) * 0.125 not in split):
                    note_length += 0.125
                else:
                    break
            note_length *= length_ratio
            # 3.保存这个音符
            piano_roll_list.append([note_iterator * 0.125, actual_note, velocity, note_length])
    return piano_roll_list


def PgList2PianoRoll(note_list, velocity=100, length_ratio=0.9, scale_adjust=0, split=None):
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for note_iterator in range(len(note_list)):
        if note_list[note_iterator] != 0:
            # 1.求这个音符的持续时间
            note_length = 0.25
            for t in range(len(note_list[note_iterator+1:])):
                if note_list[note_iterator+1:][t] == 0 and (split is None or (t + note_iterator + 1) * 0.25 not in split):
                    note_length += 0.25
                else:
                    break
            note_length *= length_ratio
            # 2.保存这个音符
            for note_dict_iterator in note_list[note_iterator]:
                piano_roll_list.append([note_iterator * 0.25, note_dict_iterator + scale_adjust, velocity, note_length])
    return piano_roll_list


def FillList2PianoRoll(note_list, velocity=100, note_length=0.5, scale_adjust=0):
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for note_iterator in range(len(note_list)):
        if note_list[note_iterator] != 0 :
            # 2.保存这个音符
            for note_dict_iterator in note_list[note_iterator]:
                piano_roll_list.append([note_iterator * 0.125, note_dict_iterator + scale_adjust, velocity, note_length])
    return piano_roll_list


def ChordList2PianoRoll(chord_list, velocity=100, length_ratio=0.9):
    # 1.音符列表解码，转成piano_roll
    piano_roll_list = []
    for note_iterator in range(0, len(chord_list), 2):
        if chord_list[note_iterator] != 0:
            # 1.1.求这个和弦对应的音符和根音
            dict_content = CHORD_DICT[chord_list[note_iterator]]
            dict_content = list(dict_content)
            root_note = (chord_list[note_iterator] - 1) // 6  # 根音
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
            piano_roll_list.append([note_iterator * CHORD_TIME_STEP, root_note, velocity, length_ratio * CHORD_TIME_STEP])
            for note_dict_iterator in dict_content:
                piano_roll_list.append([(note_iterator + 1) * CHORD_TIME_STEP, note_dict_iterator, velocity, length_ratio * CHORD_TIME_STEP])
    return piano_roll_list


def BassList2PianoRoll(note_list, velocity=100, length_ratio=0.9, split=None):
    # 1.获取音符词典
    note_dict = ReadNoteDict(1)
    # for key in note_dict:
    #     print(key, note_dict[key])
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for note_iterator in range(len(note_list)):
        if note_list[note_iterator] != 0:
            # 2.1.求这个音符的持续时间
            note_length = 1 / 8
            for t in range(len(note_list[note_iterator+1:])):
                if note_list[note_iterator+1:][t] == 0 and (split is None or (t + note_iterator + 1) * 0.125 not in split):
                    note_length += 1 / 8
                else:
                    break
            note_length *= length_ratio
            # 2.2.保存这个音符
            dict_content = note_dict[note_list[note_iterator]]
            for note_dict_iterator in dict_content:
                piano_roll_list.append([note_iterator * (1 / 8), note_dict_iterator, velocity, note_length])
    # 3.bass的所有音符集体提升一个八度（否则不好听）
    for note_iterator in range(len(piano_roll_list)):
        piano_roll_list[note_iterator][1] += 12
    return piano_roll_list


def StringList2PianoRoll(note_list, velocity=100, length_ratio=0.9, scale_adjust=0, split=None):
    new_note_list = copy.deepcopy(note_list)
    last_note_group = 0
    # 1.去除4拍内相同的音符 延长一个音符组合的持续时间
    for note_iterator in range(len(new_note_list)):
        if new_note_list[note_iterator] != 0:
            if new_note_list[note_iterator] == new_note_list[last_note_group] and note_iterator - last_note_group <= 16:
                new_note_list[note_iterator] = 0
            else:
                last_note_group = note_iterator
    # 2.音符列表解码，转成piano_roll
    piano_roll_list = []
    for note_iterator in range(len(new_note_list)):
        if new_note_list[note_iterator] != 0:
            # 1.求这个音符的持续时间
            note_length = 0.25
            for t in range(len(new_note_list[note_iterator+1:])):
                if new_note_list[note_iterator+1:][t] == 0 and (split is None or (t + note_iterator + 1) * 0.25 not in split):
                    note_length += 0.25
                else:
                    break
            note_length *= length_ratio
            # 2.保存这个音符
            for note_dict_iterator in new_note_list[note_iterator]:
                piano_roll_list.append([note_iterator * 0.25, note_dict_iterator + scale_adjust, velocity, note_length])
    return piano_roll_list
