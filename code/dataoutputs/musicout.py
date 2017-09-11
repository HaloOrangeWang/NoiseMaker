from settings import *
from interfaces.sql.sqlite import ReadNoteDict


def MusicPromote(melody_output, chord_output, drum_output):
    # 1.主旋律的最后两拍强制设为1 最后两个和弦强制设为1级大和弦
    # melody_output[-16:] = [25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # chord_output[-2:] = [1, 1]
    # 2.旋律和和弦各重复两遍空出两小节前奏 空出两小节间奏 空出一小节尾声
    melody_output = [0 for t in range(64)] + melody_output + [0 for t in range(64)] + melody_output + [0 for t in range(32)]
    chord_output = [0 for t in range(8)] + chord_output + [0 for t in range(8)] + chord_output + [0 for t in range(4)]
    # 3.加上鼓 从0.91.03开始鼓点也加入到训练内容中
    drum_output = drum_output[0:32] * 2 + drum_output + drum_output[0:32] * 2 + drum_output + drum_output[-32:]
    return melody_output, chord_output, drum_output


def NoteList2PianoRoll(note_list, time_step, velocity=100, length_ratio=0.9):
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
                if note_list[note_iterator+1:][t] == 0:
                    note_length += time_step
                else:
                    break
            note_length *= length_ratio
            # 2.2.保存这个音符
            dict_content = note_dict[note_list[note_iterator]]
            for note_dict_iterator in dict_content:
                piano_roll_list.append([note_iterator * time_step, note_dict_iterator, velocity, note_length])
    return piano_roll_list


def MelodyList2PianoRoll(melody_list, velocity=100, length_ratio=0.9):
    # 音符列表解码，转成piano_roll
    piano_roll_list = []
    for note_iterator in range(len(melody_list)):
        if melody_list[note_iterator] != 0:
            # 1.求这个音符
            actual_note = melody_list[note_iterator]  # + MELODY_LOW_NOTE - 1  # 这个音符的实际音高
            # 2.求这个音符的持续时间
            note_length = MELODY_TIME_STEP
            for t in range(len(melody_list[note_iterator+1:])):
                if melody_list[note_iterator+1:][t] == 0:
                    note_length += MELODY_TIME_STEP
                else:
                    break
            note_length *= length_ratio
            # 3.保存这个音符
            piano_roll_list.append([note_iterator * MELODY_TIME_STEP, actual_note, velocity, note_length])
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
