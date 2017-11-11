from settings import *


def NoteList2Chord(note_set, saved_chord=0, tone=TONE_MAJOR, accompany_note_set=None):
    # 处理和弦的思路
    # a.寻找合适三和弦
    # b.在寻找合适的七和弦（当同一时间的音符数量大于4的时候，a和b的顺序反过来）
    # c.与伴随音符相组合寻找和弦
    # d.记为未知和弦
    if tone == TONE_MAJOR:
        recommand_chord_list = [{0, 4, 7}, {5, 9, 0}, {7, 11, 2}, {9, 0, 4}, {2, 5, 9}, {4, 7, 11}]
    elif tone == TONE_MINOR:
        recommand_chord_list = [{0, 4, 9}, {2, 5, 9}, {4, 7, 11}, {0, 4, 7}, {5, 9, 0}, {7, 11, 2}]
    else:
        recommand_chord_list = []
    # 1.把所有的音符全部转化到0-11
    new_note_set = set()
    for note in note_set:
        new_note_set.add(note % 12)
    # 2.note_set只有一个音符的情况
    if len(new_note_set) == 1:
        # 2.1.如果这一个音符属于上一个和弦 那么返回上一个和弦
        if new_note_set.issubset(CHORD_DICT[saved_chord]):
            return saved_chord
        # 2.2.除此以外的其他情况 考虑两拍的全部音符 如果在此情况下能找到对应的和弦 则返回该和弦 否则返回0
        if accompany_note_set:
            return NoteList2Chord(note_set | accompany_note_set, saved_chord, tone, None)
        else:
            return 0
    # 3.note_set中有两个音符的情况
    if len(new_note_set) == 2:
        # 3.1.如果这两个音符属于上一个和弦 那么返回上一个和弦
        if new_note_set.issubset(CHORD_DICT[saved_chord]):
            return saved_chord
        # 3.2.如果这两个音符属于推荐和弦 返回该推荐和弦在和弦字典中的位置
        for chord_iterator in range(len(recommand_chord_list)):
            if new_note_set.issubset(recommand_chord_list[chord_iterator]):
                return CHORD_DICT.index(recommand_chord_list[chord_iterator])
        # 3.3.除此以外的其他情况 考虑两拍的全部音符 如果在此情况下能找到对应的和弦 则返回该和弦 否则返回0
        if accompany_note_set:
            return NoteList2Chord(note_set | accompany_note_set, saved_chord, tone, None)
        else:
            return 0
    # 4.note_set中有三个音符的情况 如果这个和弦在CHORD_DICT列表中 泽返回他在列表中的位置 否则返回未知和弦
    if len(new_note_set) == 3:
        try:
            return CHORD_DICT.index(new_note_set)
        except ValueError:
            return note_list_to_7chord(new_note_set)  # 如果没有合适的三和弦可供选择 看看有没有可供选择的七和弦
    # 5.note_set中有多余三个音符的情况
    if len(new_note_set) >= 4:
        # 5.1.如果有七和弦是它的子集 返回该七和弦在和弦字典中的位置
        choose_7chord = note_list_to_7chord(new_note_set)
        if choose_7chord != 0:
            # print('\t%d', choose_7chord)
            return choose_7chord
        # 5.2.如果上一个和弦是它的子集 那么返回上一个和弦
        if CHORD_DICT[saved_chord].issubset(new_note_set):
            return saved_chord
        # 5.3.如果推荐和弦中有和弦是它的子集 那么返回该推荐和弦在和弦字典中的位置
        for chord_iterator in range(len(recommand_chord_list)):
            if recommand_chord_list[chord_iterator].issubset(new_note_set):
                return CHORD_DICT.index(recommand_chord_list[chord_iterator])
        # 5.4.如果和弦字典中有和弦是它的子集 那么返回它在列表中的位置
        for chord_iterator in range(1, 73):  # 1-72号位是三和弦
            if CHORD_DICT[chord_iterator].issubset(new_note_set):
                return chord_iterator
        # 5.5.其他情况 返回未知和弦
        return 0


def note_list_to_7chord(note_set):
    """
    寻找音符列表对应的七和弦
    :param note_set: 音符列表 这里的音符列表已经全部转化到0-11之中
    :return: 七和弦在CHORD_DICT中的编号
    """
    if len(note_set) == 3:  # 有三个音的情况
        for chord_iterator in range(73, 109):
            if note_set.issubset(CHORD_DICT[chord_iterator]):  # 在CHORD_DICT中，73-108是七和弦
                return chord_iterator
        return 0
    elif len(note_set) >= 4:  # 大于等于四个音的情况
        for chord_iterator in range(73, 109):
            if CHORD_DICT[chord_iterator].issubset(note_set):
                return chord_iterator
        return 0


def ChordTo3(code_in):

    if code_in <= 72 and code_in % 6 in range(1, 5):
        return code_in
    if code_in <= 72 and code_in % 6 in [0, 5]:
        return -1
    if code_in >= 72:
        if code_in % 3 in [1, 0]:
            return ((code_in - 73) // 3) * 6 + 1
        else:
            return ((code_in - 73) // 3) * 6 + 2


def RootNote(chord, last_root):
    """
    确定一个和弦的根音。如果不能确定当前和弦则根音沿用上一个。
    方法是将根音调整至离预期根音均值与上一个根音的平均值最近的点
    :param chord: 和弦编号
    :param last_root: 上一拍的根音
    :return: 这一拍的根音
    """
    if chord == 0:
        return last_root
    if chord <= 72:
        chord_root = (chord - 1) // 6
    else:
        chord_root = (chord - 73) // 3
    if last_root == 0:
        expected_root = PIANO_GUITAR_AVERAGE_ROOT
    else:
        expected_root = last_root * 0.55 + PIANO_GUITAR_AVERAGE_ROOT * 0.45  # 预期的根音高度
    root_high_diff = chord_root - expected_root
    adjust = 12 * round(root_high_diff / 12)  # 要调整的数量
    return chord_root - adjust
