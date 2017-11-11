from settings import *
import sqlite3
# 这个文件的内容是验证内容
# 注：这个文件里的代码只检查数据集中是否有异常的数据。检查生成的音乐是否符合要求的代码在musicout.py中


def KeyPressValidation():
    """
    检查按键的位置是否正确
    """
    # 1.准备工作
    conn = sqlite3.connect(DATASET_PATH)
    # 2.从数据集中读取歌的编号为song_id且小节标注为mark_key的小节数据，并将所有不为休止符的位置全部替换为1
    rows = conn.execute('select bar_index, data, song_id from BarInfo where mark=\'main\'')
    note_location_points_list = [-4, 2, 1, 2, 0, 2, 1, 2, -3, 2, 1, 2, 0, 2, 1, 2, -3, 2, 1, 2, 0, 2, 1, 2, -3, 2, 1, 2, 0, 2, 1, 2]
    for row in rows:
        # 由于在数据库中歌的id是从1开始编号的，因此需要减一
        bar_data = eval(row[1])
        bar_point = 0
        for note_iterator in range(len(bar_data)):
            if bar_data[note_iterator] != 0:
                bar_point += note_location_points_list[note_iterator]
        # if row[2] == 138:
        #     print(row[0], bar_data)
        if bar_point > 0:
            print('第'+repr(row[2]-1)+'首歌第'+repr(row[0])+'小节得分为'+repr(bar_point))
        for note_iterator in [t*4+1 for t in range(8)]:
            if bar_data[note_iterator] != 0 and bar_data[note_iterator-1] == 0:
                print('第'+repr(row[2]-1)+'首歌第'+repr(row[0])+'小节可能存在错位')


def ChordValidation():
    # 当不确定的和弦和离调和弦在全部和弦中所占的比利过高时发出警告
    # 1.准备工作
    conn = sqlite3.connect(DATASET_PATH)
    chord_data = [[] for t in range(TRAIN_FILE_NUMBERS)]
    normal_chord = [1, 5, 6, 14, 26, 31, 43, 56, 59, 60, 70, 73, 80, 88, 96, 101]  # 1/4/5级大三和弦，2/3/6级小三和弦，7级减三和弦，1/6级挂二挂四和弦 1/4级大七和弦 5级属七和弦 2/6级小七和弦是调内和弦 其余为离调和弦
    # 2.从数据集中读取歌的编号为song_id且小节标注为mark_key的小节数据，并将所有不为休止符的位置全部替换为1
    rows = conn.execute('select bar_index, data, song_id from BarInfo where mark=\'chord\'')
    for row in rows:
        # 由于在数据库中歌的id是从1开始编号的，因此需要减一
        chord_data[row[2]-1] += eval(row[1])
    # print(chord_data[140])
    for song_iterator in chord_data:
        normal_chord_count = 0
        abnormal_chord_count = 0
        zero_chord_count = 0
        for chord_iterator in song_iterator:
            if chord_iterator in normal_chord:
                normal_chord_count += 1
            elif chord_iterator == 0:
                zero_chord_count += 1
            else:
                abnormal_chord_count += 1
        try:
            abnormal_chord_ratio = abnormal_chord_count / (normal_chord_count + abnormal_chord_count + zero_chord_count)
            zero_chord_ratio = zero_chord_count / (normal_chord_count + abnormal_chord_count + zero_chord_count)
            if abnormal_chord_ratio + zero_chord_ratio >= 0.2:
                print('第'+repr(chord_data.index(song_iterator))+'首歌离调和弦比例为%.3f,不确定和弦的比例为%.3f' % (abnormal_chord_ratio, zero_chord_ratio))
        except ZeroDivisionError:
            pass


def RunValidation():
    KeyPressValidation()
    print('\n\n\n\n\n\n')
    ChordValidation()
