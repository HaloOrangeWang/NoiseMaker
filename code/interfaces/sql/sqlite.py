import sqlite3
from settings import *


def get_raw_song_data_from_dataset(mark, tone_restrict=None):
    # 1.准备工作
    conn = sqlite3.connect(DATASET_PATH)
    raw_song_data = [{} for t in range(TRAIN_FILE_NUMBERS)]
    # 2.读取数据
    if tone_restrict is None:
        rows = conn.execute('select bar_index, data, song_id from BarInfo where mark=\'' + mark + '\'')
    else:
        rows = conn.execute('select bar_index, data, song_id from BarInfo where mark=\'' + mark + '\' and song_id in (select id from SongInfo where tone=' + str(tone_restrict) + ')')
    for row in rows:
        raw_song_data[row[2]-1][row[0]] = eval(row[1])
    return raw_song_data


def get_section_data_from_dataset():
    # 1.准备工作
    section_data = [[] for t in range(TRAIN_FILE_NUMBERS)]
    conn = sqlite3.connect(DATASET_PATH)
    # 2.从数据集中读取所有歌曲的乐段信息
    rows = conn.execute('select bar_index, bias, section_type, song_id from SectionInfo')
    for row in rows:
        # 由于在数据库中歌的id是从1开始编号的，因此需要减一
        song_dx = row[3] - 1
        bar_dx = row[0]
        bias = row[1]
        section_type = row[2]
        section_data[song_dx].append([bar_dx, bias, section_type])
    return section_data


def read_note_dict():
    """
    阅读音符词典
    :return: 音符词典的内容
    """
    conn = sqlite3.connect(DATASET_PATH)
    rows = conn.execute('select * from NoteDict')
    note_dict = {}
    for row in rows:
        note_dict[row[0] - 1] = eval(row[1])
    return note_dict


def get_tone_list():
    """
    获取每一首歌的调式信息
    :return:
    """
    conn = sqlite3.connect(DATASET_PATH)
    tone_list = [None for t in range(TRAIN_FILE_NUMBERS)]
    rows = conn.execute('select id, tone from SongInfo')
    for row in rows:
        tone_list[row[0] - 1] = row[1]
    return tone_list


def get_bpm_list():
    """
    获取每一首歌的bpm信息
    :return:
    """
    conn = sqlite3.connect(DATASET_PATH)
    bpm_list = [None for t in range(TRAIN_FILE_NUMBERS)]
    rows = conn.execute('select id, bpm from SongInfo')
    for row in rows:
        bpm_list[row[0] - 1] = row[1]
    return bpm_list


NoteDict = read_note_dict()
