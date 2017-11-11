import sqlite3
from settings import *


def GetRawSongDataFromDataset(mark, tone_restrict=None):
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


def ReadNoteDict(return_type):
    """
    阅读音符词典
    :param return_type: 如果是0则返回音符字典长度 如果为1则返回整个音符字典
    :return: 音符词典的内容及其长度
    """
    conn = sqlite3.connect(DATASET_PATH)
    # conn = sqlite3.connect('/home/whl/Documents/NoiseMaker/Noise Maker 0.90.01/data/MidiInfo.db')
    if return_type == 1:
        rows = conn.execute('select * from NoteDict')
        note_dict = {}
        for row in rows:
            note_dict[row[0] - 1] = eval(row[1])
        return note_dict
    elif return_type == 0:
        rows = conn.execute('select max(id) from NoteDict')
        return list(rows)[0][0]

NoteDict = ReadNoteDict(1)


def GetSongToneList():
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
