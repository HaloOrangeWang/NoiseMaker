from settings import *
import json
import sqlite3


def manifest():

    # 1.读取json数据
    song_info_ary = [{} for t in range(TRAIN_FILE_NUMBERS)]

    with open('../TrainData/manifest.json') as f:
        config_data = json.loads(f.read())

    skiplist = [t - 1 for t in config_data['skip']]  # 这里减一 因为json中的编号是从1开始的
    for skip_it in skiplist:
        song_info_ary[skip_it] = None

    for info in config_data['Songs']:
        song_dx = info['id'] - 1  # 在json文件中，这个编号是从1开始的，因此需要减一
        # if song_dx == 14:
        #     aaa = 4
        song_info_ary[song_dx]['name'] = info['name']  # 歌名
        song_info_ary[song_dx]['scale'] = info['scale']  # 音高
        if info['tone'] == 'major':  # 调式
            song_info_ary[song_dx]['tone'] = TONE_MAJOR
        elif info['tone'] == 'minor':
            song_info_ary[song_dx]['tone'] = TONE_MINOR
        else:
            raise ValueError
        song_info_ary[song_dx]['bpm'] = info['bpm']  # 速度
        if type(info['bias']) is str:
            song_info_ary[song_dx]['bias'] = eval(info['bias'])
        else:
            song_info_ary[song_dx]['bias'] = info['bias']

        sec_dic = {'main': SECTION_MAIN, 'middle': SECTION_MIDDLE, 'sub': SECTION_SUB, 'end': SECTION_END, 'empty': SECTION_EMPTY}  # 区段标志和对应值的对照表
        song_info_ary[song_dx]['sections'] = []
        if info['sections'] != 'NotApplicable':  # 这首歌适用于乐段的概念
            for sec in info['sections']:
                if sec[2] in sec_dic:
                    song_info_ary[song_dx]['sections'].append([sec[0], sec[1], sec_dic[sec[2]]])  # 这首歌的乐段信息
                else:
                    raise ValueError

    for song_info in song_info_ary:
        if song_info == {}:
            raise ValueError

    # 2.初始化sqlite信息
    conn = sqlite3.connect(DATASET_PATH)
    conn.execute('drop table if exists SongInfo')  # 在建表之前先删掉这个表中的原有数据
    conn.commit()
    conn.execute('create table if not exists SongInfo(id integer primary key, name varchar(30), scale integer, bpm integer, tone integer, biastime float)')
    # 3.存储配置数据
    for song_it in range(TRAIN_FILE_NUMBERS):
        if song_it not in skiplist:
            command = 'insert into SongInfo(id, name, scale, bpm, tone, biastime) values (%d, \'%s\', %d, %d, %d, %.4f)' % (song_it + 1, song_info_ary[song_it]['name'], song_info_ary[song_it]['scale'], song_info_ary[song_it]['bpm'], song_info_ary[song_it]['tone'], song_info_ary[song_it]['bias'])
            conn.execute(command)
    conn.commit()
    # 4.初始化乐段信息的sqlite表
    conn.execute('drop table if exists SectionInfo')
    conn.commit()
    conn.execute('pragma foreign_key=ON')
    conn.execute('create table if not exists SectionInfo(id integer primary key autoincrement, bar_index float, bias float, section_type integer, song_id integer, foreign key(song_id) references SongInfo(id) on update cascade)')
    # 5.存储乐段数据
    for song_it in range(TRAIN_FILE_NUMBERS):
        if song_it not in skiplist:
            for sec in song_info_ary[song_it]['sections']:
                command = 'insert into SectionInfo(bar_index, bias, section_type, song_id) values (%.4f, %.4f, %d, %d)' % (sec[0], sec[1], sec[2], song_it + 1)
                conn.execute(command)
    conn.commit()

    return song_info_ary
