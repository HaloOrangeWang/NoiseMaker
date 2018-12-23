from settings import *
import json
import sqlite3


class Manifest:

    sec_dic = {'main': DEF_SEC_MAIN, 'middle': DEF_SEC_MIDDLE, 'sub': DEF_SEC_SUB, 'end': DEF_SEC_END, 'empty': DEF_SEC_EMPTY}  # 区段标志和settings中的对应值的对照表

    def __init__(self):
        """根据json文件获取所有乐曲的配置信息"""
        self.song_info_list = [dict() for t in range(TRAIN_FILE_NUMBERS)]

        with open('../Inputs/%02d/manifest.json' % ACTIVE_MUSIC_TYPE, encoding='utf-8') as f:
            config_data = json.loads(f.read())

        self.skiplist = [t - 1 for t in config_data['skip']]  # 这里减一 因为json中的编号是从1开始的
        for song_it in self.skiplist:
            self.song_info_list[song_it] = None

        for info in config_data['Songs']:
            song_dx = info['id'] - 1  # 在json文件中，这个编号是从1开始的，因此需要减一
            self.song_info_list[song_dx]['name'] = info['name']  # 歌名
            self.song_info_list[song_dx]['scale'] = info['scale']  # 音高
            if info['tone'] == 'major':  # 调式
                self.song_info_list[song_dx]['tone'] = DEF_TONE_MAJOR
            elif info['tone'] == 'minor':
                self.song_info_list[song_dx]['tone'] = DEF_TONE_MINOR
            else:
                raise ValueError
            self.song_info_list[song_dx]['bpm'] = info['bpm']  # 速度
            if type(info['bias']) is str:
                self.song_info_list[song_dx]['bias'] = eval(info['bias'])
            else:
                self.song_info_list[song_dx]['bias'] = info['bias']

            self.song_info_list[song_dx]['sections'] = []
            if info['sections'] != 'NotApplicable':  # 这首歌适用于乐段的概念
                for sec in info['sections']:
                    if sec[2] in self.sec_dic:
                        self.song_info_list[song_dx]['sections'].append([sec[0], sec[1], self.sec_dic[sec[2]]])  # 这首歌的乐段信息
                    else:
                        raise ValueError

    def store(self):
        """把init中根据json文件整理得到的乐曲相关信息存储进sqlite中"""
        # 1.初始化sqlite信息
        conn = sqlite3.connect(PATH_RAW_DATASET)
        conn.execute('drop table if exists SongInfo')  # 在建表之前先删掉这个表中的原有数据
        conn.commit()
        conn.execute('create table if not exists SongInfo(id integer primary key, name varchar(30), scale integer, bpm integer, tone integer, biastime float)')
        # 2.存储配置数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if song_it not in self.skiplist:
                command = 'insert into SongInfo(id, name, scale, bpm, tone, biastime) values (%d, \'%s\', %d, %d, %d, %.4f)' % (song_it + 1, self.song_info_list[song_it]['name'], self.song_info_list[song_it]['scale'], self.song_info_list[song_it]['bpm'], self.song_info_list[song_it]['tone'], self.song_info_list[song_it]['bias'])
                conn.execute(command)
        conn.commit()
        # 3.初始化乐段信息的sqlite表
        conn.execute('drop table if exists SectionInfo')
        conn.commit()
        conn.execute('pragma foreign_key=ON')
        conn.execute('create table if not exists SectionInfo(id integer primary key autoincrement, bar_index float, bias float, section_type integer, song_id integer, foreign key(song_id) references SongInfo(id) on update cascade)')
        # 4.存储乐段数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if song_it not in self.skiplist:
                for sec in self.song_info_list[song_it]['sections']:
                    command = 'insert into SectionInfo(bar_index, bias, section_type, song_id) values (%.4f, %.4f, %d, %d)' % (sec[0], sec[1], sec[2], song_it + 1)
                    conn.execute(command)
        conn.commit()
