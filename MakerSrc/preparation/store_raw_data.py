from settings import *
from interfaces.chord_parse import noteset2chord
from interfaces.utils import DiaryLog
from interfaces.midi import generate_data_from_midi_file
import math
import sqlite3


class SaveMidiData:
    """这个模块的用处是将midi数据转换格式保存在sql文件中"""

    melody_marks = ['main', 'intro', 'interlude']

    def __init__(self, song_info_list, skip_list):
        self.music_data = dict()  # 整理好的music_data就在这个列表中 列表一共有4维：第一维是歌的编号 第二维是存储类型 第三维是小节序列 第四维是小节中每个音的音高
        self.chord_data = dict()  # 整理好的和弦列表。列表一共有3维，第一维是歌的编号，第二维是小节序列，第三维是小节中的和弦列表
        self.melody_data = dict()  # 整理好的主旋律列表。列比一共有三维 第一维是歌的编号 第二维是小节序列，第三维是小节中主旋律音符的列表
        self.note_dict = [[-1]]  # 音符的词典 即将音符及其组合整理成dict的形式。把第0项空出来，第0项是休止符。

        # 1.变量定义及其初始化
        scale_list = [None for t0 in range(TRAIN_FILE_NUMBERS)]
        tone_list = [None for t1 in range(TRAIN_FILE_NUMBERS)]
        bpm_list = [None for t2 in range(TRAIN_FILE_NUMBERS)]
        bias_time_list = [None for t3 in range(TRAIN_FILE_NUMBERS)]
        # 2.获取所有歌曲的信息
        for song_it in range(TRAIN_FILE_NUMBERS):
            if song_info_list[song_it] is not None:
                scale_list[song_it] = song_info_list[song_it]['scale']
                bpm_list[song_it] = song_info_list[song_it]['bpm']
                tone_list[song_it] = song_info_list[song_it]['tone']
                bias_time_list[song_it] = song_info_list[song_it]['bias']
        # 3.读取所有的文件并将里面的音符转化为可训练的形式
        for file_it in range(TRAIN_FILE_NUMBERS):
            if file_it not in skip_list:
                self.music_data[file_it] = dict()  # 初始化训练数据的第一维。注意：第一维的下标是从0开始的，而不是从1开始的。因为python的字典不能同时初始化两维
                file_path = '../Inputs/%02d/%03d.mid' % (ACTIVE_MUSIC_TYPE, file_it + 1)
                DiaryLog.warn('正在整理第%d首歌' % file_it)
                pianoroll_list = generate_data_from_midi_file(file_path, bias_time_list[file_it], scale_list[file_it])
                self.get_music_data(file_it, pianoroll_list,
                                    time_step_dic={'piano_guitar': 0.25, 'string': 0.25},
                                    eliminate=['main', 'intro', 'interlude', 'chord', 'others'])
                # 把main/intro/interlude和chord去掉意思就是主旋律和和弦部分单独处理
                self.get_melody_data(file_it, pianoroll_list)
                if 'chord' in pianoroll_list:  # 只有在这首歌有和弦的时候才训练和弦
                    self.get_chord_data(file_it, pianoroll_list, tone=tone_list[file_it])
        # 4.把这些音符存储在sqlite中
        self.save_music_data()
        self.save_chord_data()
        self.save_melody_data()

    def get_music_data(self, music_file_dx, pianoroll_dic, default_time_step=0.125, time_step_dic=None, eliminate=None):
        """
        将音符组合保存为以time_step拍为步长的数组，方便存储到sql中
        :param float default_time_step: 如果key不在time_step_dict中 默认的time_step是多少
        :param time_step_dic: 时间步长
        :param music_file_dx: Midi文件的编号
        :param pianoroll_dic: 音符列表
        :param eliminate: 那些音轨里的音符不保存
        :return:
        """
        if time_step_dic is None:
            time_step_dic = dict()
        for key in pianoroll_dic:
            time_step = default_time_step
            for time_step_key in time_step_dic:  # 从0.92.05开始 因为各项标注的音符time_step不全为1/8 所以加入time_step_dict
                if time_step_key in key:  # 比如time_step_key'piano_guitar'完全包含在key'piano_guitar1'中
                    time_step = time_step_dic[time_step_key]
            if key not in eliminate:
                if not pianoroll_dic[key]:  # 这个音轨为空 那么直接跳过这个音轨
                    continue
                self.music_data[music_file_dx][key] = []
                bar_num = int(pianoroll_dic[key][-1][0] // 4) + 1  # 小节数
                cur_note_dx = 0  # 当前音符
                # 1.逐小节读取数据
                for bar_it in range(bar_num):
                    bar_data = [0 for t in range(round(4 / time_step))]  # 一小节4拍 每拍8个音符 休止符记为0
                    # 1.1.对每一个时间步长读取音符并保存在bar_data中
                    for note_step_it in range(round(4 / time_step)):
                        # 1.1.1.读取这个时间步长中所有的音符 保存在列表中
                        raw_step_data = set()  # 将同一个时间上所有的音都存在这里。这里永set而不是list是因为要去除相同音符
                        if cur_note_dx < len(pianoroll_dic[key]):
                            while pianoroll_dic[key][cur_note_dx][0] <= bar_it * 4 + (note_step_it + 0.5) * time_step:
                                raw_step_data.add(int(pianoroll_dic[key][cur_note_dx][1]))  # 只保存音高 且为整形
                                assert abs(pianoroll_dic[key][cur_note_dx][1] - int(pianoroll_dic[key][cur_note_dx][1])) <= 0.1
                                cur_note_dx += 1
                                if cur_note_dx >= len(pianoroll_dic[key]):
                                    break  # 超过数组长度则退出
                        # 1.1.2.将列表编码为一个整数并保存在bar_data中
                        if len(raw_step_data) != 0:  # 如果这个时间步长中没有音符的话，就不用向下执行了
                            raw_step_data = list(raw_step_data)
                            raw_step_data.sort()  # 由小到大排序 防止出现[1,3]和[3,1]被视作不同元素的情况
                            if raw_step_data in self.note_dict:
                                step_data_dx = self.note_dict.index(raw_step_data)  # 检查这个音符组合有没有被保存
                            else:
                                self.note_dict.append(raw_step_data)  # 添加这个音符组合
                                step_data_dx = len(self.note_dict) - 1
                            bar_data[note_step_it] = step_data_dx  # 将这个音符保存起来
                    # 1.2.将这个小节的音符信息存储在music_data中
                    self.music_data[music_file_dx][key].append(bar_data)

    def get_melody_data(self, music_file_dx, pianoroll_dic):
        """
        （注：如果同一个时间步长中存在多个音符 取其最高者）
        :param music_file_dx: 歌的编号
        :param pianoroll_dic: pianoroll的列表
        :return: 无
        """
        self.melody_data[music_file_dx] = {mark: [] for mark in self.melody_marks}
        for mark in self.melody_marks:
            if mark not in pianoroll_dic:  # 这个音轨为空 那么直接跳过这个音轨
                continue
            if not pianoroll_dic[mark]:
                continue
            bar_num_decimal, bar_num_int = math.modf(pianoroll_dic[mark][-1][0] / 4)  # 如果最后一个音符在某小节的最后六十四分之一拍的区段内的话，它其实属于下一个小节，因此这时小节数要加一
            if bar_num_decimal >= 63 / 64:
                bar_num = int(bar_num_int + 2)
            else:
                bar_num = int(bar_num_int + 1)
            cur_note_dx = 0  # 当前音符
            # 1.逐小节读取数据
            for bar_it in range(bar_num):
                bar_melody_data = [0 for t in range(32)]  # 一小节4拍 每拍八个音符 所以一小节32个音符
                # 1.1.对每一个时间步长读取音符并保存在bar_data中
                for note_step_it in range(32):
                    # 1.1.1.读取这个时间步长中所有的音符 保存在列表中
                    cur_step_pitch = 0  # 只保存这个时间步长中音高最高的音符
                    if cur_note_dx < len(pianoroll_dic[mark]):
                        while pianoroll_dic[mark][cur_note_dx][0] <= bar_it * 4 + (note_step_it + 0.5) * 0.125:  # 处理一个时间区段内的所有音符
                            if pianoroll_dic[mark][cur_note_dx][1] > cur_step_pitch:  # 如果这个音符的音高1高于这个时间步长的最高音 那么替换这个时间步长的最高音
                                cur_step_pitch = pianoroll_dic[mark][cur_note_dx][1]
                            cur_note_dx += 1
                            if cur_note_dx >= len(pianoroll_dic[mark]):
                                break  # 超过数组长度则退出
                    # 1.1.2.将这个音符保存到bar_melody_data中
                    if cur_step_pitch != 0:
                        bar_melody_data[note_step_it] = int(cur_step_pitch)  # 音符保存时必须为整形
                        assert abs(cur_step_pitch - int(cur_step_pitch)) <= 0.1
                # 1.2.将这个小节的音符信息存储在train_data中
                self.melody_data[music_file_dx][mark].append(bar_melody_data)

    def get_chord_data(self, music_file_dx, pianoroll_dic, tone=DEF_TONE_MAJOR):
        self.chord_data[music_file_dx] = []
        bar_number = int(pianoroll_dic['chord'][-1][0] // 4) + 1  # 小节数
        cur_note_dx = 0  # 当前音符
        saved_chord = 0  # 保存上一拍的和弦
        # 逐小节读取和弦数据
        for bar_it in range(bar_number):
            bar_chord_data = [0 for t in range(4)]  # 一小节4拍 每拍1个和弦 比较奇怪的未知和弦记为0
            raw_bar_data = [set() for t in range(4)]  # 将同一个时间上所有的音都存在这里。这里永set而不是list是因为要去除相同音符
            # 1.对这个小节的每一拍读取音符并保存在bar_data中
            for beat_it in range(4):
                # 1.1.读取这个时间步长中所有的音符 保存在列表中
                if cur_note_dx < len(pianoroll_dic['chord']):
                    while pianoroll_dic['chord'][cur_note_dx][0] <= bar_it * 4 + beat_it + 0.9375:  # 处理一个时间区段内的所有音符
                        raw_bar_data[beat_it].add(int(pianoroll_dic['chord'][cur_note_dx][1]))  # 只保存音高
                        assert abs(pianoroll_dic['chord'][cur_note_dx][1] - int(pianoroll_dic['chord'][cur_note_dx][1])) <= 0.1
                        cur_note_dx += 1
                        if cur_note_dx >= len(pianoroll_dic['chord']):
                            break  # 超过数组长度则退出
            # 2.将音符列表装化为对应和弦
            for beat_it in range(4):
                # 2.1.读取这个时间步长的陪伴音符列表（即对于第2个时间步长，读取第一个时间步长的音符列表，对于第1个时间步长则读取第二个时间步长的音符列表）
                accompany_beat = int(not beat_it % 2) + 2 * (beat_it // 2)
                # 2.2.将音符列表装化为对应和弦
                if len(raw_bar_data[beat_it]) == 0:  # 这个时间区段没有音符 和弦将保存为上一个时间区段的和弦
                    bar_chord_data[beat_it] = saved_chord
                else:
                    if len(raw_bar_data[accompany_beat]) == 0:  # 陪伴音符列表为空
                        saved_chord = noteset2chord(raw_bar_data[beat_it], saved_chord, tone, None)  # 音符列表转换为和弦 并将它保存起来
                    else:  # 陪伴音符列表不为空
                        saved_chord = noteset2chord(raw_bar_data[beat_it], saved_chord, tone, raw_bar_data[accompany_beat])  # 音符列表转换为和弦 并将它保存起来
                    bar_chord_data[beat_it] = saved_chord
            # 3.将这个小节的音符信息存储在chord_data中
            self.chord_data[music_file_dx].append(bar_chord_data)

    def save_music_data(self):
        # 1.新建小节列表
        # 小节列表中有五个元素 id是主键，bar_index是该小节在歌中的位置，mark是该小节的标注(比如'main')，data是小节的内容，song_id是歌曲的id（外键）
        conn = sqlite3.connect(PATH_RAW_DATASET)
        conn.execute('drop table if exists BarInfo')
        conn.commit()
        conn.execute('pragma foreign_key=ON')
        conn.execute('create table BarInfo(id integer primary key autoincrement, bar_index integer, mark varchar(20), data text, song_id integer, foreign key(song_id) references SongInfo(id) on update cascade)')
        # 2.往小节列表中添加数据
        for song_id_key in self.music_data:
            # if song_id_key == 0:
            for mark_key in self.music_data[song_id_key]:
                for bar_it in range(len(self.music_data[song_id_key][mark_key])):
                    # 注：存储在数据库中时，歌的id从1开始编号，而在程序中，歌的id从0开始编号
                    conn.execute('insert into BarInfo(bar_index,mark,data,song_id) values (%d,\'%s\',\'%s\',%d)' % (bar_it, mark_key, repr(self.music_data[song_id_key][mark_key][bar_it]), song_id_key + 1))
        conn.commit()
        # 3.新建音符字典列表
        conn.execute('drop table if exists NoteDict')
        conn.commit()
        conn.execute('create table NoteDict(id integer primary key autoincrement, note_group text)')
        # print(self.note_dict)
        # 4.往音符字典列表添加数据
        for note_group in self.note_dict:
            conn.execute('insert into NoteDict(note_group) values (\'%s\')' % repr(note_group))
        conn.commit()

    def save_chord_data(self):
        # 由于小节列表在前面的save_train_data已经新建过了 因此这里不需要重新建表
        conn = sqlite3.connect(PATH_RAW_DATASET)
        # 往小节列表中添加数据
        for song_id_key in self.chord_data:
            for bar_it in range(len(self.chord_data[song_id_key])):
                # 注：存储在数据库中时，歌的id从1开始编号，而在程序中，歌的id从0开始编号
                conn.execute('insert into BarInfo(bar_index,mark,data,song_id) values (%d,\'chord\',\'%s\',%d)' % (bar_it, repr(self.chord_data[song_id_key][bar_it]), song_id_key + 1))
        conn.commit()

    def save_melody_data(self):
        # 由于小节列表在前面的save_train_data已经新建过了 因此这里不需要重新建表
        conn = sqlite3.connect(PATH_RAW_DATASET)
        # 往小节列表中添加数据
        for song_id_key in self.melody_data:
            for melody_mark in self.melody_marks:
                if not self.melody_data[song_id_key][melody_mark]:
                    continue  # 没有这个内容 直接continue
                for bar_it in range(len(self.melody_data[song_id_key][melody_mark])):
                    # 注：存储在数据库中时，歌的id从1开始编号，而在程序中，歌的id从0开始编号
                    conn.execute('insert into BarInfo(bar_index,mark,data,song_id) values (%d,\'%s\',\'%s\',%d)' % (bar_it, melody_mark, repr(self.melody_data[song_id_key][melody_mark][bar_it]), song_id_key + 1))
        conn.commit()
