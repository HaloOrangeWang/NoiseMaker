from settings import *
import sqlite3
import copy
from interfaces.utils import flat_array


class BaseMusicPatterns:
    """音符组合的基类,包含了把 音符组合/音符组合计数/各个训练用曲的音符组合 信息保存到数据库中 以及从数据库中恢复这些数据"""

    def __init__(self, store_count=True):
        self.common_pattern_list = []  # 常见的组合列表
        self.pattern_number_list = []  # 组合出现的次数列表
        self.flag_store_count = store_count  # 是否保存计数数据

    def store(self, table_mark):
        """
        将常见组合的数据保存到sqlite文件中
        :param table_mark: 数据表的标签
        """
        # 1.建表
        conn = sqlite3.connect(PATH_PAT_DATASET)
        conn.execute('drop table if exists %sPatterns' % table_mark)
        if self.flag_store_count:
            conn.execute('drop table if exists %sPatternNumber' % table_mark)
        conn.execute('create table %sPatterns(id integer primary key autoincrement, pattern text)' % table_mark)
        if self.flag_store_count:
            conn.execute('create table %sPatternNumber(id integer primary key autoincrement, number integer)' % table_mark)
        conn.commit()
        # 2.保存数据
        for pattern_it, pattern in enumerate(self.common_pattern_list):
            conn.execute("insert into %sPatterns(pattern) values ('%s')" % (table_mark, repr(pattern)))
            if self.flag_store_count:
                conn.execute("insert into %sPatternNumber(number) values (%d)" % (table_mark, self.pattern_number_list[pattern_it]))
            conn.commit()

    def restore(self, table_mark):
        """
        从sqlite文件中获取常见的组合列表
        :param table_mark: 数据表的标签
        """
        conn = sqlite3.connect(PATH_PAT_DATASET)
        if not hasattr(self, 'pattern_number'):  # 没有pattern_number方法 从数据库中获取pattern的数量
            # 这里减一是因为从数据表中读取的common_pattern的数量比common_pattern_number要多一项（多了第0项空值）
            self.pattern_number = list(conn.execute('select count(id) from %sPatterns' % table_mark))[0][0] - 1
        self.common_pattern_list = [None for t in range(self.pattern_number + 1)]
        self.pattern_number_list = [None for t in range(self.pattern_number + 1)]
        rows = conn.execute('select * from %sPatterns' % table_mark)
        for row in rows:
            self.common_pattern_list[row[0] - 1] = eval(row[1])
        if self.flag_store_count:
            rows = conn.execute('select * from %sPatternNumber' % table_mark)
            for row in rows:
                self.pattern_number_list[row[0] - 1] = row[1]


class CommonMusicPatterns(BaseMusicPatterns):
    """获取最常见的number种旋律组合"""

    def __init__(self, pattern_number):
        super().__init__(store_count=True)
        self.pattern_number = pattern_number

    def train(self, raw_music_data, note_time_step, pattern_time_step, multipart=False):
        """
        获取最常见的number种旋律组合
        :param multipart: 一首歌的一种表现手法是否有多个part
        :param raw_music_data: 一首歌一个音轨的数据
        :param note_time_step: 音符的时间步长
        :param pattern_time_step: 旋律组合的时间步长
        :return: 最常见的旋律组合
        """
        # :param unit: 以小节为单元还是以音符步长为单元
        common_pattern_dic = {}
        time_step_ratio = round(pattern_time_step / note_time_step)
        # 1.准备：如果一首歌的这种表现手法是否有多个part，将数据降一阶
        if multipart is False:  # 一首歌的这种表现手法是否有多个part,
            music_data = copy.deepcopy(raw_music_data)
        else:
            music_data = flat_array(raw_music_data)
        # 2.将所有的歌的这种表现手法以一定的步长分组，对每一组的音符组合进行编码，并计数
        for song_it in range(len(music_data)):  # 注意这里不是TRAIN_FILE_NUMBERS，因为可能是multipart
            if len(music_data[song_it]) != 0:
                # if unit == 'bar':  # 将以小节为单元的数组转成以音符时间步长为单元的数组
                #     music_data[song_it] = flat_array(music_data[song_it])
                beat_num = len(music_data[song_it]) * note_time_step  # 这首歌有多少拍
                try:
                    raw_data_in_1pat = [music_data[song_it][time_step_ratio * t: time_step_ratio * (t + 1)] for t in range(round(beat_num / pattern_time_step))]  # 将这首歌以pattern_time_step为步长进行分割
                    for music_pattern in raw_data_in_1pat:
                        try:
                            common_pattern_dic[str(music_pattern)] += 1
                        except KeyError:
                            common_pattern_dic[str(music_pattern)] = 1
                except KeyError:
                    pass
        # 3.截取最常见的多少种组合。按照组合的出现次数由高到底排序
        common_pattern_list_temp = sorted(common_pattern_dic.items(), key=lambda asd: asd[1], reverse=True)
        self.not_empty_pat_cnt = 0
        self.common_pat_cnt = 0
        for t in common_pattern_list_temp[1:]:
            self.not_empty_pat_cnt += t[1]
        for t in common_pattern_list_temp[1: (self.pattern_number + 1)]:
            self.common_pat_cnt += t[1]
        self.common_pattern_list = []
        self.pattern_number_list = []
        for pattern_tuple in common_pattern_list_temp[:(self.pattern_number + 1)]:
            self.common_pattern_list.append(eval(pattern_tuple[0]))
            self.pattern_number_list.append(pattern_tuple[1])


class MusicPatternEncode(object):
    """
    对于以音符时间步长为单位的音符组合，将原始音符列表转化为音符组合列表
    0.96.01: 去掉“对于以小节为单元的音符组合”这种变量保存方式
    """
    def __init__(self, common_patterns, music_data_list, note_time_step, pattern_time_step):
        time_step_ratio = round(pattern_time_step / note_time_step)
        raw_note_list = [music_data_list[time_step_ratio * t: time_step_ratio * (t + 1)] for t in range(len(music_data_list) // time_step_ratio)]  # 将音符列表按照pattern_time_step进行分割 使其变成二维数组
        self.music_pattern_list = [0 for t in range(len(music_data_list) // time_step_ratio)]  # 按照最常见的音符组合编码之后的组合列表（list形式）
        for step_it in range(0, len(raw_note_list)):
            pattern_dx = self.handle_common_patterns(raw_note_list[step_it], common_patterns)
            if pattern_dx == -1:
                pattern_dx = self.handle_rare_pattern(step_it, raw_note_list[step_it], common_patterns)
            self.music_pattern_list[step_it] = pattern_dx  # 将编码后的pattern list保存在新的pattern dict中

    @staticmethod
    def handle_common_patterns(raw_note_list, common_patterns):
        try:
            pattern_dx = common_patterns.index(raw_note_list)  # 在常见的组合列表中找到这个音符组合
        except ValueError:  # 找不到
            pattern_dx = -1
        return pattern_dx

    def handle_rare_pattern(self, pat_step_dx, raw_note_list, common_patterns):
        return len(common_patterns)


def music_pattern_decode(common_patterns, pattern_list, note_time_step, pattern_time_step):
    """
    音符组合解码。将音符组合解码为音符列表，时间步长为pattern_time_step
    :param pattern_time_step: 音符组合的时间步长
    :param note_time_step: 音符的时间步长
    :param pattern_list: 音符组合列表
    :param common_patterns: 常见的音符组合
    :return: 以note_time_step为时间步长的音符列表
    """
    note_list = []
    time_step_radio = round(pattern_time_step / note_time_step)
    for pattern in pattern_list:
        if pattern == 0 or pattern >= len(common_patterns):
            note_list = note_list + [0 for t in range(time_step_radio)]
        else:
            note_list = note_list + common_patterns[pattern]
    return note_list
