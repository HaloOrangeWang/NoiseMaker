from settings import *
from interfaces.sql.sqlite import GetRawSongDataFromDataset
from interfaces.functions import CommonMusicPatterns, MusicPatternEncode, GetDictMaxKey
from models.KMeansModel import KMeansModel
import copy


def GetMelodyAverageNoteByBar(raw_melody_data):
    """
    获取歌曲小节内音符的平均音高，以2小节为单位。空小节的平均音高为0
    :param raw_melody_data: 一首歌曲的音符列表。dict形式
    :return: 歌曲小节内音符的平均音高 二维数组第一维用空小节分割
    """
    max_key = -1
    for key in raw_melody_data:
        if key > max_key:
            max_key = key
    if max_key == -1:
        return []  # 这首歌没有主旋律 返回空列表
    average_note_list = [[]]
    key = 0
    while 1:
        if key > max_key:
            break
        if key == max_key:
            average_note_list[-1].append(sum(raw_melody_data[key])/(round(4 / MELODY_TIME_STEP) - raw_melody_data[key].count(0)))
            break
        try:
            if raw_melody_data[key] == [0 for t in range(round(4 / MELODY_TIME_STEP))]:  # 遇到空小节
                if average_note_list[-1]:
                    average_note_list.append([])
                key += 1
                continue
            elif raw_melody_data[key] != [0 for t in range(round(4 / MELODY_TIME_STEP))] and raw_melody_data[key + 1] == [0 for t in range(round(4 / MELODY_TIME_STEP))]:  # 当前小节不为空但是下一小节为空
                average_note_list[-1].append(sum(raw_melody_data[key])/(round(4 / MELODY_TIME_STEP) - raw_melody_data[key].count(0)))
                average_note_list.append([])
                key += 2
            elif raw_melody_data[key] != [0 for t in range(round(4 / MELODY_TIME_STEP))] and raw_melody_data[key + 1] != [0 for t in range(round(4 / MELODY_TIME_STEP))]:  # 这一小节与下一小节均不为空
                average_note_list[-1].append((sum(raw_melody_data[key]) + sum(raw_melody_data[key + 1]))/(round(8 / MELODY_TIME_STEP) - raw_melody_data[key].count(0) - raw_melody_data[key + 1].count(0)))
                key += 2
        except KeyError:
            key += 1
    if not average_note_list[-1]:
        average_note_list.pop(-1)
    # print(melody_cluster)
    return average_note_list


def GetMelodyProfileBySong(session, kmeans_model, cluster_center_points, raw_melody_data):
    # 这个函数的前半部分与GetMelodyClusterByBar有一些区别 所以不能直接调用
    # 1.准备工作：记录歌曲有多少个小节，以及定义一些变量
    max_key = -1
    for key in raw_melody_data:
        if key > max_key:
            max_key = key
    if max_key == -1:
        return []  # 这首歌没有主旋律 返回空列表
    melody_avr_note_list = [0 for t in range(max_key + 1)]  # 将这首歌逐两小节的音符平均音高记录在这个列表中
    key = 0
    # 2.逐2小节地记录一个区间的平均音高
    while 1:
        if key > max_key:
            break
        if key == max_key:  # 已达最后一小节
            melody_avr_note_list[key] = (sum(raw_melody_data[key])/(round(4 / MELODY_TIME_STEP) - raw_melody_data[key].count(0)))
            break
        try:
            if raw_melody_data[key] == [0 for t in range(round(4 / MELODY_TIME_STEP))]:  # 这一小节为空 不记录
                key += 1
                continue
            elif raw_melody_data[key] != [0 for t in range(round(4 / MELODY_TIME_STEP))] and raw_melody_data[key + 1] == [0 for t in range(round(4 / MELODY_TIME_STEP))]:  # 这一小节不为空 但下一小节为空
                melody_avr_note_list[key] = (sum(raw_melody_data[key])/(round(4 / MELODY_TIME_STEP) - raw_melody_data[key].count(0)))
                key += 2
            elif raw_melody_data[key] != [0 for t in range(round(4 / MELODY_TIME_STEP))] and raw_melody_data[key + 1] != [0 for t in range(round(4 / MELODY_TIME_STEP))]:  # 这一小节和下一小节均不为空
                melody_avr_note_list[key] = melody_avr_note_list[key + 1] = ((sum(raw_melody_data[key]) + sum(raw_melody_data[key + 1]))/(round(8 / MELODY_TIME_STEP) - raw_melody_data[key].count(0) - raw_melody_data[key + 1].count(0)))
                key += 2
        except KeyError:
            key += 1
    # print(melody_avr_note_list)
    # print(melody_avr_note_list)
    # 2.通过K均值算法将音高列表分类
    attachment_array = kmeans_model.run_attachment(session, cluster_center_points, melody_avr_note_list)  # 每个小节的分类情况
    # 3.将音高分类情况进行微调 如果小节旋律为空 则分类为0 否则分类为1-10
    for bar_iterator in range(max_key + 1):
        if melody_avr_note_list[bar_iterator] != 0:
            attachment_array[bar_iterator] += 1
        else:
            attachment_array[bar_iterator] = 0
    # print(attachment_array)
    return attachment_array


def DefineMelodyCluster(tone_restrict=None, train=True):
    # 1.从数据集中读取歌的编号为song_id且小节标注为main的小节数据
    raw_melody_data = GetRawSongDataFromDataset('main', tone_restrict)
    # 2.逐2小节地记录这首歌一个区间的平均音高
    melody_avr_list = []
    for song_iterator in range(len(raw_melody_data)):
        if raw_melody_data[song_iterator] != {}:
            melody_avr_list = melody_avr_list + GetMelodyAverageNoteByBar(raw_melody_data[song_iterator])
    # for t in range(len(melody_avr_list)):
    #     print(melody_avr_list[t])
    flatten_melody_avr_list = []
    for melody_avr in melody_avr_list:
        flatten_melody_avr_list = flatten_melody_avr_list + melody_avr
    # 3.使用K均值算法，对输入的音乐按照两小节的平均音高分为10类
    # sess = tf.Session()
    kmeans_model = KMeansModel(flatten_melody_avr_list, 10, 50, train)
    return kmeans_model


def CalMelodyCluster(kmeans_model, sess):
    cluster_center_points = kmeans_model.run(sess)
    return cluster_center_points


class MelodyPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, bar_index, bar_pattern_list, raw_bar_pattern_list, common_patterns):
        for bar_melody_pattern_iterator in range(len(bar_pattern_list)):
            if bar_pattern_list[bar_melody_pattern_iterator] == -1:
                assert MELODY_TIME_STEP == 1 / 8  # 这个方法只有在旋律音符的最小时间步长为1/8拍时才适用
                # 在常见的旋律列表里找不到某一个旋律组合的处理方法：
                # a.寻找符合以下条件的旋律组合
                # a1.首音不为休止符
                # a2.该旋律组合的首音/半拍音与待求旋律组合的首音/半拍音相同
                # a3.该旋律组合中所有音符与待求旋律组合对应位置的音符全部相同
                # a4.满足上述三个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
                # b.记为0
                choose_pattern = 0  # 选取的pattern
                choose_pattern_like_score = 0  # 两个旋律组合的相似程度
                for common_melody_iterator in range(1, len(common_patterns)):
                    # 1.检查首音是否为休止符
                    if common_patterns[common_melody_iterator][0] == 0:
                        continue
                    # 2.检查两个旋律组合的首音是否相同
                    if common_patterns[common_melody_iterator][0] != raw_bar_pattern_list[bar_melody_pattern_iterator][0]:
                        continue
                    # 3.检查该旋律组合中所有音符与待求旋律组合对应位置的音符是否全部相同
                    note_all_same = True
                    for note_iterator in range(len(common_patterns[common_melody_iterator])):
                        if common_patterns[common_melody_iterator][note_iterator] != 0 and common_patterns[common_melody_iterator][note_iterator] != raw_bar_pattern_list[bar_melody_pattern_iterator][note_iterator]:
                            note_all_same = False
                            break
                    if not note_all_same:
                        continue
                    # 4.求该旋律组合与待求旋律组合的差别
                    pattern_like_score = 6  # 初始的旋律组合相似度为6分 每发现一个不同音符 按权重扣分
                    note_diff_list = [10, 2, 3, 3, 6, 3, 4, 3]  # 音符差别的权重列表
                    for note_iterator in range(len(common_patterns[common_melody_iterator])):
                        if common_patterns[common_melody_iterator][note_iterator] != raw_bar_pattern_list[bar_melody_pattern_iterator][note_iterator]:
                            pattern_like_score -= note_diff_list[note_iterator]
                    # 5.如果这个旋律组合的差别是目前最小的 则保存它
                    # print(common_melody_iterator, pattern_like_score)
                    if pattern_like_score > choose_pattern_like_score:
                        choose_pattern_like_score = pattern_like_score
                        choose_pattern = common_melody_iterator
                # 6.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_MELODY_PATTERNS+1
                if choose_pattern_like_score > 0:
                    bar_pattern_list[bar_melody_pattern_iterator] = choose_pattern
                else:
                    bar_pattern_list[bar_melody_pattern_iterator] = len(common_patterns)
        return bar_pattern_list


def GetContinuousBarNumber(melody_data_dict):
    # 1.获取歌曲的小节数量
    max_key = 0
    for key in melody_data_dict:
        if key > max_key:
            max_key = key
    continuous_bar_number_list = [0 for t in range(max_key + 1)]
    # 2.获取歌曲连续小节编号
    for key in range(max_key + 1):
        try:
            if melody_data_dict[key] == [0 for t in range(round(4 / MELODY_TIME_STEP))]:
                continuous_bar_number_list[key] = 0
            elif key == 0:
                continuous_bar_number_list[key] = 1
            else:
                continuous_bar_number_list[key] = continuous_bar_number_list[key - 1] + 1
        except KeyError:
            continuous_bar_number_list[key] = 0
    return continuous_bar_number_list


class MelodyTrainData:

    input_data = []  # 输入model的数据
    output_data = []  # 从model输出的数据

    def __init__(self, tone_restrict=None):
        # 1.从数据集中读取歌的编号为song_id且小节标注为mark_key的小节数据
        self.raw_train_data = GetRawSongDataFromDataset('main', tone_restrict)
        no_tone_restrict_melody_data = GetRawSongDataFromDataset('main', None)  # 没有旋律限制的主旋律数据　用于训练其他数据
        self.raw_melody_data = copy.deepcopy(no_tone_restrict_melody_data)  # 最原始的主旋律数据
        self.continuous_bar_number_data = [[] for t in range(TRAIN_FILE_NUMBERS)]
        self.no_tone_restrict_continuous_bar_number_data = [[] for t in range(TRAIN_FILE_NUMBERS)]
        self.keypress_pattern_dict = [[0 for t in range(16)]]
        self.keypress_pattern_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 三维数组 第一维是歌曲列表 第二维是小节编号 第三维是按键的组合（步长是2拍）
        self.keypress_pattern_count = [0]
        # 2.获取最常见的主旋律组合
        self.common_melody_patterns, self.melody_pattern_number_list = CommonMusicPatterns(self.raw_train_data, number=COMMON_MELODY_PATTERN_NUMBER, note_time_step=MELODY_TIME_STEP, pattern_time_step=MELODY_PATTERN_TIME_STEP)
        # 3.生成输入输出数据
        for song_iterator in range(len(self.raw_train_data)):
            if no_tone_restrict_melody_data[song_iterator] != {}:  # 获取相关没有调式限制的相关数据
                self.get_keypress_data(song_iterator, no_tone_restrict_melody_data[song_iterator])  # 获取按键数据 当前有按键记为1 没有按键记为0
                self.no_tone_restrict_continuous_bar_number_data[song_iterator] = GetContinuousBarNumber(no_tone_restrict_melody_data[song_iterator])
                no_tone_restrict_melody_data[song_iterator] = MelodyPatternEncode(self.common_melody_patterns, no_tone_restrict_melody_data[song_iterator], MELODY_TIME_STEP, MELODY_PATTERN_TIME_STEP).music_pattern_dict
            if self.raw_train_data[song_iterator] != {}:
                # 3.1.开头补上几个空小节 便于训练开头几小节的旋律数据
                for bar_iterator in range(1 - TRAIN_MELODY_IO_BARS, 0):
                    self.raw_train_data[song_iterator][bar_iterator] = [0 for t in range(round(4 / MELODY_TIME_STEP))]
                # 3.2.获取歌曲的连续不为空的小节序号列表
                self.continuous_bar_number_data[song_iterator] = GetContinuousBarNumber(self.raw_train_data[song_iterator])
                # print(melody_profile_data)
                # 3.4.将它的主旋律编码为常见的旋律组合。如果该旋律组合不常见，则记为COMMON_MELODY_PATTERN_NUMBER+1
                self.raw_train_data[song_iterator] = MelodyPatternEncode(self.common_melody_patterns, self.raw_train_data[song_iterator], MELODY_TIME_STEP, MELODY_PATTERN_TIME_STEP).music_pattern_dict
        self.melody_pattern_data = self.raw_train_data
        self.no_tone_restrict_melody_pattern_data = no_tone_restrict_melody_data
        # print(len(self.input_data), len(self.output_data))
        # print('\n\n\n\n\n')
        # for t in self.input_data[:50]:
        #     print(t)
        # print('\n\n\n')
        # for t in self.output_data[:50]:
        #     print(t)

    def get_keypress_data(self, song_iterator, melody_data):
        for key in range(GetDictMaxKey(melody_data) + 1):
            self.keypress_pattern_data[song_iterator].append([0, 0])
            bar_keypress_data = [1 if t != 0 else 0 for t in melody_data[key]]
            raw_pattern_list = [bar_keypress_data[16 * t: 16 * (t + 1)] for t in range(2)]
            for bar_pattern_iterator, raw_pattern in enumerate(raw_pattern_list):
                if raw_pattern not in self.keypress_pattern_dict:
                    self.keypress_pattern_data[song_iterator][key][bar_pattern_iterator] = len(self.keypress_pattern_dict)
                    self.keypress_pattern_dict.append(raw_pattern)
                    self.keypress_pattern_count.append(1)
                else:
                    self.keypress_pattern_data[song_iterator][key][bar_pattern_iterator] = self.keypress_pattern_dict.index(raw_pattern)
                    self.keypress_pattern_count[self.keypress_pattern_dict.index(raw_pattern)] += 1

    def get_total_io_data(self, session, kmeans_model, cluster_center_points):
        for song_iterator in range(len(self.raw_train_data)):
            if self.raw_train_data[song_iterator] != {}:
                # 3.3.使用K均值算法，对输入的音乐按照两小节的平均音高分为10类
                melody_profile_data = GetMelodyProfileBySong(session, kmeans_model, cluster_center_points, self.raw_train_data[song_iterator])
                # 3.5.生成训练数据 输入内容是当前时间 五小节的melody_profile 过去16拍的旋律组合，输出内容是当前时间 五小节的melody_profile 和错后一拍的旋律组合
                self.get_model_io_data(self.raw_train_data[song_iterator], self.continuous_bar_number_data[song_iterator], melody_profile_data)

    def get_model_io_data(self, melody_pattern_data, continuous_bar_number_data, melody_profile_data):
        """
        在完成数据的前期处理（读取/转换等）之后，接下来就是提取有效数据输入到model中了
        :param melody_pattern_data: 一首歌旋律组合的数据
        :param continuous_bar_number_data: 一首歌主旋律连续不为空的小节列表
        :param melody_profile_data: 一首歌逐小节的melody profile列表
        :return:
        """
        for key in melody_pattern_data:
            for time_in_bar in range(4):
                try:
                    # 1.添加当前时间
                    time_add = (1 - continuous_bar_number_data[key + TRAIN_MELODY_IO_BARS] % 2) * 4  # 这个时间在2小节之中的位置
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 2.添加melody_profile 向前回溯4小节
                    if key >= 0:
                        input_time_data += melody_profile_data[key: key + TRAIN_MELODY_IO_BARS + 1]  # 添加melody_profile
                        output_time_data += melody_profile_data[key: key + TRAIN_MELODY_IO_BARS + 1]
                    else:
                        input_time_data += [0 for t in range(key, 0)] + melody_profile_data[0: key + TRAIN_MELODY_IO_BARS + 1]
                        output_time_data += [0 for t in range(key, 0)] + melody_profile_data[0: key + TRAIN_MELODY_IO_BARS + 1]
                    # 3.添加当前小节的旋律组合
                    input_time_data += melody_pattern_data[key][time_in_bar: 4]
                    output_time_data += melody_pattern_data[key][(time_in_bar + 1): 4]  # output_Data错后一拍
                    # 4.添加之后3个小节的旋律组合
                    for bar_iterator in range(key + 1, key + TRAIN_MELODY_IO_BARS):
                        input_time_data = input_time_data + melody_pattern_data[bar_iterator]
                        output_time_data = output_time_data + melody_pattern_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[0: (1 + TRAIN_MELODY_IO_BARS + 1)] + [0 for t in range(len(input_time_data) - (1 + TRAIN_MELODY_IO_BARS + 1))]
                            output_time_data = output_time_data[0: (1 + TRAIN_MELODY_IO_BARS + 1)] + [0 for t in range(len(output_time_data) - (1 + TRAIN_MELODY_IO_BARS + 1))]
                    # 5.添加最后一个小节的旋律组合
                    input_time_data += melody_pattern_data[key + TRAIN_MELODY_IO_BARS][0: time_in_bar]
                    output_time_data += melody_pattern_data[key + TRAIN_MELODY_IO_BARS][0: (time_in_bar + 1)]
                    # 6.当输出数据所在的小节与其前一小节均不为空时，该数据收录进训练集
                    if melody_pattern_data[key + TRAIN_MELODY_IO_BARS] != [0 for t in range(4)] and melody_pattern_data[key + TRAIN_MELODY_IO_BARS - 1] != [0 for t in range(4)]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass
