from interfaces.sql.sqlite import GetRawSongDataFromDataset
from settings import *
from interfaces.functions import MusicPatternEncode, CommonMusicPatterns


class BassPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, bar_index, bar_pattern_list, raw_bar_pattern_list, common_patterns):
        for bar_bass_pattern_iterator in range(len(bar_pattern_list)):
            if bar_pattern_list[bar_bass_pattern_iterator] == -1:
                # 在常见的bass列表里找不到某一个bass组合的处理方法：
                # a.寻找符合以下条件的bass组合
                # a1.首音不为休止符
                # a2.该旋律组合的首音/半拍音与待求旋律组合的首音/半拍音相同
                # a3.该旋律组合中所有音符与待求旋律组合对应位置的音符全部相同
                # a4.满足上述三个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
                # b.记为0
                choose_pattern = 0  # 选取的pattern
                choose_pattern_like_score = 0  # 两个旋律组合的相似程度
                for common_bass_iterator in range(1, len(common_patterns)):
                    # 1.检查首音是否为休止符
                    if common_patterns[common_bass_iterator][0] == 0:
                        continue
                    # 2.检查两个旋律组合的首音是否相同
                    if common_patterns[common_bass_iterator][0] != raw_bar_pattern_list[bar_bass_pattern_iterator][0]:
                        continue
                    # 3.检查该旋律组合中所有音符与待求旋律组合对应位置的音符是否全部相同
                    note_all_same = True
                    for note_iterator in range(len(common_patterns[common_bass_iterator])):
                        if common_patterns[common_bass_iterator][note_iterator] != 0 and common_patterns[common_bass_iterator][note_iterator] != raw_bar_pattern_list[bar_bass_pattern_iterator][note_iterator]:
                            note_all_same = False
                            break
                    if not note_all_same:
                        continue
                    # 4.求该旋律组合与待求旋律组合的差别
                    pattern_like_score = 6  # 初始的旋律组合相似度为6分 每发现一个不同音符 按权重扣分
                    note_diff_list = [10, 2, 3, 3, 6, 3, 4, 3, 10, 2, 3, 3, 6, 3, 4, 3]  # 音符差别的权重列表
                    for note_iterator in range(len(common_patterns[common_bass_iterator])):
                        if common_patterns[common_bass_iterator][note_iterator] != raw_bar_pattern_list[bar_bass_pattern_iterator][note_iterator]:
                            pattern_like_score -= note_diff_list[note_iterator]
                    # 5.如果这个旋律组合的差别是目前最小的 则保存它
                    # print(common_melody_iterator, pattern_like_score)
                    if pattern_like_score > choose_pattern_like_score:
                        choose_pattern_like_score = pattern_like_score
                        choose_pattern = common_bass_iterator
                # 6.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_MELODY_PATTERNS+1
                if choose_pattern_like_score > 0:
                    bar_pattern_list[bar_bass_pattern_iterator] = choose_pattern
                else:
                    bar_pattern_list[bar_bass_pattern_iterator] = len(common_patterns)
        return bar_pattern_list


class BassTrainData:

    input_data = []  # 输入model的数据
    output_data = []  # 从model输出的数据

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data):
        # 1.从数据集中读取鼓点
        raw_bass_data = GetRawSongDataFromDataset('bass', None)
        # 2.获取最常见的bass组合
        self.common_bass_patterns, __ = CommonMusicPatterns(raw_bass_data, number=COMMON_BASS_PATTERN_NUMBER, note_time_step=1/8, pattern_time_step=1)
        # 3.生成输入输出数据
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            if raw_bass_data[song_iterator] != {} and melody_pattern_data[song_iterator] != {}:
                # 3.1.开头补上几个空小节 便于训练开头的数据
                for bar_iterator in range(-TRAIN_CHORD_IO_BARS, 0):
                    raw_bass_data[song_iterator][bar_iterator] = [0 for t in range(32)]
                    melody_pattern_data[song_iterator][bar_iterator] = [0 for t in range(4)]
                    chord_data[song_iterator][bar_iterator] = [0 for t in range(4)]
                # 3.2.将一首歌的bass编码为常见的bass组合。如果该bass组合不常见，则记为common_bass_patterns+1
                raw_bass_data[song_iterator] = BassPatternEncode(self.common_bass_patterns, raw_bass_data[song_iterator], 1 / 8, 1).music_pattern_dict
                # 3.3.生成训练数据 输入内容是当前时间的编码 最近两小节+一个时间步长的主旋律 最近两小节+一个时间步长的和弦和前两小节的bass 输出内容是这个时间步长的bass
                self.get_model_io_data(raw_bass_data[song_iterator], melody_pattern_data[song_iterator], continuous_bar_number_data[song_iterator], chord_data[song_iterator])
        # print('\n\n\n\n\n')
        # for t in self.input_data:
        #     print(t)
        # print('\n\n\n')
        # for t in self.output_data:
        #     print(t)
        print(len(self.input_data), len(self.output_data))

    def get_model_io_data(self, bass_pattern_data, melody_pattern_data, continuous_bar_number_data, chord_data):
        """
        模型的训练数据包括：当前时间的编码（1-8） 过去9拍的主旋律 过去9拍的和弦 过去8拍的bass 输出数据为最后一拍的bass 共计长度为27
        :param bass_pattern_data: 一首歌的bass数据
        :param melody_pattern_data: 一首歌的主旋律组合的列表
        :param continuous_bar_number_data: 一首歌主旋律连续不为空的小节列表
        :param chord_data: 一首歌的和弦列表
        :return:
        """
        for key in bass_pattern_data:
            for time_in_bar in range(round(4)):  # bass训练的步长为1拍
                try:
                    # 1.添加当前时间的编码（0-7）
                    time_add = (1 - continuous_bar_number_data[key + TRAIN_BASS_IO_BARS] % 2) * 4
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 2.添加过去9拍的主旋律
                    input_time_data = input_time_data + melody_pattern_data[key][time_in_bar:]
                    output_time_data = output_time_data + melody_pattern_data[key][time_in_bar:]
                    for bar_iterator in range(key + 1, key + TRAIN_BASS_IO_BARS):
                        input_time_data = input_time_data + melody_pattern_data[bar_iterator]
                        output_time_data = output_time_data + melody_pattern_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = [input_time_data[0]] + [0 for t in range(len(input_time_data) - 1)]
                            output_time_data = [output_time_data[0]] + [0 for t in range(len(output_time_data) - 1)]
                    input_time_data = input_time_data + melody_pattern_data[key + TRAIN_BASS_IO_BARS][:1 + time_in_bar]
                    output_time_data = output_time_data + melody_pattern_data[key + TRAIN_BASS_IO_BARS][:1 + time_in_bar]
                    # 3.添加过去9拍的和弦
                    if melody_pattern_data[key] == [0 for t in range(4)]:
                        input_time_data = input_time_data + [0 for t in range(4 - time_in_bar)]
                        output_time_data = output_time_data + [0 for t in range(4 - time_in_bar)]
                    else:
                        input_time_data = input_time_data + chord_data[key][time_in_bar:]
                        output_time_data = output_time_data + chord_data[key][time_in_bar:]
                    for bar_iterator in range(key + 1, key + TRAIN_BASS_IO_BARS):
                        input_time_data = input_time_data + chord_data[bar_iterator]
                        output_time_data = output_time_data + chord_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[:10] + [0 for t in range(len(input_time_data) - 10)]  # 1的时间编码+9的主旋律 所以是10
                            output_time_data = output_time_data[:10] + [0 for t in range(len(output_time_data) - 10)]
                    input_time_data = input_time_data + chord_data[key + TRAIN_BASS_IO_BARS][:1 + time_in_bar]
                    output_time_data = output_time_data + chord_data[key + TRAIN_BASS_IO_BARS][:1 + time_in_bar]
                    # 4.添加过去8拍的bass
                    if melody_pattern_data[key] == [0 for t in range(4)]:  # 如果某一个小节没有主旋律 那么这个小节对应的bass也置为空
                        input_time_data = input_time_data + [0 for t in range(4 - time_in_bar)]
                        output_time_data = output_time_data + [0 for t in range(3 - time_in_bar)]
                    else:
                        input_time_data = input_time_data + bass_pattern_data[key][time_in_bar:]
                        output_time_data = output_time_data + bass_pattern_data[key][1 + time_in_bar:]
                    for bar_iterator in range(key + 1, key + TRAIN_BASS_IO_BARS):
                        input_time_data = input_time_data + bass_pattern_data[bar_iterator]
                        output_time_data = output_time_data + bass_pattern_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[:19] + [0 for t in range(len(input_time_data) - 19)]  # 1的时间编码+9的主旋律+9拍的和弦 所以是19
                            output_time_data = output_time_data[:19] + [0 for t in range(len(output_time_data) - 19)]
                    input_time_data = input_time_data + bass_pattern_data[key + TRAIN_BASS_IO_BARS][:time_in_bar]
                    output_time_data = output_time_data + bass_pattern_data[key + TRAIN_BASS_IO_BARS][:1 + time_in_bar]
                    # 5.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个小节的bass不为空
                    if melody_pattern_data[key + TRAIN_BASS_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                        output_bar_data_temp = bass_pattern_data[key + TRAIN_BASS_IO_BARS]
                        if output_bar_data_temp != [0 for t in range(4)]:
                            self.input_data.append(input_time_data)
                            self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass
