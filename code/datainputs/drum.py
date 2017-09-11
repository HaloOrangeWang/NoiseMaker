from settings import *
from interfaces.functions import MusicPatternEncode, CommonMusicPatterns
from interfaces.sql.sqlite import GetRawSongDataFromDataset


class DrumPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, bar_index, bar_pattern_list, raw_bar_pattern_list, common_patterns):
        for bar_drum_pattern_iterator in range(len(bar_pattern_list)):
            if bar_pattern_list[bar_drum_pattern_iterator] == -1:
                # 在常见的鼓点列表里找不到它的处理方法：
                # a.照抄上一小节的这个位置
                # b.记为common_drum_patterns+1
                try:
                    # 找到上一小节这个位置的鼓点组合
                    last_bar_drum_pattern_code_list = self.music_pattern_dict[bar_index - 1]
                    convert = True
                    for drum_pattern_iterator_2 in range(len(bar_pattern_list)):
                        if drum_pattern_iterator_2 != bar_drum_pattern_iterator and last_bar_drum_pattern_code_list[drum_pattern_iterator_2] != bar_pattern_list[drum_pattern_iterator_2]:
                            convert = False
                            break
                    if convert:
                        last_bar_drum_pattern_code = self.music_pattern_dict[bar_index - 1][bar_drum_pattern_iterator]
                        bar_pattern_list[bar_drum_pattern_iterator] = last_bar_drum_pattern_code
                    else:
                        bar_pattern_list[bar_drum_pattern_iterator] = len(common_patterns)
                except KeyError:
                    bar_pattern_list[bar_drum_pattern_iterator] = len(common_patterns)
        return bar_pattern_list


class DrumTrainData:

    input_data = []  # 输入model的数据
    output_data = []  # 从model输出的数据

    def __init__(self, melody_pattern_data, continuous_bar_number_data):
        # 1.从数据集中读取鼓点
        raw_drum_data = GetRawSongDataFromDataset('drum', None)
        # 2.获取最常见的鼓点组合
        self.common_drum_patterns, __ = CommonMusicPatterns(raw_drum_data, number=COMMON_DRUM_PATTERN_NUMBER, note_time_step=DRUM_TIME_STEP, pattern_time_step=DRUM_PATTERN_TIME_STEP)
        # 3.生成输入输出数据
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            if raw_drum_data[song_iterator] != {} and melody_pattern_data[song_iterator] != {}:
                # 2.1.将一首歌的鼓点编码为常见的鼓点组合。如果该鼓点组合不常见，则记为common_drum_patterns+1
                raw_drum_data[song_iterator] = DrumPatternEncode(self.common_drum_patterns, raw_drum_data[song_iterator], DRUM_TIME_STEP, DRUM_PATTERN_TIME_STEP).music_pattern_dict
                # 2.2.生成训练数据 输入内容是当前时间的编码 最近两小节+一个时间步长的主旋律和前两小节的鼓点 输出内容是这个时间步长的鼓点
                self.get_model_io_data(raw_drum_data[song_iterator], melody_pattern_data[song_iterator], continuous_bar_number_data[song_iterator])
        # print(len(self.input_data), len(self.output_data))
        # print('\n\n\n\n\n')
        # for t in self.input_data:
        #     print(t)
        # print('\n\n\n')
        # for t in self.output_data:
        #     print(t)
        print(len(self.input_data), len(self.output_data))

    def get_model_io_data(self, drum_pattern_data, melody_pattern_data, continuous_bar_number_data):
        """
        在完成数据的前期处理（读取/转换等）之后，接下来就是提取有效数据输入到model中了
        :param drum_pattern_data: 一首歌鼓点组合的数据
        :param melody_pattern_data: 一首歌主旋律组合的数据
        :param continuous_bar_number_data: 一首歌主旋律连续不为空的小节列表
        :return:
        """
        for key in drum_pattern_data:
            for time_in_bar in range(round(4 / DRUM_PATTERN_TIME_STEP)):  # 鼓机训练的步长为2拍
                try:
                    # 1.添加当前时间的编码（0-3）
                    time_add = (1 - continuous_bar_number_data[key + TRAIN_DRUM_IO_BARS] % 2) * 2
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 2.添加最近十拍的主旋律
                    input_time_data = input_time_data + melody_pattern_data[key][time_in_bar * 2:]
                    output_time_data = output_time_data + melody_pattern_data[key][time_in_bar * 2:]
                    for bar_iterator in range(key + 1, key + TRAIN_DRUM_IO_BARS):
                        input_time_data = input_time_data + melody_pattern_data[bar_iterator]
                        output_time_data = output_time_data + melody_pattern_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = [input_time_data[0]] + [0 for t in range(len(input_time_data) - 1)]
                            output_time_data = [output_time_data[0]] + [0 for t in range(len(output_time_data) - 1)]
                    input_time_data = input_time_data + melody_pattern_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]
                    output_time_data = output_time_data + melody_pattern_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]
                    # 3.添加过去2小节的鼓机
                    if melody_pattern_data[key] == [0 for t in range(4)]:
                        input_time_data = input_time_data + [0 for t in range(2 - time_in_bar)]
                        output_time_data = output_time_data + [0 for t in range(1 - time_in_bar)]
                    else:
                        input_time_data = input_time_data + drum_pattern_data[key][time_in_bar:]
                        output_time_data = output_time_data + drum_pattern_data[key][1 + time_in_bar:]
                    for bar_iterator in range(key + 1, key + TRAIN_DRUM_IO_BARS):
                        input_time_data = input_time_data + drum_pattern_data[bar_iterator]
                        output_time_data = output_time_data + drum_pattern_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[:11] + [0 for t in range(len(input_time_data) - 11)]  # 11的由来是当前时间的编码＋１０拍的主旋律
                            output_time_data = output_time_data[:11] + [0 for t in range(len(output_time_data) - 11)]
                    input_time_data = input_time_data + drum_pattern_data[key + TRAIN_DRUM_IO_BARS][:time_in_bar]
                    output_time_data = output_time_data + drum_pattern_data[key + TRAIN_DRUM_IO_BARS][:(1 + time_in_bar)]
                    # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个步长的和弦不为空
                    if melody_pattern_data[key + TRAIN_DRUM_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                        output_bar_data_temp = drum_pattern_data[key + TRAIN_DRUM_IO_BARS]
                        if output_bar_data_temp != [0 for t in range(2)]:
                            self.input_data.append(input_time_data)
                            self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass
