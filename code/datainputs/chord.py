from settings import *
from interfaces.sql.sqlite import GetRawSongDataFromDataset


class ChordTrainData:

    input_data = []  # 输入model的数据
    output_data = []  # 从model输出的数据

    def __init__(self, melody_pattern_data, continuous_bar_number_data):
        # 1.从数据集中读取歌的和弦数据
        raw_chord_data = GetRawSongDataFromDataset('chord', None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
        # 2.生成输入输出数据
        for song_iterator in range(TRAIN_FILE_NUMBERS):
            if raw_chord_data[song_iterator] != {} and melody_pattern_data[song_iterator] != {}:
                # 2.1.开头补上几个空小节 便于训练开头的数据
                for bar_iterator in range(-TRAIN_CHORD_IO_BARS, 0):
                    raw_chord_data[song_iterator][bar_iterator] = [0 for t in range(round(4 / CHORD_TIME_STEP))]
                    melody_pattern_data[song_iterator][bar_iterator] = [0 for t in range(round(4 / MELODY_PATTERN_TIME_STEP))]
                # 2.2.生成训练数据 输入内容是这两小节的主旋律和和弦 输出内容是这两拍的和弦
                self.get_model_io_data(raw_chord_data[song_iterator], melody_pattern_data[song_iterator], continuous_bar_number_data[song_iterator])
        self.chord_data = raw_chord_data
        # print('\n\n\n\n\n')
        # for t in self.input_data:
        #     print(t)
        # print('\n\n\n')
        # for t in self.output_data:
        #     print(t)
        # print(len(self.input_data), len(self.output_data))

    def get_model_io_data(self, raw_chord_data, melody_pattern_data, continuous_bar_number_data):
        """
        在完成数据的前期处理（读取/转换等）之后，接下来就是提取有效数据输入到model中了
        :param raw_chord_data: 一首歌的和弦数据
        :param melody_pattern_data: 一首歌的主旋律组合数据
        :param continuous_bar_number_data: 一首歌主旋律连续不为空的小节列表
        :return:
        """
        for key in raw_chord_data:
            for time_in_bar in range(round(4 / CHORD_GENERATE_TIME_STEP)):  # 和弦训练的步长为2拍
                try:
                    # 1.添加当前时间的编码（0-3）
                    time_add = (1 - continuous_bar_number_data[key + TRAIN_CHORD_IO_BARS] % 2) * 2
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 2.添加最近2小节多一个chord_generate_time_step的主旋律
                    input_time_data = input_time_data + melody_pattern_data[key][time_in_bar * 2:]
                    output_time_data = output_time_data + melody_pattern_data[key][time_in_bar * 2:]
                    for bar_iterator in range(key + 1, key + TRAIN_CHORD_IO_BARS):
                        input_time_data = input_time_data + melody_pattern_data[bar_iterator]
                        output_time_data = output_time_data + melody_pattern_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = [input_time_data[0]] + [0 for t in range(len(input_time_data) - 1)]
                            output_time_data = [output_time_data[0]] + [0 for t in range(len(output_time_data) - 1)]
                    input_time_data = input_time_data + melody_pattern_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]
                    output_time_data = output_time_data + melody_pattern_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]
                    # 3.添加过去2小节的和弦
                    if melody_pattern_data[key] == [0 for t in range(4)]:
                        input_time_data = input_time_data + [0 for t in range(4 - 2 * time_in_bar)]
                        output_time_data = output_time_data + [0 for t in range(2 - 2 * time_in_bar)]
                    else:
                        input_time_data = input_time_data + raw_chord_data[key][2 * time_in_bar:]
                        output_time_data = output_time_data + raw_chord_data[key][2 * (1 + time_in_bar):]
                    for bar_iterator in range(key + 1, key + TRAIN_CHORD_IO_BARS):
                        input_time_data = input_time_data + raw_chord_data[bar_iterator]
                        output_time_data = output_time_data + raw_chord_data[bar_iterator]
                        if melody_pattern_data[bar_iterator] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[:11] + [0 for t in range(len(input_time_data) - 11)]
                            output_time_data = output_time_data[:11] + [0 for t in range(len(output_time_data) - 11)]
                    input_time_data = input_time_data + raw_chord_data[key + TRAIN_CHORD_IO_BARS][:2 * time_in_bar]
                    output_time_data = output_time_data + raw_chord_data[key + TRAIN_CHORD_IO_BARS][:2 * (1 + time_in_bar)]
                    # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个步长的和弦不为空
                    if melody_pattern_data[key + TRAIN_CHORD_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                        output_bar_data_temp = raw_chord_data[key + TRAIN_CHORD_IO_BARS]
                        if output_bar_data_temp != [0 for t in range(4)]:
                            self.input_data.append(input_time_data)
                            self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass
