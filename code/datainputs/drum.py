from settings import *
from interfaces.music_patterns import MusicPatternEncode, CommonMusicPatterns
from interfaces.sql.sqlite import get_raw_song_data_from_dataset


class DrumPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, bar_index, bar_pattern_list, bar_note_list, common_patterns):
        for pattern_step_it in range(len(bar_pattern_list)):
            if bar_pattern_list[pattern_step_it] == -1:
                # 在常见的鼓点列表里找不到它的处理方法：
                # a.照抄上一小节的这个位置
                # b.记为common_drum_patterns+1
                try:
                    # 找到上一小节这个位置的鼓点组合
                    last_bar_pattern_list = self.music_pattern_dic[bar_index - 1]
                    convert = True
                    for last_bar_pattern_step_it in range(len(bar_pattern_list)):
                        if last_bar_pattern_step_it != pattern_step_it and last_bar_pattern_list[last_bar_pattern_step_it] != bar_pattern_list[last_bar_pattern_step_it]:
                            convert = False
                            break
                    if convert:
                        last_bar_pattern_dx = self.music_pattern_dic[bar_index - 1][pattern_step_it]
                        bar_pattern_list[pattern_step_it] = last_bar_pattern_dx
                    else:
                        bar_pattern_list[pattern_step_it] = len(common_patterns)
                except KeyError:
                    bar_pattern_list[pattern_step_it] = len(common_patterns)
        return bar_pattern_list


class DrumTrainData:

    def __init__(self, melody_pattern_data, continuous_bar_data):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取鼓点
        raw_drum_data = get_raw_song_data_from_dataset('drum', None)
        # 2.获取最常见的鼓点组合
        common_pattern_cls = CommonMusicPatterns(COMMON_DRUM_PATTERN_NUMBER)
        if FLAG_IS_TRAINING is True:
            common_pattern_cls.train(raw_drum_data, 0.125, 2, 'bar')
            common_pattern_cls.store('drum')
        else:
            common_pattern_cls.restore('drum')
        self.common_drum_pats = common_pattern_cls.common_pattern_list
        # 3.生成输入输出数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_drum_data[song_it] != {} and melody_pattern_data[song_it] != {}:
                # 3.1.将一首歌的鼓点编码为常见的鼓点组合。如果该鼓点组合不常见，则记为common_drum_patterns+1
                drum_pat_data = DrumPatternEncode(self.common_drum_pats, raw_drum_data[song_it], 0.125, 2).music_pattern_dic
                # 3.2.生成训练数据 输入内容是当前时间的编码 最近两小节+一个时间步长的主旋律和前两小节的鼓点 输出内容是这个时间步长的鼓点
                self.get_model_io_data(drum_pat_data, melody_pattern_data[song_it], continuous_bar_data[song_it])
        # print(len(self.input_data), len(self.output_data))
        # print('\n\n\n\n\n')
        # for t in self.input_data:
        #     print(t)
        # print('\n\n\n')
        # for t in self.output_data:
        #     print(t)
        # print(len(self.input_data), len(self.output_data))

    def get_model_io_data(self, drum_pattern_data, melody_pattern_data, continuous_bar_data):
        """
        在完成数据的前期处理（读取/转换等）之后，接下来就是提取有效数据输入到model中了
        :param drum_pattern_data: 一首歌鼓点组合的数据
        :param melody_pattern_data: 一首歌主旋律组合的数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :return:
        """
        for bar in drum_pattern_data:
            for time_in_bar in range(2):  # 鼓机训练的步长为2拍
                try:
                    # 1.添加当前时间的编码（0-3）
                    time_add = (1 - continuous_bar_data[bar + TRAIN_DRUM_IO_BARS] % 2) * 2
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 2.添加最近十拍的主旋律
                    input_time_data = input_time_data + melody_pattern_data[bar][time_in_bar * 2:]
                    output_time_data = output_time_data + melody_pattern_data[bar][time_in_bar * 2:]
                    for bar_it in range(bar + 1, bar + TRAIN_DRUM_IO_BARS):
                        input_time_data = input_time_data + melody_pattern_data[bar_it]
                        output_time_data = output_time_data + melody_pattern_data[bar_it]
                        if melody_pattern_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = [input_time_data[0]] + [0 for t in range(len(input_time_data) - 1)]
                            output_time_data = [output_time_data[0]] + [0 for t in range(len(output_time_data) - 1)]
                    input_time_data = input_time_data + melody_pattern_data[bar + TRAIN_DRUM_IO_BARS][:2 * (1 + time_in_bar)]
                    output_time_data = output_time_data + melody_pattern_data[bar + TRAIN_DRUM_IO_BARS][:2 * (1 + time_in_bar)]
                    # 3.添加过去2小节的鼓机
                    if melody_pattern_data[bar] == [0 for t in range(4)]:
                        input_time_data = input_time_data + [0 for t in range(2 - time_in_bar)]
                        output_time_data = output_time_data + [0 for t in range(1 - time_in_bar)]
                    else:
                        input_time_data = input_time_data + drum_pattern_data[bar][time_in_bar:]
                        output_time_data = output_time_data + drum_pattern_data[bar][1 + time_in_bar:]
                    for bar_it in range(bar + 1, bar + TRAIN_DRUM_IO_BARS):
                        input_time_data = input_time_data + drum_pattern_data[bar_it]
                        output_time_data = output_time_data + drum_pattern_data[bar_it]
                        if melody_pattern_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[:11] + [0 for t in range(len(input_time_data) - 11)]  # 11的由来是当前时间的编码＋１０拍的主旋律
                            output_time_data = output_time_data[:11] + [0 for t in range(len(output_time_data) - 11)]
                    input_time_data = input_time_data + drum_pattern_data[bar + TRAIN_DRUM_IO_BARS][:time_in_bar]
                    output_time_data = output_time_data + drum_pattern_data[bar + TRAIN_DRUM_IO_BARS][:(1 + time_in_bar)]
                    # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个步长的和弦不为空
                    if melody_pattern_data[bar + TRAIN_DRUM_IO_BARS] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                        output_bar_data_temp = drum_pattern_data[bar + TRAIN_DRUM_IO_BARS]
                        if output_bar_data_temp != [0 for t in range(2)]:
                            self.input_data.append(input_time_data)
                            self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass
