from settings import *
from interfaces.music_patterns import MusicPatternEncode, CommonMusicPatterns
from interfaces.utils import flat_array, DiaryLog
from interfaces.sql.sqlite import get_raw_song_data_from_dataset


class DrumPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, pat_step_dx, raw_note_list, common_patterns):
        # 在常见的鼓点列表里找不到它的处理方法：
        # a.照抄上一小节的这个位置
        # b.记为common_drum_patterns+1
        try:
            # 找到上一小节这个位置的鼓点组合
            accompany_step_dx = pat_step_dx + (lambda x: 1 if x % 2 == 0 else -1)(pat_step_dx)
            if self.music_pattern_list[accompany_step_dx - 2] == self.music_pattern_list[accompany_step_dx]:
                convert = True
            else:
                convert = False
            if convert:
                pattern_dx = self.music_pattern_list[pat_step_dx - 2]
            else:
                pattern_dx = len(common_patterns)
        except KeyError:
            pattern_dx = len(common_patterns)
        return pattern_dx


class DrumTrainData:

    def __init__(self, melody_pattern_data, continuous_bar_data):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 1.从数据集中读取鼓点，
        raw_drum_data = get_raw_song_data_from_dataset('drum', None)
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_drum_data[song_it] != dict():
                raw_drum_data[song_it] = flat_array(raw_drum_data[song_it])
            else:
                raw_drum_data[song_it] = []  # 对于没有鼓点的歌曲，将格式转化为list格式

        # 2.获取最常见的鼓点组合
        common_pattern_cls = CommonMusicPatterns(COMMON_DRUM_PAT_NUM)
        common_pattern_cls.train(raw_drum_data, 0.125, 2)
        common_pattern_cls.store('drum')
        self.common_drum_pats = common_pattern_cls.common_pattern_list

        # 3.生成输入输出数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_drum_data[song_it] and melody_pattern_data[song_it]:
                # 3.1.将一首歌的鼓点编码为常见的鼓点组合。如果该鼓点组合不常见，则记为common_drum_patterns+1
                drum_pat_data = DrumPatternEncode(self.common_drum_pats, raw_drum_data[song_it], 0.125, 2).music_pattern_list
                # 3.2.生成训练数据 输入内容是当前时间的编码 最近两小节+一个时间步长的主旋律和前两小节的鼓点 输出内容是这个时间步长的鼓点
                self.get_model_io_data(drum_pat_data, melody_pattern_data[song_it], continuous_bar_data[song_it])

        DiaryLog.warn('Generation of drum train data has finished!')

    def get_model_io_data(self, drum_pat_data, melody_pat_data, continuous_bar_data):
        """
        在完成数据的前期处理（读取/转换等）之后，接下来就是提取有效数据输入到model中了
        :param drum_pat_data: 一首歌鼓点组合的数据
        :param melody_pat_data: 一首歌主旋律组合的数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :return:
        """
        for step_it in range(-2 * TRAIN_DRUM_IO_BARS, len(drum_pat_data) - 2 * TRAIN_DRUM_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            try:
                # 1.获取当前所在的小节和所在小节的位置
                cur_bar = step_it // 2  # 第几小节
                pat_step_in_bar = step_it % 2
                beat_in_bar = (step_it % 2) * 2  # 小节内的第几拍

                # 2.添加当前音符的时间
                time_add = (1 - continuous_bar_data[cur_bar + TRAIN_DRUM_IO_BARS] % 2) * 2
                input_time_data = [pat_step_in_bar + time_add]
                output_time_data = [pat_step_in_bar + time_add]

                # 2.添加最近十拍的主旋律
                if cur_bar < 0:
                    input_time_data.extend([0 for t in range(4 - beat_in_bar)])
                    output_time_data.extend([0 for t in range(4 - beat_in_bar)])
                else:
                    input_time_data = input_time_data + melody_pat_data[(cur_bar * 4 + beat_in_bar): (cur_bar + 1) * 4]
                    output_time_data = output_time_data + melody_pat_data[(cur_bar * 4 + beat_in_bar): (cur_bar + 1) * 4]
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_DRUM_IO_BARS):
                    if bar_it < 0:
                        input_time_data.extend([0, 0, 0, 0])
                        output_time_data.extend([0, 0, 0, 0])
                    else:
                        input_time_data = input_time_data + melody_pat_data[(bar_it * 4): (bar_it + 1) * 4]
                        output_time_data = output_time_data + melody_pat_data[(bar_it * 4): (bar_it + 1) * 4]
                    if bar_it < 0 or melody_pat_data[(bar_it * 4): (bar_it + 1) * 4] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        input_time_data = [input_time_data[0]] + [0 for t in range(len(input_time_data) - 1)]
                        output_time_data = [output_time_data[0]] + [0 for t in range(len(output_time_data) - 1)]
                input_time_data = input_time_data + melody_pat_data[(cur_bar + TRAIN_DRUM_IO_BARS) * 4: (cur_bar + TRAIN_DRUM_IO_BARS) * 4 + beat_in_bar + 2]
                output_time_data = output_time_data + melody_pat_data[(cur_bar + TRAIN_DRUM_IO_BARS) * 4: (cur_bar + TRAIN_DRUM_IO_BARS) * 4 + beat_in_bar + 2]

                if len(melody_pat_data) < (cur_bar + TRAIN_DRUM_IO_BARS) * 4 + beat_in_bar + 2:
                    continue

                # 3.添加过去2小节的鼓机
                if cur_bar < 0 or melody_pat_data[(cur_bar * 4): (cur_bar + 1) * 4] == [0 for t in range(4)]:
                    input_time_data = input_time_data + [0 for t in range(2 - pat_step_in_bar)]
                    output_time_data = output_time_data + [0 for t in range(1 - pat_step_in_bar)]
                else:
                    input_time_data = input_time_data + drum_pat_data[cur_bar * 2 + pat_step_in_bar: (cur_bar + 1) * 2]
                    output_time_data = output_time_data + drum_pat_data[cur_bar * 2 + pat_step_in_bar + 1: (cur_bar + 1) * 2]
                for bar_it in range(cur_bar + 1, cur_bar + TRAIN_DRUM_IO_BARS):
                    if bar_it < 0:
                        input_time_data.extend([0, 0])
                        output_time_data.extend([0, 0])
                    else:
                        input_time_data = input_time_data + drum_pat_data[(bar_it * 2): (bar_it + 1) * 2]
                        output_time_data = output_time_data + drum_pat_data[(bar_it * 2): (bar_it + 1) * 2]
                    if melody_pat_data[(bar_it * 4): (bar_it + 1) * 4] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                        input_time_data = input_time_data[:11] + [0 for t in range(len(input_time_data) - 11)]  # 11的由来是当前时间的编码＋１０拍的主旋律
                        output_time_data = output_time_data[:11] + [0 for t in range(len(output_time_data) - 11)]
                input_time_data = input_time_data + drum_pat_data[(cur_bar + TRAIN_DRUM_IO_BARS) * 2: (cur_bar + TRAIN_DRUM_IO_BARS) * 2 + pat_step_in_bar]
                output_time_data = output_time_data + drum_pat_data[(cur_bar + TRAIN_DRUM_IO_BARS) * 2: (cur_bar + TRAIN_DRUM_IO_BARS) * 2 + pat_step_in_bar + 1]
                # 4.检查该项数据是否可以用于训练。条件是1.最后一个小节的主旋律不为空 最后一个步长的和弦不为空
                if melody_pat_data[(cur_bar + TRAIN_DRUM_IO_BARS) * 4: (cur_bar + TRAIN_DRUM_IO_BARS + 1) * 4] != [0 for t in range(4)]:  # 这段时间内主旋律不为0则进行训练
                    if drum_pat_data[(cur_bar + TRAIN_DRUM_IO_BARS) * 2: (cur_bar + TRAIN_DRUM_IO_BARS + 1) * 2] != [0 for t in range(2)]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
            except IndexError:
                pass


class DrumTestData:

    def __init__(self):
        # 1.从sqlite中读取common_drum_pattern
        common_pattern_cls = CommonMusicPatterns(COMMON_DRUM_PAT_NUM)  # 这个类可以获取常见的drum组合
        common_pattern_cls.restore('drum')  # 直接从sqlite文件中读取
        self.common_drum_pats = common_pattern_cls.common_pattern_list  # 常见的bass组合列表
