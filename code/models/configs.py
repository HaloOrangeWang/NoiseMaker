from settings import *


class MelodyConfig:  # 训练主旋律的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = COMMON_MELODY_PATTERN_NUMBER + 1  # 音符组合字典的规模 这个量是最高的音高减去最低的音高加二
    input_size = output_size = int(1 + TRAIN_MELODY_IO_BARS + 1 + TRAIN_MELODY_IO_BARS * 4 / MELODY_PATTERN_TIME_STEP)  # 输入数据的规模 最开始的1是当前时间 中间的TRAIN_MELODY_IO_BARS+1是melody_profile 后面的那部分是4小节的旋律组合
    max_max_epoch = 1 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class ChordConfig:  # 训练和弦的配置 训练和弦使用MelodyModel，不单独建一个model
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = max(len(CHORD_DICT) + 1, COMMON_MELODY_PATTERN_NUMBER + 1)  # 和弦字典的规模
    input_size = output_size = int(1 + (8 + CHORD_GENERATE_TIME_STEP) / MELODY_PATTERN_TIME_STEP + 8 / CHORD_TIME_STEP)  # 输出数据的规模。最开始的1是当前时间 中间的是两小节+chord_generate_time_step的主旋律组合 后面的是前两小节的和弦
    max_max_epoch = 1 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class DrumConfig:  # 训练鼓的配置。
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = max(COMMON_DRUM_PATTERN_NUMBER + 1, COMMON_MELODY_PATTERN_NUMBER + 1)  # 和弦字典的规模
    input_size = output_size = int(1 + (8 + DRUM_PATTERN_TIME_STEP) / MELODY_PATTERN_TIME_STEP + 8 / DRUM_PATTERN_TIME_STEP)  # 输出数据的规模。最开始的1是当前时间 中间的是两小节+drum_pattern_time_step的主旋律组合 后面的是两小节的鼓点
    max_max_epoch = 1 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class BassConfig:  # 训练bass的配置
    batch_size = 35  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = max(COMMON_BASS_PATTERN_NUMBER + 1, COMMON_MELODY_PATTERN_NUMBER + 1, len(CHORD_DICT) + 1)  # bass字典的规模
    input_size = output_size = 27  # 输出数据的规模。最开始的1是当前时间 之后是9拍的主旋律组合 在之后是9拍的和弦 最后是两小节的bass(1+9+9+8)
    max_max_epoch = 1 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class PianoGuitarConfig:  # 训练piano_guitar的配置
    batch_size = 35  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = max(COMMON_PIANO_GUITAR_PATTERN_NUMBER + 1, COMMON_MELODY_PATTERN_NUMBER + 1, len(CHORD_DICT) + 1)  # piano_guitar字典的规模
    input_size = output_size = 27  # 输出数据的规模。最开始的1是当前时间 之后是9拍的主旋律组合 在之后是9拍的和弦 最后是两小节的piano_guitar(1+9+9+8)
    max_max_epoch = 1 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class FillConfig:  # 训练加花的配置
    batch_size = 20  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = max(COMMON_FILL_PATTERN_NUMBER + 1, COMMON_MELODY_PATTERN_NUMBER + 1)  # 加花字典的规模
    input_size = output_size = 19  # 输出数据的规模。包含了当前时间 这次加花的数据及对应的主旋律数据 上次加花的数据及对应的主旋律数据
    max_max_epoch = 2 if FLAG_IS_DEBUG else 13  # 训练多少次
    max_epoch = 5  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数
