from settings import *


class MelodyConfig:  # 训练主旋律的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = COMMON_MELODY_PATTERN_NUMBER + 1  # 音符组合字典的规模
    input_size = output_size = int(1 + TRAIN_MELODY_IO_BARS + 1 + TRAIN_MELODY_IO_BARS * 4)  # 输入数据的规模 最开始的1是当前时间 中间的TRAIN_MELODY_IO_BARS+1是melody_profile 后面的那部分是4小节的旋律组合
    max_max_epoch = 5 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class MelodyConfig2:  # 训练主旋律的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = 8 + (COMMON_MELODY_PATTERN_NUMBER + 2)  # 音符组合字典的规模 前面的8是当前时间的编码 后面的是常见主旋律的编码数量
    input_size = output_size = int(TRAIN_MELODY_IO_BARS * 4)  # 输入数据的规模 4小节的时间编码和旋律组合
    max_max_epoch = 5 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class MelodyConfigNoProfile:  # 训练主旋律的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = COMMON_MELODY_PATTERN_NUMBER + 1  # 音符组合字典的规模
    input_size = output_size = int(1 + TRAIN_MELODY_IO_BARS * 4)  # 输入数据的规模 最开始的1是当前时间 后面的那部分是4小节的旋律组合
    max_max_epoch = 1 if FLAG_IS_DEBUG else 1  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class MelodyConfig3:  # 训练主旋律的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = 16 + (COMMON_MELODY_PATTERN_NUMBER + 2)  # 音符组合字典的规模 前面的16是当前时间的编码 后面的是这个乐段常见主旋律的编码数量
    input_size = output_size = int(TRAIN_MELODY_IO_BARS * 4) + 1  # 输入数据的规模 时间编码和4小节的旋律组合
    max_max_epoch = 5 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class ChordConfig:  # 训练和弦的配置 训练和弦使用MelodyModel，不单独建一个model
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = max(len(CHORD_DICT) + 1, COMMON_MELODY_PATTERN_NUMBER + 1)  # 和弦字典的规模
    input_size = output_size = int(1 + (TRAIN_CHORD_IO_BARS * 4 + 2) + TRAIN_CHORD_IO_BARS * 4)  # 输出数据的规模。最开始的1是当前时间 中间的是两小节+2拍的主旋律组合 后面的是前两小节的和弦
    max_max_epoch = 15 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.8 if FLAG_IS_DEBUG else 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class ChordConfig2:  # 训练和弦的配置 训练和弦使用MelodyModel，不单独建一个model
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = 4 + COMMON_MELODY_PATTERN_NUMBER + 2 + len(CHORD_DICT) + 1  # 和弦字典的规模。4为当前时间的编码 COMMON_MELODY_PATTERN_NUMBER + 2为常见主旋律的数量 len(CHORD_DICT) + 1为和弦字典的规模
    input_size = output_size = int(1 + TRAIN_CHORD_IO_BARS * 4)  # 输出数据的规模。最开始的1是当前时间 后面的是前两小节的和弦
    max_max_epoch = 2 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class ChordConfig3:  # 训练和弦的配置 训练和弦使用MelodyModel，不单独建一个model
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = 8 + COMMON_MELODY_PATTERN_NUMBER + 2 + len(CHORD_DICT) + 1  # 和弦字典的规模。4为当前时间的编码 COMMON_MELODY_PATTERN_NUMBER + 2为常见主旋律的数量 len(CHORD_DICT) + 1为和弦字典的规模
    input_size = output_size = TRAIN_CHORD_IO_BARS * 4  # 输出数据的规模。最开始的1是当前时间 后面的是前两小节的和弦
    max_max_epoch = 15 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.8 if FLAG_IS_DEBUG else 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class ChordConfig4:  # 训练和弦的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    # note_dict_size = 8 + COMMON_MELODY_PATTERN_NUMBER + 2 + len(CHORD_DICT) + 1  # 和弦字典的规模。4为当前时间的编码 COMMON_MELODY_PATTERN_NUMBER + 2为常见主旋律的数量 len(CHORD_DICT) + 1为和弦字典的规模
    input_size = output_size = TRAIN_CHORD_IO_BARS * 2 + 1  # 输出数据的规模。最开始的1是当前时间 后面的是前两小节的和弦
    max_max_epoch = 2 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.8 if FLAG_IS_DEBUG else 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数

    def __init__(self, cc_pat_num):
        self.note_dict_size = 4 + (COMMON_MELODY_PATTERN_NUMBER + 2) * 2 + cc_pat_num + 1  # bass字典的规模


class DrumConfig:  # 训练鼓的配置。
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = max(COMMON_DRUM_PATTERN_NUMBER + 1, COMMON_MELODY_PATTERN_NUMBER + 1)  # 和弦字典的规模
    input_size = output_size = int(1 + (TRAIN_DRUM_IO_BARS * 4 + 2) + TRAIN_DRUM_IO_BARS * 2)  # 输出数据的规模。最开始的1是当前时间 中间的是两小节+2拍的主旋律组合 后面的是两小节的鼓点
    max_max_epoch = 1 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class BassConfig:  # 训练bass的配置
    batch_size = 35  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = max(COMMON_BASS_PATTERN_NUMBER + 1, COMMON_MELODY_PATTERN_NUMBER + 1, len(CHORD_DICT) + 1)  # bass字典的规模
    input_size = output_size = int(1 + (TRAIN_BASS_IO_BARS * 4 + 2) + (TRAIN_BASS_IO_BARS * 4 + 2) + TRAIN_BASS_IO_BARS * 2)  # 输出数据的规模。最开始的1是当前时间 之后是10拍的主旋律组合 在之后是10拍的和弦 最后是两小节的bass(1+10+10+4)
    max_max_epoch = 2 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class BassConfig2:  # 训练bass的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = 4 + (COMMON_MELODY_PATTERN_NUMBER + 2) * 2 + len(CHORD_DICT) * 2 + COMMON_BASS_PATTERN_NUMBER + 2  # bass字典的规模
    input_size = output_size = 2 * TRAIN_BASS_IO_BARS + 1  # 输出数据的规模。最开始的1是当前时间 之后是10拍的主旋律组合 在之后是10拍的和弦 最后是两小节的bass(1+10+10+4)
    max_max_epoch = 5 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数


class BassConfig3:  # 训练bass的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    # note_dict_size = 4 + self.keypress_pat_num + self.rc_pat_num * 2 + COMMON_BASS_PATTERN_NUMBER + 2  # bass字典的规模
    input_size = output_size = 2 * TRAIN_BASS_IO_BARS + 1  # 输出数据的规模。最开始的1是当前时间 之后是10拍的主旋律组合 在之后是10拍的和弦 最后是两小节的bass(1+10+10+4)
    max_max_epoch = 2 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数

    def __init__(self, keypress_pat_num, rc_pat_num):
        self.note_dict_size = 4 + keypress_pat_num + rc_pat_num * 2 + COMMON_BASS_PATTERN_NUMBER + 2  # bass字典的规模


class BassConfig4:  # 训练bass的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    # note_dict_size = 4 + self.keypress_pat_num + self.rc_pat_num * 2 + COMMON_BASS_PATTERN_NUMBER + 2  # bass字典的规模
    input_size = output_size = 2 * TRAIN_BASS_IO_BARS + 1  # 输出数据的规模。最开始的1是当前时间 之后是10拍的主旋律组合 在之后是10拍的和弦 最后是两小节的bass(1+10+10+4)
    max_max_epoch = 2 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数

    def __init__(self, keypress_pat_num, rc_pat_num):
        self.note_dict_size = 4 + keypress_pat_num + rc_pat_num * 2 + COMMON_BASS_PATTERN_NUMBER + 2  # bass字典的规模


class PianoGuitarConfig:  # 训练piano_guitar的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    # note_dict_size = 4 + self.keypress_pat_num + self.rc_pat_num * 2 + COMMON_BASS_PATTERN_NUMBER + 2  # piano_guitar字典的规模
    input_size = output_size = 4 * TRAIN_PIANO_GUITAR_IO_BARS + 1  # 输出数据的规模。包含了当前时间/9拍的主旋律按键组合/9拍的和和弦根音组合/9拍的piano_guitar
    max_max_epoch = 1 if FLAG_IS_DEBUG else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数

    def __init__(self, keypress_pat_num, rc_pat_num):
        self.note_dict_size = 8 + keypress_pat_num + rc_pat_num + COMMON_PIANO_GUITAR_PATTERN_NUMBER + 2  # piano_guitar字典的规模


class StringConfig:  # 训练string的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    # note_dict_size = 4 + self.keypress_pat_num + self.rc_pat_num * 2 + COMMON_BASS_PATTERN_NUMBER + 2  # piano_guitar字典的规模
    input_size = output_size = 2 * TRAIN_STRING_IO_BARS + 1  # 输出数据的规模。包含了当前时间/10拍的主旋律骨干音组合/10拍的和和弦根音组合/10拍的string
    max_max_epoch = 4 if FLAG_IS_DEBUG else 4  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数

    def __init__(self, rc_pat_num):
        self.note_dict_size = 4 + (COMMON_CORE_NOTE_PATTERN_NUMBER + 2) + rc_pat_num * 2 + COMMON_STRING_PATTERN_NUMBER + 2  # string对照字典的规模


class StringConfig3:  # 训练string的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    # note_dict_size = 4 + self.keypress_pat_num + self.rc_pat_num * 2 + COMMON_BASS_PATTERN_NUMBER + 2  # piano_guitar字典的规模
    input_size = output_size = 2 * TRAIN_STRING_IO_BARS + 1  # 输出数据的规模。包含了当前时间/10拍的主旋律骨干音组合/10拍的和和弦根音组合/10拍的string
    max_max_epoch = 4 if FLAG_IS_DEBUG else 4  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数

    def __init__(self, rc_pat_num):
        self.note_dict_size = 4 + rc_pat_num * 2 + COMMON_STRING_PATTERN_NUMBER + 2  # string对照字典的规模


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


class IntroConfig:  # 训练前奏/间奏的配置
    batch_size = 37  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = COMMON_MELODY_PATTERN_NUMBER + 1  # 音符组合字典的规模
    input_size = output_size = int(1 + TRAIN_MELODY_IO_BARS * 4)  # 输入数据的规模 最开始的1是当前时间 后面的那部分是4小节的旋律组合
    max_max_epoch = 5 if FLAG_IS_DEBUG else 7  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 64  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数
