from settings import *


class MelodyConfig:  # 训练主旋律的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = COMMON_MELODY_PAT_NUM + 1  # 音符组合字典的规模
    input_size = output_size = int(1 + TRAIN_MELODY_IO_BARS * 4)  # 输入数据的规模 最开始的1是当前时间 后面的那部分是4小节的旋律组合
    max_max_epoch = 1 if FLAG_TEST else 1  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数
    input_dim = 1  # 输入数据为多少重编码（1重编码即为one-hot）


class IntroConfig:  # 训练前奏/间奏的配置
    batch_size = 37  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = COMMON_MELODY_PAT_NUM + 1  # 音符组合字典的规模
    input_size = output_size = int(1 + TRAIN_MELODY_IO_BARS * 4)  # 输入数据的规模 最开始的1是当前时间 后面的那部分是4小节的旋律组合
    max_max_epoch = 5 if FLAG_TEST else 7  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 64  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数
    input_dim = 1  # 输入数据为多少重编码（1重编码即为one-hot）


class ChordConfig:  # 训练和弦的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    input_size = output_size = TRAIN_CHORD_IO_BARS * 2 + 1  # 输出数据的规模。最开始的1是当前时间 后面的是前两小节的和弦
    max_max_epoch = 2 if FLAG_TEST else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.8 if FLAG_TEST else 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数
    input_dim = 4  # chord输入数据的编码为四重编码（当前时间/前一拍的旋律/后一拍的旋律/前一拍和弦）

    def __init__(self, cc_pat_num):
        self.note_dict_size = 4 + (COMMON_MELODY_PAT_NUM + 2) * 2 + cc_pat_num + 1  # 和弦输入数据的维度


class DrumConfig:  # 训练鼓的配置。
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    note_dict_size = max(COMMON_DRUM_PAT_NUM + 1, COMMON_MELODY_PAT_NUM + 1)  # 鼓点输入数据的维度
    input_size = output_size = int(1 + (TRAIN_DRUM_IO_BARS * 4 + 2) + TRAIN_DRUM_IO_BARS * 2)  # 输出数据的规模。最开始的1是当前时间 中间的是两小节+2拍的主旋律组合 后面的是两小节的鼓点
    max_max_epoch = 1 if FLAG_TEST else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数
    input_dim = 1  # 输入数据为一重编码


class BassConfig:  # 训练bass的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    input_size = output_size = 2 * TRAIN_BASS_IO_BARS + 1  # 输出数据的规模。最开始的1是当前时间 之后是10拍的主旋律组合 在之后是10拍的和弦 最后是两小节的bass(1+10+10+4)
    max_max_epoch = 2 if FLAG_TEST else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数
    input_dim = 5  # bass输入数据的编码为五重编码（当前时间/两拍的旋律按键/前一拍的和弦根音组合/后一拍的和弦根音组合/前一拍bass）

    def __init__(self, keypress_pat_num, rc_pat_num):
        self.note_dict_size = 4 + keypress_pat_num + rc_pat_num * 2 + COMMON_BASS_PAT_NUM + 2  # bass字典的规模


class PianoGuitarConfig:  # 训练piano_guitar的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    input_size = output_size = 4 * TRAIN_PG_IO_BARS + 1  # 输出数据的规模。包含了当前时间/9拍的主旋律按键组合/9拍的和和弦根音组合/9拍的piano_guitar
    max_max_epoch = 1 if FLAG_TEST else 5  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数
    input_dim = 4  # piano_guitar输入数据的编码为四重编码（当前时间/这一拍的旋律按键/这一拍的和弦根音组合/前一拍piano_guitar）

    def __init__(self, keypress_pat_num, rc_pat_num):
        self.note_dict_size = 8 + keypress_pat_num + rc_pat_num + COMMON_PG_PAT_NUM + 2  # piano_guitar字典的规模


class StringConfig:  # 训练string的配置
    batch_size = 50  # 每批数据的规模
    learning_rate = 0.01  # 初始的学习速率
    input_size = output_size = 2 * TRAIN_STRING_IO_BARS + 1  # 输出数据的规模。包含了当前时间/10拍的主旋律骨干音组合/10拍的和和弦根音组合/10拍的string
    max_max_epoch = 4 if FLAG_TEST else 4  # 训练多少次
    max_epoch = 3  # 使用初始学习速率的epoch数量
    lr_decay = 0.7  # 学习速率衰减
    rnn_hidden_size = 256  # 神经网络cell的数量
    num_layers = 2  # 神经网络层数
    input_dim = 5  # string输入数据的编码为五重编码（当前时间/两拍的旋律骨干音/前一拍的和弦根音组合/后一拍的和弦根音组合/前一拍string）

    def __init__(self, rc_pat_num):
        self.note_dict_size = 4 + (COMMON_CORE_NOTE_PAT_NUM + 2) + rc_pat_num * 2 + COMMON_STRING_PAT_NUM + 2  # string对照字典的规模
