MUSIC_TYPE_BACH = 1  # Bach类型的音乐
MUSIC_TYPE_CHILD = 2  # 儿歌
GENERATE_MUSIC_TYPE = MUSIC_TYPE_CHILD  # 生成音乐类型

TONE_MAJOR = 0  # 大调是0 小调是1
TONE_MINOR = 1

DATASET_PATH = '../data.db'  # 数据集所在路径
GENERATE_MIDIFILE_PATH = '../09302.mid'  # 生成的midi文件所在路径
LOGDIR = './log/sess'

MELODY_PATTERN_TIME_STEP = 1  # 旋律组合的时间步长
MELODY_TIME_STEP = 1/8  # 时间步长是八分之一拍
CHORD_TIME_STEP = 1  # 和弦的时间步长 即我们认为每一拍和弦变化一次
CHORD_GENERATE_TIME_STEP = 2  # 和弦生成的时间步长 我们认为每两拍生成一次和弦
DRUM_TIME_STEP = 1/8  # 鼓的时间步长
DRUM_PATTERN_TIME_STEP = 2  # 鼓点组合的时间步长
# 另注：这些time_step如果进行修改，将必然会导致程序不能正常运行，需要重新编写datainputs和pipeline部分的代码

MELODY_LOW_NOTE = 48  # 主旋律音符的最低音和最高音(超过此限制则记为0）
MELODY_HIGH_NOTE = 96
PIANO_GUITAR_AVERAGE_NOTE = 50  # 训练时的平均音高。在读取数据时将含有这个标注的音轨的平均音高调整到这个数附近。
STRING_AVERAGE_NOTE = 62
FILL_AVERAGE_NOTE = 75

PIANO_GUITAR_AVERAGE_ROOT = 48  # 预期的根音平均值
STRING_AVERAGE_ROOT = 60

COMMON_MELODY_PATTERN_NUMBER = 250  # 常见的多少种主旋律的组合 这些个组合数量最好控制在100-500之间 它们占全部pattern的比例控制在90%-95%左右
COMMON_DRUM_PATTERN_NUMBER = 300  # 常见的多少种打击乐的组合
COMMON_BASS_PATTERN_NUMBER = 200  # 常见的多少种bass的组合
COMMON_PIANO_GUITAR_PATTERN_NUMBER = 250  # 常见的多少种piano_guitar组合 250种占比78.3% 转化9.9%左右
COMMON_STRING_PATTERN_NUMBER = 120  # 常见的多少种string的组合 120种占比81.6% 转化5.6%左右
COMMON_FILL_PATTERN_NUMBER = 180  # 常见的多少种加花组合 180种占比84% 转化3.2%左右

TRAIN_FILE_NUMBERS = 169  # 训练集中有多少首歌
TRAIN_MELODY_IO_BARS = 4  # 训练主旋律一次输入的小节数量
TRAIN_CHORD_IO_BARS = 2  # 训练和弦时一次输入的小节数量
TRAIN_DRUM_IO_BARS = 2  # 训练鼓时一次输入的小节数量
TRAIN_BASS_IO_BARS = 2  # 训练bass时一次输入的小节数量
TRAIN_PIANO_GUITAR_IO_BARS = 2  # 训练piano_guitar时一次输入的小节数量
TRAIN_STRING_IO_BARS = 2  # 训练string时一次输入的小节数量

FLAG_READ_MIDI_FILES = False
FLAG_RUN_VALIDATION = False  # 是否运行验证内容 为True运行验证内容
FLAG_RUN_MAIN = True  # 是狗运行主程序 为True运行主程序
FLAG_IS_DEBUG = False  # 是否处于调试模式 如果为True 则程序运行会简化一些步骤，运行时间大约在七分钟左右（同时效果会比较差）。如果为False，程序运行会比较复杂，运行时间在一小时左右。
FLAG_IS_TRAINING = False  # 是否处于训练状态 为True则训练 为False则直接使用现成数据生成音乐

MIN_GENERATE_BAR_NUMBER = 8  # 歌曲长度的下限
MAX_GENERATE_BAR_NUMBER = 12  # 歌曲长度的上限

MAX_MELODY_GENERATE_FAIL_TIME = 25  # 最大可接受的的主旋律生成的不合要求的次数。超过此次数则打回从头生成
MAX_CHORD_GENERATE_FAIL_TIME = 30  # 最大可接受的的和弦生成的不合要求的次数。超过此次数则打回从头生成
MAX_DRUM_GENERATE_FAIL_TIME = 10  # 最大可接受的的打击乐生成的不合要求的次数。超过此次数则打回从头生成
MAX_BASS_GENERATE_FAIL_TIME = 60  # 最大可接受的的bass生成的不合要求的次数。超过此次数则打回从头生成
MAX_PIANO_GUITAR_GENERATE_FAIL_TIME = 100  # 最大可接受的的piano_guitar生成的不合要求的次数。超过此次数则打回从头生成
MAX_FILL_GENERATE_FAIL_TIME = 50

SYSARGS = [  # 传入到程序中的参数
    {'name': 'outputpath', 'type': str, 'default': None},
]

CHORD_DICT = [  # 和弦组合的词典 第1-6列分别是 大三 小三 增三 减三 挂二 挂四 共72个和弦
    {-1},  # 把0号位空出来 用作未知和弦
    {0, 4, 7}, {0, 3, 7}, {0, 4, 8}, {0, 3, 6}, {0, 2, 7}, {0, 5, 7},
    {1, 5, 8}, {1, 4, 8}, {1, 5, 9}, {1, 4, 7}, {1, 3, 8}, {1, 6, 8},
    {2, 6, 9}, {2, 5, 9}, {2, 6, 10}, {2, 5, 8}, {2, 4, 9}, {2, 7, 9},
    {3, 7, 10}, {3, 6, 10}, {3, 7, 11}, {3, 6, 9}, {3, 5, 10}, {3, 8, 10},
    {4, 8, 11}, {4, 7, 11}, {4, 8, 0}, {4, 7, 10}, {4, 6, 11}, {4, 9, 11},
    {5, 9, 0}, {5, 8, 0}, {5, 9, 1}, {5, 8, 11}, {5, 7, 0}, {5, 10, 0},
    {6, 10, 1}, {6, 9, 1}, {6, 10, 2}, {6, 9, 0}, {6, 8, 1}, {6, 11, 1},
    {7, 11, 2}, {7, 10, 2}, {7, 11, 3}, {7, 10, 1}, {7, 9, 2}, {7, 0, 2},
    {8, 0, 3}, {8, 11, 3}, {8, 0, 4}, {8, 11, 2}, {8, 10, 3}, {8, 1, 3},
    {9, 1, 4}, {9, 0, 4}, {9, 1, 5}, {9, 0, 3}, {9, 11, 4}, {9, 2, 4},
    {10, 2, 5}, {10, 1, 5}, {10, 2, 6}, {10, 1, 4}, {10, 0, 5}, {10, 3, 5},
    {11, 3, 6}, {11, 2, 6}, {11, 3, 7}, {11, 2, 5}, {11, 1, 6}, {11, 4, 6},
    # 73-108是七和弦，七和弦包括大七 小七 属七（另外几种七和弦不常见 视作未知和弦）
    {0, 4, 7, 11}, {0, 3, 7, 10}, {0, 4, 7, 10},
    {1, 5, 8, 0}, {1, 4, 8, 11}, {1, 5, 8, 11},
    {2, 6, 9, 1}, {2, 5, 9, 0}, {2, 6, 9, 0},
    {3, 7, 10, 2}, {3, 6, 10, 1}, {3, 7, 10, 1},
    {4, 8, 11, 3}, {4, 7, 11, 2}, {4, 8, 11, 2},
    {5, 9, 0, 4}, {5, 8, 0, 3}, {5, 9, 0, 3},
    {6, 10, 1, 5}, {6, 9, 1, 4}, {6, 10, 1, 4},
    {7, 11, 2, 6}, {7, 10, 2, 5}, {7, 11, 2, 5},
    {8, 0, 3, 7}, {8, 11, 3, 6}, {8, 0, 3, 6},
    {9, 1, 4, 8}, {9, 0, 4, 7}, {9, 1, 4, 7},
    {10, 2, 5, 9}, {10, 1, 5, 8}, {10, 2, 5, 8},
    {11, 3, 6, 10}, {11, 2, 6, 9}, {11, 3, 6, 9},
]
