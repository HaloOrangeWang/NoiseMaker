MUSIC_TYPE_BACH = 1  # Bach类型的音乐
MUSIC_TYPE_CHILD = 2  # 儿歌
GENERATE_MUSIC_TYPE = MUSIC_TYPE_CHILD  # 生成音乐类型

TONE_MAJOR = 0  # 大调是0 小调是1
TONE_MINOR = 1

SECTION_EMPTY = 0  # 乐段的类型 包括了空、主歌、过渡副歌和尾声
SECTION_MAIN = 1
SECTION_MIDDLE = 2
SECTION_SUB = 3
SECTION_END = 4

DATASET_PATH = '../TrainData/rawdata.db'  # 数据集所在路径
PATTERN_DATASET_PATH = '../TrainData/patterndata.db'  # pattern数据集的路径
GENERATE_MIDIFILE_PATH = '../Outputs/096.mid'  # 生成的midi文件所在路径
LOGDIR = '../TrainData/TfLog/sess'
DIARY_PATH = '../Diary/%s/%02d%02d%02d-%04d.txt'  # 程序运行时的日志输出文件地址 ../Diary/Train(Test)/年月日-id.txt
SOUNDFONT_PATH = '../sf2/FluidR3_GM.sf2'

TRAIN_DATA_RADIO = 0.9  # 多大比例的数据用于训练（其余的用于验证）

MELODY_LOW_NOTE = 48  # 主旋律音符的最低音和最高音(超过此限制则记为0）
MELODY_HIGH_NOTE = 96
PIANO_GUITAR_AVERAGE_NOTE = 50  # 训练时的平均音高。在读取数据时将含有这个标注的音轨的平均音高调整到这个数附近。
STRING_AVERAGE_NOTE = 62
FILL_AVERAGE_NOTE = 75

PIANO_GUITAR_AVERAGE_ROOT = 48  # 预期的根音平均值
BASS_AVERAGE_ROOT = 36
STRING_AVERAGE_ROOT = 60

COMMON_MELODY_PATTERN_NUMBER = 250  # 常见的多少种主旋律的组合 这些个组合数量最好控制在100-500之间 它们占全部pattern的比例控制在90%-95%左右
COMMON_DRUM_PATTERN_NUMBER = 300  # 常见的多少种打击乐的组合
COMMON_BASS_PATTERN_NUMBER = 180  # 常见的多少种bass的组合 180种占比85.9% 转化3.7%左右
COMMON_PIANO_GUITAR_PATTERN_NUMBER = 350  # 常见的多少种piano_guitar组合 350种占比81.6% 转化8.8%左右
COMMON_STRING_PATTERN_NUMBER = 200  # 常见的多少种string的组合 200种占比87.3% 转化3%左右
COMMON_FILL_PATTERN_NUMBER = 180  # 常见的多少种加花组合 180种占比84% 转化3.2%左右
COMMON_CORE_NOTE_PATTERN_NUMBER = 400  # 常见的多少种骨干音组合（时长为2拍） 400种占比88.6% 转换1.3%左右

TRAIN_FILE_NUMBERS = 169  # 训练集中有多少首歌
TRAIN_MELODY_IO_BARS = 4  # 训练主旋律一次输入的小节数量
TRAIN_CHORD_IO_BARS = 2  # 训练和弦时一次输入的小节数量
TRAIN_DRUM_IO_BARS = 2  # 训练鼓时一次输入的小节数量
TRAIN_BASS_IO_BARS = 2  # 训练bass时一次输入的小节数量
TRAIN_PIANO_GUITAR_IO_BARS = 2  # 训练piano_guitar时一次输入的小节数量
TRAIN_STRING_IO_BARS = 2  # 训练string时一次输入的小节数量

FLAG_READ_MIDI_FILES = False
FLAG_RUN_VALIDATION = False  # 是否运行验证内容 为True运行验证内容
FLAG_RUN_MAIN = True  # 是否运行主程序 为True运行主程序
FLAG_IS_DEBUG = False  # 是否处于调试模式 如果为True 则程序运行会简化一些步骤，运行时间大约在七分钟左右（同时效果会比较差）。如果为False，程序运行会比较复杂，运行时间在一小时左右。
FLAG_IS_TRAINING = True  # 是否处于训练状态 为True则训练 为False则直接使用现成数据生成音乐

MIN_GENERATE_BAR_NUMBER = float('nan')  # 歌曲长度的下限。这两个变量即将被删除
MAX_GENERATE_BAR_NUMBER = float('nan')  # 歌曲长度的上限
GENERATE_TONE = TONE_MAJOR  # 输出音乐的调式

MAX_MELODY_GENERATE_FAIL_TIME = 100  # 最大可接受的的主旋律生成的不合要求的次数。超过此次数则打回从头生成
MAX_INTRO_GENERATE_FAIL_TIME = 100  # 最大可接受的的前奏生成的不合要求的次数。超过此次数则打回从头生成
MAX_CHORD_GENERATE_FAIL_TIME = 30  # 最大可接受的的和弦生成的不合要求的次数。超过此次数则打回从头生成
MAX_DRUM_GENERATE_FAIL_TIME = 10  # 最大可接受的的打击乐生成的不合要求的次数。超过此次数则打回从头生成
MAX_BASS_GENERATE_FAIL_TIME = 60  # 最大可接受的的bass生成的不合要求的次数。超过此次数则打回从头生成
MAX_PIANO_GUITAR_GENERATE_FAIL_TIME = 100  # 最大可接受的的piano_guitar生成的不合要求的次数。超过此次数则打回从头生成
MAX_STRING_GENERATE_FAIL_TIME = 50
MAX_FILL_GENERATE_FAIL_TIME = 50
MAX_1SONG_GENERATE_FAIL_TIME = 10

GENERATE_WAV = False  # 是否输出WAV文件（输出wav文件需要安装）

REL_NOTE_COMPARE_DICT = {  # 和根音之间的绝对音高差和相对音高差的对照表
    'Major': [[0, 0], [1, -1], [1, 0], [2, -1], [2, 0], [3, 0], [3, 1], [4, 0], [5, -1], [5, 0], [6, -1], [6, 0]],  # 大调
    'Minor': [[0, 0], [1, -1], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [5, 0], [5, 1], [6, 0], [6, 1]],  # 小调
}

SYSARGS = [  # 传入到程序中的参数
    {'name': 'outputpath', 'type': str, 'default': None},
    {'name': 'diaryId', 'type': int, 'default': -1},
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
