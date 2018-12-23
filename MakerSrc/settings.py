# 路径的配置信息。包括原始数据集/标准化数据集/json文件/输出数据/日志的文件路径
PATH_RAW_DATASET = '../TrainData/rawdata.db'  # 数据集所在路径
PATH_PAT_DATASET = '../TrainData/patterndata.db'  # pattern数据集的路径
PATH_GENERATE_MIDIFILE = '../Outputs/test1.mid'  # 生成的midi文件所在路径
PATH_TFLOG = '../TrainData/TfLog/sess'  # 日志的输出路径
PATH_DIARY = '../Diary/%s/%02d%02d%02d-%04d.txt'  # 程序运行时的日志输出文件地址 ../Diary/Train(Test)/年月日-id.txt
PATH_SOUNDFONT = '../sf2/FluidR3_GM.sf2'  # mid转wav的必要文件路径
PATH_PATTERNLOG = '../TrainData/PatternLog'

# 程序运行内容的标志位
PROGRAM_MODULES = {
    "Main": 0,  # 0代表运行主程序（生成输出数据）
    "Manifest": 1,  # 1代表运行读取原始数据集文件并规范化保存的内容
    "Check": 2  # 2代表运行数据标注的校验模块
}
ACTIVE_MODULE = PROGRAM_MODULES["Main"]  # 运行上面几个模块中的哪一个
FLAG_TEST = True  # 是否处于调试模式 如果为True 则程序运行会简化一些步骤，运行时间大约在七分钟左右（同时效果会比较差）。如果为False，程序运行会比较复杂，运行时间在一小时左右。
FLAG_TRAINING = True  # 是否处于训练状态 为True则训练 为False则直接使用现成数据生成音乐

# 传入到程序中的参数
PROGRAM_ARGS = [
    {'name': 'outputpath', 'type': str, 'default': None},
    {'name': 'diaryId', 'type': int, 'default': -1},
    {'name': 'MelodyMethod', 'type': str, 'default': 'full'}
]

# 训练乐曲的类型
MUSIC_TYPES = {
    "Bach": 1,  # Bach类型的音乐
    "Child": 2,  # 儿歌
}
ACTIVE_MUSIC_TYPE = MUSIC_TYPES["Child"]  # 生成音乐类型

# 调式的类型标志位。大调是0, 小调是1
DEF_TONE_MAJOR = 0
DEF_TONE_MINOR = 1

# 乐段的类型标志位。0-4分别是空/主/过渡/副/尾声
DEF_SEC_EMPTY = 0
DEF_SEC_MAIN = 1
DEF_SEC_MIDDLE = 2
DEF_SEC_SUB = 3
DEF_SEC_END = 4

# 训练集占全部数据的比例是多大
TRAIN_DATA_RADIO = 0.9

# 训练集中的歌曲的数量
TRAIN_FILE_NUMBERS = 169

# 各个音轨在格式转化和训练时的调整平均音高。在读取数据时将含有这个标注的音轨的平均音高调整到这个数附近。
PG_AVR_NOTE = 50
STRING_AVR_NOTE = 62
FILL_AVR_NOTE = 75
BASS_AVR_NOTE = 38

# 常见的组合数量。选择组合数量的标准是80%的训练数据是常见组合，10%的训练数据可以转化为某一种常见组合，还有10%的训练数据会被舍弃
COMMON_MELODY_PAT_NUM = 250  # 常见的主旋律的组合数量
COMMON_CORE_NOTE_PAT_NUM = 400  # 常见的骨干音组合数量（时长为2拍） 400种占比88.6% 转换1.3%左右
COMMON_DRUM_PAT_NUM = 300  # 常见的打击乐的组合数量
COMMON_BASS_PAT_NUM = 180  # 常见的bass的组合数量 180种占比85.9% 转化3.7%左右
COMMON_PG_PAT_NUM = 350  # 常见的piano_guitar组合数量 350种占比81.6% 转化8.8%左右
COMMON_STRING_PAT_NUM = 200  # 常见的string的组合数量 200种占比87.3% 转化3%左右

# 训练数据的长度。训练各个音轨时输入的小节数量是多少
TRAIN_MELODY_IO_BARS = 4  # 训练主旋律一次输入的小节数量
TRAIN_CHORD_IO_BARS = 2  # 训练和弦时一次输入的小节数量
TRAIN_DRUM_IO_BARS = 2  # 训练鼓时一次输入的小节数量
TRAIN_BASS_IO_BARS = 2  # 训练bass时一次输入的小节数量
TRAIN_PG_IO_BARS = 2  # 训练piano_guitar时一次输入的小节数量
TRAIN_STRING_IO_BARS = 2  # 训练string时一次输入的小节数量
TRAIN_INTRO_IO_BARS = 4  # 训练intro/interlude时一次输入的小节数量

# 输出的歌曲调式是什么（大调/小调）
GENERATE_TONE = DEF_TONE_MAJOR

# 各个音轨在输出时（除主旋律/和弦/鼓点以外），预期的基础输出音高均值（average root）
PG_AVR_NOTE_OUT = 62
BASS_AVR_NOTE_OUT = 38
STRING_AVR_NOTE_OUT = 50
FILL_AVR_NOTE_OUT = 72

# 各个音轨在输出的时候，最多允许校验失败的次数时多少次（超过此次数则整首歌打回重新生成）
MAX_GEN_MELODY_FAIL_TIME = 100
MAX_GEN_INTRO_FAIL_TIME = 100
MAX_GEN_CHORD_FAIL_TIME = 30
MAX_GEN_DRUM_FAIL_TIME = 10
MAX_GEN_BASS_FAIL_TIME = 60
MAX_GEN_PG_FAIL_TIME = 100
MAX_GEN_STRING_FAIL_TIME = 50
MAX_GEN_FILL_FAIL_TIME = 50
MAX_GEN_1SONG_FAIL_TIME = 10

# 除了生成mid文件以外，是否还生成wav文件
FLAG_GEN_WAV = False

# 音符和根音之间的绝对音高差和相对音高差的对照表
REL_NOTE_COMPARE_DIC = {
    'Major': [[0, 0], [1, -1], [1, 0], [2, -1], [2, 0], [3, 0], [3, 1], [4, 0], [5, -1], [5, 0], [6, -1], [6, 0]],  # 大调
    'Minor': [[0, 0], [1, -1], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [5, 0], [5, 1], [6, 0], [6, 1]],  # 小调
}

# 和弦组合的词典
# 前12行 第1-6列分别是 大三 小三 增三 减三 挂二 挂四 共72个和弦
# 13-24行 第1-3列分别是 大七 小七 属七（另外几种七和弦不常见 视作未知和弦）
CHORD_LIST = [
    {-1},
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

GEN_SEC_CHOICES = [  # 生成时可供选择的乐段类型
    # [(0, "main"), (8, "sub"), (12, "empty")],
    # [(0, "main"), (8, "sub"), (13, "empty")],
    # [(0, "main"), (8, "sub"), (14, "empty")],
    # [(0, "main"), (8, "sub"), (16, "empty")],
    [(0, "main"), (4, "middle"), (8, "sub"), (16, "empty")],
    [(0, "main"), (4, "middle"), (8, "sub"), (17, "empty")],
    [(0, "main"), (8, "middle"), (12, "sub"), (16, "empty")],
    [(0, "main"), (8, "middle"), (12, "sub"), (17, "empty")],
    [(0, "main"), (8, "middle"), (12, "sub"), (18, "empty")],
    [(0, "main"), (8, "middle"), (12, "sub"), (20, "empty")],
    [(0, "main"), (4, "main"), (8, "middle"), (12, "sub"), (20, "empty")],
    [(0, "main"), (4, "main"), (8, "middle"), (12, "sub"), (20, "empty")],
    [(0, "main"), (8, "main"), (16, "sub"), (24, "empty")],
    [(0, "main"), (8, "main"), (16, "sub"), (25, "empty")],
    [(0, "main"), (4, "main"), (8, "middle"), (16, "sub"), (24, "empty")],
    [(0, "main"), (4, "main"), (8, "middle"), (16, "sub"), (25, "empty")],
    # [(0, "main"), (4, "main"), (8, "middle"), (16, "sub"), (20, "sub"), (24, "empty")],
    # [(0, "main"), (4, "main"), (8, "middle"), (16, "sub"), (20, "sub"), (25, "empty")],
]
