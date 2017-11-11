import sqlite3
from settings import *
from interfaces.midi.midi import GenerateDataFromMidiFile
from interfaces.chord_parse import NoteList2Chord


def StoreMidiFileInfo():
    midi_number = TRAIN_FILE_NUMBERS
    scale_list = [5, -4, 5, 0, -7, -9, 4, -4, 0, -12, -10, -1, -12, None, 6, 0, 0, -12, 2, 8,
                  7, 24, -7, 0, 14, 3, -7, 5, -10, 3, 3, 15, None, -6, -2, -2, 7, 7, 0, 2,
                  -9, 0, -9, -2, 0, None, 4, 6, -7, -5, 2, -6, None, 0, -9, -9, -12, 0, -12, -5,
                  3, 1, 0, -8, 3, -1, 0, 0, None, 7, -7, -7, 4, -7, 2, -12, 12, -12, -5, 3,
                  0, -7, None, 0, 0, -11, 0, 0, -8, -3, 5, 5, 0, -10, -7, 2, -7, 2, 4, -3,
                  4, -14, 2, 0, -8, -14, -2, -10, 2, -12, 5, None, 0, 5, -10, -5, -2, 4, 2, -15,
                  -12, 3, -17, 2, 0, -12, 5, -9, 0, -10, 4, -8, 2, 5, -12, 0, -10, -7, 4, -12,
                  0, 0, -8, -7, -3, -1, -5, -5, 19, -7, -12, -5, -2, 0, -12, -8, -7, -5, 2, 9,
                  -5, -5, -7, -12, 0, -5, -9, -5, 0]
    tone_list = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, None, 0, 0, 0, 0, 1, 1,
                 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, None, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, None, 0, 0, 1, 0, 0, 0, None, 0, 0, 0, 1, 0, 0, 1,
                 0, 0, 0, 0, 1, 0, 0, 0, None, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0,
                 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 1, 0, 0]
    bpm_list = [100, 58, 96, 84, 130, 88, 64, 62, 100, 141, 108, 101, 90, None, 102, 84, 115, 120, 100, 112,
                125, 85, 180, 108, 120, 122, 120, 110, 140, 71, 183, 97, None, 165, 120, 120, 128, 160, 110, 121,
                76, 110, 60, 110, 96, None, 120, 62, 66, 96, 125, 160, None, 140, 71, 120, 104, 142, 100, 104,
                86, 75, 130, 140, 160, 67, 100, 105, None, 90, 80, 158, 78, 122, 88, 112, 135, 102, 120, 80,
                137, 126, None, 106, 106, 96, 123, 120, 108, 100, 100, 96, 105, 78, 96, 125, 110, 98, 100, 80,
                67, 120, 128, 128, 124, 113, 110, 118, 120, 76, 100, None, 120, 90, 65, 154, 70, 104, 155, 108,
                68, 145, 60, 120, 138, 100, 130, 60, 66, 100, 142, 87, 160, 145, 88, 70, 130, 120, 100, 100,
                96, 96, 75, 120, 74, 58, 73, 94, 114, 120, 115, 100, 64, 100, 115, 114, 100, 150, 61, 120,
                130, 110, 120, 120, 125, 80, 124, 80, 109]
    name_list = ['买报歌', '鲁冰花', '娃哈哈', '小燕子', '两只老虎', '让我们荡起双桨', '妈妈的吻', '听妈妈讲那过去的事情', '洋娃娃和小熊跳舞', '春天在哪里', '六一的歌', '小小少年', '小红帽', '阿童木之歌', '爸爸去哪儿', '毕业歌', '表情歌', '捕鱼歌', '采蘑菇的小姑娘', '采蘑菇的小姑娘',
                 '草原赞歌', '虫儿飞', '大风歌', '大头儿子小头爸爸', '丢手绢', '豆豆龙', '读书郎', '读书郎', '多年以前', '歌唱二小放牛郎', '歌声与微笑', '恭喜恭喜', '共产儿童团歌', '国旗真美丽', '红星闪闪', '红星闪闪', '花仙子', '花仙子', '活泼乐曲', '机器猫',
                 '教师节快乐', '蓝精灵之歌', '懒惰虫', '劳动最光荣', '雷锋，请听我回答', '铃儿响叮当', '铃儿响叮当', '妈妈的吻', '妈妈格桑拉', '猫和老鼠', '沐浴在阳光下', '男孩和女孩', '泥娃娃', '牛奶歌', '泼水歌', '青草莓 红苹果', '捉泥鳅', '人人齐欢笑', '如果你开心你就拍拍手', '如今家乡山连山',
                 '赛船', '三个和尚', '三只小熊', '少年，少年，祖国的春天', '拾稻穗的小姑娘', '世上只有妈妈好', '数鸭子', '数鸭子', '数字歌', '思索', '送别李叔同', '王老先生有块地', '望月亮', '问候歌', '蜗牛与黄鹂鸟', '我爱北京天安门', '我是一条小青龙', '我愿做个好小孩', '我在马路上捡到一分钱', '西风的话',
                 '向前冲', '小号手', '小花狗，不回家', '小螺号', '小螺号', '小毛驴', '小蜜蜂', '小松树', '只要妈妈笑一笑', '小兔子乖乖', '小小世界', '小小萤火虫', '星仔走天涯', '小燕子', '幸福拍手歌', '学习雷锋好榜样', '爷爷为我打月饼', '一分钱', '一个师傅仨徒弟', '一只大雁在站岗',
                 '雨花石', '找爸爸', '找朋友', '找朋友', '只要妈妈露笑脸', '中國少年先鋒隊隊歌', '种太陽', '捉泥鳅', '最美丽', '放风筝', '粉刷匠', '上学歌', '小星星', '玛丽有只小羊羔', '啊个好人', '菠菜', '采菱', '打电话', '稻田里的火鸡', '邋遢大王',
                 '风中的精灵', '葫芦兄弟', '黄鹂鸟', '回收歌', '开心往前飞', '看星星', '兰花草', '懒惰虫', '懒羊羊当大厨', '去远方', '稍息', '生产队里养了一群小鸭子', '圣诞老人进城', '舒克贝塔', '童谣', '我爱洗澡', '小娃娃吃西瓜', '校园多美好', '星星点灯', '醒来了',
                 '雪人', '一加一等于小可爱', '知了', '侏儒赛跑', '直至消失', '鲁冰花', '两地书', '二月里来', '北方佬', '卡布利鸟', '大胖呆', '她会绕着山那边回來', '寂寞的眼', '小草', '彩虹妹妹', '我们的村庄', '我们的田野', '我是一只小鸭子', '旋转木马', '春天到了',
                 '渔光曲', '美丽的河流', '老人', '老黑爵', '长腿叔叔', '静夜思', '驼铃', '鳟鱼', '小雨点']
    bias_time_list = [0.5, 0, 0, 0, -1 / 24, -1 / 24, 0, -3, 0, 2, 0, 0, 0, None, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, -1, -1 / 24, -1 / 24, 0, 0, -2, -1 / 32, None, 0, -2, -2, 0, 0, 0, -11 / 24,
                      0, 0, 0, -2, 0, None, 7 / 16, 0, -1 / 96, 0, 0, 0, None, 0, 0, 0, 0, -3, 0, -1 / 96,
                      -1 / 48, 0, 0, 0, 0, 0, 0, 1 + 7 / 24, None, 0, 0, 1, 0, 1, 0, 0, 0.5, 0, 0.25, 0,
                      1, -1 / 24, None, 0, 0, 0, 0, 0, -2, 0, 0, -2, 1, 0, 0, 0, 0, 0, 0, -(2 + 1 / 24),
                      -1 / 96, 0, 0, 0, -(2 + 1 / 24), -1, 0, 0, -1 / 24, 0, 5 / 96, None, 0, 1 / 96, -2, 0, 0, 0, -3, 0,
                      0, 0, 0, 0, 0, 0, 25 / 48, 0, -1, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1 / 32, -1 / 24, -1 / 24, 2, -2.5, 0, -2, -2.5, 0, 0, 0, 0, 0, -3.5, 0,
                      -1 / 24, 0, 0, 0, 0, -1 / 24, -1 / 24, -1 / 24, 0]
    error_list = [13, 32, 45, 52, 68, 82, 111]  # 这几首歌从训练集中排外，不作为训练样本
    # 1.初始化
    conn = sqlite3.connect(DATASET_PATH)
    conn.execute('drop table if exists SongInfo')  # 在建表之前先删掉这个表中的原有数据
    conn.commit()
    conn.execute('create table if not exists SongInfo(id integer primary key, name varchar(30), scale integer, bpm integer, tone integer, biastime float)')
    # 2.存储
    for t in range(midi_number):
        if t not in error_list:
            command = 'insert into SongInfo(id, name, scale, bpm, tone, biastime) values (' + repr(t + 1) + ',\'' + name_list[t] + '\',' + repr(scale_list[t]) + ',' + repr(bpm_list[t]) + ',' + repr(tone_list[t]) + ',' + repr(bias_time_list[t]) + ')'
            conn.execute(command)
    conn.commit()


class SaveMidiData:
    # 这个模块的用处是将midi数据保存在sql文件中
    music_data = {}  # 整理好的music_data就在这个列表中 列表一共有4维：第一维是歌的编号 第二维是存储类型 第三维是小节序列 第四维是小节中每个音的音高
    chord_data = {}  # 整理好的和弦列表。列表一共有3维，第一维是歌的编号，第二维是小节序列，第三维是小节中的和弦列表
    melody_data = {}  # 整理好的主旋律列表。列比一共有三维 第一维是歌的编号 第二维是小节序列，第三维是小节中主旋律音符的列表
    note_dict = [[-1]]  # 音符的词典 即将音符及其组合整理成dict的形式。把第0项空出来，第0项是休止符。

    def __init__(self):
        # 1.变量定义及其初始化
        scale_list = [None for t in range(TRAIN_FILE_NUMBERS)]
        tone_list = [None for t in range(TRAIN_FILE_NUMBERS)]
        bpm_list = [None for t in range(TRAIN_FILE_NUMBERS)]
        bias_time_list = [None for t in range(TRAIN_FILE_NUMBERS)]
        error_list = [t for t in range(TRAIN_FILE_NUMBERS)]  # 这几首歌从训练集中排外，不作为训练样本
        # 2.读取所有歌曲的信息
        conn = sqlite3.connect(DATASET_PATH)
        rows = conn.execute('select * from SongInfo')
        for row in rows:
            song_id = row[0] - 1
            error_list.remove(song_id)
            scale_list[song_id] = row[2]
            bpm_list[song_id] = row[3]
            tone_list[song_id] = row[4]
            bias_time_list[song_id] = row[5]
        # print(error_list)
        # print(scale_list)
        # print(tone_list)
        # print(bpm_list)
        # print(start_time_list)
        # 3.读取所有的文件并将里面的音符转化为可训练的形式
        for train_file_iterator in range(TRAIN_FILE_NUMBERS):
            if train_file_iterator not in error_list:
                self.music_data[train_file_iterator] = {}  # 初始化训练数据的第一维。注意：第一维的下标是从0开始的，而不是从1开始的。因为python的字典不能同时初始化两维
                file_path = '../MidiData/%02d/%03d.mid' % (GENERATE_MUSIC_TYPE, train_file_iterator + 1)
                print(train_file_iterator)
                # print(file_path, bpm_list[train_file_iterator], start_time_list[train_file_iterator], scale_list[train_file_iterator])
                midi_data = GenerateDataFromMidiFile(file_path, bias_time_list[train_file_iterator], scale_list[train_file_iterator])
                self.get_music_data(train_file_iterator, midi_data.pianoroll_list, time_step_dict={'piano_guitar': 0.25, 'string': 0.25}, eliminate=['main', 'chord', 'others'])
                # 把main和chord去掉意思就是主旋律和和弦部分单独处理
                if 'main' in midi_data.pianoroll_list:
                    self.get_melody_data(train_file_iterator, midi_data.pianoroll_list)
                if 'chord' in midi_data.pianoroll_list:  # 只有在这首歌有和弦的时候才训练和弦
                    self.get_chord_data(train_file_iterator, midi_data.pianoroll_list, tone=tone_list[train_file_iterator])
            # if train_file_iterator in [0, 2]:
            #     print(self.note_dict)
            #     for t in range(len(self.music_data[train_file_iterator]['piano_guitar1'])):
            #         print(t, self.music_data[train_file_iterator]['piano_guitar1'][t])
            #     print('\n\n\n')
            #     for t in range(len(self.chord_data[train_file_iterator])):
            #         print(t, self.chord_data[train_file_iterator][t])
            #     print('\n\n\n')
            #     for t in range(len(self.melody_data[train_file_iterator])):
            #         print(t, self.melody_data[train_file_iterator][t])
            #     print('\n\n\n')
        # 5.把这些音符存储在sqlite中
        self.save_music_data()
        self.save_chord_data()
        self.save_melody_data()

    def get_music_data(self, music_number_index, pianoroll_list, default_time_step=1 / 8, time_step_dict={}, eliminate=None):
        """
        将音符组合保存为以time_step拍为步长的数组，方便存储到sql中
        :param default_time_step: 如果key不在time_step_dict中 默认的time_step是多少
        :param time_step_dict: 时间步长
        :param music_number_index: Midi文件的编号
        :param pianoroll_list: 音符列表
        :param eliminate: 那些音轨里的音符不保存
        :return:
        """
        for key in pianoroll_list:
            time_step = default_time_step
            for time_step_key in time_step_dict:  # 从0.92.05开始 因为各项标注的音符time_step不全为1/8 所以加入time_step_dict
                if time_step_key in key:  # 比如time_step_key'piano_guitar'完全包含在key'piano_guitar1'中
                    time_step = time_step_dict[time_step_key]
            # print(key, time_step)
            if key not in eliminate:
                if not pianoroll_list[key]:  # 这个音轨为空 那么直接跳过这个音轨
                    continue
                self.music_data[music_number_index][key] = []
                bar_number = int(pianoroll_list[key][-1][0] // 4) + 1  # 小节数
                current_note_ = 0  # 当前音符
                # 1.逐小节读取数据
                for bar_iterator in range(bar_number):
                    bar_data = [0 for t in range(round(4 / time_step))]  # 一小节4拍 每拍1/TIME_STEP个音符 休止符记为0
                    # 1.1.对每一个时间步长读取音符并保存在bar_data中
                    for note_time_iterator in range(round(4 / time_step)):
                        # 1.1.1.读取这个时间步长中所有的音符 保存在列表中
                        raw_time_data = set()  # 将同一个时间上所有的音都存在这里。这里永set而不是list是因为要去除相同音符
                        if current_note_ < len(pianoroll_list[key]):
                            while pianoroll_list[key][current_note_][0] <= bar_iterator * 4 + (note_time_iterator + 0.5) * time_step:
                                raw_time_data.add(pianoroll_list[key][current_note_][1])  # 只保存音高
                                current_note_ += 1
                                if current_note_ >= len(pianoroll_list[key]):
                                    break  # 超过数组长度则退出
                        # 1.1.2.将列表编码为一个整数并保存在bar_data中
                        if len(raw_time_data) != 0:  # 如果这个时间步长中没有音符的话，就不用向下执行了
                            raw_time_data = list(raw_time_data)
                            raw_time_data.sort()  # 由小到大排序 防止出现[1,3]和[3,1]被视作不同元素的情况
                            try:
                                time_data_index = self.note_dict.index(raw_time_data)  # 检查这个音符组合有没有被保存
                            except ValueError:
                                self.note_dict.append(raw_time_data)  # 添加这个音符组合
                                time_data_index = len(self.note_dict) - 1
                            bar_data[note_time_iterator] = time_data_index  # 将这个音符保存起来
                    # 1.2.将这个小节的音符信息存储在music_data中
                    self.music_data[music_number_index][key].append(bar_data)
                    # print(self.train_data[0])
                    # print(len(self.note_dict))

    def get_melody_data(self, music_number_index, pianoroll_list):
        """
        （注：如果同一个时间步长中存在多个音符 取其最高者）
        :param music_number_index: 歌的编号
        :param pianoroll_list: pianoroll的列表
        :return: 无
        """
        self.melody_data[music_number_index] = []
        bar_number = int(pianoroll_list['main'][-1][0] // 4) + 1  # 小节数
        current_note_ = 0  # 当前音符
        # 1.逐小节读取数据
        for bar_iterator in range(bar_number):
            bar_melody_data = [0 for t in range(32)]  # 一小节4拍 没拍1/TIME_STEP个和弦 比较奇怪的未知和弦记为0
            # 1.1.对每一个时间步长读取音符并保存在bar_data中
            for note_time_iterator in range(32):
                # 1.1.1.读取这个时间步长中所有的音符 保存在列表中
                high_time_note = 0  # 这个时间步长中最高的音符
                if current_note_ < len(pianoroll_list['main']):
                    while pianoroll_list['main'][current_note_][0] <= bar_iterator * 4 + (note_time_iterator + 0.5) * 0.125:  # 处理一个时间区段内的所有音符
                        if pianoroll_list['main'][current_note_][1] > high_time_note:  # 如果这个音符的音高1高于这个时间步长的最高音 那么替换这个时间步长的最高音
                            high_time_note = pianoroll_list['main'][current_note_][1]
                        current_note_ += 1
                        if current_note_ >= len(pianoroll_list['main']):
                            break  # 超过数组长度则退出
                # 1.1.2.将这个音符保存到bar_melody_data中
                if high_time_note != 0:
                    bar_melody_data[note_time_iterator] = high_time_note
            # 1.2.将这个小节的音符信息存储在train_data中
            self.melody_data[music_number_index].append(bar_melody_data)
        # print(self.melody_data[music_number_index])

    def get_chord_data(self, music_number_index, pianoroll_list, tone=TONE_MAJOR):
        # print(music_number_index)
        self.chord_data[music_number_index] = []
        bar_number = int(pianoroll_list['chord'][-1][0] // 4) + 1  # 小节数
        current_note_ = 0  # 当前音符
        saved_chord = 0  # 保存上一个时间区段的和弦
        # 逐小节读取数据
        for bar_iterator in range(bar_number):
            bar_chord_data = [0 for t in range(round(4 / CHORD_TIME_STEP))]  # 一小节4拍 没拍1/TIME_STEP个和弦 比较奇怪的未知和弦记为0
            raw_bar_data = [set() for t in range(round(4 / CHORD_TIME_STEP))]  # 将同一个时间上所有的音都存在这里。这里永set而不是list是因为要去除相同音符
            # 1.对每一个时间步长读取音符并保存在bar_data中
            for note_time_iterator in range(round(4 / CHORD_TIME_STEP)):
                # 1.1.读取这个时间步长中所有的音符 保存在列表中
                if current_note_ < len(pianoroll_list['chord']):
                    while pianoroll_list['chord'][current_note_][0] <= bar_iterator * 4 + (note_time_iterator + 0.9375) * CHORD_TIME_STEP:  # 处理一个时间区段内的所有音符
                        raw_bar_data[note_time_iterator].add(pianoroll_list['chord'][current_note_][1])  # 只保存音高
                        current_note_ += 1
                        if current_note_ >= len(pianoroll_list['chord']):
                            break  # 超过数组长度则退出
            # 2.将音符列表装化为对应和弦
            for note_time_iterator in range(round(4 / CHORD_TIME_STEP)):
                # 2.1.读取这个时间步长的陪伴音符列表（即对于第2个时间步长，读取第一个时间步长的音符列表，对于第1个时间步长则读取第二个时间步长的音符列表）
                accompany_note_time = int(not note_time_iterator % 2) + 2 * (note_time_iterator // 2)
                # 2.2.将音符列表装化为对应和弦
                if len(raw_bar_data[note_time_iterator]) == 0:  # 这个时间区段没有音符 和弦将保存为上一个时间区段的和弦
                    bar_chord_data[note_time_iterator] = saved_chord
                else:
                    if len(raw_bar_data[accompany_note_time]) == 0:  # 陪伴音符列表为空
                        saved_chord = NoteList2Chord(raw_bar_data[note_time_iterator], saved_chord, tone, None)  # 音符列表转换为和弦 并将它保存起来
                    else:  # 陪伴音符列表不为空
                        saved_chord = NoteList2Chord(raw_bar_data[note_time_iterator], saved_chord, tone, raw_bar_data[accompany_note_time])  # 音符列表转换为和弦 并将它保存起来
                    bar_chord_data[note_time_iterator] = saved_chord
            # 3.将这个小节的音符信息存储在chord_data中
            self.chord_data[music_number_index].append(bar_chord_data)
        # print(self.chord_train_data[music_number_index])
        # print(len(self.note_dict))

    def save_music_data(self):
        # 1.新建小节列表
        # 小节列表中有五个元素 id是主键，bar_index是该小节在歌中的位置，mark是该小节的标注(比如'main')，data是小节的内容，song_id是歌曲的id（外键）
        conn = sqlite3.connect(DATASET_PATH)
        conn.execute('drop table if exists BarInfo')
        conn.commit()
        conn.execute('pragma foreign_key=ON')
        conn.execute('create table BarInfo(id integer primary key autoincrement, bar_index integer, mark varchar(20), data text, song_id integer, foreign key(song_id) references SongInfo(id) on update cascade)')
        # 2.往小节列表中添加数据
        for song_id_key in self.music_data:
            # if song_id_key == 0:
            for mark_key in self.music_data[song_id_key]:
                for bar_iterator in range(len(self.music_data[song_id_key][mark_key])):
                    # 注：存储在数据库中时，歌的id从1开始编号，而在程序中，歌的id从0开始编号
                    conn.execute('insert into BarInfo(bar_index,mark,data,song_id) values (' + repr(bar_iterator) + ',\'' + mark_key + '\',\'' + repr(self.music_data[song_id_key][mark_key][bar_iterator]) + '\',' + repr(song_id_key + 1) + ')')
        conn.commit()
        # 3.新建音符字典列表
        conn.execute('drop table if exists NoteDict')
        conn.commit()
        conn.execute('create table NoteDict(id integer primary key autoincrement, note_group text)')
        # print(self.note_dict)
        # 4.往音符字典列表添加数据
        for note_group in self.note_dict:
            conn.execute('insert into NoteDict(note_group) values (\'' + repr(note_group) + '\')')
        conn.commit()
        # rows = conn.execute('select * from NoteDict')
        # print('NoteDict')
        # for row in rows:
        #     print(row)
        # rows = conn.execute('select * from BarInfo where song_id=0')
        # print('BarInfo')
        # for row in rows:
        #     print(row)

    def save_chord_data(self):
        # 由于小节列表在前面的save_train_data已经新建过了 因此这里不需要重新建表
        conn = sqlite3.connect(DATASET_PATH)
        # 往小节列表中添加数据
        for song_id_key in self.chord_data:
            for bar_iterator in range(len(self.chord_data[song_id_key])):
                # 注：存储在数据库中时，歌的id从1开始编号，而在程序中，歌的id从0开始编号
                conn.execute('insert into BarInfo(bar_index,mark,data,song_id) values (' + repr(bar_iterator) + ',\'chord\',\'' + repr(self.chord_data[song_id_key][bar_iterator]) + '\',' + repr(song_id_key + 1) + ')')
        conn.commit()

    def save_melody_data(self):
        # 由于小节列表在前面的save_train_data已经新建过了 因此这里不需要重新建表
        conn = sqlite3.connect(DATASET_PATH)
        # 往小节列表中添加数据
        for song_id_key in self.melody_data:
            for bar_iterator in range(len(self.melody_data[song_id_key])):
                # 注：存储在数据库中时，歌的id从1开始编号，而在程序中，歌的id从0开始编号
                conn.execute('insert into BarInfo(bar_index,mark,data,song_id) values (' + repr(bar_iterator) + ',\'main\',\'' + repr(self.melody_data[song_id_key][bar_iterator]) + '\',' + repr(song_id_key + 1) + ')')
        conn.commit()
