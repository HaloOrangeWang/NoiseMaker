from settings import *
import sqlite3
from interfaces.sql.sqlite import get_section_data_from_dataset, get_raw_song_data_from_dataset, get_tone_list
from interfaces.utils import DiaryLog, last_not0_number_in_array
import copy
import math

# 这个文件的内容是验证内容
# 注：这个文件里的代码只检查数据集中是否有异常的数据。检查生成的音乐是否符合要求的代码在musicout.py中


def keypress_validation():
    """
    检查按键的位置是否正确
    """
    # 1.准备工作
    conn = sqlite3.connect(DATASET_PATH)
    # 2.从数据集中读取歌的编号为song_id且小节标注为mark_key的小节数据，并将所有不为休止符的位置全部替换为1
    rows = conn.execute('select bar_index, data, song_id from BarInfo where mark=\'main\'')
    note_location_points_list = [-4, 2, 1, 2, 0, 2, 1, 2, -3, 2, 1, 2, 0, 2, 1, 2, -3, 2, 1, 2, 0, 2, 1, 2, -3, 2, 1, 2, 0, 2, 1, 2]
    for row in rows:
        # 由于在数据库中歌的id是从1开始编号的，因此需要减一
        bar_data = eval(row[1])
        bar_point = 0
        for step_it in range(len(bar_data)):
            if bar_data[step_it] != 0:
                bar_point += note_location_points_list[step_it]
        # if row[2] == 138:
        #     print(row[0], bar_data)
        if bar_point > 0:
            print('第'+repr(row[2]-1)+'首歌第'+repr(row[0])+'小节得分为'+repr(bar_point))
        for step_it in [t*4+1 for t in range(8)]:
            if bar_data[step_it] != 0 and bar_data[step_it-1] == 0:
                print('第'+repr(row[2]-1)+'首歌第'+repr(row[0])+'小节可能存在错位')


def chord_validation():
    # 当不确定的和弦和离调和弦在全部和弦中所占的比利过高时发出警告
    # 1.准备工作
    conn = sqlite3.connect(DATASET_PATH)
    chord_data = [[] for t in range(TRAIN_FILE_NUMBERS)]
    normal_chord = [1, 5, 6, 14, 26, 31, 43, 56, 59, 60, 70, 73, 80, 88, 96, 101]  # 1/4/5级大三和弦，2/3/6级小三和弦，7级减三和弦，1/6级挂二挂四和弦 1/4级大七和弦 5级属七和弦 2/6级小七和弦是调内和弦 其余为离调和弦
    # 2.从数据集中读取歌的编号为song_id且小节标注为mark_key的小节数据，并将所有不为休止符的位置全部替换为1
    rows = conn.execute('select bar_index, data, song_id from BarInfo where mark=\'chord\'')
    for row in rows:
        # 由于在数据库中歌的id是从1开始编号的，因此需要减一
        chord_data[row[2]-1] += eval(row[1])
    # print(chord_data[140])
    for song_it in chord_data:
        normal_chord_count = 0
        abnormal_chord_count = 0
        zero_chord_count = 0
        for chord_it in song_it:
            if chord_it in normal_chord:
                normal_chord_count += 1
            elif chord_it == 0:
                zero_chord_count += 1
            else:
                abnormal_chord_count += 1
        try:
            abnormal_chord_ratio = abnormal_chord_count / (normal_chord_count + abnormal_chord_count + zero_chord_count)
            zero_chord_ratio = zero_chord_count / (normal_chord_count + abnormal_chord_count + zero_chord_count)
            if abnormal_chord_ratio + zero_chord_ratio >= 0.2:
                print('第'+repr(chord_data.index(song_it))+'首歌离调和弦比例为%.3f,不确定和弦的比例为%.3f' % (abnormal_chord_ratio, zero_chord_ratio))
        except ZeroDivisionError:
            pass


def section_validation():
    # 检查乐段标注是否正确
    # 1.从数据集中读取所有歌曲的乐段信息
    section_data = get_section_data_from_dataset()
    # 2.从数据集中读取所有歌曲的主旋律信息
    raw_melody_data = get_raw_song_data_from_dataset('main', None)
    tone_list = get_tone_list()
    # 3.分析乐段信息 判断是否合理
    for song_it in range(TRAIN_FILE_NUMBERS):
        if section_data[song_it]:
            section_data_1song = copy.deepcopy(section_data[song_it])
            section_data_1song.sort()  # 按照小节先后顺序排序
            # 3.1.检查空白段的小节是否有音符 以及非空白段是否存在空小节（即空白段和非空白段的分界是否合理）
            start_bar_dx = 0
            for sec_it in range(len(section_data_1song)):
                section_type = section_data_1song[sec_it][2]
                if sec_it != len(section_data_1song) - 1:
                    end_bar_dx = int(min(section_data_1song[sec_it + 1][0], len(raw_melody_data[song_it])))  # 从上一个乐段的结束处验证到这个乐段的结束处
                    end_bias = section_data_1song[sec_it + 1][1]
                else:
                    end_bar_dx = len(raw_melody_data[song_it])
                    end_bias = 0
                for bar_it in range(start_bar_dx, end_bar_dx):
                    if bar_it != end_bar_dx - 1 or end_bias >= 0:
                        if (raw_melody_data[song_it][bar_it] == [0] * 32) ^ (section_type == SECTION_EMPTY):  # 标注是否为空与实际是否为空的异或 如果异或为真 则说明标注有问题
                            DiaryLog.warn('第%d首歌的第%d小节的实际休止情况与标注情况不相符' % (song_it, bar_it))
                    else:  # 下一个乐段提前开始 则排除掉最后的几拍
                        end_bar_step = int((end_bias + 4) * 8)
                        if (raw_melody_data[song_it][bar_it][0: end_bar_step] == [0] * end_bar_step) ^ (section_type == SECTION_EMPTY):
                            DiaryLog.warn('第%d首歌的第%d小节的实际休止情况与标注情况不相符' % (song_it, bar_it))
                start_bar_dx = end_bar_dx  # 进入下一个乐段的判断
            # 3.2.检查每个乐段的最后一个音符是否为1/3/5（大调）或3/5/6（小调）即检查内部
            for sec_it in range(len(section_data_1song)):
                section_type = section_data_1song[sec_it][2]
                if section_type == SECTION_EMPTY:
                    continue  # 不处理空乐段
                if sec_it != len(section_data_1song) - 1:
                    if section_data_1song[sec_it + 1][0] == int(section_data_1song[sec_it + 1][0]):  # 下一乐段是从整数小节开始的
                        end_bar_dx = int(min(section_data_1song[sec_it + 1][0], len(raw_melody_data[song_it])))  # 这个乐段的最后一小节
                        end_bias = section_data_1song[sec_it + 1][1]
                        if end_bias >= 0:
                            bar_data = raw_melody_data[song_it][end_bar_dx - 1]
                        else:
                            end_bar_step = int((end_bias + 4) * 8)
                            bar_data = raw_melody_data[song_it][end_bar_dx - 1][0: end_bar_step]
                    else:
                        end_bar_dx_decimal, end_bar_dx_int = math.modf(section_data_1song[sec_it + 1][0])
                        end_bar_dx = min(int(end_bar_dx_int), len(raw_melody_data[song_it]))
                        end_bias = end_bar_dx_decimal * 4 + section_data_1song[sec_it + 1][1]  # 下个乐段具体起始于哪一拍
                        if 8 > end_bias >= 4:  # 下个乐段的起始拍在下一小节
                            if end_bar_dx == len(raw_melody_data[song_it]):
                                bar_data = raw_melody_data[song_it][end_bar_dx - 1]
                            else:
                                bar_data = raw_melody_data[song_it][end_bar_dx]
                        elif end_bias >= 0:  # 下个乐段的起始拍在这一小节
                            if end_bar_dx == len(raw_melody_data[song_it]):
                                bar_data = raw_melody_data[song_it][end_bar_dx - 1]
                            else:
                                end_bar_step = int(end_bias * 8)
                                bar_data = raw_melody_data[song_it][end_bar_dx - 1][end_bar_step:] + raw_melody_data[song_it][end_bar_dx][:end_bar_step]
                        elif end_bias >= -4:  # 下个乐段的起始拍在上一小节
                            end_bar_step = int((end_bias + 4) * 8)
                            bar_data = raw_melody_data[song_it][end_bar_dx - 2][end_bar_step:] + raw_melody_data[song_it][end_bar_dx - 1][:end_bar_step]
                        else:  # 其他情况是不允许出现的
                            raise ValueError
                else:
                    bar_data = raw_melody_data[song_it][len(raw_melody_data[song_it]) - 1]
                last_note = last_not0_number_in_array(bar_data, reverse=True)
                if tone_list[song_it] == TONE_MAJOR:
                    if last_note % 12 not in [0, 4, 7]:
                        DiaryLog.warn('第%d首歌(tone=%d)的第%d小节为一个乐段的结束小节，这个小节的最后一个音符不太合适' % (song_it, tone_list[song_it], end_bar_dx))
                elif tone_list[song_it] == TONE_MINOR:
                    if last_note % 12 not in [4, 7, 9]:
                        DiaryLog.warn('第%d首歌(tone=%d)的第%d小节为一个乐段的结束小节，这个小节的最后一个音符不太合适' % (song_it, tone_list[song_it], end_bar_dx))
                else:
                    raise ValueError
        else:
            pass
            # print('第%d首歌没有乐段' % song_it)


def json_validation():

    class OriginData:  # 0.95.10之前的Hardcode在脚本中的歌曲数据

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

    # 检查json文件里写的内容与旧版本Hardcode的歌曲信息是否一致
    # 1.从数据集中读取曲目的歌名、调式等信息
    conn = sqlite3.connect(DATASET_PATH)
    rows = conn.execute('select * from SongInfo')
    for row in rows:
        song_id = row[0] - 1
        scale = row[2]
        bpm = row[3]
        tone = row[4]
        bias_time = row[5]

        old_scale = OriginData.scale_list[song_id]
        old_bpm = OriginData.bpm_list[song_id]
        old_tone = OriginData.tone_list[song_id]
        old_bias_time = OriginData.bias_time_list[song_id]

        if scale != old_scale:
            DiaryLog.warn('第%d首歌在0.95.10之前的音高为%d, 在manifest.json中的音高为%d' % (song_id + 1, old_scale, scale))
        if bpm != old_bpm:
            DiaryLog.warn('第%d首歌在0.95.10之前的节奏为%dbpm, 在manifest.json中的节奏为%dbpm' % (song_id + 1, old_bpm, bpm))
        if tone != old_tone:
            DiaryLog.warn('第%d首歌在0.95.10之前的调式为%d, 在manifest.json中的调式为%d' % (song_id + 1, old_tone, tone))
        if abs(bias_time - old_bias_time) >= 1e-4:
            DiaryLog.warn('第%d首歌在0.95.10之前的时间调整为%.4f, 在manifest.json中的时间调整为%.4f' % (song_id + 1, old_bias_time, bias_time))


def run_validation():
    keypress_validation()
    print('\n\n\n\n\n\n')
    chord_validation()
