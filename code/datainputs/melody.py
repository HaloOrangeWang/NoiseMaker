from settings import *
from interfaces.music_patterns import CommonMusicPatterns, MusicPatternEncode, MusicPatternEncodeStep
from interfaces.utils import get_dict_max_key, DiaryLog
from interfaces.sql.sqlite import get_raw_song_data_from_dataset, get_section_data_from_dataset, NoteDict
from models.KMeansModel import KMeansModel
import datainputs.functions as fnc
import copy
import math


class MelodyPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, bar_index, bar_pattern_list, bar_note_list, common_patterns):
        for pattern_step_it in range(len(bar_pattern_list)):
            if bar_pattern_list[pattern_step_it] == -1:
                # 在常见的旋律列表里找不到某一个旋律组合的处理方法：
                # a.寻找符合以下条件的旋律组合
                # a1.首音不为休止符
                # a2.该旋律组合的首音/半拍音与待求旋律组合的首音/半拍音相同
                # a3.该旋律组合中所有音符与待求旋律组合对应位置的音符全部相同
                # a4.满足上述三个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
                # b.记为0
                choose_pattern = 0  # 选取的pattern
                choose_pattern_like_score = 0  # 两个旋律组合的相似程度
                for pattern_it in range(1, len(common_patterns)):
                    # 1.检查首音是否为休止符
                    if common_patterns[pattern_it][0] == 0:
                        continue
                    # 2.检查两个旋律组合的首音是否相同
                    if common_patterns[pattern_it][0] != bar_note_list[pattern_step_it][0]:
                        continue
                    # 3.检查该旋律组合中所有音符与待求旋律组合对应位置的音符是否全部相同
                    note_all_same = True
                    for note_it in range(len(common_patterns[pattern_it])):
                        if common_patterns[pattern_it][note_it] != 0 and common_patterns[pattern_it][note_it] != bar_note_list[pattern_step_it][note_it]:
                            note_all_same = False
                            break
                    if not note_all_same:
                        continue
                    # 4.求该旋律组合与待求旋律组合的差别
                    pattern_like_score = 6  # 初始的旋律组合相似度为6分 每发现一个不同音符 按权重扣分
                    note_diff_list = [10, 2, 3, 3, 6, 3, 4, 3]  # 音符差别的权重列表
                    for note_it in range(len(common_patterns[pattern_it])):
                        if common_patterns[pattern_it][note_it] != bar_note_list[pattern_step_it][note_it]:
                            pattern_like_score -= note_diff_list[note_it]
                    # 5.如果这个旋律组合的差别是目前最小的 则保存它
                    # print(common_melody_iterator, pattern_like_score)
                    if pattern_like_score > choose_pattern_like_score:
                        choose_pattern_like_score = pattern_like_score
                        choose_pattern = pattern_it
                # 6.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_MELODY_PATTERNS+1
                if choose_pattern_like_score > 0:
                    bar_pattern_list[pattern_step_it] = choose_pattern
                else:
                    bar_pattern_list[pattern_step_it] = len(common_patterns)
        return bar_pattern_list


class CoreNotePatternEncode(MusicPatternEncodeStep):

    def handle_rare_pattern(self, pattern_dx, raw_note_list, common_patterns):
        # 在常见的core_note列表里找不到某一个core_note组合的处理方法：
        # a.寻找符合以下条件的core_note组合
        # a1.所有的休止情况完全相同
        # a2.休止情况不相同的 音高差异必须为12的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为common_patterns+1
        if pattern_dx == -1:
            choose_pattern = 0  # 选取的pattern
            min_diff_note = 10  # 最小差异的旋律组合
            for common_pat_it in range(1, len(common_patterns)):
                diff_note_count = 0
                # 1.1.检查该旋律组合是否符合要求
                note_satisfactory = True
                for note_it in range(len(common_patterns[common_pat_it])):
                    # 1.1.1.检查休止情况是否相同
                    if bool(common_patterns[common_pat_it][note_it]) ^ bool(raw_note_list[note_it]):  # 一个为休止符 一个不为休止符
                        note_satisfactory = False
                        break
                    if common_patterns[common_pat_it][note_it] == 0:  # 如果这个时间步长是休止符的话，直接进入下个时间步长
                        continue
                    # 1.1.2.查看音高差异是否为12的倍数
                    if (common_patterns[common_pat_it][note_it] - raw_note_list[note_it]) % 12 != 0:
                        note_satisfactory = False
                        break
                    # 1.1.3.计算该组合与待求组合之间的差异程度
                    if common_patterns[common_pat_it][note_it] != raw_note_list[note_it]:
                        diff_note_count += 1
                if not note_satisfactory:
                    continue
                # 1.2.如果找到符合要求的组合 将其记录下来
                if diff_note_count < min_diff_note:
                    choose_pattern = common_pat_it
                    min_diff_note = diff_note_count
            # 2.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_CORE_NOTE_PATTERN_NUMBER+1
            if choose_pattern != 0:
                pattern_dx = choose_pattern
            else:
                pattern_dx = COMMON_CORE_NOTE_PATTERN_NUMBER + 1
        # print(bar_pattern_list, '\n')
        return pattern_dx


def get_continuous_bar_number(melody_data_dic):
    """
    获取一首歌连续的小节数
    :param melody_data_dic: 这首歌的音符列表
    :return: 这首歌的连续小节数
    """
    # 1.获取歌曲的小节数量
    max_key = get_dict_max_key(melody_data_dic)
    continuous_bar_number_list = [0 for t in range(max_key + 1)]
    # 2.获取歌曲连续小节编号
    for key in range(max_key + 1):
        try:
            if melody_data_dic[key] == [0 for t in range(32)]:
                continuous_bar_number_list[key] = 0
            elif key == 0:
                continuous_bar_number_list[key] = 1
            else:
                continuous_bar_number_list[key] = continuous_bar_number_list[key - 1] + 1
        except KeyError:
            continuous_bar_number_list[key] = 0
    return continuous_bar_number_list


def melody_core_note(flattern_melody_data):
    """
    寻找主旋律的核心音符
    :param flattern_melody_data: 主旋律 展开形式 相对音高
    :return: 核心音列表
    """
    # TODO 不同调式的rel_note和core_note
    core_note_list = []
    for step_it in range(0, len(flattern_melody_data)):
        if flattern_melody_data[step_it] == 0:  # 如果这个时间步长没有音符
            if flattern_melody_data[step_it - step_it % 8] != 0:  # 先观察这一拍的第一个时间步长有没有音符 如果有则取之
                core_note_list.append(flattern_melody_data[step_it - step_it % 8])
            else:  # 如果这个时间步长没有音符 这一拍的第一个时间步长也没有音符 回溯前四拍 都没有则置为0
                find_note = False
                for note_it in range(step_it, max(-1, step_it - 33), -1):
                    if flattern_melody_data[note_it] != 0:
                        core_note_list.append(flattern_melody_data[note_it])
                        find_note = True
                        break
                if find_note is False:
                    core_note_list.append(0)
        else:  # 这一拍有音符
            if flattern_melody_data[step_it - step_it % 8] != 0:  # 不是这一拍的第一个时间步长 那么先观察这一拍的第一个时间步长有没有音符 如果有则取之 如果没有则使用这个时间步长的
                core_note_list.append(flattern_melody_data[step_it - step_it % 8])
            else:
                core_note_list.append(flattern_melody_data[step_it])
    return core_note_list


def melody_core_note_for_chord(raw_melody_data):
    """
    寻找主旋律的核心音符 这里生成的骨干音符组合主要是为了给和弦训练使用
    与上面的function不同之处在于 这里得到的核心音符列表收录了每一拍的第一个音符以及位于半拍位置上且长度大于半怕的音符。同时，没有音符的step的值直接记为零
    :param raw_melody_data: 主旋律 展开形式 绝对音高
    :return: 核心音列表 对12取余数形式
    """
    core_note_list = []
    flattern_melody_data = []
    for key in range(get_dict_max_key(raw_melody_data) + 1):
        flattern_melody_data.extend(raw_melody_data[key])
    for step_it in range(0, len(flattern_melody_data)):
        if flattern_melody_data[step_it] == 0:  # 如果这个时间步长没有音符
            if step_it % 8 == 0:  # 如果这是某一拍的第一个音符
                find_note = False
                for note_it in range(step_it, max(-1, step_it - 33), -8):  # 回溯前四拍的第一个音符 都没有则置为0
                    if flattern_melody_data[note_it] != 0:
                        core_note_list.append(flattern_melody_data[note_it])
                        find_note = True
                        break
                if find_note is False:
                    core_note_list.append(0)
            else:
                core_note_list.append(0)
        else:  # 这一拍有音符
            if step_it % 8 == 0:  # 这是一拍的第一个音 直接收录
                core_note_list.append(flattern_melody_data[step_it])
            elif step_it % 4 == 0:  # 不是这一拍的第一个时间步长 那么先观察这一拍的第一个时间步长有没有音符 如果有则取之 如果没有则使用这个时间步长的
                if flattern_melody_data[step_it + 1: step_it + 4] == [0, 0, 0]:
                    core_note_list.append(flattern_melody_data[step_it])
                else:
                    core_note_list.append(0)
            else:
                core_note_list.append(0)
    for notelist_it in range(0, len(core_note_list), 16):  # 逐两拍将主旋律对12取余数
        core_note_list[notelist_it: notelist_it + 16] = fnc.melody_note_div_12(core_note_list[notelist_it: notelist_it + 16])
    return core_note_list


def get_scale_shift_value(raw_melody_data, section_data):
    """
    获取主旋律连续两小节（不跨越乐段）的音高变化得分 分数为（音高变化/时间差）的平方和
    :param section_data: 这首歌的乐段数据
    :param raw_melody_data: 一首歌的主旋律数据
    :return:
    """
    score_ary = []
    if section_data:  # 训练集中的部分歌没有乐段
        section_data.sort()  # 按照小节先后顺序排序
    for bar_it in range(0, len(raw_melody_data) - 2):
        shift_score = 0
        sec_dx = -1  # 记录这一拍属于这首歌的第几个乐段
        if section_data:  # 有乐段的情况下 当这个小节和下一小节均不为空且不跨越乐段时进行收录
            for sec_it in range(len(section_data)):
                if section_data[sec_it][0] * 4 + section_data[sec_it][1] > bar_it * 4:  # 将区段的起始拍和当前拍进行比较 如果起始拍在当前拍之前则说明是属于这个区段
                    sec_dx = sec_it - 1
                    break
            if sec_dx == -1:
                sec_dx = len(section_data) - 1  # 属于这首歌的最后一个区段
            if section_data[sec_dx][2] == SECTION_EMPTY:  # 这个乐段是间奏 不进行分数选择
                continue
            if sec_dx != len(section_data) - 1:
                if section_data[sec_dx + 1][0] * 4 + section_data[sec_dx + 1][1] < (bar_it + 2) * 4:
                    continue  # 出现跨越乐段的情况，不收录
        else:
            if raw_melody_data[bar_it] == [0] * 32 or raw_melody_data[bar_it + 1] == [0] * 32:
                continue  # 没有乐段的情况下 这一小节和下一小节均不能为空
        last_note = -1  # 上一个音符的音高
        last_note_step = -1  # 上一个音符所在的位置
        note_count = 0  # 两小节一共多少音符
        for cal_bar_it in range(bar_it, bar_it + 2):
            for step_it in range(32):
                if raw_melody_data[cal_bar_it][step_it] != 0:  # 计算变化分
                    if last_note > 0:
                        step_diff = step_it + (cal_bar_it - bar_it) * 32 - last_note_step
                        shift_score += (raw_melody_data[cal_bar_it][step_it] - last_note) * (raw_melody_data[cal_bar_it][step_it] - last_note) / (step_diff * step_diff)
                    last_note = raw_melody_data[cal_bar_it][step_it]
                    last_note_step = step_it + (cal_bar_it - bar_it) * 32
                    note_count += 1
        if note_count == 1:  # 只有一个音符的情况下，音高差异分为0分
            score_ary.append(0)
        elif note_count > 1:
            score_ary.append(shift_score / (note_count - 1))
    return score_ary


def diff_notes_in_1sec(raw_melody_data, section_data):
    score_ary = []
    if not section_data:  # 如果这首歌没有乐段信息，则直接返回空值
        return []
    section_data.sort()  # 按照小节先后顺序排序
    for sec_it in range(len(section_data)):
        if section_data[sec_it][2] == SECTION_EMPTY:
            continue
        start_step_dx = min(int(section_data[sec_it][0] * 32), len(raw_melody_data) * 32)
        if sec_it == len(section_data) - 1:
            end_step_dx = len(raw_melody_data) * 32
        else:
            end_step_dx = min(int(section_data[sec_it + 1][0] * 32), len(raw_melody_data) * 32)
        if end_step_dx - start_step_dx < 4 * 32:
            continue  # 至少要达到4小节才有继续的意义
        middle_step_dx = int((start_step_dx + end_step_dx) / 2)
        note_count_1half = [0] * 12  # 将各个音符与它们持续时间的对数乘积
        note_count_2half = [0] * 12
        keypress_count_1half = [0] * 32
        keypress_count_2half = [0] * 32
        last_note_step = -1  # 上一个音符所在的位置
        last_note = -1
        for bar_it in range(start_step_dx // 32, end_step_dx // 32 + 1):
            for step_it in range(32):
                if bar_it * 32 + step_it < start_step_dx:
                    continue
                if bar_it * 32 + step_it == end_step_dx:  # 到达了最后一拍
                    keypress_count_2half[last_note_step % 32] += math.log2(end_step_dx - last_note_step) + 1
                    note_count_2half[last_note % 12] += math.log2(end_step_dx - last_note_step) + 1
                    break
                if raw_melody_data[bar_it][step_it] == 0:  # 这一拍没音符 直接跳过
                    continue
                if last_note_step < middle_step_dx:
                    if last_note_step != -1:
                        keypress_count_1half[last_note_step % 32] += math.log2(bar_it * 32 + step_it - last_note_step) + 1
                        note_count_1half[last_note % 12] += math.log2(bar_it * 32 + step_it - last_note_step) + 1
                else:
                    if last_note_step != -1:
                        keypress_count_2half[last_note_step % 32] += math.log2(bar_it * 32 + step_it - last_note_step) + 1
                        note_count_2half[last_note % 12] += math.log2(bar_it * 32 + step_it - last_note_step) + 1
                last_note_step = bar_it * 32 + step_it
                last_note = raw_melody_data[last_note_step // 32][last_note_step % 32]
        note_diff = sum([abs(note_count_2half[t] - note_count_1half[t]) for t in range(12)]) / (sum(note_count_1half) + sum(note_count_2half))
        keypress_diff = sum([abs(keypress_count_2half[t] - keypress_count_1half[t]) for t in range(32)]) / (sum(keypress_count_1half) + sum(keypress_count_2half))
        score_ary.append(note_diff + keypress_diff * 2)
    return score_ary


class MelodyProfile(object):
    """使用K-Means训练音符的平均音高"""

    def __init__(self):
        pass

    @staticmethod
    def get_average_note_by_2bar(raw_melody_data):
        """
        获取歌曲小节内音符的平均音高，以2小节为单位。空小节的平均音高为0
        :param raw_melody_data: 一首歌曲的音符列表。dict形式
        :return: 歌曲小节内音符的平均音高 二维数组第一维用空小节分割
        """
        bar_num = get_dict_max_key(raw_melody_data)  # 这首歌的主旋律一共有多少个小节
        if bar_num == -1:
            return []  # 这首歌没有主旋律 返回空列表
        average_note_list = [[]]
        bar = 0  # 当前小节
        while 1:
            if bar > bar_num:
                break
            if bar == bar_num:
                average_note_list[-1].append(sum(raw_melody_data[bar])/(32 - raw_melody_data[bar].count(0)))
                break
            try:
                if raw_melody_data[bar] == [0 for t in range(32)]:  # 遇到空小节
                    if average_note_list[-1]:
                        average_note_list.append([])
                    bar += 1
                    continue
                elif raw_melody_data[bar] != [0 for t in range(32)] and raw_melody_data[bar + 1] == [0 for t in range(32)]:  # 当前小节不为空但是下一小节为空
                    average_note_list[-1].append(sum(raw_melody_data[bar])/(32 - raw_melody_data[bar].count(0)))
                    average_note_list.append([])
                    bar += 2
                elif raw_melody_data[bar] != [0 for t in range(32)] and raw_melody_data[bar + 1] != [0 for t in range(32)]:  # 这一小节与下一小节均不为空
                    average_note_list[-1].append((sum(raw_melody_data[bar]) + sum(raw_melody_data[bar + 1]))/(64 - raw_melody_data[bar].count(0) - raw_melody_data[bar + 1].count(0)))
                    bar += 2
            except KeyError:
                bar += 1
        if not average_note_list[-1]:
            average_note_list.pop(-1)
        # print(melody_cluster)
        return average_note_list

    def define_cluster_model(self, tone_restrict=None, train=True):
        """获取所有歌曲逐两小节的平均音高，并确定模型"""
        # 1.从数据集中读取歌的编号为song_id且小节标注为main的小节数据
        raw_melody_data = get_raw_song_data_from_dataset('main', tone_restrict)
        # 2.逐2小节地记录每一首歌所有区间的平均音高
        melody_avr_list = []
        for song_iterator in range(len(raw_melody_data)):
            if raw_melody_data[song_iterator] != {}:
                melody_avr_list.extend(self.get_average_note_by_2bar(raw_melody_data[song_iterator]))  # 这里'+'换成了ｅｘｔｅｎｄ
        # for t in range(len(melody_avr_list)):
        #     print(melody_avr_list[t])
        flatten_melody_avr_list = []
        for melody_avr in melody_avr_list:  # 将二维数组展成一维数组
            flatten_melody_avr_list.extend(melody_avr)
        # 3.使用K均值算法，对输入的音乐按照两小节的平均音高分为10类
        self.kmeans_model = KMeansModel(flatten_melody_avr_list, 10, 50, train)

    def get_cluster_center_points(self, session, train=True):
        """
        使用K-Means算法获取10个中心点
        :param session: tf.session
        :param train: 是否是训练模式
        :return: 十个中心点
        """
        if train is True:  # 训练状态 获取平均音高
            self.cluster_center_points = self.kmeans_model.cal_centers(session)
        else:  # 非训练模式 从文件中获取平均音高
            self.cluster_center_points = self.kmeans_model.restore_centers(session)
        DiaryLog.warn('平均音高的十个中心点分别为: ' + repr(self.cluster_center_points))

    def get_melody_profile_by_song(self, session, raw_melody_data):
        # 这个函数的前半部分与GetMelodyClusterByBar有一些区别 所以不能直接调用
        # 1.准备工作：记录歌曲有多少个小节，以及定义一些变量
        bar_num = get_dict_max_key(raw_melody_data)
        if bar_num == -1:
            return []  # 这首歌没有主旋律 返回空列表
        melody_avr_note_list = [0 for t in range(bar_num + 1)]  # 将这首歌逐两小节的音符平均音高记录在这个列表中
        bar = 0
        # 2.逐2小节地记录一个区间的平均音高
        while 1:
            if bar > bar_num:
                break
            if bar == bar_num:  # 已达最后一小节
                melody_avr_note_list[bar] = (sum(raw_melody_data[bar]) / (32 - raw_melody_data[bar].count(0)))
                break
            try:
                if raw_melody_data[bar] == [0 for t in range(32)]:  # 这一小节为空 不记录
                    bar += 1
                    continue
                elif raw_melody_data[bar] != [0 for t in range(32)] and raw_melody_data[bar + 1] == [0 for t in range(32)]:  # 这一小节不为空 但下一小节为空
                    melody_avr_note_list[bar] = (sum(raw_melody_data[bar]) / (32 - raw_melody_data[bar].count(0)))
                    bar += 2
                elif raw_melody_data[bar] != [0 for t in range(32)] and raw_melody_data[bar + 1] != [0 for t in range(32)]:  # 这一小节和下一小节均不为空
                    melody_avr_note_list[bar] = melody_avr_note_list[bar + 1] = ((sum(raw_melody_data[bar]) + sum(raw_melody_data[bar + 1])) / (64 - raw_melody_data[bar].count(0) - raw_melody_data[bar + 1].count(0)))
                    bar += 2
            except KeyError:
                bar += 1
        # print(melody_avr_note_list)
        # print(melody_avr_note_list)
        # 2.通过K均值算法将音高列表分类
        attachment_array = self.kmeans_model.run_attachment(session, self.cluster_center_points, melody_avr_note_list)  # 每个小节的分类情况
        # 3.将音高分类情况进行微调 如果小节旋律为空 则分类为0 否则分类为1-10
        for bar_it in range(bar_num + 1):
            if melody_avr_note_list[bar_it] != 0:
                attachment_array[bar_it] += 1
            else:
                attachment_array[bar_it] = 0
        # print(attachment_array)
        return attachment_array


class MelodyTrainData:

    def __init__(self, tone_restrict=None):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        # 注：后缀为nres的表示没有调式限制
        # 1.从数据集中读取歌的编号为song_id且小节标注为mark_key的小节数据
        self.raw_train_data = get_raw_song_data_from_dataset('main', tone_restrict)
        raw_melody_data_nres = get_raw_song_data_from_dataset('main', None)  # 没有旋律限制的主旋律数据　用于训练其他数据
        self.raw_melody_data = copy.deepcopy(raw_melody_data_nres)  # 最原始的无调式限制的主旋律数据
        self.continuous_bar_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 每一首歌连续的小节计数
        self.continuous_bar_data_nres = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 每一首歌无限制的小节计数

        self.keypress_pat_ary = [[0 for t in range(16)]]  # 主旋律按键组合的对照表
        self.keypress_pat_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 二维数组 第一维是歌曲列表 第二维是按键的组合（步长是2拍）
        self.keypress_pat_count = [0]  # 各种按键组合的计数

        self.rel_melody_data_nres = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 相对音高列表
        self.core_note_ary_nres = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 每一首歌每一拍的骨干音列表
        self.raw_core_note_nres_for_chord = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 为和弦训练提供的另一组骨干音列表 上面的变量是原始数据 下面的列表是组合数据
        self.core_note_pat_nres_for_chord = [[] for t in range(TRAIN_FILE_NUMBERS)]

        # 2.获取最常见的主旋律组合
        common_pattern_cls = CommonMusicPatterns(COMMON_MELODY_PATTERN_NUMBER)  # 这个类可以获取常见的主旋律组合
        if FLAG_IS_TRAINING is True:  # 训练模式
            common_pattern_cls.train(self.raw_train_data, 0.125, 1, 'bar')
            common_pattern_cls.store('melody')  # 存储在sqlite文件中
        else:
            common_pattern_cls.restore('melody')  # 直接从sqlite文件中读取
        self.common_melody_pats = common_pattern_cls.common_pattern_list  # 常见的旋律组合列表
        self.melody_pats_num_list = common_pattern_cls.pattern_number_list  # 这些旋律组合出现的次数列表

        # 3.逐歌曲获取音符组合数据
        self.melody_pat_data = [{} for t in range(TRAIN_FILE_NUMBERS)]  # 有调式限制的旋律组合数据
        self.melody_pat_data_nres = [{} for t in range(TRAIN_FILE_NUMBERS)]  # 无调式限制的旋律组合数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            # 3.1.获取没有调式限制的旋律数据
            if raw_melody_data_nres[song_it] != {}:  # 获取相关没有调式限制的相关数据
                # 3.1.1.获取以歌曲为单位的主旋律
                flatten_melody_data = []
                for key in range(get_dict_max_key(raw_melody_data_nres[song_it]) + 1):
                    flatten_melody_data.extend(raw_melody_data_nres[song_it][key])
                # 3.1.2.获取按键数据
                self.get_keypress_data(song_it, raw_melody_data_nres[song_it])  # 获取按键数据 当前有按键记为1 没有按键记为0
                # 3.1.3.获取相对音高数据
                if tone_restrict == TONE_MAJOR:
                    self.rel_melody_data_nres[song_it] = fnc.one_song_rel_notelist_melody(flatten_melody_data, TONE_MAJOR, 72)
                elif tone_restrict == TONE_MINOR:
                    self.rel_melody_data_nres[song_it] = fnc.one_song_rel_notelist_melody(flatten_melody_data, TONE_MINOR, 69)
                else:
                    raise ValueError
                # 3.1.4.获取骨干音及骨干音的常见组合列表
                self.core_note_ary_nres[song_it] = melody_core_note(self.rel_melody_data_nres[song_it])
                self.raw_core_note_nres_for_chord[song_it] = melody_core_note_for_chord(raw_melody_data_nres[song_it])
                # 3.1.5.将它的主旋律编码为常见的旋律组合。如果该旋律组合不常见，则记为COMMON_MELODY_PATTERN_NUMBER+1
                self.continuous_bar_data_nres[song_it] = get_continuous_bar_number(raw_melody_data_nres[song_it])
                self.melody_pat_data_nres[song_it] = MelodyPatternEncode(self.common_melody_pats, raw_melody_data_nres[song_it], 0.125, 1).music_pattern_dic
            # 3.2.获取有调式限制的旋律数据
            if self.raw_train_data[song_it] != {}:
                # 3.2.1.开头补上几个空小节 便于训练开头几小节的旋律数据
                for bar_it in range(1 - TRAIN_MELODY_IO_BARS, 0):
                    self.raw_train_data[song_it][bar_it] = [0 for t in range(32)]
                # 3.2.2.获取歌曲的连续不为空的小节序号列表
                self.continuous_bar_data[song_it] = get_continuous_bar_number(self.raw_train_data[song_it])
                # print(melody_profile_data)
                # 3.2.3.将它的主旋律编码为常见的旋律组合。如果该旋律组合不常见，则记为COMMON_MELODY_PATTERN_NUMBER+1
                self.melody_pat_data[song_it] = MelodyPatternEncode(self.common_melody_pats, self.raw_train_data[song_it], 0.125, 1).music_pattern_dic
        # 4.获取无调式限制的常见骨干音组合列表 并对原始的骨干音列表进行编码
        core_note_pattern_cls = CommonMusicPatterns(COMMON_CORE_NOTE_PATTERN_NUMBER)
        if FLAG_IS_TRAINING is True:  # 训练模式
            core_note_pattern_cls.train(self.raw_core_note_nres_for_chord, 0.125, 2)
            core_note_pattern_cls.store('core_note')  # 存储在sqlite文件中
        else:
            core_note_pattern_cls.restore('core_note')  # 直接从sqlite文件中读取
        self.common_corenote_pats = core_note_pattern_cls.common_pattern_list  # 常见的旋律组合列表
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_melody_data_nres[song_it] != {}:  # 获取相关没有调式限制的相关数据
                self.core_note_pat_nres_for_chord[song_it] = CoreNotePatternEncode(self.common_corenote_pats, self.raw_core_note_nres_for_chord[song_it], 0.125, 2).music_pattern_ary

    def get_keypress_data(self, song_dx, raw_melody_data):
        """
        逐两拍获取一首歌音符打击的情况
        :param int song_dx: 这首歌在歌曲列表中的编号
        :param raw_melody_data: 这首歌原始的主旋律数据
        :return:
        """
        self.keypress_pat_data[song_dx] = [0 for t in range((get_dict_max_key(raw_melody_data) + 1) * 2)]  # 按键组合数据以两拍为单位
        for bar in range(get_dict_max_key(raw_melody_data) + 1):
            bar_keypress_data = [1 if t != 0 else 0 for t in raw_melody_data[bar]]  # 一个小节的按键数据
            bar_pattern_list = [bar_keypress_data[16 * t: 16 * (t + 1)] for t in range(2)]
            for bar_pattern_it, raw_pattern in enumerate(bar_pattern_list):
                if raw_pattern not in self.keypress_pat_ary:
                    self.keypress_pat_data[song_dx][bar * 2 + bar_pattern_it] = len(self.keypress_pat_ary)
                    self.keypress_pat_ary.append(raw_pattern)
                    self.keypress_pat_count.append(1)
                else:
                    self.keypress_pat_data[song_dx][bar * 2 + bar_pattern_it] = self.keypress_pat_ary.index(raw_pattern)
                    self.keypress_pat_count[self.keypress_pat_ary.index(raw_pattern)] += 1

    def get_model_io_data(self, session, melody_profile):
        for song_it in range(len(self.raw_train_data)):
            if self.raw_train_data[song_it] != {}:
                # 1.使用K均值算法，对输入的音乐按照两小节的平均音高分为10类
                melody_profile_data = melody_profile.get_melody_profile_by_song(session, self.raw_train_data[song_it])
                # 2.生成这首歌的训练数据 输入内容是当前时间 五小节的melody_profile 过去16拍的旋律组合，输出内容是当前时间 五小节的melody_profile 和错后一拍的旋律组合
                self.get_model_io_data_1song(self.melody_pat_data[song_it], self.continuous_bar_data[song_it], melody_profile_data)
        # print(len(self.input_data), len(self.output_data))
        # print('\n\n\n\n\n')
        # for t in self.input_data[:50]:
        #     print(t)
        # print('\n\n\n')
        # for t in self.output_data[:50]:
        #     print(t)

    def get_model_io_data_1song(self, melody_pat_data, continuous_bar_data, melody_profile_data):
        """
        将一首歌的数据输入到model中
        :param melody_pat_data: 一首歌旋律组合的数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param melody_profile_data: 一首歌逐小节的melody profile列表
        :return:
        """
        for bar in melody_pat_data:
            for time_in_bar in range(4):
                try:
                    # 1.添加当前时间
                    time_add = (1 - continuous_bar_data[bar + TRAIN_MELODY_IO_BARS] % 2) * 4  # 这个时间在2小节之中的位置
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 2.添加melody_profile 向前回溯4小节
                    if bar >= 0:
                        input_time_data += melody_profile_data[bar: bar + TRAIN_MELODY_IO_BARS + 1]  # 添加melody_profile
                        output_time_data += melody_profile_data[bar: bar + TRAIN_MELODY_IO_BARS + 1]
                    else:
                        input_time_data += [0 for t in range(bar, 0)] + melody_profile_data[0: bar + TRAIN_MELODY_IO_BARS + 1]
                        output_time_data += [0 for t in range(bar, 0)] + melody_profile_data[0: bar + TRAIN_MELODY_IO_BARS + 1]
                    # 3.添加当前小节的旋律组合
                    input_time_data += melody_pat_data[bar][time_in_bar: 4]
                    output_time_data += melody_pat_data[bar][(time_in_bar + 1): 4]  # output_Data错后一拍
                    # 4.添加之后3个小节的旋律组合
                    for bar_it in range(bar + 1, bar + TRAIN_MELODY_IO_BARS):
                        input_time_data = input_time_data + melody_pat_data[bar_it]
                        output_time_data = output_time_data + melody_pat_data[bar_it]
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[0: (1 + TRAIN_MELODY_IO_BARS + 1)] + [0 for t in range(len(input_time_data) - (1 + TRAIN_MELODY_IO_BARS + 1))]
                            output_time_data = output_time_data[0: (1 + TRAIN_MELODY_IO_BARS + 1)] + [0 for t in range(len(output_time_data) - (1 + TRAIN_MELODY_IO_BARS + 1))]
                    # 5.添加最后一个小节的旋律组合
                    input_time_data += melody_pat_data[bar + TRAIN_MELODY_IO_BARS][0: time_in_bar]
                    output_time_data += melody_pat_data[bar + TRAIN_MELODY_IO_BARS][0: (time_in_bar + 1)]
                    # 6.当输出数据所在的小节与其前一小节均不为空时，该数据收录进训练集
                    if melody_pat_data[bar + TRAIN_MELODY_IO_BARS] != [0 for t in range(4)] and melody_pat_data[bar + TRAIN_MELODY_IO_BARS - 1] != [0 for t in range(4)]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass


class MelodyTrainData2(MelodyTrainData):

    def __init__(self, tone_restrict=None):
        super().__init__(tone_restrict)
        # 5.生成每首歌的训练数据
        for song_it in range(len(self.raw_train_data)):
            if self.raw_train_data[song_it] != {}:
                self.get_model_io_data_1song(self.melody_pat_data[song_it], self.continuous_bar_data[song_it], None)

    def get_model_io_data_1song(self, melody_pat_data, continuous_bar_data, melody_profile_data):
        """
        输入内容为当前时间的编码 过去四小节的主旋律 MelodyProfile 输出内容为这一拍的主旋律
        :param melody_pat_data: 一首歌旋律组合的数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param melody_profile_data: None
        :return:
        """
        for bar_dx in melody_pat_data:
            for time_in_bar in range(4):
                try:
                    input_time_data = []
                    output_time_data = []
                    melody1_code_add_base = 8  # 主旋律数据编码增加的基数
                    # 1.添加当前小节的时间和旋律组合
                    if bar_dx >= 0:
                        time_add = 0 if continuous_bar_data[bar_dx] == 0 else (1 - continuous_bar_data[bar_dx] % 2) * 4
                        input_time_data.extend([[time_add + t, melody_pat_data[bar_dx][t] + melody1_code_add_base] for t in range(time_in_bar, 4)])
                        output_time_data.extend([melody_pat_data[bar_dx][t] + melody1_code_add_base for t in range((time_in_bar + 1), 4)])
                    else:
                        input_time_data.extend([[t, melody1_code_add_base] for t in range(time_in_bar, 4)])
                        output_time_data.extend([melody1_code_add_base for t in range((time_in_bar + 1), 4)])
                    # 2.添加之后3个小节的旋律组合
                    for bar_it in range(bar_dx + 1, bar_dx + TRAIN_MELODY_IO_BARS):
                        if bar_it > 0:
                            time_add = 0 if continuous_bar_data[bar_it] == 0 else (1 - continuous_bar_data[bar_it] % 2) * 4
                            input_time_data.extend([[time_add + t, melody_pat_data[bar_it][t] + melody1_code_add_base] for t in range(4)])
                            output_time_data.extend([melody_pat_data[bar_it][t] + melody1_code_add_base for t in range(4)])
                        else:
                            input_time_data.extend([[t, melody1_code_add_base] for t in range(4)])
                            output_time_data.extend([melody1_code_add_base for t in range(4)])
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data[:-4] = [[t[0] % 4, melody1_code_add_base] for t in input_time_data[:-4]]
                            output_time_data[:-4] = [melody1_code_add_base for t in range(len(output_time_data) - 4)]
                    # 3.添加最后一个小节的旋律组合
                    time_add = (1 - continuous_bar_data[bar_dx + TRAIN_MELODY_IO_BARS] % 2) * 4
                    input_time_data.extend([[time_add + t, melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS][t] + melody1_code_add_base] for t in range(time_in_bar)])
                    output_time_data.extend([melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS][t] + melody1_code_add_base for t in range(time_in_bar + 1)])
                    # 4.当输出数据所在的小节与其前一小节均不为空时，该数据收录进训练集
                    if melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS] != [0 for t in range(4)] and melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS - 1] != [0 for t in range(4)]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass


class MelodyTrainDataNoProfile(MelodyTrainData2):

    def __init__(self, tone_restrict=None):
        super().__init__(tone_restrict)
        # 6.获取每首歌的乐段信息
        self.section_data = get_section_data_from_dataset()
        # 7.生成每首歌的旋律变化累积幅度数据
        shift_score_ary = []
        diff_note_insec_score_ary = []
        for song_it in range(len(self.raw_train_data)):
            if self.raw_train_data[song_it] != {}:
                shift_score_ary.extend(get_scale_shift_value(self.raw_melody_data[song_it], copy.deepcopy(self.section_data[song_it])))
                diff_note_insec_score_ary.extend(diff_notes_in_1sec(self.raw_melody_data[song_it], copy.deepcopy(self.section_data[song_it])))
        # 8.找出前95%所在位置
        shift_score_ary = sorted(shift_score_ary)
        prob_095_dx = int(len(shift_score_ary) * 0.95 + 1)
        self.ShiftConfidenceLevel = shift_score_ary[prob_095_dx]
        diff_note_insec_score_ary = sorted(diff_note_insec_score_ary)
        prob_095_dx = int(len(diff_note_insec_score_ary) * 0.95 + 1)
        self.DfNoteConfidenceLevel = diff_note_insec_score_ary[prob_095_dx]

    def get_model_io_data_1song(self, melody_pat_data, continuous_bar_data, melody_profile_data):
        """
        输入内容为当前时间的编码 过去四小节的主旋律 MelodyProfile 输出内容为这一拍的主旋律
        :param melody_pat_data: 一首歌旋律组合的数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param melody_profile_data: None
        :return:
        """
        for bar in melody_pat_data:
            for time_in_bar in range(4):
                try:
                    # 1.添加当前时间
                    time_add = (1 - continuous_bar_data[bar + TRAIN_MELODY_IO_BARS] % 2) * 4  # 这个时间在2小节之中的位置
                    input_time_data = [time_in_bar + time_add]
                    output_time_data = [time_in_bar + time_add]
                    # 3.添加当前小节的旋律组合
                    input_time_data += melody_pat_data[bar][time_in_bar: 4]
                    output_time_data += melody_pat_data[bar][(time_in_bar + 1): 4]  # output_Data错后一拍
                    # 4.添加之后3个小节的旋律组合
                    for bar_it in range(bar + 1, bar + TRAIN_MELODY_IO_BARS):
                        input_time_data = input_time_data + melody_pat_data[bar_it]
                        output_time_data = output_time_data + melody_pat_data[bar_it]
                        if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                            input_time_data = input_time_data[0: (1 + TRAIN_MELODY_IO_BARS + 1)] + [0 for t in range(len(input_time_data) - (1 + TRAIN_MELODY_IO_BARS + 1))]
                            output_time_data = output_time_data[0: (1 + TRAIN_MELODY_IO_BARS + 1)] + [0 for t in range(len(output_time_data) - (1 + TRAIN_MELODY_IO_BARS + 1))]
                    # 5.添加最后一个小节的旋律组合
                    input_time_data += melody_pat_data[bar + TRAIN_MELODY_IO_BARS][0: time_in_bar]
                    output_time_data += melody_pat_data[bar + TRAIN_MELODY_IO_BARS][0: (time_in_bar + 1)]
                    # 6.当输出数据所在的小节与其前一小节均不为空时，该数据收录进训练集
                    if melody_pat_data[bar + TRAIN_MELODY_IO_BARS] != [0 for t in range(4)] and melody_pat_data[bar + TRAIN_MELODY_IO_BARS - 1] != [0 for t in range(4)]:
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass


class MelodyTrainData3(MelodyTrainData):

    melody1_code_add_base = 16  # 主旋律数据编码增加的基数

    def __init__(self, tone_restrict=None):
        super().__init__(tone_restrict)
        # 5.获取每首歌的乐段信息
        self.section_data = get_section_data_from_dataset()
        # 6.生成每首歌的训练数据 输入内容是当前时间 过去16拍的旋律组合 输出内容是错后一拍的旋律组合
        for song_it in range(len(self.raw_train_data)):
            if self.raw_train_data[song_it] != {}:
                self.get_model_io_data_1song(self.melody_pat_data[song_it], self.continuous_bar_data[song_it], self.section_data[song_it])

    def get_model_io_data_1song(self, melody_pat_data, continuous_bar_data, melody_section_data):
        """
        输入内容为当前时间的编码 过去四小节的主旋律 输出内容为这一拍的主旋律
        时间编码为时间在乐段中的位置，0-8表示非过渡状态，9-16表示过渡状态。
        主旋律的起点是max(四小节前，乐段的起点)，而不以上一拍是否为空拍来判断
        :param melody_pat_data: 一首歌旋律组合的数据，dict形式
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        :param melody_section_data: 这首歌的乐段信息
        :return:
        """
        if melody_section_data:  # 训练集中的部分歌没有乐段
            melody_section_data.sort()  # 按照小节先后顺序排序
        melody_max_key = get_dict_max_key(melody_pat_data)
        for bar_dx in range(1 - TRAIN_MELODY_IO_BARS, 1 - TRAIN_MELODY_IO_BARS + melody_max_key):
            for time_in_bar in range(4):
                try:
                    if melody_section_data:
                        # 1.有乐段的情况
                        # 1.1.找出这一拍属于一首歌的哪一个乐段
                        sec_dx = -1  # 记录这一拍属于这首歌的第几个乐段
                        for sec_it in range(len(melody_section_data)):
                            if melody_section_data[sec_it][0] * 4 + melody_section_data[sec_it][1] > (bar_dx + TRAIN_MELODY_IO_BARS) * 4 + time_in_bar:  # 将区段的起始拍和当前拍进行比较 如果起始拍在当前拍之前则说明是属于这个区段
                                sec_dx = sec_it - 1
                                break
                        if sec_dx == -1:
                            sec_dx = len(melody_section_data) - 1  # 属于这首歌的最后一个区段
                        if melody_section_data[sec_dx][2] == SECTION_EMPTY:  # 这个乐段是间奏 不进行训练
                            continue
                        # 1.2.添加当前时间
                        sec_start_maj_beat = int(melody_section_data[sec_dx][0] * 4)  # 这个乐段的起始重拍（即不计算前面的
                        sec_start_first_beat = int(melody_section_data[sec_dx][0] * 4 + melody_section_data[sec_dx][1])  # 这个乐段真正的起始拍
                        if (bar_dx + TRAIN_MELODY_IO_BARS) * 4 + time_in_bar - sec_start_first_beat < 0:  # 距离乐段开始不足3拍 且前一个段落是empty的，不加入训练集
                            continue
                        if (bar_dx + TRAIN_MELODY_IO_BARS) * 4 + time_in_bar - sec_start_first_beat <= 2:
                            if sec_start_first_beat < 4 or (melody_pat_data[sec_start_first_beat // 4][:sec_start_first_beat % 4] == [0] * (sec_start_first_beat % 4) and melody_pat_data[sec_start_first_beat // 4 - 1][sec_start_first_beat % 4:] == [0] * (4 - sec_start_first_beat % 4)):
                                continue
                        time_add = ((bar_dx + TRAIN_MELODY_IO_BARS) * 4 + time_in_bar - sec_start_maj_beat) % 8
                        if sec_dx == len(melody_section_data) - 1:
                            sec_end_beat = len(melody_pat_data) * 4
                        else:
                            sec_end_beat = min(len(melody_pat_data) * 4, melody_section_data[sec_dx + 1][0] * 4)  # 这个乐段的结尾重拍在第多少拍
                        if 5 <= sec_end_beat - ((bar_dx + TRAIN_MELODY_IO_BARS) * 4 + time_in_bar) + time_add % 4 <= 8:  # 这一拍处在距离乐段结束1-2小节的位置
                            time_add = 8 + time_add % 4
                        elif sec_end_beat - ((bar_dx + TRAIN_MELODY_IO_BARS) * 4 + time_in_bar) + time_add % 4 <= 4:  # 这一拍处在距离乐段结束不足1小节的位置
                            time_add = 12 + time_add % 4
                        input_time_data = [time_add]
                        output_time_data = [time_add]
                        # 1.3.添加当前小节的旋律组合
                        for beat_it in range(time_in_bar, 4):
                            if bar_dx * 4 + beat_it < sec_start_first_beat:
                                input_time_data.append(self.melody1_code_add_base)
                            else:
                                input_time_data.append(self.melody1_code_add_base + melody_pat_data[bar_dx][beat_it])
                            if beat_it != time_in_bar:  # output_time_data没有第一拍
                                if bar_dx * 4 + beat_it < sec_start_first_beat:
                                    output_time_data.append(self.melody1_code_add_base)
                                else:
                                    output_time_data.append(self.melody1_code_add_base + melody_pat_data[bar_dx][beat_it])
                        # 1.4.添加后三个小节的旋律组合
                        for bar_it in range(bar_dx + 1, bar_dx + TRAIN_MELODY_IO_BARS):
                            for beat_it in range(4):
                                if bar_it >= bar_dx + TRAIN_MELODY_IO_BARS - 2 and (bar_dx + TRAIN_MELODY_IO_BARS) * 4 + time_in_bar - sec_start_first_beat <= 3:  # 对于乐段的第一个小节 训练时增加前面乐段的最后两小节
                                    input_time_data.append(self.melody1_code_add_base + melody_pat_data[bar_it][beat_it])
                                    output_time_data.append(self.melody1_code_add_base + melody_pat_data[bar_it][beat_it])
                                elif bar_it * 4 + beat_it < sec_start_first_beat:
                                    input_time_data.append(self.melody1_code_add_base)
                                    output_time_data.append(self.melody1_code_add_base)
                                else:
                                    input_time_data.append(self.melody1_code_add_base + melody_pat_data[bar_it][beat_it])
                                    output_time_data.append(self.melody1_code_add_base + melody_pat_data[bar_it][beat_it])
                        # 1.5.添加最后一小节的旋律组合
                        for beat_it in range(time_in_bar + 1):
                            if beat_it != time_in_bar:  # input_time_data没有最后一拍
                                # if (bar_dx + TRAIN_MELODY_IO_BARS) * 4 + beat_it < sec_start_first_beat:
                                #     input_time_data.append(self.melody1_code_add_base)
                                # else:
                                input_time_data.append(self.melody1_code_add_base + melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS][beat_it])
                            # if (bar_dx + TRAIN_MELODY_IO_BARS) * 4 + beat_it < sec_start_first_beat:
                            #     output_time_data.append(self.melody1_code_add_base)
                            # else:
                            output_time_data.append(self.melody1_code_add_base + melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS][beat_it])
                        # 1.6.将该数据收录进训练集
                        self.input_data.append(input_time_data)
                        self.output_data.append(output_time_data)
                    else:
                        # 2.没有乐段的情况 仍用之前的方式生成
                        # 2.1.添加当前时间
                        time_add = (1 - continuous_bar_data[bar_dx + TRAIN_MELODY_IO_BARS] % 2) * 4  # 这个时间在2小节之中的位置
                        input_time_data = [time_in_bar + time_add]
                        output_time_data = [time_in_bar + time_add]
                        # 2.2.添加当前小节的旋律组合
                        input_time_data.extend([t + self.melody1_code_add_base for t in melody_pat_data[bar_dx][time_in_bar: 4]])
                        output_time_data.extend([t + self.melody1_code_add_base for t in melody_pat_data[bar_dx][(time_in_bar + 1): 4]])  # output_Data错后一拍
                        # 2.3.添加之后3个小节的旋律组合
                        for bar_it in range(bar_dx + 1, bar_dx + TRAIN_MELODY_IO_BARS):
                            input_time_data.extend([t + self.melody1_code_add_base for t in melody_pat_data[bar_it]])
                            output_time_data.extend([t + self.melody1_code_add_base for t in melody_pat_data[bar_it]])
                            if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                                input_time_data = [input_time_data[0]] + [self.melody1_code_add_base for t in range(len(input_time_data) - 1)]
                                output_time_data = [output_time_data[0]] + [self.melody1_code_add_base for t in range(len(output_time_data) - 1)]
                        # 2.4.添加最后一个小节的旋律组合
                        input_time_data += [t + self.melody1_code_add_base for t in melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS][0: time_in_bar]]
                        output_time_data += [t + self.melody1_code_add_base for t in melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS][0: (time_in_bar + 1)]]
                        # 2.5.当输出数据所在的小节与其前一小节均不为空时，该数据收录进训练集
                        if melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS] != [0 for t in range(4)] and melody_pat_data[bar_dx + TRAIN_MELODY_IO_BARS - 1] != [0 for t in range(4)]:
                            self.input_data.append(input_time_data)
                            self.output_data.append(output_time_data)
                except KeyError:
                    pass
                except IndexError:
                    pass
