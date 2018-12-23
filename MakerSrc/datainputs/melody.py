from settings import *
from models.KMeansModel import KMeansModel
from interfaces.sql.sqlite import get_raw_song_data_from_dataset, get_section_data_from_dataset
from interfaces.utils import flat_array
from interfaces.music_patterns import BaseMusicPatterns, CommonMusicPatterns, MusicPatternEncode
from validations.melody import ShiftConfidenceCheck, DiffNoteConfidenceCheck
from interfaces.utils import DiaryLog
import numpy as np
import copy
import math
import os


def melody_note_div_12(note_list):
    """
    将一个音符列表全部-n*12 但不能出现负数
    :param note_list: 音符列表(绝对音高形式)
    :return: 处理之后的音符列表
    """
    min_note = math.inf  # 音高最低的音符
    output_notelist = []
    for note in note_list:
        if note <= min_note and note != 0:
            min_note = note
    pitch_adj = 12 * ((min_note - 1) // 12)
    for note in note_list:
        if note == 0:
            output_notelist.append(0)
        else:
            output_notelist.append(note - pitch_adj)
    return output_notelist


def get_continuous_bar_cnt(raw_melody_data):
    """
    获取一首歌连续的小节数
    :param raw_melody_data: 这首歌的音符列表
    :return: 这首歌的连续小节数
    """
    # 1.获取歌曲的小节数量
    bar_num = math.ceil(len(raw_melody_data) / 32)
    continuous_bar_cnt_list = [0 for t in range(bar_num)]
    # 2.获取歌曲连续小节编号
    for key in range(bar_num):
        if raw_melody_data[key * 32: (key + 1) * 32] == [0 for t in range(32)]:
            continuous_bar_cnt_list[key] = 0
        elif key == 0:
            continuous_bar_cnt_list[key] = 1
        else:
            continuous_bar_cnt_list[key] = continuous_bar_cnt_list[key - 1] + 1
    return continuous_bar_cnt_list


def melody_core_note(raw_melody_data, continuous_bar_data, section_data):
    """
    寻找主旋律的核心音符 这里生成的骨干音符组合主要是为了给和弦训练使用
    得到的核心音符列表收录了每一拍的第一个音符以及位于半拍位置上且长度大于半怕的音符。同时，没有音符的step的值直接记为零
    :param raw_melody_data: 主旋律 以步长为单位 绝对音高
    :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
    :param section_data: 一首歌的乐段数据
    :return: 核心音列表 对12取余数形式
    """
    core_note_list = []
    for step_it in range(len(raw_melody_data)):
        # 1.间奏处没有core_note
        if section_data:
            sec_dx = -1  # 记录这一拍属于这首歌的第几个乐段
            for sec_it in range(len(section_data)):
                if section_data[sec_it][0] * 4 + section_data[sec_it][1] > step_it // 8:  # 将区段的起始拍和当前拍进行比较 如果起始拍在当前拍之前则说明是属于这个区段
                    sec_dx = sec_it - 1
                    break
            if sec_dx == -1:
                sec_dx = len(section_data) - 1  # 属于这首歌的最后一个区段
            if section_data[sec_dx][2] == DEF_SEC_EMPTY:  # 这个乐段是间奏 不进行训练
                core_note_list.append(0)
                continue
        else:
            if continuous_bar_data[step_it // 32] == 0:
                core_note_list.append(0)
                continue

        # 2.对一个步长 确定此时的核心音符
        if raw_melody_data[step_it] == 0:  # 如果这个时间步长没有音符
            if step_it % 8 == 0:  # 如果这是某一拍的第一个音符
                find_note = False
                for note_it in range(step_it, max(-1, step_it - 33), -8):  # 回溯前四拍的第一个音符 都没有则置为0
                    if raw_melody_data[note_it] != 0:
                        core_note_list.append(raw_melody_data[note_it])
                        find_note = True
                        break
                if find_note is False:
                    core_note_list.append(0)
            else:  # 不是某一拍的第一个音符 则置为0
                core_note_list.append(0)
        else:  # 这一拍有音符
            if step_it % 8 == 0:  # 这是一拍的第一个音 直接收录
                core_note_list.append(raw_melody_data[step_it])
            elif step_it % 4 == 0:  # 不是这一拍的第一个时间步长 那么先观察这一拍的第一个时间步长有没有音符 如果有则取之 如果没有则使用这个时间步长的
                if raw_melody_data[step_it + 1: step_it + 4] == [0, 0, 0]:
                    core_note_list.append(raw_melody_data[step_it])
                else:
                    core_note_list.append(0)
            else:
                core_note_list.append(0)
    for notelist_it in range(0, len(core_note_list), 16):  # 逐两拍将主旋律对12取余数
        core_note_list[notelist_it: notelist_it + 16] = melody_note_div_12(core_note_list[notelist_it: notelist_it + 16])
    return core_note_list


class MelodyPatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, pat_step_dx, raw_note_list, common_patterns):
        # 在常见的旋律列表里找不到某一个旋律组合的处理方法：
        # a.寻找符合以下条件的旋律组合
        # a1.首音不为休止符
        # a2.该旋律组合的首音/半拍音与待求旋律组合的首音/半拍音相同
        # a3.该旋律组合中所有音符与待求旋律组合对应位置的音符全部相同
        # a4.满足上述三个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为COMMON_MELODY_PAT_NUM+1
        choose_pattern = 0  # 选取的pattern
        choose_pattern_like_score = 0  # 两个旋律组合的相似程度
        for pattern_it in range(1, len(common_patterns)):
            # 1.检查首音是否为休止符
            if common_patterns[pattern_it][0] == 0:
                continue
            # 2.检查两个旋律组合的首音是否相同
            if common_patterns[pattern_it][0] != raw_note_list[0]:
                continue
            # 3.检查该旋律组合中所有音符与待求旋律组合对应位置的音符是否全部相同
            note_all_same = True
            for note_it in range(len(common_patterns[pattern_it])):
                if common_patterns[pattern_it][note_it] != 0 and common_patterns[pattern_it][note_it] != raw_note_list[note_it]:
                    note_all_same = False
                    break
            if not note_all_same:
                continue
            # 4.求该旋律组合与待求旋律组合的差别
            pattern_like_score = 6  # 初始的旋律组合相似度为6分 每发现一个不同音符 按权重扣分
            note_diff_list = [10, 2, 3, 3, 6, 3, 4, 3]  # 音符差别的权重列表
            for note_it in range(len(common_patterns[pattern_it])):
                if common_patterns[pattern_it][note_it] != raw_note_list[note_it]:
                    pattern_like_score -= note_diff_list[note_it]
            # 5.如果这个旋律组合的差别是目前最小的 则保存它
            # print(common_melody_iterator, pattern_like_score)
            if pattern_like_score > choose_pattern_like_score:
                choose_pattern_like_score = pattern_like_score
                choose_pattern = pattern_it
        # 6.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_MELODY_PATTERNS+1
        if choose_pattern_like_score > 0:
            pattern_dx = choose_pattern
        else:
            pattern_dx = len(common_patterns)
        return pattern_dx


class CoreNotePatternEncode(MusicPatternEncode):

    def handle_rare_pattern(self, pat_step_dx, raw_note_list, common_patterns):
        # 在常见的core_note列表里找不到某一个core_note组合的处理方法：
        # a.寻找符合以下条件的core_note组合
        # a1.所有的休止情况完全相同
        # a2.休止情况不相同的 音高差异必须为12的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为common_patterns+1
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
        # 2.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_CORE_NOTE_PAT_NUM+1
        if choose_pattern != 0:
            pattern_dx = choose_pattern
        else:
            pattern_dx = COMMON_CORE_NOTE_PAT_NUM + 1
        return pattern_dx


class MelodyProfile(object):
    """使用K-Means训练音符的平均音高"""

    def __init__(self):
        pass

    @staticmethod
    def get_average_note_by_2bar(raw_melody_data):
        """
        获取歌曲小节内音符的平均音高，以2小节为单位。空小节的平均音高为0
        :param raw_melody_data: 一首歌曲的音符列表。以步长为单位 绝对音高
        :return: 歌曲小节内音符的平均音高 二维数组 其中第一维的分割依据为空小节
        """
        # TODO 第一维的分割依据可以调整为按照乐段来分割
        if not raw_melody_data:
            return []  # 这首歌没有主旋律 返回空列表

        bar_num = len(raw_melody_data) // 32  # 这首歌的主旋律一共有多少个小节
        average_note_list = [[]]
        bar_dx = 0  # 当前小节
        while 1:
            if bar_dx >= bar_num:
                break
            if bar_dx == bar_num - 1:
                average_note_list[-1].append(sum(raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32])/(32 - raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32].count(0)))
                break
            # try:
            if raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32] == [0 for t in range(32)]:  # 遇到空小节
                if average_note_list[-1]:
                    average_note_list.append([])
                bar_dx += 1
                continue
            elif raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32] != [0 for t in range(32)] and raw_melody_data[(bar_dx + 1) * 32: (bar_dx + 2) * 32] == [0 for t in range(32)]:  # 当前小节不为空但是下一小节为空
                average_note_list[-1].append(sum(raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32])/(32 - raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32].count(0)))
                average_note_list.append([])
                bar_dx += 2
            else:  # 这一小节与下一小节均不为空
                average_note_list[-1].append((sum(raw_melody_data[bar_dx * 32: (bar_dx + 2) * 32]))/(64 - raw_melody_data[bar_dx * 32: (bar_dx + 2) * 32].count(0)))
                bar_dx += 2
            # except IndexError:
            #     bar += 1
        if not average_note_list[-1]:
            average_note_list.pop(-1)
        return average_note_list

    def define_cluster_model(self, raw_melody_data, tone_restrict=None):
        """获取所有歌曲逐两小节的平均音高，并确定模型"""
        # 1.从数据集中读取歌的编号为song_id且小节标注为main的小节数据
        # raw_melody_data = get_raw_song_data_from_dataset('main', tone_restrict)
        # 2.逐2小节地记录每一首歌所有区间的平均音高
        melody_avr_list = []
        for song_it in range(len(raw_melody_data)):
            if raw_melody_data[song_it]:
                melody_avr_list.extend(self.get_average_note_by_2bar(raw_melody_data[song_it]))  # 这里'+'换成了extend
        flatten_melody_avr_list = []
        for melody_avr in melody_avr_list:  # 将二维数组展成一维数组
            flatten_melody_avr_list.extend(melody_avr)
        # 3.使用K均值算法，对输入的音乐按照两小节的平均音高分为10类
        self.kmeans_model = KMeansModel(flatten_melody_avr_list, 10, 50, True)

    def define_test_model(self):
        self.kmeans_model = KMeansModel([-1], 10, 50, False)

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
        if not raw_melody_data:
            return []  # 这首歌没有主旋律 返回空列表

        bar_num = len(raw_melody_data) // 32  # 这首歌的主旋律一共有多少个小节
        melody_avr_note_list = [0 for t in range(bar_num + 1)]  # 将这首歌逐两小节的音符平均音高记录在这个列表中
        bar_dx = 0
        # 2.逐2小节地记录一个区间的平均音高
        while 1:
            if bar_dx >= bar_num:
                break
            if bar_dx == bar_num - 1:  # 已达最后一小节
                melody_avr_note_list[bar_dx] = (sum(raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32]) / (32 - raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32].count(0)))
                break
            # try:
            if raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32] == [0 for t in range(32)]:  # 这一小节为空 不记录
                bar_dx += 1
                continue
            elif raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32] != [0 for t in range(32)] and raw_melody_data[(bar_dx + 1) * 32: (bar_dx + 2) * 32] == [0 for t in range(32)]:  # 这一小节不为空 但下一小节为空
                melody_avr_note_list[bar_dx] = (sum(raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32]) / (32 - raw_melody_data[bar_dx * 32: (bar_dx + 1) * 32].count(0)))
                bar_dx += 2
            else:  # 这一小节和下一小节均不为空
                melody_avr_note_list[bar_dx] = melody_avr_note_list[bar_dx + 1] = sum(raw_melody_data[bar_dx * 32: (bar_dx + 2) * 32]) / (64 - raw_melody_data[bar_dx * 32: (bar_dx + 2) * 32].count(0))
                bar_dx += 2
            # except KeyError:
            #     bar += 1
        # 2.通过K均值算法将音高列表分类
        attachment_array = self.kmeans_model.run_attachment(session, self.cluster_center_points, melody_avr_note_list)  # 每个小节的分类情况
        # 3.将音高分类情况进行微调 如果小节旋律为空 则分类为0 否则分类为1-10
        for bar_it in range(bar_num + 1):
            if melody_avr_note_list[bar_it] != 0:
                attachment_array[bar_it] += 1
            else:
                attachment_array[bar_it] = 0
        return attachment_array


class MelodyTrainData:

    def __init__(self, tone_restrict=None):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据

        self.continuous_bar_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 每一首歌连续的小节计数
        self.continuous_bar_data_nres = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 每一首歌无限制的小节计数

        self.keypress_pat_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 二维数组 第一维是歌曲列表 第二维是按键的组合（步长是2拍）
        self.all_keypress_pats = [[0 for t in range(16)]]  # 主旋律按键组合的对照表
        self.keypress_pat_count = [0]  # 各种按键组合的计数

        self.core_note_ary_nres = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 每一首歌每一拍的骨干音列表
        self.core_note_pat_nres = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 骨干音组合数据

        self.melody_pat_data = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 有调式限制的旋律组合数据
        self.melody_pat_data_nres = [[] for t in range(TRAIN_FILE_NUMBERS)]  # 无调式限制的旋律组合数据

        self.ShiftConfidence = ShiftConfidenceCheck()  # 计算训练集中所有歌曲相邻两小节的音高变化情况
        self.DiffNoteConfidence = DiffNoteConfidenceCheck()  # 计算训练集中所有歌曲所有段落前半段和后半段的按键和音高上的差异

        # 1.从数据集中读取所有歌曲的主旋律数据，并变更为以音符步长为单位的列表
        self.raw_train_data = get_raw_song_data_from_dataset('main', tone_restrict)
        self.raw_melody_data = get_raw_song_data_from_dataset('main', None)  # 没有旋律限制的主旋律数据　用于训练其他数据
        self.section_data = get_section_data_from_dataset()
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.raw_train_data[song_it] != dict():
                self.raw_train_data[song_it] = flat_array(self.raw_train_data[song_it])
            else:
                self.raw_train_data[song_it] = []  # 对于没有主旋律的歌曲，将格式转化为list格式
            if self.raw_melody_data[song_it] != dict():
                self.raw_melody_data[song_it] = flat_array(self.raw_melody_data[song_it])
            else:
                self.raw_train_data[song_it] = []
        raw_melody_data_nres = copy.deepcopy(self.raw_melody_data)

        # 2.获取最常见的主旋律组合
        common_pattern_cls = CommonMusicPatterns(COMMON_MELODY_PAT_NUM)  # 这个类可以获取常见的主旋律组合
        common_pattern_cls.train(self.raw_train_data, 0.125, 1, False)
        common_pattern_cls.store('melody')  # 存储在sqlite文件中
        self.common_melody_pats = common_pattern_cls.common_pattern_list  # 常见的旋律组合列表
        self.melody_pats_num_list = common_pattern_cls.pattern_number_list  # 这些旋律组合出现的次数列表

        # 3.逐歌曲获取连续不为空的小节列表/按键数据/骨干音数据/
        for song_it in range(TRAIN_FILE_NUMBERS):
            # 3.1.获取没有调式限制的旋律数据
            if raw_melody_data_nres[song_it]:  # 获取相关没有调式限制的相关数据
                # 3.1.1.获取旋律的按键数据
                self.get_keypress_data(song_it, raw_melody_data_nres[song_it])  # 获取按键数据 当前有按键记为1 没有按键记为0
                # 3.1.3.将它的主旋律编码为常见的旋律组合。如果该旋律组合不常见，则记为COMMON_MELODY_PAT_NUM+1
                self.continuous_bar_data_nres[song_it] = get_continuous_bar_cnt(raw_melody_data_nres[song_it])
                # 3.1.2.获取骨干音及骨干音的常见组合列表
                self.core_note_ary_nres[song_it] = melody_core_note(raw_melody_data_nres[song_it], self.continuous_bar_data_nres[song_it], self.section_data[song_it])
            # 3.2.获取有调式限制的旋律数据
            if self.raw_train_data[song_it]:
                # 3.2.1.获取歌曲的连续不为空的小节序号列表
                self.continuous_bar_data[song_it] = get_continuous_bar_cnt(self.raw_train_data[song_it])
        # 3.3.存储按键数据的组合列表
        keyperss_pattern_cls = BaseMusicPatterns()
        keyperss_pattern_cls.common_pattern_list = self.all_keypress_pats
        keyperss_pattern_cls.pattern_number_list = self.keypress_pat_count
        keyperss_pattern_cls.store('keypress')

        # 4.获取无调式限制的常见骨干音组合列表
        core_note_pattern_cls = CommonMusicPatterns(COMMON_CORE_NOTE_PAT_NUM)
        core_note_pattern_cls.train(self.core_note_ary_nres, 0.125, 2)
        core_note_pattern_cls.store('core_note')  # 存储在sqlite文件中
        self.common_corenote_pats = core_note_pattern_cls.common_pattern_list  # 常见的旋律组合列表

        # 5.根据常见的音符组合，对原始的旋律音符和骨干音列表进行编码
        for song_it in range(TRAIN_FILE_NUMBERS):
            # 5.1.编码无调式限制的主旋律及其骨干音
            if raw_melody_data_nres[song_it]:  # 没有调式限制的相关数据
                self.melody_pat_data_nres[song_it] = MelodyPatternEncode(self.common_melody_pats, raw_melody_data_nres[song_it], 0.125, 1).music_pattern_list
                self.core_note_pat_nres[song_it] = CoreNotePatternEncode(self.common_corenote_pats, self.core_note_ary_nres[song_it], 0.125, 2).music_pattern_list
            # 5.2.编码有调式限制的主旋律。如果该旋律组合不常见，则记为COMMON_MELODY_PAT_NUM+1
            if self.raw_train_data[song_it]:
                self.melody_pat_data[song_it] = MelodyPatternEncode(self.common_melody_pats, self.raw_train_data[song_it], 0.125, 1).music_pattern_list

        # 6.生成每首歌的旋律变化累积幅度的数据
        # 6.1.生成每首歌的相邻两小节的音高变化情况，和每个段落的前半部分与后半部分的差异
        for song_it in range(TRAIN_FILE_NUMBERS):
            if raw_melody_data_nres[song_it]:
                self.ShiftConfidence.train_1song(raw_melody_data=raw_melody_data_nres[song_it], section_data=self.section_data[song_it])
                self.DiffNoteConfidence.train_1song(raw_melody_data=raw_melody_data_nres[song_it], section_data=self.section_data[song_it])
        # 6.2.找出旋律变化和段落内差异前95%所在位置
        self.ShiftConfidence.calc_confidence_level(0.95)
        self.DiffNoteConfidence.calc_confidence_level(0.95)
        self.ShiftConfidence.store('melody_shift')  # 把shift_confidence_level和diff_note保存到sqlite中
        self.DiffNoteConfidence.store('melody_diffnote')

        # 7.生成每首歌的训练数据
        for song_it in range(TRAIN_FILE_NUMBERS):
            if self.raw_train_data[song_it]:
                self.get_model_io_data(self.melody_pat_data[song_it], self.continuous_bar_data[song_it])
        np.save(os.path.join(PATH_PATTERNLOG, 'MelodyInputData.npy'), self.input_data)  # 在generate的时候要比较生成数据和训练集是否雷同，因此这个也要存储

        DiaryLog.warn('Generation of melody train data has finished!')

    def get_keypress_data(self, song_dx, raw_melody_data):
        """
        逐两拍获取一首歌音符打击的情况
        :param int song_dx: 这首歌在歌曲列表中的编号
        :param raw_melody_data: 这首歌原始的主旋律数据
        :return:
        """
        pat_num = len(raw_melody_data) // 16  # 这首歌一共应该有多少个按键组合
        self.keypress_pat_data[song_dx] = [0 for t in range(pat_num)]  # 按键组合数据以两拍为单位
        for pat_step_it in range(pat_num):
            keypress_pat_data_1step = [1 if t != 0 else 0 for t in raw_melody_data[pat_step_it * 16: (pat_step_it + 1) * 16]]  # 这两拍的按键数据
            if keypress_pat_data_1step not in self.all_keypress_pats:
                # 这个按键组合不在按键组合列表中。新建这个按键组合
                self.keypress_pat_data[song_dx][pat_step_it] = len(self.all_keypress_pats)
                self.all_keypress_pats.append(keypress_pat_data_1step)
                self.keypress_pat_count.append(1)
            else:
                # 这个按键组合在按键组合列表中
                pat_dx = self.all_keypress_pats.index(keypress_pat_data_1step)
                self.keypress_pat_data[song_dx][pat_step_it] = self.all_keypress_pats.index(keypress_pat_data_1step)
                self.keypress_pat_count[pat_dx] += 1

    def get_model_io_data(self, melody_pat_data, continuous_bar_data):
        """
        输入内容为当前时间的编码 过去四小节的主旋律 MelodyProfile 输出内容为这一拍的主旋律
        :param melody_pat_data: 一首歌旋律组合的数据
        :param continuous_bar_data: 一首歌主旋律连续不为空的小节列表
        """
        for step_it in range(-4 * TRAIN_MELODY_IO_BARS, len(melody_pat_data) - 4 * TRAIN_MELODY_IO_BARS):  # 从负拍开始 方便训练前几拍的音符
            # 1.获取当前所在的小节和所在小节的位置
            cur_bar = step_it // 4  # 第几小节
            pat_step_in_bar = step_it % 4  # 小节内的第几个步长

            # 2.添加当前音符的时间
            time_add = (1 - continuous_bar_data[cur_bar + TRAIN_MELODY_IO_BARS] % 2) * 4
            input_time_data = [pat_step_in_bar + time_add]  # 这个时间在2小节之中的位置
            output_time_data = [pat_step_in_bar + time_add]

            # 3.添加当前小节的旋律组合
            if cur_bar < 0:
                input_time_data.extend([0 for t in range(4 - pat_step_in_bar)])
                output_time_data.extend([0 for t in range(3 - pat_step_in_bar)])
            else:
                input_time_data.extend(melody_pat_data[cur_bar * 4 + pat_step_in_bar: cur_bar * 4 + 4])
                output_time_data.extend(melody_pat_data[cur_bar * 4 + pat_step_in_bar + 1: cur_bar * 4 + 4])  # output_data错后一拍

            # 4.添加后三个小节的旋律组合
            for bar_it in range(cur_bar + 1, cur_bar + TRAIN_MELODY_IO_BARS):
                if bar_it < 0:  # 当处于负拍时 这个小节对应的旋律置为空
                    input_time_data.extend([0, 0, 0, 0])
                    output_time_data.extend([0, 0, 0, 0])
                else:
                    input_time_data.extend(melody_pat_data[bar_it * 4: (bar_it + 1) * 4])
                    output_time_data.extend(melody_pat_data[bar_it * 4: (bar_it + 1) * 4])
                if bar_it < 0 or melody_pat_data[bar_it * 4: (bar_it + 1) * 4] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
                    input_time_data = [input_time_data[0]] + [0 for t in range(len(input_time_data) - 1)]
                    output_time_data = [output_time_data[0]] + [0 for t in range(len(output_time_data) - 1)]

            # 5.添加最后一个小节的旋律组合
            input_time_data.extend(melody_pat_data[(cur_bar + TRAIN_MELODY_IO_BARS) * 4: (cur_bar + TRAIN_MELODY_IO_BARS) * 4 + pat_step_in_bar])
            output_time_data.extend(melody_pat_data[(cur_bar + TRAIN_MELODY_IO_BARS) * 4: (cur_bar + TRAIN_MELODY_IO_BARS) * 4 + (pat_step_in_bar + 1)])

            # 6.当输出数据所在的小节与其前一小节均不为空时，该数据收录进训练集
            if cur_bar + TRAIN_MELODY_IO_BARS - 1 < 0 or melody_pat_data[(cur_bar + TRAIN_MELODY_IO_BARS - 1) * 4: (cur_bar + TRAIN_MELODY_IO_BARS) * 4] != [0 for t in range(4)]:
                if melody_pat_data[(cur_bar + TRAIN_MELODY_IO_BARS) * 4: (cur_bar + TRAIN_MELODY_IO_BARS + 1) * 4] != [0 for t in range(4)]:
                    self.input_data.append(input_time_data)
                    self.output_data.append(output_time_data)


class MelodyTestData:

    def __init__(self):
        # 1.从sqlite中读取common_melody_pattern
        common_pattern_cls = CommonMusicPatterns(COMMON_MELODY_PAT_NUM)  # 这个类可以获取常见的主旋律组合
        common_pattern_cls.restore('melody')  # 直接从sqlite文件中读取
        self.common_melody_pats = common_pattern_cls.common_pattern_list  # 常见的旋律组合列表
        self.melody_pats_num_list = common_pattern_cls.pattern_number_list  # 这些旋律组合出现的次数列表

        # 2.从sqlite中读取所有按键组合数据
        keyperss_pattern_cls = BaseMusicPatterns()
        keyperss_pattern_cls.restore('keypress')
        self.all_keypress_pats = keyperss_pattern_cls.common_pattern_list  # 常见的旋律组合列表
        self.keypress_pat_count = keyperss_pattern_cls.pattern_number_list  # 这些旋律组合出现的次数列表

        # 3.从sqlite中读取常见的骨干音组合数据
        core_note_pattern_cls = CommonMusicPatterns(COMMON_CORE_NOTE_PAT_NUM)  # 这个类可以获取常见的骨干音组合
        core_note_pattern_cls.restore('core_note')  # 直接从sqlite文件中读取
        self.common_corenote_pats = core_note_pattern_cls.common_pattern_list  # 常见的旋律组合列表

        # 4.从sqlite中读取每首歌的相邻两小节的音高变化情况，和每个段落的前半部分与后半部分的差异的合理的阈值
        self.ShiftConfidence = ShiftConfidenceCheck()  # 计算训练集中所有歌曲相邻两小节的音高变化情况
        self.DiffNoteConfidence = DiffNoteConfidenceCheck()
        self.ShiftConfidence.restore('melody_shift')
        self.DiffNoteConfidence.restore('melody_diffnote')

        # 5.获取主旋律模型的输入数据，用于和生成结果进行比对
        self.input_data = np.load(os.path.join(PATH_PATTERNLOG, 'MelodyInputData.npy')).tolist()

        DiaryLog.warn('Restoring of melody associated data has finished!')
