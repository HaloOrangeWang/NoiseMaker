from settings import *
from pipelines.functions import BaseLstmPipeline, music_pattern_prediction, melody_pattern_prediction_unique, keypress_encode
from datainputs.melody import MelodyProfile, MelodyTrainData, MelodyTestData, CoreNotePatternEncode
from models.configs import MelodyConfig
from interfaces.utils import DiaryLog
from interfaces.music_patterns import music_pattern_decode
from validations.melody import keypress_check, section_begin_check, section_end_check, melody_end_check
import tensorflow as tf
import numpy as np
import copy
import random


def get_first_melody_pat(pat_num_list, min_pat_dx, max_pat_dx):
    """
    按照主旋律的各种组合在训练集中出现的频率，随机生成主旋律的第一个组合
    :param pat_num_list: 各种pattern在训练集中出现的次数
    :param min_pat_dx: 选取的pattern在序列中的最小值
    :param max_pat_dx: 选取的pattern在序列中的最大值
    :return: 选取的pattern的代码
    """
    row_sum = sum(pat_num_list[min_pat_dx: (max_pat_dx + 1)])
    random_value = random.random() * row_sum
    for element in range(len(pat_num_list[min_pat_dx: (max_pat_dx + 1)])):
        random_value -= pat_num_list[min_pat_dx: (max_pat_dx + 1)][element]
        if random_value < 0:
            return element + min_pat_dx
    return max_pat_dx


class MelodyPipeline(BaseLstmPipeline):

    def __init__(self, is_train, tone_restrict):
        super().__init__()

        self.is_train = is_train
        self.tone_restrict = tone_restrict  # 这次生成大调还是小调
        if self.is_train:
            self.train_data = MelodyTrainData(tone_restrict)  # 生成训练所用的数据
        else:
            self.train_data = MelodyTestData()
        with tf.variable_scope('cluster'):
            self.melody_profile = MelodyProfile()
            if self.is_train:
                self.melody_profile.define_cluster_model(self.train_data.raw_melody_data, tone_restrict=tone_restrict)
            else:
                self.melody_profile.define_test_model()

    # noinspection PyAttributeOutsideInit
    def prepare(self):
        self.config = MelodyConfig()
        self.test_config = MelodyConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'MelodyModel'

    def get_center_points(self, session, is_train=True):
        """计算音符平均音高的中心点,这些中心点对"""
        self.melody_profile.get_cluster_center_points(session, is_train)

    # noinspection PyAttributeOutsideInit
    def generate_init(self):
        # 生成过程中的常量
        self.section_data = copy.copy(GEN_SEC_CHOICES[random.randint(0, len(GEN_SEC_CHOICES) - 1)])  # 为生成的歌曲选择一个乐段类型

        # 生成过程中的一些变量
        self.rollback_times = 0  # 旋律被打回重新生成的次数
        self.sec_end_cluster = 0  # 每个乐段结束的最后两小节的melody profile
        self.trans_back_times = 0  # 为防止持续失败 如果过渡段验证连续失败10次 则把整个乐段完全重新生成
        self.pat_step_dx = 0  # 生成完第几个pattern了
        self.sec_dx = 0  # 生成到第几个段落了
        self.sec_profile_list = np.array([0 for t in range(len(self.section_data))])  # 每一个乐段的melody profile

        # 生成结果
        self.melody_out_notes = []  # 音符输出
        self.melody_out_pats = []  # 输出的音符组合形式

    def rollback(self, step_num):
        """
        将主旋律的输出回滚step_num个步长（pat_step_dx减少，output_notes和output_pats去掉最后几个数）
        :param step_num: 需要回滚的步长数量
        """
        self.pat_step_dx -= step_num  # 重新生成这一小节的音乐
        self.melody_out_notes = self.melody_out_notes[:len(self.melody_out_notes) - step_num * 8]
        self.melody_out_pats = self.melody_out_pats[:len(self.melody_out_pats) - step_num]

    def generate_by_step(self, session):
        """生成一个步长的音符组合。当出现起始步长或段落相同的情况，会连续生成多个步长"""
        # 1.如果输入数据为空 随机生成一个起始pattern作为乐曲的开始
        if self.pat_step_dx == 0:
            start_pattern = get_first_melody_pat(self.train_data.melody_pats_num_list, 1, COMMON_MELODY_PAT_NUM)  # 随机选取开始的pattern 注意 开始的第一个pattern不能为空
            self.melody_out_pats.append(start_pattern)
            self.melody_out_notes.extend(music_pattern_decode(self.train_data.common_melody_pats, [start_pattern], 0.125, 1))  # 将第一个旋律组合解码 得到最初的8个音符的列表
            DiaryLog.warn('第%d次生成开始: 主旋律开始的组合编码是%d, 组合为%s' % (self.rollback_times + 1, start_pattern, repr(self.melody_out_notes)))
            self.pat_step_dx = 1

        # 2.如果这一拍是某一个段落的第一拍，且这个段落与前一个段落相同，则除了过渡小节以外，其他的全部复制前一个乐段相同位置的值
        # 2.1.找到这一拍属于第几个乐段
        sec_dx = -1
        for sec_it in range(len(self.section_data)):
            if self.section_data[sec_it][0] > self.pat_step_dx // 4:
                sec_dx = sec_it - 1
                break
        self.sec_dx = sec_dx
        # 2.2.判断这个乐段和上一个乐段是否一致 如果一致则“除了过渡小节以外，其他的全部复制前一个乐段相同位置的值”
        if self.sec_dx >= 1 and self.section_data[self.sec_dx][1] == self.section_data[self.sec_dx - 1][1]:
            beat_bias = self.pat_step_dx - self.section_data[self.sec_dx][0] * 4
            if self.section_data[self.sec_dx][0] - self.section_data[self.sec_dx - 1][0] <= 7:
                trans_bar_num = 1  # 对于不足7小节的乐段来说，有1个过渡小节，除此以外有2个过渡小节
            else:
                trans_bar_num = 2
            if beat_bias < (self.section_data[self.sec_dx][0] - self.section_data[self.sec_dx - 1][0] - trans_bar_num) * 4:
                self.melody_out_notes.extend(self.melody_out_notes[(self.section_data[self.sec_dx - 1][0] * 4 + beat_bias) * 8: self.section_data[self.sec_dx][0] * 32 - trans_bar_num * 32])
                self.melody_out_pats.extend(self.melody_out_pats[(self.section_data[self.sec_dx - 1][0] * 4 + beat_bias): self.section_data[self.sec_dx][0] * 4 - trans_bar_num * 4])
                self.pat_step_dx += self.section_data[self.sec_dx][0] * 4 - self.section_data[self.sec_dx - 1][0] * 4 - trans_bar_num * 4 - beat_bias

        # 3.使用LstmModel生成一个步长的音符
        # 3.1.判断这个步长能否为空拍。判断依据为
        # a.当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空。
        # b.不能连续两个小节的第一个步长为空。
        # c.一个乐段的第一个步长不能为空。
        # d.不能连续四拍为空
        flag_allow_empty = True
        if self.pat_step_dx % 4 == 0:
            if self.pat_step_dx > 8 and self.melody_out_notes[-64:].count(0) <= 56:
                flag_allow_empty = False  # 当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空
            if self.melody_out_pats[self.pat_step_dx - 4] == 0:
                flag_allow_empty = False  # 不能连续两个小节的第一个步长为空
            if self.section_data[self.sec_dx][0] == self.pat_step_dx // 4:
                flag_allow_empty = False  # 一个乐段的第一个步长不能为空
        if self.pat_step_dx >= 3 and self.melody_out_pats[-3:] == [0, 0, 0]:
            flag_allow_empty = False  # 不能连续四拍为空
        # 3.2.逐时间步长生成test model输入数据
        time_add = self.pat_step_dx % 4 + 4 * ((self.pat_step_dx // 4 - self.section_data[self.sec_dx][0]) % 2)
        melody_input = [[time_add]]  # 当前时间的编码
        # 对于不是一个乐段的第一小节，训练数据只从这个乐段的第一拍开始
        if self.pat_step_dx - self.section_data[self.sec_dx][0] * 4 <= 3:
            lookback_beats = 16
        else:
            lookback_beats = min(16, self.pat_step_dx - self.section_data[self.sec_dx][0] * 4)
        if len(self.melody_out_pats) >= lookback_beats:  # melody_out_pat_list足够长的情况
            melody_input[0].extend([0] * (16 - lookback_beats) + self.melody_out_pats[-lookback_beats:])  # 最近4小节的旋律组合
        else:  # melody_out_pat_list不够长的情况
            melody_input[0].extend([0] * (16 - len(self.melody_out_pats)) + self.melody_out_pats)
        # 3.3.生成输出的音符
        melody_predict = self.predict(session, melody_input)  # LSTM预测 得到二维数组predict
        if self.pat_step_dx <= 3:  # 还没到五拍，不可能出现“连续五拍与某个常见组合雷同”的情况
            out_pat_dx = music_pattern_prediction(melody_predict, int(not flag_allow_empty), COMMON_MELODY_PAT_NUM)
        else:
            try:  # 人为排除与训练集中数据雷同的主旋律组合
                out_pat_dx = melody_pattern_prediction_unique(melody_predict, int(not flag_allow_empty), COMMON_MELODY_PAT_NUM, self.melody_out_pats, self.train_data.input_data)
            except RuntimeError:
                DiaryLog.warn('在第%d个步长, 主旋律第%02d次打回，难以选出不雷同的组合, 最后四拍音符为%s' % (self.pat_step_dx, self.rollback_times, repr(self.melody_out_notes[-32:])))
                self.rollback(4)
                self.rollback_times += 1  # 被打回重新生成了一次
                return
        self.melody_out_pats.append(out_pat_dx)
        self.melody_out_notes.extend(music_pattern_decode(self.train_data.common_melody_pats, [out_pat_dx], 0.125, 1))
        self.pat_step_dx += 1

    def calc_profile(self, session):
        cluster_ary = self.melody_profile.get_melody_profile_by_song(session, self.melody_out_notes[self.section_data[self.sec_dx][0] * 32: self.section_data[self.sec_dx + 1][0] * 32])  # 计算这个乐段的cluster
        section_end_cluster = cluster_ary[-1]  # 过渡小节的cluster
        # section_profile_ary.append([section_data[sec_dx][0], np.mean(cluster_ary)])  # 整个乐段的cluster
        self.sec_profile_list[self.sec_dx] = np.mean(cluster_ary)  # 整个乐段的cluster

    def check_1step(self, session):
        # 1.检查音符按键位置是否符合要求。在整小节处检验
        if self.pat_step_dx % 4 == 0:
            if not keypress_check(self.melody_out_notes[-32:]):  # 检查这一小节的音乐是否符合要求 如果不符合要求 则返工 重新生成这一小节的音乐
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，按键位置有异常, 最后四拍音符为%s' % (self.pat_step_dx, self.rollback_times, self.melody_out_notes[-32:]))
                self.rollback(4)
                self.rollback_times += 1
                return

        # 2.检查两小节内的音高变化幅度是否过大。在整小节处且前两小节不跨越乐段时检验
        if self.pat_step_dx % 4 == 0 and self.pat_step_dx >= 8 and (self.pat_step_dx // 4 - 1) != self.section_data[self.sec_dx][0]:
            shift_score = self.train_data.ShiftConfidence.evaluate(melody_note_list=self.melody_out_notes[-64:])
            if not self.train_data.ShiftConfidence.compare(shift_score):  # 这两小节的音高变化是否在一定限制幅度内
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，音高变化的分数为%.4f,高于临界值%.4f' % (self.pat_step_dx, self.rollback_times, shift_score, self.train_data.ShiftConfidence.confidence_level))
                self.rollback(8)
                self.rollback_times += 1
                return

        # 3.检查第一段副歌的cluster是否比之前所有段落的cluster都大（更高）。在第一段副歌结束时进行检验
        if self.pat_step_dx % 4 == 0 and self.section_data[self.sec_dx][1] == "sub" and (self.sec_dx >= 1 and self.section_data[self.sec_dx - 1][1] != "sub") and self.pat_step_dx // 4 == self.section_data[self.sec_dx + 1][0]:
            if self.sec_profile_list[self.sec_dx] <= max(self.sec_profile_list[:self.sec_dx]):
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，第一段副歌的cluster不是最大, 各个乐段的cluster分别为%s' % (self.pat_step_dx, self.rollback_times, repr(self.sec_profile_list)))
                rollback_beats = (self.section_data[self.sec_dx + 1][0] - self.section_data[self.sec_dx][0]) * 4  # 回退多少拍
                self.rollback(rollback_beats)  # 重新生成这一整个乐段的音乐
                self.rollback_times += 1  # 被打回重新生成了一次
                return

        # 4.第一段副歌前两拍的cluster必须必前面一个乐段收尾部分的cluster要大。在第一段副歌的第二拍结束后检验
        if self.section_data[self.sec_dx][1] == "sub" and (self.sec_dx >= 1 and self.section_data[self.sec_dx - 1][1] != "sub") and self.pat_step_dx == self.section_data[self.sec_dx][0] * 4 + 2:
            if self.melody_out_notes[-16:] != [0 for t in range(16)]:
                start_note_cluster = self.melody_profile.get_melody_profile_by_song(session, self.melody_out_notes[-16:] * 4)[1]  # 初始音符所在的cluster
            else:
                start_note_cluster = -1  # 如果副歌的前两拍为空 则认为是不合理的
            if self.melody_out_notes[-16:] == [0] * 16 or start_note_cluster < self.sec_end_cluster:
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，第一段副歌的开始音符的cluster:%d小于前一段落末尾处的cluster:%d, 最后两拍音符为%s' % (self.pat_step_dx, self.rollback_times, start_note_cluster, self.sec_end_cluster, repr(self.melody_out_notes[-16:])))
                self.rollback(2)  # 重新生成这两拍的音乐
                self.rollback_times += 1  # 被打回重新生成了一次
                return

        # 5.每一个乐段的第一个小节必须要有一半的时长为C/Am的弦内音。在每个段落第一小节即将结束时检查
        if self.pat_step_dx == self.section_data[self.sec_dx][0] * 4 + 4:  # 每一个乐段的第一个小节必须要有一半的时长为弦内音
            steps_in_chord = section_begin_check(self.melody_out_notes[-32:], self.tone_restrict)
            if steps_in_chord < 16:
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，乐段的第一个小节只有%d个弦内音, 最后四拍音符为%s' % (self.pat_step_dx, self.rollback_times, steps_in_chord, repr(self.melody_out_notes[-32:])))
                self.rollback(4)  # 重新生成这一小节的音乐
                self.rollback_times += 1  # 被打回重新生成了一次
                return

        # 6.middle段落和main段落的整体音高cluster相差不能超过2。在第一个middle段落即将结束时检验
        if self.pat_step_dx % 4 == 0 and self.section_data[self.sec_dx][1] == "middle" and self.sec_dx >= 1 and self.pat_step_dx // 4 == self.section_data[self.sec_dx + 1][0]:
            if abs(self.sec_profile_list[self.sec_dx] - self.sec_profile_list[self.sec_dx - 1]) > 2:
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，middle段的cluster与main段的cluster相差太大, 各个乐段的cluster分别为%s' % (self.pat_step_dx, self.rollback_times, repr(self.sec_profile_list)))
                rollback_beats = (self.section_data[self.sec_dx + 1][0] - self.section_data[self.sec_dx][0]) * 4  # 回退多少拍
                self.rollback(rollback_beats)  # 重新生成这一整个乐段的音乐
                self.rollback_times += 1  # 被打回重新生成了一次
                return

        # 7.一个段落内部，按键和音符内容相差不能过多。在一个段落即将结束时检验
        if self.pat_step_dx % 4 == 0 and self.pat_step_dx // 4 == self.section_data[self.sec_dx + 1][0]:  # 段落内差异程度检验
            sec_step_num = (self.section_data[self.sec_dx + 1][0] - self.section_data[self.sec_dx][0]) * 32
            dfnote_score = self.train_data.DiffNoteConfidence.evaluate(melody_note_list=self.melody_out_notes[-sec_step_num:])
            if not self.train_data.DiffNoteConfidence.compare(dfnote_score):
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，统一段落前一半和后一半音符变化的分数为%.4f,高于临界值%.4f' % (self.pat_step_dx, self.rollback_times, dfnote_score, self.train_data.DiffNoteConfidence.confidence_level))
                rollback_beats = (self.section_data[self.sec_dx + 1][0] - self.section_data[self.sec_dx][0]) * 4  # 回退多少小节
                self.rollback(rollback_beats)  # 重新生成这一整个乐段的音乐
                self.rollback_times += 1  # 被打回重新生成了一次

        # 8.检查每个乐段的结束阶段部分是否符合要求
        if self.pat_step_dx % 4 == 0 and self.pat_step_dx // 4 == self.section_data[self.sec_dx + 1][0]:  # 过渡段校验
            if not section_end_check(self.melody_out_notes[-32:], self.tone_restrict):
                # 检验失败则重新生成过去两拍。但如果连续失败十次，则需要重新生成整个乐段
                if self.trans_back_times >= 10:
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回, 其中该次过渡段连续失败超过10次, 将整个乐段重新生成' % (self.pat_step_dx, self.rollback_times))
                    rollback_beats = (self.section_data[self.sec_dx + 1][0] - self.section_data[self.sec_dx][0]) * 4  # 回退多少小节
                    self.rollback(rollback_beats)  # 重新生成这一整个乐段的音乐
                    self.rollback_times += 1  # 被打回重新生成了一次
                    self.trans_back_times = 0
                else:
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回, 其中该次过渡段连续失败%d次, 过渡段不符合要求, 最后四拍音符为%s' % (self.pat_step_dx, self.rollback_times, self.trans_back_times, repr(self.melody_out_notes[-32:])))
                    self.rollback(8)  # 重新生成两小节过渡段的音乐
                    self.rollback_times += 1  # 被打回重新生成了一次
                    self.trans_back_times += 1
                return

        # 9.主旋律结束阶段的检验。在最后一个步长执行检验
        if self.pat_step_dx // 4 // 4 == self.section_data[-1][0]:
            if not melody_end_check(self.melody_out_notes[-32:], self.tone_restrict):  # 结束条件是最后音符是dol或la
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回,收束不符合要求, 最后四拍音符为%s' % (self.pat_step_dx, self.rollback_times, repr(self.melody_out_notes[-32:])))
                self.rollback(8)  # 重新生成最后两小节的音乐
                self.rollback_times += 1  # 被打回重新生成了一次

    def generate(self, session):
        self.generate_init()
        while True:
            self.generate_by_step(session)  # 生成一个步长
            if self.pat_step_dx % 4 == 0 and self.section_data[self.sec_dx + 1][0] == self.pat_step_dx // 4:
                self.calc_profile(session)  # 生成过渡段的最后一拍后，计算这一个段落的melody profile
            self.check_1step(session)  # 对这个步长生成的数据进行校验

            # 退出的条件有二：一是生成失败的次数过多 打回重新生成，二是生成完成
            if self.rollback_times >= MAX_GEN_MELODY_FAIL_TIME:
                DiaryLog.warn('主旋律被打回次数超过%d次,重新生成。\n\n\n' % self.rollback_times)
                raise RuntimeError
            if self.pat_step_dx // 4 == self.section_data[-1][0]:
                assert self.pat_step_dx == len(self.melody_out_notes) // 8
                break
        core_note_pat_list = CoreNotePatternEncode(self.train_data.common_corenote_pats, self.melody_out_notes, 0.125, 2).music_pattern_list
        keypress_pat_list = keypress_encode(self.melody_out_notes, self.train_data.all_keypress_pats)
        DiaryLog.warn('主旋律的输出:' + repr(self.melody_out_notes) + '\n\n\n')
        return self.melody_out_notes, self.melody_out_pats, core_note_pat_list, keypress_pat_list


class MelodyPipelineGen1Sec(MelodyPipeline):

    # noinspection PyAttributeOutsideInit
    def generate_init(self):
        # 生成过程中的一些变量
        self.rollback_times = 0  # 旋律被打回重新生成的次数
        self.pat_step_dx = 0  # 生成完第几个pattern了
        self.flag_end = False  # 生成是否结束

        # 生成结果
        self.melody_out_notes = []  # 音符输出
        self.melody_out_pats = []  # 输出的音符组合形式

    def generate_by_step(self, session):
        # 1.如果输入数据为空 随机生成一个起始pattern作为乐曲的开始
        if self.pat_step_dx == 0:
            start_pattern = get_first_melody_pat(self.train_data.melody_pats_num_list, 1, COMMON_MELODY_PAT_NUM)  # 随机选取开始的pattern 注意 开始的第一个pattern不能为空
            self.melody_out_pats.append(start_pattern)
            self.melody_out_notes.extend(music_pattern_decode(self.train_data.common_melody_pats, [start_pattern], 0.125, 1))  # 将第一个旋律组合解码 得到最初的8个音符的列表
            DiaryLog.warn('第%d次生成开始: 主旋律开始的组合编码是%d, 组合为%s' % (self.rollback_times + 1, start_pattern, repr(self.melody_out_notes)))
            self.pat_step_dx = 1

        # 2.使用LstmModel生成一个步长的音符
        # 2.1.判断这个步长能否为空拍。判断依据为
        # a.当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空。
        # b.不能连续两个小节的第一个步长为空。
        # c.一个乐段的第一个步长不能为空。
        # d.不能连续四拍为空
        flag_allow_empty = True
        if self.pat_step_dx % 4 == 0:
            if self.pat_step_dx > 8 and self.melody_out_notes[-64:].count(0) <= 56:
                flag_allow_empty = False  # 当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空
            if self.melody_out_pats[self.pat_step_dx - 4] == 0:
                flag_allow_empty = False  # 不能连续两个小节的第一个步长为空
        if self.pat_step_dx >= 3 and self.melody_out_pats[-3:] == [0, 0, 0]:
            flag_allow_empty = False  # 不能连续四拍为空
        # 2.2.逐时间步长生成test model输入数据
        time_add = self.pat_step_dx % 8
        melody_input = [[time_add]]  # 当前时间的编码
        # 对于不是一个乐段的第一小节，训练数据只从这个乐段的第一拍开始
        if len(self.melody_out_pats) >= 16:  # melody_out_pat_list足够长的情况
            melody_input[0].extend(self.melody_out_pats[-16:])  # 最近4小节的旋律组合
        else:  # melody_out_pat_list不够长的情况
            melody_input[0].extend([0] * (16 - len(self.melody_out_pats)) + self.melody_out_pats)
        # 2.3.生成输出的音符
        melody_predict = self.predict(session, melody_input)  # LSTM预测 得到二维数组predict
        if self.pat_step_dx <= 3:  # 还没到五拍，不可能出现“连续五拍与某个常见组合雷同”的情况
            out_pat_dx = music_pattern_prediction(melody_predict, int(not flag_allow_empty), COMMON_MELODY_PAT_NUM)
        else:
            try:  # 人为排除与训练集中数据雷同的主旋律组合
                out_pat_dx = melody_pattern_prediction_unique(melody_predict, int(not flag_allow_empty), COMMON_MELODY_PAT_NUM, self.melody_out_pats, self.train_data.input_data)
            except RuntimeError:
                DiaryLog.warn('在第%d个步长, 主旋律第%02d次打回，难以选出不雷同的组合, 最后四拍音符为%s' % (self.pat_step_dx, self.rollback_times, repr(self.melody_out_notes[-32:])))
                self.rollback(4)
                self.rollback_times += 1  # 被打回重新生成了一次
                return
        self.melody_out_pats.append(out_pat_dx)
        self.melody_out_notes.extend(music_pattern_decode(self.train_data.common_melody_pats, [out_pat_dx], 0.125, 1))
        self.pat_step_dx += 1

    def check_1step(self, session):
        # 1.检查音符按键位置是否符合要求。在整小节处检验
        if self.pat_step_dx % 4 == 0:
            if not keypress_check(self.melody_out_notes[-32:]):  # 检查这一小节的音乐是否符合要求 如果不符合要求 则返工 重新生成这一小节的音乐
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，按键位置有异常, 最后四拍音符为%s' % (self.pat_step_dx, self.rollback_times, self.melody_out_notes[-32:]))
                self.rollback(4)
                self.rollback_times += 1
                return

        # 2.检查两小节内的音高变化幅度是否过大。在整小节处检验
        if self.pat_step_dx % 4 == 0 and self.pat_step_dx >= 8:
            shift_score = self.train_data.ShiftConfidence.evaluate(melody_note_list=self.melody_out_notes[-64:])
            if not self.train_data.ShiftConfidence.compare(shift_score):  # 这两小节的音高变化是否在一定限制幅度内
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，音高变化的分数为%.4f,高于临界值%.4f' % (self.pat_step_dx, self.rollback_times, shift_score, self.train_data.ShiftConfidence.confidence_level))
                self.rollback(8)
                self.rollback_times += 1
                return

        # 3.主旋律结束阶段的检验。如果主旋律已经生成了8/10/12小节，且符合结束条件，则结束。如果达到12小节仍未达到结束条件，则打回重新生成
        if self.pat_step_dx % 8 == 0 and self.pat_step_dx // 4 in [8, 10, 12]:
            if melody_end_check(self.melody_out_notes[-32:], self.tone_restrict):
                self.flag_end = True
                return
        if self.pat_step_dx // 4 >= 12:  # 到达这里说明超过12小节，收束仍然不符合要求
            DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回,收束不符合要求, 最后四拍音符为%s' % (self.pat_step_dx, self.rollback_times, repr(self.melody_out_notes[-32:])))
            self.rollback(8)  # 重新生成最后两小节的音乐
            self.rollback_times += 1  # 被打回重新生成了一次

    def generate(self, session):
        self.generate_init()
        while True:
            self.generate_by_step(session)  # 生成一个步长
            self.check_1step(session)  # 对这个步长生成的数据进行校验

            # 退出的条件有二：一是生成失败的次数过多 打回重新生成，二是生成完成
            if self.rollback_times >= MAX_GEN_MELODY_FAIL_TIME:
                DiaryLog.warn('主旋律被打回次数超过%d次,重新生成。\n\n\n' % self.rollback_times)
                raise RuntimeError
            if self.flag_end:
                assert self.pat_step_dx == len(self.melody_out_notes) // 8
                break
        self.section_data = [(0, "main"), (self.pat_step_dx // 4, "empty")]  # 生成只有一个段落的section数据
        core_note_pat_list = CoreNotePatternEncode(self.train_data.common_corenote_pats, self.melody_out_notes, 0.125, 2).music_pattern_list
        keypress_pat_list = keypress_encode(self.melody_out_notes, self.train_data.all_keypress_pats)
        DiaryLog.warn('主旋律的输出:' + repr(self.melody_out_notes) + '\n\n\n')
        return self.melody_out_notes, self.melody_out_pats, core_note_pat_list, keypress_pat_list
