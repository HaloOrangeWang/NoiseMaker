from settings import *
from pipelines.functions import BaseLstmPipeline, music_pattern_prediction, melody_pattern_prediction_unique, keypress_encode
from datainputs.intro import IntroTrainData, IntroTestData
from datainputs.melody import CoreNotePatternEncode
from models.configs import IntroConfig
from interfaces.utils import DiaryLog
from interfaces.music_patterns import music_pattern_decode
from validations.intro import intro_end_check
import numpy as np
import random


def get_intro_beginning(melody_output_len, section_data):
    """
    随机得到一种前奏开始生成的方式
    :param melody_output_len: 主旋律的音符长度
    :param section_data: 主旋律的乐段数据
    :return: 前奏开始时的编码（0/1/2/3），intro接入主旋律的哪个小节之后
    """
    random_value = random.random()
    if random_value <= 0.25:  # 以第一段主歌的第一小节作为前奏的开头
        for sec_it in range(0, len(section_data)):
            if section_data[sec_it][1] == "main" and (sec_it == 0 or section_data[sec_it - 1][1] != "main"):
                return 0, section_data[sec_it][0]
    elif random_value <= 0.5:  # 以第一段副歌的第一小节作为前奏的开头
        for sec_it in range(0, len(section_data)):
            if section_data[sec_it][1] == "sub" and (sec_it == 0 or section_data[sec_it - 1][1] != "sub"):
                return 1, section_data[sec_it][0]
    elif random_value <= 0.75:  # 从第一段主歌的结尾处开始生成
        for sec_it in range(0, len(section_data)):
            if section_data[sec_it][1] == "main" and (sec_it == 0 or section_data[sec_it - 1][1] != "main"):
                if sec_it == len(section_data) - 1:
                    return 2, melody_output_len // 32
                else:
                    return 2, section_data[sec_it + 1][0]
    else:  # 从第一段副歌的结尾处开始生成
        for sec_it in range(0, len(section_data)):
            if section_data[sec_it][1] == "sub" and (sec_it == 0 or section_data[sec_it - 1][1] != "sub"):
                if sec_it == len(section_data) - 1:
                    return 3, melody_output_len // 32
                else:
                    return 3, section_data[sec_it + 1][0]


class IntroPipeline(BaseLstmPipeline):

    def __init__(self, is_train, melody_pipe_cls, tone_restrict):
        super().__init__()
        self.melody_pipe_cls = melody_pipe_cls
        self.tone_restrict = tone_restrict  # 这次生成大调还是小调
        if is_train:
            self.train_data = IntroTrainData(melody_pipe_cls.train_data.raw_melody_data, melody_pipe_cls.train_data.melody_pat_data_nres, melody_pipe_cls.train_data.common_melody_pats, melody_pipe_cls.train_data.section_data, melody_pipe_cls.train_data.continuous_bar_data_nres)
        else:
            self.train_data = IntroTestData()

    # noinspection PyAttributeOutsideInit
    def prepare(self):
        self.config = IntroConfig()
        self.test_config = IntroConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'IntroModel'

    # noinspection PyAttributeOutsideInit
    def generate_init(self, session, melody_out_notes, melody_out_pats):
        self.melody_out_notes = melody_out_notes
        self.melody_out_pats = melody_out_pats

        # 生成过程中的常量和类型
        while True:
            self.intro_bar_num = [2, 4, 6, 8][random.randint(0, 3)]  # 前奏长多少小节（长度必须在正式内容的十分之一到正式内容的三分之一之间，且是2，4，6，8中的偶数）
            if (len(melody_out_notes) // 32) / 10 <= self.intro_bar_num <= (len(melody_out_notes) // 32) / 3:
                break

        # 生成过程中的一些变量
        self.rollback_times = 0  # 被打回重新生成的次数
        self.beat_dx = 0  # 生成完第几个pattern了
        self.start_mark = 0  # 前奏开始时的编码（0/1/2/3）
        self.m_bar_dx = 0  # intro接入主旋律的哪个小节之后

        # 生成结果
        self.intro_out_notes = []
        self.intro_out_pats = []

    def rollback(self, step_num):
        """
        将主旋律的输出回滚step_num个步长（pat_step_dx减少，output_notes和output_pats去掉最后几个数）
        :param step_num: 需要回滚的步长数量
        """
        self.beat_dx -= step_num  # 重新生成这一小节的音乐
        self.intro_out_notes = self.intro_out_notes[:len(self.intro_out_notes) - step_num * 8]
        self.intro_out_pats = self.intro_out_pats[:len(self.intro_out_pats) - step_num]

    def generate_by_step(self, session):
        # 1.随机选取前奏和间奏的开始方式（25%几率是从主歌开头开始的，25%是从副歌开头开始的，25%是从主歌结尾开始的，25%是从副歌结尾开始的）
        if self.beat_dx == 0:
            self.start_mark, self.m_bar_dx = get_intro_beginning(len(self.melody_out_notes), self.melody_pipe_cls.section_data)
            DiaryLog.warn('前奏生成的方式编码为%d' % self.start_mark)
            if self.start_mark in [0, 1]:  # 直接以某个段落的开头作为前奏的开头
                self.intro_out_notes.extend(self.melody_out_notes[self.m_bar_dx * 32: (self.m_bar_dx + 1) * 32])
                self.intro_out_pats.extend(self.melody_out_pats[self.m_bar_dx * 4: (self.m_bar_dx + 1) * 4])
                self.beat_dx += 4

        # 2.使用LstmModel生成一个步长的音符
        # 2.1.判断这个步长能否为空拍。判断依据为
        # a.当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空。
        # b.不能连续两个小节的第一个步长为空。
        # c.第一个步长不能为空
        # d.不能连续四拍为空
        flag_allow_empty = True
        if self.beat_dx % 4 == 0:
            if self.beat_dx > 8 and self.melody_out_notes[-64:].count(0) <= 56:
                flag_allow_empty = False  # 当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空
            if self.beat_dx >= 4 and self.intro_out_pats[self.beat_dx - 4] == 0:
                flag_allow_empty = False  # 不能连续两个小节的第一个步长为空
            if self.beat_dx == 0:
                flag_allow_empty = False  # 前奏的第一个步长不能为空
            if self.beat_dx >= 3 and self.intro_out_pats[-3:] == [0, 0, 0]:
                flag_allow_empty = False  # 不能连续四拍为空
        # 2.2.逐时间步长生成test model输入数据
        intro_input = [[self.beat_dx % 4]]  # 当前时间的编码
        if self.start_mark in [0, 1]:
            if self.beat_dx >= 16:
                intro_input[0].extend(self.intro_out_pats[-16:])  # 最近4小节的旋律组合
            else:
                intro_input[0].extend([0] * (16 - self.beat_dx) + self.intro_out_pats)
        elif self.start_mark in [2, 3]:
            if self.beat_dx >= 16:
                intro_input[0].extend(self.intro_out_pats[-16:])  # 最近4小节的旋律组合
            else:
                intro_input[0].extend(self.melody_out_pats[self.m_bar_dx * 4 + self.beat_dx - 16: self.m_bar_dx * 4] + self.intro_out_pats)  # 不足4小节的话，前面添加第一段主歌或副歌的最后部分
        # 2.3.生成输出的音符
        intro_predict = self.predict(session, intro_input)  # LSTM预测 得到二维数组predict
        if (self.beat_dx <= 3 and self.start_mark in [2, 3]) or (self.beat_dx <= 7 and self.start_mark in [0, 1]):
            out_pat_dx = music_pattern_prediction(intro_predict, int(not flag_allow_empty), COMMON_MELODY_PAT_NUM)  # 随机生成一个输出旋律组合
        else:
            try:  # 人为排除与训练集中数据雷同的主旋律组合
                out_pat_dx = melody_pattern_prediction_unique(intro_predict, int(not flag_allow_empty), COMMON_MELODY_PAT_NUM, self.melody_out_pats, self.train_data.input_data)
            except ValueError:
                DiaryLog.warn('在第%d个步长, 前奏第%02d次打回，难以选出不雷同的组合, 最后四拍音符为%s' % (self.beat_dx, self.rollback_times, repr(self.intro_out_notes[-32:])))
                self.rollback(4)
                self.rollback_times += 1  # 被打回重新生成了一次
                return
        self.intro_out_pats.append(out_pat_dx)
        self.intro_out_notes.extend(music_pattern_decode(self.melody_pipe_cls.train_data.common_melody_pats, [out_pat_dx], 0.125, 1))
        self.beat_dx += 1

    def check_1step(self, session):
        # 1.检查前奏的结束部分是否符合要求。在最后一个步长执行检验
        if self.beat_dx == self.intro_bar_num * 4:
            if not intro_end_check(self.intro_out_notes, self.tone_restrict):  # 检查前奏的结束阶段是否符合要求
                DiaryLog.warn('在第%d个pattern, 前奏第%02d次打回，前奏的结束阶段不符合要求, 最后四拍音符为%s' % (self.beat_dx, self.rollback_times, repr(self.intro_out_notes[-32:])))
                self.rollback(8)
                self.rollback_times += 1
                return

        # 2.检查前奏和主旋律的连接部分是否符合要求。在最后一个步长执行检验
        if self.beat_dx == self.intro_bar_num * 4:
            shift_score = self.train_data.ShiftConfidence.evaluate(melody_list=self.melody_out_notes, intro_list=self.intro_out_notes[-64:])
            if not self.train_data.ShiftConfidence.compare(shift_score):  # 这两小节的音高变化是否在一定限制幅度内
                DiaryLog.warn('在第%d个pattern, 前奏第%02d次打回，和主旋律开始部分的连接情况得分为%.4f,高于临界值%.4f' % (self.beat_dx, self.rollback_times, shift_score, self.train_data.ShiftConfidence.confidence_level))
                self.rollback(4)
                self.rollback_times += 1
                return

        # 3.检查前奏平均音高的情况（前奏的平均音高要么跟主歌相差1以内，要么跟副歌相差1以内）。在最后一个步长执行检验
        if self.beat_dx == self.intro_bar_num * 4:
            cluster_ary = self.melody_pipe_cls.melody_profile.get_melody_profile_by_song(session, self.intro_out_notes)  # 计算前奏的cluster
            intro_cluster = np.mean(cluster_ary)
            flag_cluster_right = False
            for sec_it in range(len(self.melody_pipe_cls.section_data)):
                if self.melody_pipe_cls.section_data[sec_it][1] in ["main", "sub"] and abs(self.melody_pipe_cls.sec_profile_list[sec_it] - intro_cluster) <= 1.5:
                    flag_cluster_right = True
                    break
            if flag_cluster_right is False:  # cluster校验不通过
                DiaryLog.warn('在第%d个pattern, 前奏第%02d次打回，cluster与主旋律的cluster相差太大, 前奏的cluster为%.4f, 各个乐段的cluster分别为%s' % (self.beat_dx, self.rollback_times, float(intro_cluster), repr(self.melody_pipe_cls.sec_profile_list)))
                rollback_beats = self.intro_bar_num * 4  # 回退多少小节
                self.rollback(rollback_beats)
                self.rollback_times += 1

    def generate(self, session, melody_out_notes, melody_out_pats):
        self.generate_init(session, melody_out_notes, melody_out_pats)
        while True:
            self.generate_by_step(session)
            self.check_1step(session)

            if self.rollback_times >= MAX_GEN_INTRO_FAIL_TIME:
                DiaryLog.warn('前奏被打回次数超过%d次,重新生成。\n\n\n' % self.rollback_times)
                raise RuntimeError
            if self.beat_dx == self.intro_bar_num * 4:
                assert self.beat_dx == len(self.intro_out_notes) // 8
                break

        core_note_pat_list = CoreNotePatternEncode(self.melody_pipe_cls.train_data.common_corenote_pats, self.intro_out_notes, 0.125, 2).music_pattern_list
        keypress_pat_list = keypress_encode(self.intro_out_notes, self.melody_pipe_cls.train_data.all_keypress_pats)
        DiaryLog.warn('前奏的输出:' + repr(self.intro_out_notes) + '\n\n\n')
        return self.intro_out_notes, self.intro_out_pats, core_note_pat_list, keypress_pat_list
