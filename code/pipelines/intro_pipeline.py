from settings import *
from dataoutputs.validation import intro_end_check, intro_connect_check
from datainputs.intro import IntroTrainData
from interfaces.music_patterns import music_pattern_decode
from pipelines.functions import BaseLstmPipeline, music_pattern_prediction, melody_pattern_prediction_unique
from models.configs import IntroConfig
from interfaces.utils import DiaryLog
import random
import numpy as np


class IntroPipeline(BaseLstmPipeline):

    def __init__(self, melody_pipeline, tone_restrict):  # raw_melody_data, melody_pat_data, common_melody_pats, section_data, continuous_bar_data, melody_input_data, tone_restrict):
        self.melody_pipeline = melody_pipeline
        self.tone_restrict = tone_restrict  # 这次生成大调还是小调
        self.train_data = IntroTrainData(melody_pipeline.train_data.raw_melody_data, melody_pipeline.train_data.melody_pat_data_nres, melody_pipeline.train_data.common_melody_pats, melody_pipeline.train_data.section_data, melody_pipeline.train_data.continuous_bar_data_nres)
        super().__init__()

    def prepare(self):
        self.config = IntroConfig()
        self.test_config = IntroConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'IntroModel'

    @staticmethod
    def get_intro_beginning(melody_output_len, section_data):
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

    def generate(self, session, melody_output, melody_pat_output):
        # 1.数据预备处理
        generate_fail_time = 0  # 前奏被打回重新生成的次数
        pat_step_dx = 0  # 生成到第几个pattern了
        intro_output = []  # 音符输出
        intro_pattern_list = []  # 把melody_pattern_dict转换成list的形式
        while True:
            self.intro_bars = [2, 4, 6, 8][random.randint(0, 3)]  # 前奏长多少小节（长度必须在正式内容的十分之一到正式内容的三分之一之间）
            if (len(melody_output) // 32) / 10 <= self.intro_bars <= (len(melody_output) // 32) / 3:
                break
        # 2.逐时间步长生成数据
        while True:
            # 2.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_BASS_GENERATE_FAIL_TIME:
                DiaryLog.warn('前奏被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 2.2.随机选取前奏和间奏的开始方式（25%几率是从主歌开头开始的，25%是从副歌开头开始的，25%是从主歌结尾开始的，25%是从副歌结尾开始的）
            if pat_step_dx == 0:
                start_mark, m_bar_dx = self.get_intro_beginning(len(melody_output), self.melody_pipeline.section_data)
                DiaryLog.warn('前奏生成的方式编码为%d' % start_mark)
                if start_mark in [0, 1]:  # 直接以某个段落的开头作为前奏的开头
                    intro_output.extend(melody_output[m_bar_dx * 32: (m_bar_dx + 1) * 32])
                    intro_pattern_list.extend(melody_pat_output[m_bar_dx * 4: (m_bar_dx + 1) * 4])
                    pat_step_dx = 4
            # 2.3.计算每个小节的第一个时间步长能否为空。判断依据为 a.当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空。b.不能连续两个小节的第一个步长为空。
            allow_empty_step = True
            if pat_step_dx % 4 == 0:
                if pat_step_dx > 8 and melody_output[-64:].count(0) <= 56:
                    allow_empty_step = False  # 当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空
                if pat_step_dx >= 4 and intro_pattern_list[pat_step_dx - 4] == 0:
                    allow_empty_step = False  # 不能连续两个小节的第一个步长为空
                if pat_step_dx == 0:
                    allow_empty_step = False  # 前奏的第一个步长为空
            # 2.4.逐时间步长生成test model输入数据
            intro_input = [[pat_step_dx % 4]]  # 当前时间的编码
            if start_mark in [0, 1]:
                if pat_step_dx >= 16:
                    intro_input[0].extend(intro_pattern_list[-16:])  # 最近4小节的旋律组合
                else:
                    intro_input[0].extend([0] * (16 - pat_step_dx) + intro_pattern_list)
            elif start_mark in [2, 3]:
                if pat_step_dx >= 16:
                    intro_input[0].extend(intro_pattern_list[-16:])  # 最近4小节的旋律组合
                else:
                    intro_input[0].extend(melody_pat_output[m_bar_dx * 4 + pat_step_dx - 16: m_bar_dx * 4] + intro_pattern_list)  # 不足4小节的话，前面添加第一段主歌或副歌的最后部分
            # 2.5.生成输出的音符
            intro_predict = self.predict(session, intro_input)  # LSTM预测 得到二维数组predict
            if (pat_step_dx <= 3 and start_mark in [2, 3]) or (pat_step_dx <= 7 and start_mark in [0, 1]):
                output_intro_pattern = music_pattern_prediction(intro_predict, int(not allow_empty_step), COMMON_MELODY_PATTERN_NUMBER)  # 随机生成一个输出旋律组合
            else:
                try:  # 人为排除与训练集中主旋律数据雷同的主旋律组合
                    output_intro_pattern = melody_pattern_prediction_unique(intro_predict, int(not allow_empty_step), COMMON_MELODY_PATTERN_NUMBER, intro_pattern_list, self.melody_pipeline.train_data.input_data, 0)
                except ValueError:
                    DiaryLog.warn('在第%d个pattern, 前奏第%02d次打回，难以选出不雷同的组合' % (pat_step_dx, generate_fail_time) + repr(intro_output[-32:]))
                    pat_step_dx -= 4  # 重新生成这一小节的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    intro_output = intro_output[:len(intro_output) - 32]
                    intro_pattern_list = intro_pattern_list[:len(intro_pattern_list) - 4]
                    continue
            intro_pattern_list.append(output_intro_pattern)
            intro_output.extend(music_pattern_decode(self.melody_pipeline.train_data.common_melody_pats, [output_intro_pattern], 0.125, 1))
            pat_step_dx += 1
            # 2.6.检查连续四拍为空的情况
            if intro_output[-32:] == [0] * 32:
                DiaryLog.warn('在第%d个pattern, 前奏第%02d次打回，连续四拍为空' % (pat_step_dx, generate_fail_time) + repr(intro_output[-32:]))
                pat_step_dx -= 4  # 重新生成这一小节的前奏
                generate_fail_time += 1  # 被打回重新生成了一次
                intro_output = intro_output[:len(intro_output) - 32]
                intro_pattern_list = intro_pattern_list[:len(intro_pattern_list) - 4]
                continue
            # 2.7.检查前奏的最后几拍的情况
            if pat_step_dx == self.intro_bars * 4:
                if not intro_end_check(intro_output, self.tone_restrict):  # 检查前奏的结束阶段是否符合要求
                    DiaryLog.warn('在第%d个pattern, 前奏第%02d次打回，前奏的结束阶段不符合要求' % (pat_step_dx, generate_fail_time) + repr(intro_output[-64:]))
                    pat_step_dx -= 8
                    generate_fail_time += 1  # 被打回重新生成了一次
                    intro_output = intro_output[:len(intro_output) - 64]
                    intro_pattern_list = intro_pattern_list[:len(intro_pattern_list) - 8]
                    continue
                shift_score = intro_connect_check(intro_output[-64:], melody_output)
                if shift_score > self.train_data.ShiftConfidenceLevel:  # 这一小节的音高变化是否在一定限制幅度内
                    DiaryLog.warn('在第%d个pattern, 前奏第%02d次打回，和主旋律开始部分的连接情况得分为%.4f,高于临界值%.4f' % (pat_step_dx, generate_fail_time, shift_score, self.train_data.ShiftConfidenceLevel) + repr(intro_output[-32:]))
                    pat_step_dx -= 4  # 重新生成这两小节的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    intro_output = intro_output[:len(intro_output) - 32]
                    intro_pattern_list = intro_pattern_list[:len(intro_pattern_list) - 4]
                    continue
            # 2.8.检查平均音高的情况（前奏的平均音高要么跟主歌相差1以内，要么跟副歌相差1以内）
            if pat_step_dx == self.intro_bars * 4:
                cluster_ary = self.melody_pipeline.melody_profile.get_melody_profile_by_song(session, {t: intro_output[t * 32: (t + 1) * 32] for t in range(self.intro_bars)})  # 计算前奏的cluster
                intro_cluster = np.mean(cluster_ary)
                flag_cluster_right = False
                for sec_it in range(len(self.melody_pipeline.section_data)):
                    if self.melody_pipeline.section_data[sec_it][1] in ["main", "sub"] and abs(self.melody_pipeline.section_profile_ary[sec_it] - intro_cluster) <= 1.5:
                        flag_cluster_right = True
                        break
                if flag_cluster_right is False:  # cluster校验不通过
                    DiaryLog.warn('在第%d个pattern, 前奏第%02d次打回，cluster与主旋律的cluster相差太大, 前奏的cluster为%d, 各个乐段的cluster分别为' % (pat_step_dx, generate_fail_time, intro_cluster) + repr(self.melody_pipeline.section_profile_ary))
                    rollback_beats = self.intro_bars * 4  # 回退多少小节
                    pat_step_dx -= rollback_beats  # 重新生成这一整个前奏
                    generate_fail_time += 1  # 被打回重新生成了一次
                    intro_output = intro_output[:len(intro_output) - rollback_beats * 8]
                    intro_pattern_list = intro_pattern_list[:len(intro_pattern_list) - rollback_beats]
                    continue
            if pat_step_dx >= self.intro_bars * 4:  # 结束生成的条件
                break
        # 3.输出
        DiaryLog.warn('前奏和间奏的输出:' + repr(intro_output) + '\n\n\n')
        return intro_output
