from dataoutputs.validation import melody_check, melody_cluster_check, melody_end_check, section_end_check, \
    melody_similarity_check, section_begin_check, melody_confidence_check, dfnote_confidence_check
from models.configs import MelodyConfig, MelodyConfig2, MelodyConfigNoProfile, MelodyConfig3
from datainputs.melody import MelodyProfile, MelodyTrainData, MelodyPatternEncode, MelodyTrainData2, MelodyTrainDataNoProfile, MelodyTrainData3
from settings import *
from interfaces.music_patterns import music_pattern_decode
from pipelines.functions import BaseLstmPipeline, BaseLstmPipelineMultiCode, music_pattern_prediction, \
    get_first_melody_pat, melody_pattern_prediction_unique
from interfaces.utils import DiaryLog
import tensorflow as tf
import random
import copy
import numpy as np


class MelodyPipeline(BaseLstmPipeline):

    def __init__(self, tone_restrict, train=True):
        self.train = train
        self.tone_restrict = tone_restrict  # 这次生成大调还是小调
        with tf.variable_scope('cluster'):
            self.melody_profile = MelodyProfile()
            self.melody_profile.define_cluster_model(tone_restrict=tone_restrict, train=train)
            self.train_data = MelodyTrainData(tone_restrict=self.tone_restrict)
        super().__init__()

    def prepare(self):
        self.config = MelodyConfig()
        self.test_config = MelodyConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'MelodyModel'

    def get_train_data(self, session, train=True):
        """计算音符平均音高的中心点,并根据这些中心点获取主旋律LSTM模型的输入输出数据"""
        self.melody_profile.get_cluster_center_points(session, train)
        self.train_data.get_model_io_data(session, self.melody_profile)

    def generate(self, session, melody_generate_input=None, begin_empty_bar_num=0):
        # 1.数据预备处理
        no_input_melody = not melody_generate_input  # 是否没有预输入的音乐片段
        if no_input_melody:
            melody_generate_input = [0 for t in range(128)]  # 如果没有输入的旋律 前面补上4个空小节
            begin_empty_bar_num = 4  # 最开始的空小节数量
        assert len(melody_generate_input) >= round(128)  # melody_generate_input在4-6小节之间
        assert len(melody_generate_input) <= round(192)
        melody_output = [t for t in melody_generate_input]  # 初始输出 在最后会把最开始的空小节去掉
        generate_fail_time = 0  # 旋律被打回重新生成的次数
        pat_step_dx = 0  # 生成到第几个pattern了
        # 2.获取melody_generate_input的melody profile
        if not no_input_melody:  # 有初始主旋律的情况
            melody_cluster_input = {int(t / 32): melody_generate_input[t:t + 32] for t in range(0, len(melody_generate_input), 32)}  # 获取歌曲的melody profile需要将主旋律转换成dict形式
            melody_profile = self.melody_profile.get_melody_profile_by_song(session, melody_cluster_input)
            initial_melody_profile_length = len(melody_profile)
            initial_melody_cluster = melody_profile[-1]  # 初始音乐的最后一小节的melody_cluster
            # melody_profile += [melody_profile[-1] for t in range(10)]
        else:  # 没有初始主旋律的情况
            initial_melody_profile_length = 4
            melody_profile = [0 for t in range(4)]  # 四个小节的0
        # 3.将melody_generate_input转化为常见的melody_pattern
        if not no_input_melody:  # 有初始主旋律的情况
            melody_pattern_input = {int(t / 32): melody_generate_input[t:t + 32] for t in range(0, len(melody_generate_input), 32)}  # 获取melody pattern 也需要将主旋律转换成dict的形式
            melody_pattern_dict = MelodyPatternEncode(self.train_data.common_melody_pats, melody_pattern_input, 0.125, 1).music_pattern_dic
            melody_pattern_list = []  # 把melody_pattern_dict转换成list的形式
            for key in range(round(len(melody_generate_input) / 32)):  # 遍历这首歌的所有小节
                melody_pattern_list += melody_pattern_dict[key]
        # 4.生成这首歌曲的melody_profile 生成策略是 1-MIN_GENERATE_BAR_NUMBER小节由初始音符的cluster向中音dol所在的cluster渐变 MIN_GENERATE_BAR_NUMBER小节以后全部为中音dol所在的cluster
        if self.tone_restrict == TONE_MAJOR:
            mid_note_melody_cluster = self.melody_profile.get_melody_profile_by_song(session, {0: [72 for t in range(32)], 1: [72 for t in range(32)]})[1]  # 求全部是中音dol/la的情况下的melody_cluster
        elif self.tone_restrict == TONE_MINOR:
            mid_note_melody_cluster = self.melody_profile.get_melody_profile_by_song(session, {0: [69 for t in range(32)], 1: [69 for t in range(32)]})[1]  # 求全部是中音dol/la的情况下的melody_cluster
        else:
            raise ValueError
        min_generate_cluster_number = int((MIN_GENERATE_BAR_NUMBER - len(melody_generate_input) / 32 + begin_empty_bar_num) / 2)  # 最小生成的cluster数量
        if not no_input_melody:
            melody_profile += [round((initial_melody_cluster * ((min_generate_cluster_number + 1) - t) + mid_note_melody_cluster * t) / (min_generate_cluster_number + 1)) for t in range(1, (min_generate_cluster_number + 1)) for t0 in range(2)]
            melody_profile += [mid_note_melody_cluster for t in range(MAX_GENERATE_BAR_NUMBER - MIN_GENERATE_BAR_NUMBER)]
        # 5.逐时间步长生成数据
        while True:
            # 5.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_MELODY_GENERATE_FAIL_TIME:
                DiaryLog.warn('主旋律被打回次数过多,重新生成。\n\n\n' + repr(generate_fail_time))
                return None
            # 5.2.如果输入数据为空 随机生成一个起始pattern作为乐曲的开始
            if no_input_melody and pat_step_dx == 0:
                melody_pattern_list = [0 for t in range(16)]  # 前四小节的melody_pattern肯定都是0
                start_pattern = get_first_melody_pat(self.train_data.melody_pats_num_list, 1, COMMON_MELODY_PATTERN_NUMBER)  # 随机选取开始的pattern 注意 开始的第一个pattern不能为空
                melody_pattern_list.append(start_pattern)
                melody_output += music_pattern_decode(self.train_data.common_melody_pats, [start_pattern], 0.125, 1)  # 将第一个旋律组合解码 得到最初的8个音符的列表
                # 由于开始音符可能会发生变化 因此，每生成一个开始音符都要重新生成一遍melody_cluster
                start_note_cluster = self.melody_profile.get_melody_profile_by_song(session, {0: melody_output[-8:] * 4, 1: melody_output[-8:] * 4})[1]  # 初始音符所在的cluster
                min_generate_cluster_number = int(MIN_GENERATE_BAR_NUMBER / 2)
                melody_profile = melody_profile[:4] + [round((start_note_cluster * (min_generate_cluster_number - t) + mid_note_melody_cluster * t) / min_generate_cluster_number) for t in range(min_generate_cluster_number) for t0 in range(2)]  # melody_profile为从开始音符所在的cluster到中音dol所在的cluster的为期八小节的过度，加上4小节的中音dol所在的cluster
                melody_profile += [mid_note_melody_cluster for t in range(MAX_GENERATE_BAR_NUMBER - MIN_GENERATE_BAR_NUMBER)]
                DiaryLog.warn('主旋律开始组合' + repr(melody_output[-8:]) + repr(start_pattern))
                DiaryLog.warn('主旋律的第一个Profile' + repr(melody_profile))
                pat_step_dx = 1
            # 5.3.逐时间步长生成test model输入数据
            # melody_input = [[round(iterator % 4)]]
            # melody_input[0] += melody_pattern_list[-16:]  # 最近4小节
            bar_add = int(pat_step_dx % 8 >= 4)
            melody_input = [[round(pat_step_dx % 4) + bar_add * 4]]  # 当前时间的编码
            melody_input[0] += melody_profile[int(pat_step_dx / 4) + initial_melody_profile_length - 4: int(pat_step_dx / 4) + initial_melody_profile_length + 1]  # 过去５小节的melody_profile
            melody_input[0] += melody_pattern_list[-16:]  # 最近4小节的旋律组合
            # print('          input', iterator, melody_input[0], end=' ')
            # 5.4.生成输出的音符
            melody_predict = self.predict(session, melody_input)  # LSTM预测 得到二维数组predict
            if pat_step_dx % 4 == 0:  # 每小节的第一拍不能为空 其余三拍可以为空
                output_melody_pattern = music_pattern_prediction(melody_predict, 1, COMMON_MELODY_PATTERN_NUMBER)  # 随机生成一个输出旋律组合
            else:
                output_melody_pattern = music_pattern_prediction(melody_predict, 0, COMMON_MELODY_PATTERN_NUMBER)  # 随机生成一个输出旋律组合
            melody_pattern_list.append(output_melody_pattern)
            melody_output.extend(music_pattern_decode(self.train_data.common_melody_pats, [output_melody_pattern], 0.125, 1))
            # print(list(melody_predict[-1]))
            # print(output_melody_pattern)
            # print('\n\n')
            pat_step_dx += 1
            # 5.5.检查生成的旋律
            if pat_step_dx % 4 == 0:
                if not melody_check(melody_output[-32:]):  # 检查这一小节的音乐是否符合要求 如果不符合要求 则返工 重新生成这一小节的音乐
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，按键位置和音高有异常' % (pat_step_dx, generate_fail_time) + repr(melody_output[-32:]))
                    pat_step_dx -= 4  # 重新生成这一小节的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - 32]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 4]
                    continue
            if pat_step_dx >= 12 and pat_step_dx % 4 == 0:  # 检查是否存在与训练集中的歌曲雷同的情况 如果出现雷同 重新生成三小节的音乐
                if not melody_similarity_check(melody_output[-96:], self.train_data.raw_melody_data):  # 存在雷同 重新生成三小节音乐
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，与训练集中的曲目雷同' % (pat_step_dx, generate_fail_time) + repr(melody_output[-96:]))
                    pat_step_dx -= 12  # 重新生成这3小节的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - 96]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 12]
                    continue
            if pat_step_dx % 8 == 0:
                # print(MelodyClusterCheck(melody.kmeans_sess, melody.kmeans_model, melody.cluster_center_points, melody_cluster_check_input, 7, 0))
                if not melody_cluster_check(session, self.melody_profile, melody_output[-64:], melody_profile[int(pat_step_dx / 4) + initial_melody_profile_length - 1], 3):  # 检查这两小节的音乐cluster是否符合要求 如果不符合要求 则返工 重新生成这两1小节的音乐
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，平均音高不合要求' % (pat_step_dx, generate_fail_time) + repr(melody_output[-64:]))
                    pat_step_dx -= 8
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - 64]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 8]
                    continue
                else:
                    melody_cluster_input = {int(t / 32): melody_output[-64:][t:t + 32] for t in range(0, 64, 32)}
                    latest_melody_cluster = self.melody_profile.get_melody_profile_by_song(session, melody_cluster_input)
                    melody_profile = melody_profile[:(int(pat_step_dx / 4) + initial_melody_profile_length - 2)] + latest_melody_cluster + melody_profile[(int(pat_step_dx / 4) + initial_melody_profile_length):]  # 更新前一个区段的melody_profile 同时后一个区段的的melody_profile为两个6
                    # print(iterator, 'Profile', melody_profile)
            # 5.6.判断是否符合结束生成的条件 如果最后一个音是dol，且总小节数在8-16之间的偶数 则结束生成。
            bar_generate = int(pat_step_dx / 4)
            if bar_generate + initial_melody_profile_length - begin_empty_bar_num >= MIN_GENERATE_BAR_NUMBER and pat_step_dx % 8 == 0:
                if melody_end_check(melody_output[-32:], self.tone_restrict):  # 结束条件是最后音符是dol或la
                    break
            if bar_generate + initial_melody_profile_length - begin_empty_bar_num >= MAX_GENERATE_BAR_NUMBER:  # 到max小节仍未结束 重新生成最后两小节的主旋律
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回,收束不符合要求' % (pat_step_dx, generate_fail_time) + repr(melody_output[-64:]))
                pat_step_dx -= 8  # 重新生成最后两小节的音乐
                generate_fail_time += 1  # 被打回重新生成了一次
                melody_output = melody_output[:len(melody_output) - 64]
                melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 8]
        # 6.输出
        melody_output = melody_output[32 * begin_empty_bar_num:]  # 把前面的空小节去掉
        DiaryLog.warn('主旋律的输出:' + repr(melody_output) + '\n\n\n')
        return melody_output


class MelodyPipeline2(BaseLstmPipelineMultiCode):

    def __init__(self, tone_restrict, train=True):
        self.train = train
        self.tone_restrict = tone_restrict  # 这次生成大调还是小调
        self.train_data = MelodyTrainData2(tone_restrict)  # 生成训练所用的数据
        super().__init__(2)  # chord输入数据的编码为二重编码

    def prepare(self):
        self.config = MelodyConfig2()
        self.test_config = MelodyConfig2()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'MelodyModel'


class MelodyPipelineNoProfile(BaseLstmPipeline):

    section_choice = [  # 可供选择的乐段类型
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

    def __init__(self, tone_restrict, train=True):
        self.train = train
        self.tone_restrict = tone_restrict  # 这次生成大调还是小调
        self.train_data = MelodyTrainDataNoProfile(tone_restrict)  # 生成训练所用的数据
        with tf.variable_scope('cluster'):
            self.melody_profile = MelodyProfile()
            self.melody_profile.define_cluster_model(tone_restrict=tone_restrict, train=train)
        super().__init__()

    def prepare(self):
        self.config = MelodyConfigNoProfile()
        self.test_config = MelodyConfigNoProfile()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'MelodyModel'

    def get_center_points(self, session, train=True):
        """计算音符平均音高的中心点,并根据这些中心点获取主旋律LSTM模型的输入输出数据"""
        self.melody_profile.get_cluster_center_points(session, train)

    def generate(self, session):
        # 1.数据预备处理
        generate_fail_time = 0  # 旋律被打回重新生成的次数
        pat_step_dx = 0  # 生成到第几个pattern了
        self.section_data = copy.copy(self.section_choice[random.randint(0, len(self.section_choice) - 1)])
        melody_output = []  # 音符输出
        melody_pattern_list = []  # 把melody_pattern_dict转换成list的形式
        self.section_profile_ary = np.array([0 for t in range(len(self.section_data))])  # 每一个乐段的melody profile
        section_end_cluster = 0  # 每个乐段结束的最后两小节的melody profile
        transit_fail_time = 0  # 为防止持续失败 如果过渡段验证连续失败10次 则把整个乐段完全重新生成
        # 2.逐时间步长生成数据
        while True:
            # 2.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_MELODY_GENERATE_FAIL_TIME:
                DiaryLog.warn('主旋律被打回次数过多,重新生成。\n\n\n' + repr(generate_fail_time))
                return None
            # 2.2.如果输入数据为空 随机生成一个起始pattern作为乐曲的开始
            if pat_step_dx == 0:
                start_pattern = get_first_melody_pat(self.train_data.melody_pats_num_list, 1, COMMON_MELODY_PATTERN_NUMBER)  # 随机选取开始的pattern 注意 开始的第一个pattern不能为空
                melody_pattern_list.append(start_pattern)
                melody_output += music_pattern_decode(self.train_data.common_melody_pats, [start_pattern], 0.125, 1)  # 将第一个旋律组合解码 得到最初的8个音符的列表
                DiaryLog.warn('主旋律开始组合' + repr(melody_output[-8:]) + '\t' + repr(start_pattern))
                pat_step_dx = 1
            # 2.3.如果这一拍是某一个段落的第一拍，且这个段落与前一个段落相同，则直接复制到过渡小节前面
            sec_dx = -1
            for sec_it in range(len(self.section_data)):
                if self.section_data[sec_it][0] > pat_step_dx // 4:
                    sec_dx = sec_it - 1
                    break  # 找到这一拍属于第几个乐段
            if sec_dx >= 1 and self.section_data[sec_dx][1] == self.section_data[sec_dx - 1][1]:  # 这个乐段与前一个乐段属于同一类型 则直接复制到过渡两小节的前面
                beat_bias = pat_step_dx - self.section_data[sec_dx][0] * 4
                if self.section_data[sec_dx][0] - self.section_data[sec_dx - 1][0] <= 7:
                    trans_bar_len = 1  # 对于不足7小节的乐段来说，有1个过渡小节，除此以外有2个过渡小节
                else:
                    trans_bar_len = 2
                if beat_bias < (self.section_data[sec_dx][0] - self.section_data[sec_dx - 1][0] - trans_bar_len) * 4:
                    melody_output.extend(melody_output[(self.section_data[sec_dx - 1][0] * 4 + beat_bias) * 8: self.section_data[sec_dx][0] * 32 - trans_bar_len * 32])
                    melody_pattern_list.extend(melody_pattern_list[(self.section_data[sec_dx - 1][0] * 4 + beat_bias): self.section_data[sec_dx][0] * 4 - trans_bar_len * 4])
                    pat_step_dx += self.section_data[sec_dx][0] * 4 - self.section_data[sec_dx - 1][0] * 4 - trans_bar_len * 4 - beat_bias
            # 2.4.计算每个小节的第一个时间步长能否为空。判断依据为 a.当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空。b.不能连续两个小节的第一个步长为空。c.一个乐段的第一个步长不能为空。
            allow_empty_step = True
            if pat_step_dx % 4 == 0:
                if pat_step_dx > 8 and melody_output[-64:].count(0) <= 56:
                    allow_empty_step = False  # 当过去两小节有八个（含）以上音符时，下一小节的第一个步长不能为空
                if melody_pattern_list[pat_step_dx - 4] == 0:
                    allow_empty_step = False  # 不能连续两个小节的第一个步长为空
                if self.section_data[sec_dx][0] == pat_step_dx // 4:
                    allow_empty_step = False  # 一个乐段的第一个步长不能为空
            # 2.5.逐时间步长生成test model输入数据
            time_add = pat_step_dx % 4 + 4 * ((pat_step_dx // 4 - self.section_data[sec_dx][0]) % 2)
            melody_input = [[time_add]]  # 当前时间的编码
            # 对于不是一个乐段的第一小节，训练数据只从这个乐段的第一拍开始
            if pat_step_dx - self.section_data[sec_dx][0] * 4 <= 3:
                lookback_beats = 16
            else:
                lookback_beats = min(16, pat_step_dx - self.section_data[sec_dx][0] * 4)
            if len(melody_pattern_list) >= lookback_beats:
                melody_input[0].extend([0] * (16 - lookback_beats) + melody_pattern_list[-lookback_beats:])  # 最近4小节的旋律组合
            else:
                melody_input[0].extend([0] * (16 - len(melody_pattern_list)) + melody_pattern_list)
            # 2.6.生成输出的音符
            melody_predict = self.predict(session, melody_input)  # LSTM预测 得到二维数组predict
            # if pat_step_dx % 4 == 0:  # 每小节的第一拍不能为空 其余三拍可以为空
            #     output_melody_pattern = music_pattern_prediction(melody_predict, 1, COMMON_MELODY_PATTERN_NUMBER)  # 随机生成一个输出旋律组合
            # else:
            #     output_melody_pattern = music_pattern_prediction(melody_predict, 0, COMMON_MELODY_PATTERN_NUMBER)  # 随机生成一个输出旋律组合
            if pat_step_dx <= 3:
                output_melody_pattern = music_pattern_prediction(melody_predict, int(not allow_empty_step), COMMON_MELODY_PATTERN_NUMBER)
            else:
                try:  # 人为排除与训练集中数据雷同的主旋律组合
                    output_melody_pattern = melody_pattern_prediction_unique(melody_predict, int(not allow_empty_step), COMMON_MELODY_PATTERN_NUMBER, melody_pattern_list, self.train_data.input_data, 0)
                except ValueError:
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，难以选出不雷同的组合' % (pat_step_dx, generate_fail_time) + repr(melody_output[-32:]))
                    pat_step_dx -= 4  # 重新生成这一小节的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - 32]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 4]
                    continue
            melody_pattern_list.append(output_melody_pattern)
            melody_output.extend(music_pattern_decode(self.train_data.common_melody_pats, [output_melody_pattern], 0.125, 1))
            # 2.7.检查连续四拍为空的情况
            if melody_output[-32:] == [0] * 32:
                DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，连续四拍为空' % (pat_step_dx, generate_fail_time) + repr(melody_output[-32:]))
                pat_step_dx -= 3  # 重新生成这一小节的音乐 这里是减3，因为此时pat_step_dx尚未加上最后哪一拍
                generate_fail_time += 1  # 被打回重新生成了一次
                melody_output = melody_output[:len(melody_output) - 32]
                melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 4]
                continue
            # 2.8.对于每一个乐段的最后一拍 计算这个乐段的profile
            if pat_step_dx % 4 == 3 and self.section_data[sec_dx + 1][0] - (pat_step_dx // 4) == 1:  # 过渡段的最后一拍
                cluster_ary = self.melody_profile.get_melody_profile_by_song(session, {t - self.section_data[sec_dx][0]: melody_output[t * 32: (t + 1) * 32] for t in range(self.section_data[sec_dx][0], self.section_data[sec_dx + 1][0])})  # 计算这个乐段的cluster
                section_end_cluster = cluster_ary[-1]  # 过渡小节的cluster
                # section_profile_ary.append([section_data[sec_dx][0], np.mean(cluster_ary)])  # 整个乐段的cluster
                self.section_profile_ary[sec_dx] = np.mean(cluster_ary)  # 整个乐段的cluster
            pat_step_dx += 1
            # 2.9.检查生成的旋律
            if pat_step_dx % 4 == 0:
                if not melody_check(melody_output[-32:]):  # 检查这一小节的音乐是否符合要求 如果不符合要求 则返工 重新生成这一小节的音乐
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，按键位置有异常' % (pat_step_dx, generate_fail_time) + repr(melody_output[-32:]))
                    pat_step_dx -= 4  # 重新生成这一小节的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - 32]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 4]
                    continue
            if pat_step_dx % 4 == 0 and pat_step_dx >= 8 and (pat_step_dx // 4 + 1) != self.section_data[sec_dx + 1][0]:  # 音高变化的检查
                shift_score = melody_confidence_check(melody_output[-64:])
                if shift_score > self.train_data.ShiftConfidenceLevel:  # 这两小节的音高变化是否在一定限制幅度内
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，音高变化的分数为%.4f,高于临界值%.4f' % (pat_step_dx, generate_fail_time, shift_score, self.train_data.ShiftConfidenceLevel) + repr(melody_output[-32:]))
                    pat_step_dx -= 8  # 重新生成这两小节的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - 64]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 8]
                    continue
            # if pat_step_dx >= 12 and pat_step_dx % 4 == 0:  # 检查是否存在与训练集中的歌曲雷同的情况 如果出现雷同 重新生成三小节的音乐
            #     if not melody_similarity_check(melody_output[-96:], self.train_data.raw_melody_data):  # 存在雷同 重新生成三小节音乐
            #         DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，与训练集中的曲目雷同' % (pat_step_dx, generate_fail_time) + repr(melody_output[-96:]))
            #         pat_step_dx -= 12  # 重新生成这3小节的音乐
            #         generate_fail_time += 1  # 被打回重新生成了一次
            #         melody_output = melody_output[:len(melody_output) - 96]
            #         melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 12]
            #         continue
            if pat_step_dx % 4 == 0 and self.section_data[sec_dx][1] == "sub" and (sec_dx >= 1 and self.section_data[sec_dx - 1][1] != "sub") and pat_step_dx // 4 == self.section_data[sec_dx + 1][0]:  # 第一段副歌的cluster必须比前面所有段落的cluster都大
                if self.section_profile_ary[sec_dx] <= max(self.section_profile_ary[:sec_dx]):
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，第一段副歌的cluster不是最大, 各个乐段的cluster分别为' % (pat_step_dx, generate_fail_time) + repr(self.section_profile_ary))
                    rollback_beats = (self.section_data[sec_dx + 1][0] - self.section_data[sec_dx][0]) * 4  # 回退多少小节
                    pat_step_dx -= rollback_beats  # 重新生成这一整个乐段的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - rollback_beats * 8]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - rollback_beats]
                    continue
            if self.section_data[sec_dx][1] == "sub" and (sec_dx >= 1 and self.section_data[sec_dx - 1][1] != "sub") and pat_step_dx == self.section_data[sec_dx][0] * 4 + 2:  # 第一段副歌前两拍的cluster必须必前面一个乐段收尾部分的cluster要大
                if melody_output[-16:] != [0] * 16:
                    start_note_cluster = self.melody_profile.get_melody_profile_by_song(session, {0: melody_output[-16:] * 2, 1: melody_output[-16:] * 2})[1]  # 初始音符所在的cluster
                else:
                    start_note_cluster = -1  # 如果副歌的前两拍为空 则认为是不合理的
                if melody_output[-16:] == [0] * 16 or start_note_cluster < section_end_cluster:
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，第一段副歌的开始音符的cluster:%d小于前一段落末尾处的cluster:%d' % (pat_step_dx, generate_fail_time, start_note_cluster, section_end_cluster) + repr(melody_output[-16:]))
                    pat_step_dx -= 2  # 重新生成这一整个乐段的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - 16]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 2]
                    continue
            if pat_step_dx == self.section_data[sec_dx][0] * 4 + 4:  # 每一个乐段的第一个小节必须要有一半的时长为弦内音
                steps_in_chord = section_begin_check(melody_output[-32:])
                if steps_in_chord < 16:
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，乐段的第一个小节只有%d个弦内音' % (pat_step_dx, generate_fail_time, steps_in_chord) + repr(melody_output[-32:]))
                    pat_step_dx -= 4  # 重新生成这一整个乐段的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - 32]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 4]
                    continue
            if pat_step_dx % 4 == 0 and self.section_data[sec_dx][1] == "middle" and sec_dx >= 1 and pat_step_dx // 4 == self.section_data[sec_dx + 1][0]:  # middle的cluster和main相比, 整体音高相差不能超过2
                if abs(self.section_profile_ary[sec_dx] - self.section_profile_ary[sec_dx - 1]) > 2:
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，middle段的cluster与main段的cluster相差太大, 各个乐段的cluster分别为' % (pat_step_dx, generate_fail_time) + repr(self.section_profile_ary))
                    rollback_beats = (self.section_data[sec_dx + 1][0] - self.section_data[sec_dx][0]) * 4  # 回退多少小节
                    pat_step_dx -= rollback_beats  # 重新生成这一整个乐段的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - rollback_beats * 8]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - rollback_beats]
                    continue
            if pat_step_dx % 4 == 0 and pat_step_dx // 4 == self.section_data[sec_dx + 1][0]:  # 段落内差异程度检验
                sec_step_num = (self.section_data[sec_dx + 1][0] - self.section_data[sec_dx][0]) * 32
                dfnote_score = dfnote_confidence_check(melody_output[-sec_step_num:])
                if dfnote_score > self.train_data.DfNoteConfidenceLevel:
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回，统一段落前一半和后一半音符变化的分数为%.4f,高于临界值%.4f' % (pat_step_dx, generate_fail_time, dfnote_score, self.train_data.DfNoteConfidenceLevel))
                    rollback_beats = (self.section_data[sec_dx + 1][0] - self.section_data[sec_dx][0]) * 4  # 回退多少小节
                    pat_step_dx -= rollback_beats  # 重新生成这一整个乐段的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - rollback_beats * 8]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - rollback_beats]
            if pat_step_dx % 4 == 0 and pat_step_dx // 4 == self.section_data[sec_dx + 1][0]:  # 过渡段校验
                # if (section_data[sec_dx + 1][0] - section_data[sec_dx][0]) % 2 == 0:  # 对于长度为奇数的乐段 最后一个音符需要多余3拍 否则多余2拍即可
                #     last_note_min_len = 16
                # else:
                #     last_note_min_len = 24
                if not section_end_check(melody_output[-32:], self.tone_restrict):  # , last_note_min_len):
                    if transit_fail_time >= 10:
                        DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回, 其中该次过渡段连续失败超过10次, 将整个乐段重新生成' % (pat_step_dx, generate_fail_time) + repr(melody_output[-64:]))
                        # if sec_dx >= 1:
                        rollback_beats = (self.section_data[sec_dx + 1][0] - self.section_data[sec_dx][0]) * 4  # 回退多少小节
                        # else:
                        #     rollback_beats = (section_data[sec_dx][0]) * 4
                        pat_step_dx -= rollback_beats  # 重新生成这一整个乐段的音乐
                        generate_fail_time += 1  # 被打回重新生成了一次
                        melody_output = melody_output[:len(melody_output) - rollback_beats * 8]
                        melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - rollback_beats]
                        transit_fail_time = 0
                    else:
                        DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回, 其中该次过渡段连续失败%d次, 过渡段不符合要求' % (pat_step_dx, generate_fail_time, transit_fail_time) + repr(melody_output[-64:]))
                        pat_step_dx -= 8  # 重新生成两小节过渡段的音乐
                        generate_fail_time += 1  # 被打回重新生成了一次
                        melody_output = melody_output[:len(melody_output) - 64]
                        melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 8]
                        transit_fail_time += 1
                    continue
            if pat_step_dx // 4 == self.section_data[-1][0]:
                if not melody_end_check(melody_output[-32:], self.tone_restrict):  # 结束条件是最后音符是dol或la
                    DiaryLog.warn('在第%d个pattern, 主旋律第%02d次打回,收束不符合要求' % (pat_step_dx, generate_fail_time) + repr(melody_output[-64:]))
                    pat_step_dx -= 8  # 重新生成最后两小节的音乐
                    generate_fail_time += 1  # 被打回重新生成了一次
                    melody_output = melody_output[:len(melody_output) - 64]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 8]
                else:
                    break  # 主旋律生成完成
        # 3.输出
        DiaryLog.warn('主旋律的输出:' + repr(melody_output) + '\n\n\n')
        return melody_output
