from datainputs.strings import StringTrainData, StringTrainData2, StringTrainData3
from datainputs.melody import melody_core_note_for_chord, CoreNotePatternEncode
from interfaces.chord_parse import chord_rootnote, c_group
from interfaces.note_format import get_abs_notelist_chord
from pipelines.functions import BaseLstmPipelineMultiCode, root_chord_encode, pat_predict_addcode
from models.HmmModel import HmmModel
from models.configs import StringConfig, StringConfig3
from interfaces.utils import DiaryLog
from dataoutputs.validation import string_chord_check, pg_confidence_check, piano_guitar_end_check
import tensorflow as tf
import numpy as np
from settings import *


class StringPipeline:

    def __init__(self, melody_pat_data, continuous_bar_data, chord_cls):
        self.melody_pat_data = melody_pat_data  # 主旋律组合的数据
        self.continuous_bar_data = continuous_bar_data  # 连续小节数据
        self.chord_cls = chord_cls  # 和弦训练用的类（需要用到其中一个方法

    def prepare(self):  # 为防止出现不在同一个graph中的错误 待到prepare再定义模型
        self.train_data = StringTrainData(self.melody_pat_data, self.continuous_bar_data, self.chord_cls)
        self.variable_scope_name = 'StringModel'
        with tf.variable_scope(self.variable_scope_name):
            self.model = HmmModel(transfer=self.train_data.transfer, emission=self.train_data.emission, pi=self.train_data.pi)  # 定义隐式马尔科夫模型

    def model_definition(self, chord_output):
        # 1.首先根据和弦寻找该和弦的根音
        self.root_data = []
        rc_pattern_list = []
        for chord_it in range(len(chord_output)):
            if chord_it == 0:
                self.root_data.append(chord_rootnote(chord_output[0], 0, STRING_AVERAGE_ROOT))
            else:
                self.root_data.append(chord_rootnote(chord_output[chord_it], self.root_data[chord_it - 1], STRING_AVERAGE_ROOT))
        # 2.将和弦和根音组合进行编码
        for chord_it in range(0, len(chord_output), 2):  # 每2拍生成一次
            try:
                rc_pattern_list.append(self.train_data.rc_pattern_count.index([self.root_data[chord_it], chord_output[chord_it]]) - 1)  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要减一
            except ValueError:  # 根音-和弦对照表中没有这个根音-和弦组合
                DiaryLog.warn('chord_output中出现了根音-和弦对照表中找不到和根音-和弦组合, 是' + repr([self.root_data[chord_it], chord_output[chord_it]]))
                rc_pattern_list.append(0)
        # print('root_data', self.root_data)
        # print('pattern_list', rc_pattern_list)
        # print('rc_dict', self.train_data.rc_pattern_dict)
        # 3.定义模型
        with tf.variable_scope(self.variable_scope_name):
            ob_seq = tf.constant(rc_pattern_list, dtype=tf.int32)
            self.model.define_viterbi(ob_seq, len(rc_pattern_list))

    def generate(self, session):
        # 1.运行隐马尔科夫模型 得到输出序列
        [states_seq, state_prob] = session.run([self.model.state_seq, self.model.state_prob])
        # print('state_seq', states_seq)
        # print('state_prob', state_prob)
        # 2.将相对音高转化为绝对音高
        output_note_list = []
        for pattern_it, pattern in enumerate(states_seq):
            rel_note_list = self.train_data.common_string_pats[pattern + 1]  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要加一
            # print('String'rel_note_list)
            for rel_note_group in rel_note_list:
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    output_note_list.append(rel_note_group)
                else:
                    output_note_list.append(get_abs_notelist_chord(rel_note_group, self.root_data[pattern_it * 2]))
        DiaryLog.warn('String的最终输出为:' + repr(output_note_list))
        return output_note_list


class StringPipeline2(BaseLstmPipelineMultiCode):

    def __init__(self, melody_pat_data, continuous_bar_data, corenote_data, corenote_pat_ary, chord_cls):
        self.train_data = StringTrainData2(melody_pat_data, continuous_bar_data, corenote_data, corenote_pat_ary, chord_cls)
        super().__init__(5)  # piano_guitar输入数据的编码为四重编码

    def prepare(self):
        self.config = StringConfig(self.train_data.rc_pat_num)
        self.test_config = StringConfig(self.train_data.rc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'StringModel'

    def generate(self, session, melody_output, chord_output, corenote_pat_ary, melody_beat_num):
        string_output = []  # string输出
        string_pattern_output = []  # string输出的pattern形式
        # 1.数据预处理 将melody_output和chord_output转化为pattern
        melody_output_dic = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}  # melody_pattern的dic形式
        corenote_output = melody_core_note_for_chord(melody_output_dic)  # 转化成骨干音的形式
        corenote_pat_output = CoreNotePatternEncode(corenote_pat_ary, corenote_output, 0.125, 2).music_pattern_ary
        root_data, rc_pat_output = root_chord_encode(chord_output, self.train_data.rc_pattern_count, STRING_AVERAGE_ROOT)
        generate_fail_time = 0  # 本次string的生成一共被打回去多少次
        bar_num = len(melody_output) // 32  # 主旋律一共有多少个小节
        # 2.定义90%string差异分判断的相关参数 包括次验证失败时的备选数据
        confidence_fail_time = [0] * (bar_num + 1)  # 如果自我检查连续失败十次 则直接继续
        string_choose_bk = [[1, 1, 1, 1] for t in range(bar_num + 1)]  # 备选string。为防止死循环，如果连续十次验证失败，则使用备选string
        string_abs_note_bk = [[0] * 32 for t in range(bar_num + 1)]
        diff_score_bk = [np.inf] * (bar_num + 1)  # 备选方案对应的差异函数
        corenote_code_add_base = 4  # 主旋律骨干音数据编码增加的基数
        rc1_code_add_base = 4 + (COMMON_CORE_NOTE_PATTERN_NUMBER + 2)  # 和弦第一拍数据编码增加的基数
        rc2_code_add_base = 4 + (COMMON_CORE_NOTE_PATTERN_NUMBER + 2) + self.train_data.rc_pat_num  # 和弦第二拍数据编码增加的基数
        string_code_add_base = 4 + (COMMON_CORE_NOTE_PATTERN_NUMBER + 2) + self.train_data.rc_pat_num * 2  # string数据编码增加的基数
        # 3.逐两拍生成数据
        beat_dx = 2  # 准备生成第几个拍的数据
        while beat_dx <= len(melody_output) // 8:  # 遍历整个主旋律 步长是2拍
            # 3.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_STRING_GENERATE_FAIL_TIME:
                DiaryLog.warn('string被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 3.2.生成输入数据。生成当前时间的编码,过去10拍的主旋律和过去10拍的和弦和过去10拍的string
            string_prediction_input = list()
            for backward_beat_dx in range(beat_dx - 10, beat_dx, 2):
                cur_step = backward_beat_dx // 2  # 第几个bass的步长
                if cur_step < 0:
                    string_prediction_input.append([cur_step % 2, corenote_code_add_base, rc1_code_add_base, rc2_code_add_base, string_code_add_base])
                elif cur_step < 1:
                    string_prediction_input.append([cur_step % 4, corenote_pat_output[cur_step] + corenote_code_add_base, rc_pat_output[backward_beat_dx] + rc1_code_add_base, rc_pat_output[backward_beat_dx + 1] + rc2_code_add_base, string_code_add_base])
                else:
                    string_prediction_input.append([cur_step % 4, corenote_pat_output[cur_step] + corenote_code_add_base, rc_pat_output[backward_beat_dx] + rc1_code_add_base, rc_pat_output[backward_beat_dx + 1] + rc2_code_add_base, string_pattern_output[cur_step - 1] + string_code_add_base])
            # 3.3.生成输出数据
            string_predict = self.predict(session, [string_prediction_input])  # LSTM预测 得到二维数组predict
            flag_allow_ept = True  # 这两拍是否允许为空
            if beat_dx == 2:  # 首拍的string不能为空
                flag_allow_ept = False
            elif beat_dx >= 6 and string_pattern_output[-2:] == [0, 0]:  # 不能连续六拍为空
                flag_allow_ept = False
            elif beat_dx >= 4 and c_group(chord_output[beat_dx - 1]) != c_group(chord_output[beat_dx - 3]) and c_group(chord_output[beat_dx - 2]) != c_group(chord_output[beat_dx - 4]):  # 和弦发生变化后 string不能为空
                flag_allow_ept = False
            if flag_allow_ept is True:
                string_out_pattern = pat_predict_addcode(string_predict, string_code_add_base, 0, COMMON_STRING_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的string组合string_out_pattern
            else:
                string_out_pattern = pat_predict_addcode(string_predict, string_code_add_base, 1, COMMON_STRING_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的string组合string_out_pattern
            string_pattern_output.append(string_out_pattern)  # 添加到最终的string输出列表中
            rel_note_list = self.train_data.common_string_pats[string_out_pattern]  # 将新生成的string组合变为相对音高列表
            for rel_note_group in rel_note_list:
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    string_output.append(0)
                else:
                    string_output.append(get_abs_notelist_chord(rel_note_group, root_data[beat_dx - 2]))
            beat_dx += 2
            # 3.4.检查生成的string
            if beat_dx >= 10 and beat_dx % 4 == 2 and not string_chord_check(string_output[-32:], chord_output[(beat_dx - 10): (beat_dx - 2)]):  # bass与同时期的和弦差异过大
                # print('在第%d拍, 鼓点第%02d次打回，第一拍为空拍' % (beat_dx, generate_fail_time), drum_output[-16:])
                DiaryLog.warn('在第%d拍, string第%02d次打回，与同时期和弦差异太大' % (beat_dx, generate_fail_time) + repr(string_output[-32:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的string
                string_output = string_output[:(-32)]
                string_pattern_output = string_pattern_output[:(-4)]
                generate_fail_time += 1
                continue
            if beat_dx >= 14 and beat_dx % 8 == 6:  # 每生成了奇数小节之后进行校验
                total_diff_score = pg_confidence_check(string_output[-48:], chord_output[(beat_dx - 10): (beat_dx - 2)])  # 根据训练集90%string差异分判断的校验法
                if total_diff_score >= self.train_data.ConfidenceLevel:
                    DiaryLog.warn('第%d拍,string的误差分数为%.4f,高于临界值%.4f' % (beat_dx, total_diff_score, self.train_data.ConfidenceLevel))
                    curbar = (beat_dx - 2) // 4 - 1  # 当前小节 减一
                    confidence_fail_time[curbar] += 1
                    if total_diff_score <= diff_score_bk[curbar]:
                        string_abs_note_bk[curbar] = string_output[-32:]
                        string_choose_bk[curbar] = string_pattern_output[-4:]
                        diff_score_bk[curbar] = total_diff_score
                    beat_dx -= 8  # 检查不合格 重新生成这两小节的string
                    string_output = string_output[:(-32)]
                    string_pattern_output = string_pattern_output[:(-4)]  # 检查不合格 重新生成这两小节的string
                    if confidence_fail_time[curbar] >= 10:
                        DiaryLog.warn('第%d拍,string使用备选方案,误差函数值为%.4f' % (beat_dx, diff_score_bk[curbar]))
                        string_output = string_output + string_abs_note_bk[curbar]
                        string_pattern_output = string_pattern_output + string_choose_bk[curbar]
                        beat_dx += 8
                    else:
                        continue
            if beat_dx in [melody_beat_num + 2, len(melody_output) // 8 + 2] and not piano_guitar_end_check(string_output):  # 最后一个和弦不是1级大和弦
                DiaryLog.warn('在%d拍, string第%02d次打回，最后一个音是弦外音' % (beat_dx, generate_fail_time) + repr(string_output[-32:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的string
                string_output = string_output[:(-32)]
                string_pattern_output = string_pattern_output[:(-4)]
                generate_fail_time += 1
                continue
        DiaryLog.warn('string的组合输出: ' + repr(string_pattern_output))
        DiaryLog.warn('string的最终输出: ' + repr(string_output))
        return string_output


class StringPipeline3(BaseLstmPipelineMultiCode):

    def __init__(self, melody_pat_data, continuous_bar_data, corenote_data, corenote_pat_ary, chord_cls):
        self.train_data = StringTrainData3(melody_pat_data, continuous_bar_data, corenote_data, corenote_pat_ary, chord_cls)
        super().__init__(4)  # piano_guitar输入数据的编码为四重编码

    def prepare(self):
        self.config = StringConfig3(self.train_data.rc_pat_num)
        self.test_config = StringConfig3(self.train_data.rc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'StringModel'
