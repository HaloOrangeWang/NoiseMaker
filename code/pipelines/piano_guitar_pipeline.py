from models.HmmModel import HmmModel
from models.configs import PianoGuitarConfig
from datainputs.piano_guitar import PianoGuitarTrainData, PianoGuitarTrainData2
from dataoutputs.validation import pg_chord_check, pg_confidence_check, piano_guitar_end_check
from settings import *
from pipelines.functions import BaseLstmPipelineMultiCode, keypress_encode, root_chord_encode, pat_predict_addcode
from interfaces.utils import DiaryLog
from interfaces.chord_parse import chord_rootnote
from interfaces.note_format import get_abs_notelist_chord
import tensorflow as tf
import numpy as np


class PianoGuitarPipeline:

    def __init__(self, melody_pat_data, continuous_bar_data, chord_cls, keypress_pat_data, keypress_pat_dic):
        self.melody_pat_data = melody_pat_data  # 主旋律组合的数据
        self.continuous_bar_data = continuous_bar_data  # 连续小节数据
        self.chord_cls = chord_cls  # 和弦训练用的类（需要用到其中一个方法
        self.keypress_pat_data = keypress_pat_data  # 按键组合的数据
        self.keypress_pat_dic = keypress_pat_dic  # 按键组合的对照表

    def prepare(self):  # 为防止出现不在同一个graph中的错误 待到prepare再定义模型
        self.train_data = PianoGuitarTrainData(self.melody_pat_data, self.continuous_bar_data, self.chord_cls, self.keypress_pat_data, self.keypress_pat_dic)
        self.keypress_pat_dic = self.keypress_pat_dic
        self.rhythm_variable_scope_name = 'PgRhythmModel'
        self.final_variable_scope_name = 'PianoGuitarModel'
        self.error_flag = 0
        # print(keypress_pattern_dict)
        with tf.variable_scope(self.rhythm_variable_scope_name):
            self.rhythm_model = HmmModel(transfer=self.train_data.pg_rhythm_data.transfer, emission=self.train_data.pg_rhythm_data.emission, pi=self.train_data.pg_rhythm_data.pi)  # 定义节拍隐马尔科夫模型
        with tf.variable_scope(self.final_variable_scope_name):
            self.final_model = HmmModel(transfer=self.train_data.transfer, emission=self.train_data.emission_final, pi=self.train_data.pi)

    def rhy_model_definition(self, melody_output):
        # 1.首先把主旋律的输出转化为按键组合
        keypress_pat_list = []
        for melody_step_it in range(0, len(melody_output), 16):
            keypress_list = [1 if t != 0 else 0 for t in melody_output[melody_step_it: melody_step_it + 16]]
            try:
                keypress_dx = self.keypress_pat_dic.index(keypress_list) - 1
                if keypress_dx == -1:
                    keypress_dx = keypress_pat_list[-1]
                keypress_pat_list.append(keypress_dx)
            except ValueError:
                DiaryLog.warn('melody_output中出现了中keypress_pat_dic中不存在的旋律组合, 是' + repr(keypress_list))
                keypress_pat_list.append(keypress_pat_list[-1])
            except IndexError:
                DiaryLog.warn('melody_output中出现了中keypress_pat_dic中不存在的旋律组合, 是' + repr(keypress_list))
                keypress_pat_list.append(keypress_pat_list[-1])
        # print(keypress_pattern_list)
        # 2.定义模型
        with tf.variable_scope(self.rhythm_variable_scope_name):
            ob_seq = tf.constant(keypress_pat_list, dtype=tf.int32)
            self.rhythm_model.define_viterbi(ob_seq, len(keypress_pat_list))

    def final_model_definition(self, chord_output):
        # 1.首先根据和弦寻找该和弦的根音
        self.root_data = []
        self.rc_pattern_list = []
        for chord_it in range(len(chord_output)):
            if chord_it == 0:
                self.root_data.append(chord_rootnote(chord_output[0], 0, PIANO_GUITAR_AVERAGE_ROOT))
            else:
                self.root_data.append(chord_rootnote(chord_output[chord_it], self.root_data[chord_it - 1], PIANO_GUITAR_AVERAGE_ROOT))
        # 2.将和弦和根音组合进行编码
        for chord_it in range(len(chord_output)):
            try:
                self.rc_pattern_list.append(self.train_data.rc_pattern_count.index([self.root_data[chord_it], chord_output[chord_it]]) - 1)  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要减一
            except ValueError:  # 根音-和弦对照表中没有这个根音-和弦组合
                DiaryLog.warn('chord_output中出现了根音-和弦对照表中找不到和根音-和弦组合, 是' + repr([self.root_data[chord_it], chord_output[chord_it]]))
                self.rc_pattern_list.append(0)
        # print('root_data', self.root_data)
        # print('pattern_list', rc_pattern_list)
        # print('rc_dict', self.train_data.rc_pattern_dict)
        # 3.定义模型
        with tf.variable_scope(self.final_variable_scope_name):
            self.final_ob_seq = tf.placeholder(tf.int32, [len(self.rc_pattern_list)])
            self.final_model.define_viterbi(self.final_ob_seq, len(self.rc_pattern_list))

    def rhythm_generate(self, session):
        # 1.运行隐马尔科夫模型 得到输出序列
        [states_seq, state_prob] = session.run([self.rhythm_model.state_seq, self.rhythm_model.state_prob])
        DiaryLog.warn('PianoGuitar的节奏输出为:' + repr(states_seq))
        return states_seq

    def final_generate(self, session, rhythm_output):
        keypress_list = []
        # 1.把rhythm_output解码并逐拍进行重编码
        for rhythm in rhythm_output:
            keypress = self.train_data.pg_rhythm_data.rhythm_pattern_dict[rhythm + 1]  # 变成[0,1,0,0,1,0,1]这样的列表形式
            keypress_list.append(8 * keypress[0] + 4 * keypress[1] + 2 * keypress[2] + keypress[3])
            keypress_list.append(8 * keypress[4] + 4 * keypress[5] + 2 * keypress[6] + keypress[7])
        # print('keypress_list', keypress_list)
        # 2.把rhythm和和弦根音进行组合
        for rc_it in range(len(self.rc_pattern_list)):
            self.rc_pattern_list[rc_it] = self.rc_pattern_list[rc_it] * 16 + keypress_list[rc_it]
        # 3.运行隐马尔科夫模型 得到输出序列
        [states_seq, state_prob] = session.run([self.final_model.state_seq, self.final_model.state_prob], feed_dict={self.final_ob_seq: self.rc_pattern_list})
        # print('state_seq', states_seq)
        # print('state_prob', state_prob)
        # 2.将相对音高转化为绝对音高
        output_note_list = []
        for pattern_it, pattern in enumerate(states_seq):
            rel_note_list = self.train_data.common_pg_pats[pattern + 1]  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要加一
            for rel_note_group in rel_note_list:
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    output_note_list.append(rel_note_group)
                else:
                    output_note_list.append(get_abs_notelist_chord(rel_note_group, self.root_data[pattern_it]))
        DiaryLog.warn('PianoGuitar的最终输出为:' + repr(output_note_list))
        return output_note_list


class PianoGuitarPipeline2(BaseLstmPipelineMultiCode):

    def __init__(self, melody_pat_data, continuous_bar_data, keypress_pat_data, keypress_pat_ary, chord_cls):
        self.train_data = PianoGuitarTrainData2(melody_pat_data, continuous_bar_data, keypress_pat_data, keypress_pat_ary, chord_cls)
        super().__init__(4)  # piano_guitar输入数据的编码为四重编码

    def prepare(self):
        self.config = PianoGuitarConfig(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config = PianoGuitarConfig(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'PianoGuitarModel'

    def generate(self, session, melody_output, chord_output, keypress_pats, melody_beat_num):
        pg_output = []  # piano_guitar输出
        pg_pattern_output = []  # piano_guitar输出的pattern形式
        # 1.数据预处理 将melody_output和chord_output转化为pattern
        keypress_pat_output = keypress_encode(melody_output, keypress_pats)
        root_data, rc_pat_output = root_chord_encode(chord_output, self.train_data.rc_pattern_count, PIANO_GUITAR_AVERAGE_ROOT)
        generate_fail_time = 0  # 本次piano_guitar的生成一共被打回去多少次
        bar_num = len(melody_output) // 32  # 主旋律一共有多少个小节
        # 2.定义90%piano_guitar差异分判断的相关参数 包括次验证失败时的备选数据
        confidence_fail_time = [0] * (bar_num + 1)  # 如果自我检查连续失败十次 则直接继续
        pg_choose_bk = [[1, 1, 1, 1] for t in range(bar_num + 1)]  # 备选piano_guitar。为防止死循环，如果连续十次验证失败，则使用备选piano_guitar
        pg_abs_note_bk = [[0] * 32 for t in range(bar_num + 1)]
        diff_score_bk = [np.inf] * (bar_num + 1)  # 备选方案对应的差异函数
        keypress_code_add_base = 8  # 主旋律数据编码增加的基数
        rc_code_add_base = 8 + self.train_data.keypress_pat_num  # 和弦数据编码增加的基数
        pg_code_add_base = 8 + self.train_data.keypress_pat_num + self.train_data.rc_pat_num  # piano_guitar数据编码增加的基数
        # 3.逐拍生成数据
        beat_dx = 1  # 准备生成第几个拍的数据
        while beat_dx <= len(melody_output) // 8:  # 遍历整个主旋律 步长是2拍
            # 3.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_PIANO_GUITAR_GENERATE_FAIL_TIME:
                DiaryLog.warn('piano guitar被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 3.2.生成输入数据。生成当前时间的编码,过去9拍的主旋律和过去9拍的和弦和过去9拍的piano guitar
            pg_prediction_input = list()
            for backward_beat_dx in range(beat_dx - 9, beat_dx):
                cur_keypress_step = backward_beat_dx // 2  # 第几个keypress的步长(keypress的步长是2拍)
                if backward_beat_dx < 0:
                    pg_prediction_input.append([backward_beat_dx % 4, keypress_code_add_base, rc_code_add_base, pg_code_add_base])
                elif backward_beat_dx < 1:
                    pg_prediction_input.append([backward_beat_dx % 8, keypress_pat_output[cur_keypress_step] + keypress_code_add_base, rc_pat_output[backward_beat_dx] + rc_code_add_base, pg_code_add_base])
                else:
                    pg_prediction_input.append([backward_beat_dx % 8, keypress_pat_output[cur_keypress_step] + keypress_code_add_base, rc_pat_output[backward_beat_dx] + rc_code_add_base, pg_pattern_output[backward_beat_dx - 1] + pg_code_add_base])
            # 3.3.生成输出数据
            pg_predict = self.predict(session, [pg_prediction_input])  # LSTM预测 得到二维数组predict
            pg_out_pattern = pat_predict_addcode(pg_predict, pg_code_add_base, 0, COMMON_PIANO_GUITAR_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这拍的piano_guitar组合pg_out_pattern
            pg_pattern_output.append(pg_out_pattern)  # 添加到最终的piano_guitar输出列表中
            rel_note_list = self.train_data.common_pg_pats[pg_out_pattern]  # 将新生成的piano_guitar组合变为相对音高列表
            # print('     rnote_output', rel_note_list)
            for rel_note_group in rel_note_list:
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    pg_output.append(0)
                else:
                    pg_output.append(get_abs_notelist_chord(rel_note_group, root_data[beat_dx - 1]))
            beat_dx += 1
            # 3.4.检查生成的piano_guitar
            if beat_dx >= 9 and beat_dx % 4 == 1 and not pg_chord_check(pg_output[-32:], chord_output[(beat_dx - 9): (beat_dx - 1)]):  # bass与同时期的和弦差异过大
                # print('在第%d拍, 鼓点第%02d次打回，第一拍为空拍' % (beat_dx, generate_fail_time), drum_output[-16:])
                DiaryLog.warn('在第%d拍, piano_guitar第%02d次打回，与同时期和弦差异太大' % (beat_dx, generate_fail_time) + repr(pg_output[-32:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                pg_output = pg_output[:(-32)]
                pg_pattern_output = pg_pattern_output[:(-8)]
                generate_fail_time += 1
                continue
            if beat_dx >= 13 and beat_dx % 8 == 5:  # 每生成了奇数小节之后进行校验
                total_diff_score = pg_confidence_check(pg_output[-48:], chord_output[(beat_dx - 9): (beat_dx - 1)])  # 根据训练集90%bass差异分判断的校验法
                if total_diff_score >= self.train_data.ConfidenceLevel:
                    DiaryLog.warn('第%d拍,piano_guitar的误差分数为%.4f,高于临界值%.4f' % (beat_dx, total_diff_score, self.train_data.ConfidenceLevel))
                    curbar = (beat_dx - 1) // 4 - 1  # 当前小节 减一
                    confidence_fail_time[curbar] += 1
                    if total_diff_score <= diff_score_bk[curbar]:
                        pg_abs_note_bk[curbar] = pg_output[-32:]
                        pg_choose_bk[curbar] = pg_pattern_output[-8:]
                        diff_score_bk[curbar] = total_diff_score
                    beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                    pg_output = pg_output[:(-32)]
                    pg_pattern_output = pg_pattern_output[:(-8)]  # 检查不合格 重新生成这两小节的piano_guitar
                    if confidence_fail_time[curbar] >= 10:
                        DiaryLog.warn('第%d拍,piano_guitar使用备选方案,误差函数值为%.4f' % (beat_dx, diff_score_bk[curbar]))
                        pg_output = pg_output + pg_abs_note_bk[curbar]
                        pg_pattern_output = pg_pattern_output + pg_choose_bk[curbar]
                        beat_dx += 8
                    else:
                        continue
            if beat_dx in [melody_beat_num + 1, len(melody_output) // 8 + 1] and not piano_guitar_end_check(pg_output):  # 最后一个和弦不是1级大和弦
                DiaryLog.warn('在%d拍, piano_guitar第%02d次打回，最后一个音是弦外音' % (beat_dx, generate_fail_time) + repr(pg_output[-32:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的piano_guitar
                pg_output = pg_output[:(-32)]
                pg_pattern_output = pg_pattern_output[:(-8)]
                generate_fail_time += 1
                continue
        DiaryLog.warn('piano_guitar的组合输出: ' + repr(pg_pattern_output))
        DiaryLog.warn('piano_guitar的最终输出: ' + repr(pg_output))
        return pg_output
