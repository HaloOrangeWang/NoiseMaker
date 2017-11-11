from dataoutputs.validation import PianoGuitarCheck
from models.configs import PianoGuitarConfig
from models.HmmModel import HmmModel, HmmModel_2
from datainputs.piano_guitar import PianoGuitarTrainData, PianoGuitarTrainData_2, PianoGuitarTrainData_3
from datainputs.melody import MelodyPatternEncode
from settings import *
from dataoutputs.predictions import MusicPatternPrediction
from interfaces.functions import MusicPatternDecode
from models.LstmPipeline import BaseLstmPipeline
from interfaces.chord_parse import RootNote
from dataoutputs.musicout import GetAbsNoteList
import tensorflow as tf


class PianoGuitarPipeline(BaseLstmPipeline):

    x1 = 0
    x2 = 0

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data, tone_restrict):
        self.tone_restrict = tone_restrict  # 这次生成大调还是小调
        self.train_data = PianoGuitarTrainData(melody_pattern_data, continuous_bar_number_data, chord_data)
        super().__init__()

    def prepare(self):
        self.config = PianoGuitarConfig()
        self.test_config = PianoGuitarConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'PianoGuitarModel'

    def generate(self, session, melody_output, chord_output, common_melody_patterns):
        pg_output = []  # piano_guitar输出
        pg_pattern_output = []  # piano_guitar输出的pattern形式
        # 1.数据预处理 将melody_output转化为pattern
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dict = MelodyPatternEncode(common_melody_patterns, melody_pattern_input, 1 / 8, 1).music_pattern_dict
        melody_output_pattern = []  # 把melody_output编码为melody_pattern的形式
        generate_fail_time = 0  # 本次bass的生成一共被打回去多少次
        for key in range(round(len(melody_output) / 32)):  # 遍历这首歌的所有小节 把melody_pattern由dict形式转成list形式
            melody_output_pattern += melody_pattern_dict[key]
        # 2.逐拍生成数据
        generate_pg_iterator = 1
        while generate_pg_iterator <= len(melody_output_pattern):  # 遍历整个主旋律 步长是1拍
            # 2.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_PIANO_GUITAR_GENERATE_FAIL_TIME:
                print('piano_guitar restart ', generate_fail_time)
                return None
            # 2.2.生成输入数据
            # 2.2.1.生成当前时间的编码,过去9拍的主旋律和过去9拍的和弦
            pg_prediction_input = [(generate_pg_iterator - 1) % 8]  # 先保存当前时间的编码
            if generate_pg_iterator < 9:  # 还没有到第9拍 旋律输入在前面补0 否则使用对应位置的旋律和和弦
                last_bars_melody = [0 for t in range(9 - generate_pg_iterator)] + melody_output_pattern[:generate_pg_iterator]
                last_bars_chord = [0 for t in range(9 - generate_pg_iterator)] + chord_output[:generate_pg_iterator]
            else:
                last_bars_melody = melody_output_pattern[(generate_pg_iterator - 9): generate_pg_iterator]
                last_bars_chord = chord_output[(generate_pg_iterator - 9): generate_pg_iterator]
            pg_prediction_input += last_bars_melody + last_bars_chord
            # 2.2.2.生成过去8拍的piano_guitar 如果不足8拍则补0
            required_pg_length = 4 * TRAIN_PIANO_GUITAR_IO_BARS
            if len(pg_pattern_output) < required_pg_length:
                last_bars_pg = [0 for t in range(required_pg_length - len(pg_pattern_output))] + pg_pattern_output
            else:
                last_bars_pg = pg_pattern_output[-required_pg_length:]
            pg_prediction_input += last_bars_pg
            # print('          input', generate_pg_iterator, pg_prediction_input)
            # 2.3.生成输出数据
            pg_predict = self.predict(session, [pg_prediction_input])  # LSTM预测 得到二维数组predict
            if generate_pg_iterator % 8 == 1:  # 每两小节的第一拍不能为空
                pg_out_pattern = MusicPatternPrediction(pg_predict, 1, COMMON_PIANO_GUITAR_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的PIANO_GUITAR组合pg_out_pattern
            else:
                pg_out_pattern = MusicPatternPrediction(pg_predict, 0, COMMON_PIANO_GUITAR_PATTERN_NUMBER)
            pg_pattern_output.append(pg_out_pattern)  # 添加到最终的和弦输出列表中
            pg_output += MusicPatternDecode(self.train_data.common_pg_patterns, [pg_out_pattern], 1 / 4, 1)  # 将新生成的piano_guitar组合解码为音符列表
            # 添加到最终的和弦输出列表中
            # print(list(pg_predict[-1]))
            # print(pg_out_pattern, pg_pattern_output)
            # print('\n\n')
            generate_pg_iterator += 1  # 步长是1拍
            # 2.4.检查生成的piano_guitar
            # if generate_pg_iterator >= 9 and generate_pg_iterator % 4 == 1 and not PianoGuitarCheck(pg_output[-32:], chord_output[(generate_pg_iterator - 9): (generate_pg_iterator - 1)], self.tone_restrict):  # bass与同时期的和弦差异过大
            #     print(generate_pg_iterator, '1False%02d' % generate_fail_time, pg_output)
            #     generate_pg_iterator -= 8  # 检查不合格 重新生成这两小节的bass
            #     pg_output = pg_output[:(-64)]
            #     pg_pattern_output = pg_pattern_output[:(-8)]
            #     generate_fail_time += 1
            #     continue
            if generate_pg_iterator >= 9 and generate_pg_iterator % 8 == 1:
                self.x1 += PianoGuitarCheck(pg_output[-32:], chord_output[(generate_pg_iterator - 9): (generate_pg_iterator - 1)], self.tone_restrict)
                self.x2 += (4 - PianoGuitarCheck(pg_output[-32:], chord_output[(generate_pg_iterator - 9): (generate_pg_iterator - 1)], self.tone_restrict))
                # print(generate_pg_iterator, PianoGuitarCheck(pg_output[-32:], chord_output[(generate_pg_iterator - 9): (generate_pg_iterator - 1)], self.tone_restrict))
        print(self.x1, self.x2)#, pg_pattern_output)
        # print('piano_guitar', pg_output)
        return pg_output


class PianoGuitarPipeline_2:

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data):
        self.train_data = PianoGuitarTrainData_2(melody_pattern_data, continuous_bar_number_data, chord_data)
        self.model = HmmModel(transfer=self.train_data.transfer, emission=self.train_data.emission, pi=self.train_data.pi)  # 定义隐式马尔科夫模型
        self.variable_scope_name = 'PianoGuitarModel'

    def model_definition(self, chord_output):
        # 1.首先根据和弦寻找该和弦的根音
        self.root_data = []
        rc_pattern_list = []
        for chord_iterator in range(len(chord_output)):
            if chord_iterator == 0:
                self.root_data.append(RootNote(chord_output[0], 0))
            else:
                self.root_data.append(RootNote(chord_output[chord_iterator], self.root_data[chord_iterator - 1]))
        # 2.将和弦和根音组合进行编码
        for chord_iterator in range(len(chord_output)):
            rc_pattern_list.append(self.train_data.rc_pattern_dict.index([self.root_data[chord_iterator], chord_output[chord_iterator]]) - 1)  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要减一
        # print('root_data', self.root_data)
        # print('pattern_list', rc_pattern_list)
        # print('rc_dict', self.train_data.rc_pattern_dict)
        # 3.定义模型
        self.model.define_viterbi(rc_pattern_list)

    def generate(self, session):
        # １.运行隐马尔科夫模型 得到输出序列
        [states_seq, state_prob] = session.run([self.model.state_seq, self.model.state_prob])
        print('state_seq', states_seq)
        # print('state_prob', state_prob)
        # 2.将相对音高转化为绝对音高
        output_note_list = []
        for pattern_iterator, pattern in enumerate(states_seq):
            rel_note_list = self.train_data.common_pg_patterns[pattern + 1]  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要加一
            for note_iterator, rel_note_group in enumerate(rel_note_list):
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    output_note_list.append(rel_note_group)
                else:
                    output_note_list.append(GetAbsNoteList(rel_note_group, self.root_data[pattern_iterator]))
        print(output_note_list)
        return output_note_list


class PianoGuitarPipeline_3:

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data, keypress_pattern_data, keypress_pattern_dict):
        self.train_data = PianoGuitarTrainData_3(melody_pattern_data, continuous_bar_number_data, chord_data, keypress_pattern_data, keypress_pattern_dict)
        self.keypress_pattern_dict = keypress_pattern_dict
        self.train_data.combine_rck_2()
        self.rhythm_variable_scope_name = 'PgRhythmModel'
        self.final_variable_scope_name = 'PianoGuitarModel'
        self.error_flag = 0
        # print(keypress_pattern_dict)
        with tf.variable_scope(self.rhythm_variable_scope_name):
            self.rhythm_model = HmmModel_2(transfer=self.train_data.pg_rhythm_data.transfer, emission=self.train_data.pg_rhythm_data.emission, pi=self.train_data.pg_rhythm_data.pi)  # 定义节拍隐马尔科夫模型
        with tf.variable_scope(self.final_variable_scope_name):
            self.final_model = HmmModel_2(transfer=self.train_data.transfer, emission=self.train_data.emission_final, pi=self.train_data.pi)

    def rhy_model_definition(self, melody_output):
        # 1.首先把主旋律的输出转化为按键组合
        keypress_pattern_list = []
        for melody_iterator in range(0, len(melody_output), 16):
            keypress_list = [1 if t != 0 else 0 for t in melody_output[melody_iterator: melody_iterator + 16]]
            try:
                keypress_index = self.keypress_pattern_dict.index(keypress_list) - 1
                if keypress_index == -1:
                    keypress_index = keypress_pattern_list[-1]
                keypress_pattern_list.append(keypress_index)
            except ValueError:
                # raise ValueError("")
                keypress_pattern_list.append(1)
            except IndexError:
                # raise ValueError("")
                keypress_pattern_list.append(1)
        # print(keypress_pattern_list)
        # 2.定义模型
        with tf.variable_scope(self.rhythm_variable_scope_name):
            ob_seq = tf.constant(keypress_pattern_list, dtype=tf.int32)
            self.rhythm_model.define_viterbi(ob_seq, len(keypress_pattern_list))

    def final_model_definition(self, chord_output):
        # 1.首先根据和弦寻找该和弦的根音
        self.root_data = []
        self.rc_pattern_list = []
        for chord_iterator in range(len(chord_output)):
            if chord_iterator == 0:
                self.root_data.append(RootNote(chord_output[0], 0))
            else:
                self.root_data.append(RootNote(chord_output[chord_iterator], self.root_data[chord_iterator - 1]))
        # 2.将和弦和根音组合进行编码
        for chord_iterator in range(len(chord_output)):
            self.rc_pattern_list.append(self.train_data.rc_pattern_dict.index([self.root_data[chord_iterator], chord_output[chord_iterator]]) - 1)  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要减一
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
        print('rhy_out', states_seq)
        return states_seq

    def final_generate(self, session, rhythm_output):
        keypress_list = []
        # 1.把rhythm_output解码并逐拍进行重编码
        for rhythm in rhythm_output:
            keypress = self.train_data.pg_rhythm_data.rhythm_pattern_dict[rhythm + 1]
            keypress_list.append(8 * keypress[0] + 4 * keypress[1] + 2 * keypress[2] + keypress[3])
            keypress_list.append(8 * keypress[4] + 4 * keypress[5] + 2 * keypress[6] + keypress[7])
        # print('keypress_list', keypress_list)
        # 2.把rhythm和和弦根音进行组合
        for rc_iterator in range(len(self.rc_pattern_list)):
            self.rc_pattern_list[rc_iterator] = self.rc_pattern_list[rc_iterator] * 16 + keypress_list[rc_iterator]
        # 3.运行隐马尔科夫模型 得到输出序列
        [states_seq, state_prob] = session.run([self.final_model.state_seq, self.final_model.state_prob], feed_dict={self.final_ob_seq: self.rc_pattern_list})
        # print('state_seq', states_seq)
        # print('state_prob', state_prob)
        # 2.将相对音高转化为绝对音高
        output_note_list = []
        for pattern_iterator, pattern in enumerate(states_seq):
            rel_note_list = self.train_data.common_pg_patterns[pattern + 1]  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要加一
            for note_iterator, rel_note_group in enumerate(rel_note_list):
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    output_note_list.append(rel_note_group)
                else:
                    output_note_list.append(GetAbsNoteList(rel_note_group, self.root_data[pattern_iterator]))
        print(output_note_list)
        return output_note_list
