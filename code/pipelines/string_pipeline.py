from datainputs.strings import StringTrainData
from datainputs.piano_guitar import RootNote
from dataoutputs.musicout import GetAbsNoteList
from models.HmmModel import HmmModel_2
import tensorflow as tf


class StringPipeline:

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data):
        self.train_data = StringTrainData(melody_pattern_data, continuous_bar_number_data, chord_data)
        self.variable_scope_name = 'StringModel'
        with tf.variable_scope(self.variable_scope_name):
            self.model = HmmModel_2(transfer=self.train_data.transfer, emission=self.train_data.emission, pi=self.train_data.pi)  # 定义隐式马尔科夫模型

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
        for chord_iterator in range(0, len(chord_output), 2):  # 每2拍生成一次
            rc_pattern_list.append(self.train_data.rc_pattern_dict.index([self.root_data[chord_iterator], chord_output[chord_iterator]]) - 1)  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要减一
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
        for pattern_iterator, pattern in enumerate(states_seq):
            rel_note_list = self.train_data.common_string_patterns[pattern + 1]  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要加一
            for note_iterator, rel_note_group in enumerate(rel_note_list):
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    output_note_list.append(rel_note_group)
                else:
                    output_note_list.append(GetAbsNoteList(rel_note_group, self.root_data[pattern_iterator * 2]))
        print(output_note_list)
        return output_note_list
