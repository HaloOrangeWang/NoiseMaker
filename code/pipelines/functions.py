from settings import *
from datainputs.chord import chord_rootnote
import tensorflow as tf
from models.LstmModel import LstmModel, LstmModelMultiCode
import random
import numpy as np
import copy
from interfaces.utils import DiaryLog


class BaseLstmPipeline(object):

    def __init__(self):
        self.prepare()
        self.input_data = tf.placeholder(tf.int32, [None, self.config.input_size])  # 主旋律模型的输入数据
        self.output_data = tf.placeholder(tf.int32, [None, self.config.output_size])  # 主旋律模型的预期输出数据
        self.learning_rate = tf.placeholder(tf.float32, [])  # 从0.90.03开始，学习速率将变为一个变量
        with tf.variable_scope(self.variable_scope_name, reuse=None):
            self.train_model = LstmModel(input_data=self.input_data, output_data=self.output_data, config=self.config, learning_rate=self.learning_rate, is_training=True)  # 定义训练模型
        with tf.variable_scope(self.variable_scope_name, reuse=True):
            self.valid_model = LstmModel(input_data=self.input_data, output_data=self.output_data, config=self.config, learning_rate=self.learning_rate, is_training=False, is_valid=True)  # 定义验证模型
        with tf.variable_scope(self.variable_scope_name, reuse=True):
            self.test_model = LstmModel(input_data=self.input_data, output_data=None, config=self.test_config, learning_rate=self.learning_rate, is_training=False)  # 定义测试的模型

    def prepare(self):
        """prepare方法主要定义config,输入数据和variable_scope的名称"""
        self.config = None
        self.test_config = None
        self.test_config.batch_size = 1
        self.train_data = None
        self.variable_scope_name = None

    def run_epoch(self, session, pattern_number=-1):
        assert len(self.train_data.input_data) == len(self.train_data.output_data)
        # 1.随机选出一些数据用于训练 其余的用于验证
        all_disrupt_input = np.array([[self.train_data.input_data[t], self.train_data.output_data[t]] for t in range(len(self.train_data.input_data))])
        np.random.shuffle(all_disrupt_input)  # 将原始数据打乱顺序
        train_data_num = int(len(self.train_data.input_data) * TRAIN_DATA_RADIO)  # 一共有多少组向量用于训练
        train_data = all_disrupt_input[0: train_data_num]  # 训练数据
        valid_data = all_disrupt_input[train_data_num:]  # 验证数据
        # 2.进行训练
        for epoch in range(self.config.max_max_epoch):
            # 2.1.训练数据
            num_batch = len(train_data) // self.config.batch_size
            lr_decay = self.config.lr_decay ** max(epoch + 1 - self.config.max_epoch, 0.0)  # 求出学习速率的衰减值
            train_disrupt_input = copy.deepcopy(train_data)  # 每一次训练都重新打乱模型的输入数据
            np.random.shuffle(train_disrupt_input)
            DiaryLog.warn('%s的输入向量个数为%d, 一共分为%d个batch.' % (self.variable_scope_name, len(train_data), num_batch))
            for batch in range(num_batch):
                batches_input = train_disrupt_input[self.config.batch_size * batch: self.config.batch_size * (batch + 1), 0]
                batches_output = train_disrupt_input[self.config.batch_size * batch: self.config.batch_size * (batch + 1), 1]
                loss, _, __ = session.run([
                    self.train_model.total_loss,
                    self.train_model.last_state,
                    self.train_model.train_op,
                ], feed_dict={self.input_data: batches_input, self.output_data: batches_output, self.learning_rate: self.config.learning_rate * lr_decay})
                if batch % 10 == 0:
                    DiaryLog.warn('Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
            # 2.2.用验证集评价训练结果 评价指标是准确率
            num_right_predict = 0  # 一共有多少组数据验证正确/错误
            num_wrong_predict = 0
            num_valid_batch = len(valid_data) // self.config.batch_size
            DiaryLog.warn('%s验证集的输入向量个数为%d, 一共分为%d个batch.' % (self.variable_scope_name, len(valid_data), num_valid_batch))
            for batch in range(num_valid_batch):
                batches_input = valid_data[self.config.batch_size * batch: self.config.batch_size * (batch + 1), 0]
                batches_output = valid_data[self.config.batch_size * batch: self.config.batch_size * (batch + 1), 1]
                predict_matrix, __ = session.run([  # 验证模型的输出结果 共batch_size行 input_size列
                    self.valid_model.prediction,
                    self.valid_model.last_state
                ], feed_dict={self.input_data: batches_input, self.output_data: batches_output})
                right_num, wrong_num = self.valid(predict_matrix, batches_output, pattern_number)  # 由于各个模型的评价方式并不完全相同 这里单独写个函数出来用于继承
                num_right_predict += right_num
                num_wrong_predict += wrong_num
            accuracy = num_right_predict / (num_right_predict + num_wrong_predict)  # 正确率
            DiaryLog.warn('%s的第%d个Epoch, 验证正确的个数为%d, 验证错误的个数为%d, 正确率为%.4f' % (self.variable_scope_name, epoch, num_right_predict, num_wrong_predict, accuracy))

    def valid(self, predict_matrix, batches_output, pattern_number):
        """
        验证方法
        :param pattern_number: 组合数量。这个参数用于将本应舍弃的训练向量排除在验证结果之外
        :param predict_matrix: 模型的输出结果
        :param batches_output: 本应该的输出结果
        :return: 训练正确的向量数量和训练错误的向量数量
        """
        predict_value = predict_matrix[:, -1]
        right_value = batches_output[:, -1]
        if pattern_number == -1:
            right_num = sum(predict_value == right_value)
            wrong_num = sum(predict_value != right_value)
        else:
            right_num = sum((predict_value == right_value) & (right_value != pattern_number + 1))  # 不对输出结果为pattern_number+1的向量判断其正确与否
            wrong_num = sum((predict_value != right_value) & (right_value != pattern_number + 1))
        return right_num, wrong_num

    def predict(self, session, predict_input):
        [predict_matrix, last_state] = session.run([self.test_model.prediction, self.test_model.last_state], feed_dict={self.input_data: predict_input})  # LSTM预测 得到二维数组predict
        return predict_matrix


class BaseLstmPipelineMultiCode(object):

    def __init__(self, dim_len):
        self.prepare()
        self.input_data = tf.placeholder(tf.int32, [None, self.config.input_size, dim_len])  # 主旋律模型的输入数据 dim_len为编码的重数
        self.output_data = tf.placeholder(tf.int32, [None, self.config.output_size])  # 主旋律模型的预期输出数据
        self.learning_rate = tf.placeholder(tf.float32, [])  # 从0.90.03开始，学习速率将变为一个变量
        with tf.variable_scope(self.variable_scope_name, reuse=None):
            self.train_model = LstmModelMultiCode(input_data=self.input_data, output_data=self.output_data, config=self.config, learning_rate=self.learning_rate, is_training=True)  # 定义训练模型
        with tf.variable_scope(self.variable_scope_name, reuse=True):
            self.valid_model = LstmModelMultiCode(input_data=self.input_data, output_data=self.output_data, config=self.config, learning_rate=self.learning_rate, is_training=False, is_valid=True)  # 定义验证模型
        with tf.variable_scope(self.variable_scope_name, reuse=True):
            self.test_model = LstmModelMultiCode(input_data=self.input_data, output_data=None, config=self.test_config, learning_rate=self.learning_rate, is_training=False)  # 定义测试的模型

    def prepare(self):
        """prepare方法主要定义config,输入数据和variable_scope的名称"""
        self.config = None
        self.test_config = None
        self.test_config.batch_size = 1
        self.train_data = None
        self.variable_scope_name = None

    def run_epoch(self, session, pattern_number=-1):
        assert len(self.train_data.input_data) == len(self.train_data.output_data)
        # 1.随机选出一些数据用于训练 其余的用于验证
        random_array = np.random.permutation(len(self.train_data.input_data))
        all_disrupt_input = np.array(self.train_data.input_data)[random_array]
        all_disrupt_output = np.array(self.train_data.output_data)[random_array]  # 打乱顺序的原始数据
        train_data_num = int(len(self.train_data.input_data) * TRAIN_DATA_RADIO)  # 一共有多少组向量用于训练
        train_data_input = all_disrupt_input[0: train_data_num]  # 训练数据输入
        train_data_output = all_disrupt_output[0: train_data_num]  # 训练数据输出
        valid_data_input = all_disrupt_input[train_data_num:]  # 验证数据输入
        valid_data_output = all_disrupt_output[train_data_num:]  # 验证数据输出
        # 2.进行训练
        for epoch in range(self.config.max_max_epoch):
            # 2.1.训练数据
            num_batch = len(train_data_input) // self.config.batch_size
            lr_decay = self.config.lr_decay ** max(epoch + 1 - self.config.max_epoch, 0.0)  # 求出学习速率的衰减值
            random_array = np.random.permutation(len(train_data_input))  # # 每一次训练都重新打乱模型的输入数据
            train_disrupt_input = copy.deepcopy(train_data_input[random_array])
            train_disrupt_output = copy.deepcopy(train_data_output[random_array])
            DiaryLog.warn('%s的输入向量个数为%d, 一共分为%d个batch.' % (self.variable_scope_name, len(train_data_input), num_batch))
            for batch in range(num_batch):
                batches_input = train_disrupt_input[self.config.batch_size * batch: self.config.batch_size * (batch + 1)]
                batches_output = train_disrupt_output[self.config.batch_size * batch: self.config.batch_size * (batch + 1)]
                loss, _, __ = session.run([
                    self.train_model.total_loss,
                    self.train_model.last_state,
                    self.train_model.train_op,
                ], feed_dict={self.input_data: batches_input, self.output_data: batches_output, self.learning_rate: self.config.learning_rate * lr_decay})
                if batch % 10 == 0:
                    DiaryLog.warn('Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
            # 2.2.用验证集评价训练结果 评价指标是准确率
            num_right_predict = 0  # 一共有多少组数据验证正确/错误
            num_wrong_predict = 0
            num_valid_batch = len(valid_data_input) // self.config.batch_size
            DiaryLog.warn('%s验证集的输入向量个数为%d, 一共分为%d个batch.' % (self.variable_scope_name, len(valid_data_input), num_valid_batch))
            for batch in range(num_valid_batch):
                batches_input = valid_data_input[self.config.batch_size * batch: self.config.batch_size * (batch + 1)]
                batches_output = valid_data_output[self.config.batch_size * batch: self.config.batch_size * (batch + 1)]
                predict_matrix, __ = session.run([  # 验证模型的输出结果 共batch_size行 input_size列
                    self.valid_model.prediction,
                    self.valid_model.last_state
                ], feed_dict={self.input_data: batches_input, self.output_data: batches_output})
                right_num, wrong_num = self.valid(predict_matrix, batches_output, pattern_number)  # 由于各个模型的评价方式并不完全相同 这里单独写个函数出来用于继承
                num_right_predict += right_num
                num_wrong_predict += wrong_num
            accuracy = num_right_predict / (num_right_predict + num_wrong_predict)  # 正确率
            DiaryLog.warn('%s的第%d个Epoch, 验证正确的个数为%d, 验证错误的个数为%d, 正确率为%.4f' % (self.variable_scope_name, epoch, num_right_predict, num_wrong_predict, accuracy))

    def valid(self, predict_matrix, batches_output, pattern_number):
        """
        验证方法
        :param pattern_number: 组合数量。这个参数用于将本应舍弃的训练向量排除在验证结果之外
        :param predict_matrix: 模型的输出结果
        :param batches_output: 本应该的输出结果
        :return: 训练正确的向量数量和训练错误的向量数量
        """
        predict_value = predict_matrix[:, -1]
        right_value = batches_output[:, -1]
        if pattern_number == -1:
            right_num = sum(predict_value == right_value)
            wrong_num = sum(predict_value != right_value)
        else:
            right_num = sum((predict_value == right_value) & (right_value != pattern_number + 1))  # 不对输出结果为pattern_number+1的向量判断其正确与否
            wrong_num = sum((predict_value != right_value) & (right_value != pattern_number + 1))
        return right_num, wrong_num

    def predict(self, session, predict_input):
        [predict_matrix, last_state] = session.run([self.test_model.prediction, self.test_model.last_state], feed_dict={self.input_data: predict_input})  # LSTM预测 得到二维数组predict
        return predict_matrix


def music_pattern_prediction(predict_matrix, min_pattern_number, max_pattern_number):
    while True:
        row_sum = sum(predict_matrix[-1][min_pattern_number: (max_pattern_number + 1)])
        random_value = random.random() * row_sum
        for element in range(len(predict_matrix[-1][min_pattern_number: (max_pattern_number + 1)])):  # 最多只能到common_melody_pattern
            random_value -= predict_matrix[-1][min_pattern_number:][element]
            if random_value <= 0:
                return element + min_pattern_number


def melody_pattern_prediction_unique(predict_matrix, min_pattern_number, max_pattern_number, melody_pat_ary, input_data, code_add_base):
    # predict_vect = predict_matrix[-1][min_pattern_number: (max_pattern_number + 1)]
    valid_ary = np.ones([len(predict_matrix[-1])])
    for input_ary_dx in range(len(input_data)):
        if input_data[input_ary_dx][-5: -1] == melody_pat_ary[-4:] and input_data[input_ary_dx][-5: -1].count(code_add_base) <= 1:
            valid_ary[input_data[input_ary_dx][-1]] = 0
    predict_vect = copy.deepcopy(predict_matrix[-1])  
    for pat_it in range(len(predict_vect)):
        if pat_it < min_pattern_number or pat_it > max_pattern_number:
            predict_vect[pat_it] = 0
        elif valid_ary[pat_it] == 0:
            predict_vect[pat_it] = 0
    if sum(predict_vect) <= 0.01:
        raise ValueError
    elif sum(predict_vect) <= 0.6:
        return np.argmax(predict_vect)
    else:
        row_sum = sum(predict_vect)
        random_value = random.random() * row_sum
        for element in range(len(predict_vect)):
            random_value -= predict_vect[element]
            if random_value <= 0:
                return element + min_pattern_number


def pat_predict_addcode(predict_matrix, base_code_add, min_pattern_number, max_pattern_number):
    min_pat_dx = base_code_add + min_pattern_number
    max_pat_dx = base_code_add + max_pattern_number
    while True:
        row_sum = sum(predict_matrix[-1][min_pat_dx: (max_pat_dx + 1)])
        random_value = random.random() * row_sum
        for element in range(max_pat_dx - min_pat_dx + 1):  # 最多只能到common_melody_pattern
            random_value -= predict_matrix[-1][min_pat_dx:][element]
            if random_value <= 0:
                return element + min_pattern_number  # 最后返回的是在common_pat_ary中的位置 而不是增加了base_code_add之后的结果


def chord_prediction(predict_matrix):
    """
    生成和弦的方法。
    :param predict_matrix: 和弦预测的二维矩阵
    :return: 和弦预测的一维向量
    """
    chord_number = len(CHORD_DICT) - 1  # 和弦词典中一共有多少种和弦
    predict_matrix = predict_matrix[-2:]
    row_sum_array = []
    for note_it in range(1, (1 + chord_number)):
        row_sum_array.append(sum(predict_matrix[:, note_it]))
    while True:
        row_sum = sum(row_sum_array)
        random_value = random.random() * row_sum
        for element in range(len(CHORD_DICT) - 1):
            random_value -= row_sum_array[element]
            if random_value < 0:
                return element + 1


def chord_prediction_3(predict_matrix, add_base=0):
    """
    生成和弦的方法。这个方法增加了基准和弦量（即chord_dict中的零号和弦对应的matrix索引值不是零，而是增加了一些数的值）
    :param add_base: 和弦增加的基准
    :param predict_matrix: 和弦预测的二维矩阵
    :return: 和弦预测的一维向量
    """
    chord_number = len(CHORD_DICT) - 1  # 和弦词典中一共有多少种和弦
    predict_matrix = predict_matrix[-2:]
    row_sum_array = []
    for note_it in range(1 + add_base, (1 + add_base + chord_number)):
        row_sum_array.append(sum(predict_matrix[:, note_it]))
    while True:
        row_sum = sum(row_sum_array)
        random_value = random.random() * row_sum
        for element in range(len(CHORD_DICT) - 1):
            random_value -= row_sum_array[element]
            if random_value < 0:
                return element + 1


def get_first_melody_pat(pat_num_list, min_pat_dx, max_pat_dx):
    """
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


def imitate_pattern_decode(cur_step, imitate_note_diff, init_time_diff, imitate_spd_ratio, melody_rel_note_list, tone, root_note):
    if tone == TONE_MAJOR:
        rel_list = [0, 2, 4, 5, 7, 9, 11]
    else:
        rel_list = [0, 2, 3, 5, 7, 8, 10]
    abs_note_output = [0 for t in range(8)]
    final_time_diff = init_time_diff + 8 * (1 - imitate_spd_ratio)
    imitate_time_diff = init_time_diff
    for note_it, note in enumerate(melody_rel_note_list[cur_step - init_time_diff: cur_step + 8 - final_time_diff]):
        if note != 0 and note_it + imitate_time_diff == int(note_it + imitate_time_diff):  # 这个时间步长的主旋律有音符
            rel_root_notelist = [[t[0] + imitate_note_diff, t[1]] for t in note]
            abs_note_list = [12 * (t[0] // 7) + rel_list[t[0] % 7] + t[1] + root_note for t in rel_root_notelist]
            abs_note_output[note_it] = abs_note_list
        imitate_time_diff -= (1 - imitate_spd_ratio)
    return abs_note_output


def keypress_encode(melody_output, keypress_pats):
    """
    根据主旋律的输出编码为按键组合
    :param melody_output: 主旋律的输出
    :param keypress_pats: 按键组合的列表
    :return: 编码为按键组合后的结果
    """
    keypress_pat_ary = []
    for melody_step_it in range(0, len(melody_output), 16):
        keypress_list = [1 if t != 0 else 0 for t in melody_output[melody_step_it: melody_step_it + 16]]
        try:
            keypress_dx = keypress_pats.index(keypress_list)
            if keypress_dx == -1:
                keypress_dx = keypress_pat_ary[-1]
            keypress_pat_ary.append(keypress_dx)
        except ValueError:
            DiaryLog.warn('melody_output中出现了中keypress_pat_dic中不存在的旋律组合, 是' + repr(keypress_list))
            keypress_pat_ary.append(keypress_pat_ary[-1])
        except IndexError:
            DiaryLog.warn('melody_output中出现了中keypress_pat_dic中不存在的旋律组合, 是' + repr(keypress_list))
            keypress_pat_ary.append(keypress_pat_ary[-1])
    return keypress_pat_ary


def root_chord_encode(chord_output, root_chord_pats, base_rootnote):
    """
    根据和弦的输出编码为根音-和弦组合
    :param chord_output: 和弦的输出
    :param root_chord_pats: 根音-和弦组合的列表
    :param base_rootnote: 基准根音音高
    :return: 根音数据和根音-和弦组合
    """
    root_data = []
    rc_pat_list = []
    for chord_it in range(len(chord_output)):
        if chord_it == 0:
            root_data.append(chord_rootnote(chord_output[0], 0, base_rootnote))
        else:
            root_data.append(chord_rootnote(chord_output[chord_it], root_data[chord_it - 1], base_rootnote))
    # 2.将和弦和根音组合进行编码
    for chord_it in range(len(chord_output)):
        try:
            rc_pat_list.append(root_chord_pats.index([root_data[chord_it], chord_output[chord_it]]))  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要减一
        except ValueError:  # 根音-和弦对照表中没有这个根音-和弦组合
            DiaryLog.warn('chord_output中出现了根音-和弦对照表中找不到和根音-和弦组合, 是' + repr([root_data[chord_it], chord_output[chord_it]]))
            rc_pat_list.append(0)
    return root_data, rc_pat_list
