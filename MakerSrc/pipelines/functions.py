from settings import *
from models.LstmModel import LstmModel
from interfaces.utils import DiaryLog
from interfaces.chord_parse import get_chord_root_pitch
import tensorflow as tf
import random
import numpy as np
import copy


class BaseLstmPipeline(object):

    def __init__(self):
        self.prepare()
        if self.config.input_dim == 1:  # 区分one-hot编码和多重编码
            self.input_data = tf.placeholder(tf.int32, [None, self.config.input_size])  # 模型的输入数据
        else:
            self.input_data = tf.placeholder(tf.int32, [None, self.config.input_size, self.config.input_dim])  # 主旋律模型的输入数据 dim_len为编码的重数

        self.output_data = tf.placeholder(tf.int32, [None, self.config.output_size])  # 模型的预期输出数据
        self.learning_rate = tf.placeholder(tf.float32, [])  # 从0.90.03开始，学习速率将变为一个变量
        with tf.variable_scope(self.variable_scope_name, reuse=None):
            self.train_model = LstmModel(input_data=self.input_data, output_data=self.output_data, config=self.config, learning_rate=self.learning_rate, is_training=True)  # 定义训练模型
        with tf.variable_scope(self.variable_scope_name, reuse=True):
            self.valid_model = LstmModel(input_data=self.input_data, output_data=self.output_data, config=self.config, learning_rate=self.learning_rate, is_training=False, is_valid=True)  # 定义验证模型
        with tf.variable_scope(self.variable_scope_name, reuse=True):
            self.test_model = LstmModel(input_data=self.input_data, output_data=None, config=self.test_config, learning_rate=self.learning_rate, is_training=False)  # 定义测试的模型

    # noinspection PyAttributeOutsideInit
    def prepare(self):
        """prepare方法主要定义config,输入数据、variable_scope、generate类的名称"""
        self.config = None
        self.test_config = None
        self.test_config.batch_size = 1
        self.train_data = None
        self.variable_scope_name = None

    def run_epoch(self, session, pattern_number=-1):
        if self.config.input_dim == 1:
            run_func = self.run_epoch_one_hot
        elif self.config.input_dim >= 2:
            run_func = self.run_epoch_multi_code
        else:
            raise ValueError
        run_func(session, pattern_number)

    def run_epoch_one_hot(self, session, pattern_number=-1):
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

    def run_epoch_multi_code(self, session, pattern_number=-1):
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


def music_pattern_prediction(predict_matrix, min_pat_dx, max_pat_dx):
    """
    根据概率矩阵生成下一个音符组合的内容
    :param predict_matrix: 输出的概率矩阵
    :param min_pat_dx: 预测的结果“矩阵最后一行的索引值”应当在min_pat_dx至max_pat_dx之间
    :param max_pat_dx: 预测的结果“矩阵最后一行的索引值”应当在min_pat_dx至max_pat_dx之间
    :return: 生成的最后一拍结果（“矩阵最后一行的索引值”）
    """
    cut_list = predict_matrix[-1][min_pat_dx: (max_pat_dx + 1)]  # 从predict_matrix中截取一部分作为待选的概率向量
    row_sum = sum(cut_list)
    random_value = random.random() * row_sum
    for element in range(len(cut_list)):
        random_value -= cut_list[element]
        if random_value <= 0:
            return element + min_pat_dx


def melody_pattern_prediction_unique(predict_matrix, min_pat_dx, max_pat_dx, melody_out_pat_list, input_data):
    """
    根据概率矩阵生成下一个音符组合的内容，但不能和训练集中的某一个组合连续五拍相同（避免雷同）
    :param predict_matrix: 输出的概率矩阵
    :param min_pat_dx: 预测的结果“矩阵最后一行的索引值”应当在min_pat_dx至max_pat_dx之间
    :param max_pat_dx: 预测的结果“矩阵最后一行的索引值”应当在min_pat_dx至max_pat_dx之间
    :param melody_out_pat_list: 已经输出的组合
    :param input_data: 训练集中的输入数据项
    :return: 生成的最后一拍结果（“矩阵最后一行的索引值”）
    """
    predict_list = copy.deepcopy(predict_matrix[-1])  # 这个数组是选取下一个步长输出内容的基准，表示common_patten中选取各个pattern的概率
    # 1.将和某个输入向量最后五拍完全一致的pattern置为不可选取（概率置零）
    for input_ary_dx in range(len(input_data)):
        if input_data[input_ary_dx][-5: -1] == melody_out_pat_list[-4:] and input_data[input_ary_dx][-5: -1].count(0) <= 1:
            predict_list[input_data[input_ary_dx][-1]] = 0
    # 2.将小于min_pat_dx或大于max_pat_dx的pattern置为不可选取
    for pat_it in range(len(predict_list)):
        if pat_it < min_pat_dx or pat_it > max_pat_dx:
            predict_list[pat_it] = 0
    # 3.根据剩余音符的输出概率总和决定输出策略
    if sum(predict_list) <= 0.01:
        raise RuntimeError  # 剩余音符的输出概率总和不足0.01，直接回滚4拍
    elif sum(predict_list) <= 0.6:  # 剩余音符的输出概率总和在0.01-0.6之间，直接选概率最高的
        return np.argmax(predict_list)
    else:  # 其余情况 按照概率选择一个
        row_sum = sum(predict_list)
        random_value = random.random() * row_sum
        for element in range(len(predict_list)):
            random_value -= predict_list[element]
            if random_value <= 0:
                return element + min_pat_dx


def pat_predict_addcode(predict_matrix, base_code_add, min_pattern_number, max_pattern_number):
    """在存在“编码位移”的情况下（因为多重编码），根据概率矩阵生成下一个步长的内容"""
    min_pat_dx = base_code_add + min_pattern_number
    max_pat_dx = base_code_add + max_pattern_number
    cut_list = predict_matrix[-1][min_pat_dx: (max_pat_dx + 1)]  # 从predict_matrix中截取一部分作为待选的概率向量
    row_sum = sum(cut_list)
    random_value = random.random() * row_sum
    for element in range(max_pat_dx - min_pat_dx + 1):
        random_value -= cut_list[element]
        if random_value <= 0:
            return element + min_pattern_number  # 最后返回的是在common_pat_ary中的位置 而不是增加了base_code_add之后的结果


def keypress_encode(melody_output, keypress_pats):
    """
    根据主旋律的输出编码为按键组合
    :param melody_output: 主旋律的输出
    :param keypress_pats: 按键组合的列表
    :return: 编码为按键组合后的结果
    """
    keypress_pat_list = []
    for melody_step_it in range(0, len(melody_output), 16):
        keypress_list = [1 if t != 0 else 0 for t in melody_output[melody_step_it: melody_step_it + 16]]
        try:
            keypress_dx = keypress_pats.index(keypress_list)
            if keypress_dx == -1:
                keypress_dx = keypress_pat_list[-1]
            keypress_pat_list.append(keypress_dx)
        except ValueError:
            DiaryLog.warn('melody_output中出现了中keypress_pat_list中不存在的旋律组合, 是' + repr(keypress_list))
            keypress_pat_list.append(keypress_pat_list[-1])
        except IndexError:
            DiaryLog.warn('melody_output中出现了中keypress_pat_list中不存在的旋律组合, 是' + repr(keypress_list))
            keypress_pat_list.append(keypress_pat_list[-1])
    return keypress_pat_list


def root_chord_encode(chord_out, all_rc_pats, base_rootnote):
    """
    根据和弦的输出编码为根音-和弦组合
    :param chord_out: 和弦的输出
    :param all_rc_pats: 根音-和弦组合的列表
    :param base_rootnote: 基准根音音高
    :return: 根音数据和根音-和弦组合
    """
    root_data = []
    rc_pat_list = []
    for chord_it in range(len(chord_out)):
        if chord_it == 0:
            root_data.append(get_chord_root_pitch(chord_out[0], 0, base_rootnote))
        else:
            root_data.append(get_chord_root_pitch(chord_out[chord_it], root_data[chord_it - 1], base_rootnote))
    # 2.将和弦和根音组合进行编码
    for chord_it in range(len(chord_out)):
        try:
            rc_pat_list.append(all_rc_pats.index([root_data[chord_it], chord_out[chord_it]]))  # 由于前面获取状态转移矩阵和输出矩阵时将所有的和弦根音组合和PG组合都减了1 因此在这里要减一
        except ValueError:  # 根音-和弦对照表中没有这个根音-和弦组合
            DiaryLog.warn('chord_output中出现了根音-和弦对照表中找不到和根音-和弦组合, 是' + repr([root_data[chord_it], chord_out[chord_it]]))
            rc_pat_list.append(0)
    return root_data, rc_pat_list
