import tensorflow as tf
from .LstmModel import LstmModel_1
from .configs import MelodyConfig
import random
import numpy as np
from datainputs.melody import MelodyTrainData


class BaseLstmPipeline(object):

    def __init__(self):
        self.prepare()
        self.input_data = tf.placeholder(tf.int32, [None, self.config.input_size])  # 主旋律模型的输入数据
        self.output_data = tf.placeholder(tf.int32, [None, self.config.output_size])  # 主旋律模型的预期输出数据
        self.learning_rate = tf.placeholder(tf.float32, [])  # 从0.90.03开始，学习速率将变为一个变量
        with tf.variable_scope(self.variable_scope_name, reuse=None):
            self.train_model = LstmModel_1(input_data=self.input_data, output_data=self.output_data, config=self.config, learning_rate=self.learning_rate, is_training=True)  # 定义训练模型
        with tf.variable_scope(self.variable_scope_name, reuse=True):
            self.test_model = LstmModel_1(input_data=self.input_data, output_data=None, config=self.test_config, learning_rate=self.learning_rate, is_training=False)  # 定义测试的模型

    def prepare(self):
        self.config = MelodyConfig()
        self.test_config = MelodyConfig()
        self.test_config.batch_size = 1
        self.train_data = MelodyTrainData(None, None, None)
        self.variable_scope_name = 'Model'

    def run_epoch(self, session):
        for epoch in range(self.config.max_max_epoch):
            num_batch = len(self.train_data.input_data) // self.config.batch_size
            lr_decay = self.config.lr_decay ** max(epoch + 1 - self.config.max_epoch, 0.0)  # 求出学习速率的衰减值
            disrupt_input = [[self.train_data.input_data[t], self.train_data.output_data[t]] for t in range(len(self.train_data.input_data))]  # 下面两行是打乱顺序输入到model中
            random.shuffle(disrupt_input)
            print(len(self.train_data.input_data), num_batch)
            for batch in range(num_batch):
                batches_input = np.array(disrupt_input)[self.config.batch_size * batch: self.config.batch_size * (batch + 1), 0]
                batches_output = np.array(disrupt_input)[self.config.batch_size * batch: self.config.batch_size * (batch + 1), 1]
                loss, _, __ = session.run([
                    self.train_model.total_loss,
                    self.train_model.last_state,
                    self.train_model.train_op,
                ], feed_dict={self.input_data: batches_input, self.output_data: batches_output, self.learning_rate: self.config.learning_rate * lr_decay})
                if batch % 10 == 0:
                    print('Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))

    def predict(self, session, predict_input):
        [predict_matrix, last_state] = session.run([self.test_model.prediction, self.test_model.last_state], feed_dict={self.input_data: predict_input})  # LSTM预测 得到二维数组predict
        return predict_matrix
