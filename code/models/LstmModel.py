import inspect
import tensorflow as tf


class LstmModel_1:
    def __init__(self, input_data, output_data, config, learning_rate, is_training=False):
        # 1.定义LSTM的cell
        if 'reuse' in inspect.getargspec(tf.nn.rnn_cell.BasicLSTMCell.__init__).args:
            cell = tf.contrib.rnn.BasicLSTMCell(config.rnn_hidden_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(config.rnn_hidden_size, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * config.num_layers, state_is_tuple=True)

        # 2.初始化state和inputs
        initial_state = cell.zero_state(config.batch_size, tf.float32)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable('embedding', initializer=tf.random_uniform([config.note_dict_size + 1, config.rnn_hidden_size], -1.0, 1.0))
            inputs = tf.nn.embedding_lookup(embedding, input_data)

        # 3.获取LSTM计算的输出
        with tf.variable_scope("RNN"):
            outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        output = tf.reshape(outputs, [-1, config.rnn_hidden_size])

        # 4.根据LSTM的输出，经过WX+B计算得到真正的输出logits
        weights = tf.get_variable("weights", [config.rnn_hidden_size, config.note_dict_size + 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable("bias", [config.note_dict_size + 1], dtype=tf.float32, initializer=tf.zeros_initializer())
        logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)

        # 5.分为训练状态和测试状态
        if is_training:
            # 5.1.训练状态，将logits与期望输出值作比较 得到误差函数 并设计优化器使误差函数最小化
            labels =tf.one_hot(tf.reshape(output_data, [-1]), depth=config.note_dict_size + 1)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            total_loss = tf.reduce_mean(loss)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

            self._initial_state = initial_state
            self._output = output
            self._total_loss = total_loss
            self._train_op = train_op
            self._last_state = last_state
        else:
            # 5.2.测试状态，输出logits的处理结果
            prediction = tf.nn.softmax(logits)
            self._initial_state = initial_state
            self._last_state = last_state
            self._prediction = prediction

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def output(self):
        return self._output

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def last_state(self):
        return self._last_state

    @property
    def prediction(self):
        return self._prediction
