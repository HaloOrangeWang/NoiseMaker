import tensorflow as tf
import numpy as np
import random


class KMeansModel:
    """
    计算K均值的model.
    """
    def __init__(self, input_values, cluster_number, iterate_times=10, training=True):
        # 1.接受输入的向量及分类个数
        self.cluster_number = cluster_number
        if training is True:
            self.input_values = input_values
            self.input_size = len(input_values)
            self.iterate_times = iterate_times  # 每次计算K均值的迭代次数
            self.train_model()  # 先初始化model中的Variable
        else:
            self.test_model()

    def train_model(self):
        """
        定义这个model中的各种变量。不写在init中的原因是每次运行时都需要重新定义一遍
        """
        # 1.从原来的数组中随机取一些(cluster_number个)数作为中心点
        vector_indices = list(range(self.input_size))
        random.shuffle(vector_indices)
        self.center_points = tf.get_variable('center', initializer=np.float32([0 for t in range(self.cluster_number)]), dtype=tf.float32)
        self.center_placeholder = tf.placeholder(tf.float32, [self.cluster_number])
        self.center_assigns = tf.assign(self.center_points, self.center_placeholder)  # 选取的中心点
        # 2.每个输入值的隶属关系，初始化时每个输入值的分类均为0。它们将在后续的操作中被分配到合适的类
        self.attachments = tf.get_variable('attachment', initializer=[0 for t in range(self.input_size)], dtype=tf.int32)
        self.attachment_placeholder = tf.placeholder(tf.int32, [self.input_size])
        self.attachment_assigns = tf.assign(self.attachments, self.attachment_placeholder)
        # 3.计算输入值到各个中心点之间的距离 从而确定该输入值的类别
        self.input_value_placeholder = tf.placeholder(tf.float32, [self.cluster_number])  # 存放输入点（复制cluster_number次)
        self.cluster_placeholder = tf.placeholder(tf.float32, [self.cluster_number])
        distances = tf.abs(tf.subtract(self.input_value_placeholder, self.cluster_placeholder))  # 计算输入值到中心点之间的距离
        self.cluster_assignment = tf.argmin(distances, 0)  # 找出距离最近的那个中心点 隶属于它
        # 4.计算新的中心点位置
        self.mean_input = tf.placeholder(tf.float32, [None])  # 这里存放所有隶属于一个中心点的值距该中心点的距离值
        self.mean_op = tf.reduce_mean(self.mean_input, 0)  # 计算它们的平均值

    def test_model(self):
        self.center_points = tf.get_variable('center', initializer=np.float32([0 for t in range(self.cluster_number)]), dtype=tf.float32)

        self.input_value_placeholder = tf.placeholder(tf.float32, [self.cluster_number])  # 存放输入点（复制cluster_number次)
        self.cluster_placeholder = tf.placeholder(tf.float32, [self.cluster_number])
        distances = tf.abs(tf.subtract(self.input_value_placeholder, self.cluster_placeholder))  # 计算输入值到中心点之间的距离
        self.cluster_assignment = tf.argmin(distances, 0)  # 找出距离最近的那个中心点 隶属于它

    def calculate(self, session, operate_times=10):
        """
        用K均值算法对输入数据进行分类
        :param session:
        :param operate_times: 迭代步数
        :return: 中心点列表和输入数据的隶属情况
        """
        for cal_iterator in range(operate_times):
            # 1.首先遍历所有的向量,计算每个值距每个中心点的距离并重新分类
            attachment_vector = []  # 存储分类情况
            # print(session.run(self.center_points))
            for input_iterator in range(self.input_size):
                attachment_vector.append(session.run(self.cluster_assignment, feed_dict={self.input_value_placeholder: [self.input_values[input_iterator] for t in range(self.cluster_number)], self.cluster_placeholder: session.run(self.center_points)}))
            session.run(self.attachment_assigns, feed_dict={self.attachment_placeholder: attachment_vector})
            # 2.最大化的步骤。计算输入值到新的中心点之间的距离使得距离的平方和最小
            location_vector = []
            for cluster_iterator in range(self.cluster_number):
                assigned_values = [self.input_values[t] for t in range(self.input_size) if attachment_vector[t] == cluster_iterator]
                location_vector.append(session.run(self.mean_op, feed_dict={self.mean_input: np.array(assigned_values)}))
            session.run(self.center_assigns, feed_dict={self.center_placeholder: location_vector})  # 为每个中心点分配更合适的值
        return session.run(self.center_points), session.run(self.attachments)  # 返回中心点的位置和输入值隶属情况

    def run(self, session):
        # 计算中心点
        while True:
            centerpoints_legal = True  # 这里判断计算出的中心点有没有NaN
            # 1.初始化
            vector_indices = list(range(self.input_size))
            random.shuffle(vector_indices)
            session.run(self.center_assigns, feed_dict={self.center_placeholder: [self.input_values[vector_indices[t]] for t in range(self.cluster_number)]})
            session.run(self.attachment_assigns, feed_dict={self.attachment_placeholder: [0 for t in range(self.input_size)]})
            # 2.计算中心点
            centerpoint_array, result = self.calculate(session=session, operate_times=self.iterate_times)
            # 3.判断计算出的中心点是否有NaN，如果有 则重新计算
            for centerpoint in centerpoint_array:
                if np.isnan(centerpoint):
                    centerpoints_legal = False
                    break
            if not centerpoints_legal:
                # print(centerpoint_array)
                print('Center Point Failed!')
            # 4.如果数组符合要求，则对其进行排序后返回
            if centerpoints_legal:
                centerpoint_array.sort()  # 由小到大排序
                # attachment_vector = []  # 存储分类情况
                # for input_iterator in range(self.input_size):
                #     attachment_vector.append(sess.run(self.cluster_assignment, feed_dict={self.input_value_placeholder: [self.input_values[input_iterator] for t in range(self.cluster_number)], self.cluster_placeholder: centerpoint_array}))
                return centerpoint_array

    def restore_centers(self, session):
        centerpoint_array = session.run(self.center_points)
        centerpoint_array.sort()  # 由小到大排序
        return centerpoint_array

    def run_attachment(self, session, centerpoint_array, input_vector):
        attachment_vector = []  # 存储分类情况
        for input_iterator in range(len(input_vector)):
            attachment_vector.append(session.run(self.cluster_assignment, feed_dict={self.input_value_placeholder: [input_vector[input_iterator] for t in range(len(centerpoint_array))], self.cluster_placeholder: centerpoint_array}))
        return attachment_vector
