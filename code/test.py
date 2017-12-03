import copy

from settings import *


class TestCase:

    def G1(self):
        import tensorflow as tf
        from models.KMeansModel import KMeansModel
        logpath = './log/test'
        with tf.Graph().as_default():
            with tf.variable_scope('ABC'):
                kmeans_model = KMeansModel([1, 1, 2, 3, 3, 7, 7, 8, 9, 9, 35, 35, 36, 37, 37, 20, 20, 21, 22, 22], 4, 10)
            init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(logdir=logpath)
            with sv.managed_session() as sess:
                sess.run(init_op)
                cluster_center_points = kmeans_model.run(sess)
                print(cluster_center_points)
                sv.saver.save(sess, logpath, global_step=sv.global_step)

    def G2(self):
        import tensorflow as tf
        from models.KMeansModel import KMeansModel
        logpath = './log/test'
        with tf.Graph().as_default():
            with tf.variable_scope('ABC'):
                kmeans_model = KMeansModel(None, 4, training=False)
            init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(logdir=logpath)
            with sv.managed_session() as sess:
                sess.run(init_op)
                sv.saver.restore(sess, logpath)
                c = kmeans_model.restore_centers(sess)
                print(c)

    def run(self):
        self.G2()
        # self.G2()


if __name__ == '__main__':
    test_case = TestCase()
    test_case.run()
