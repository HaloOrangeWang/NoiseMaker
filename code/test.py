from settings import *


class TestCase:

    def G1(self):
        from pipelines.melody_pipeline import MelodyPipeline
        import tensorflow as tf

        melody = MelodyPipeline()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)  # 初始化
            # 2.训练
            melody.run_epoch(sess)
            # 3.生成主旋律
            melody.generate(sess, None)

    def run(self):
        from pipelines.melody_pipeline import MelodyPipeline
        import tensorflow as tf

        melody = MelodyPipeline()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)  # 初始化
            # 2.训练
            melody.run_epoch(sess)
            # 3.生成主旋律
            melody.generate(sess, None)
