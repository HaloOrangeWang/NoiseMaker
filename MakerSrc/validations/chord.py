from settings import *
import tensorflow as tf
import numpy as np


def chord_check(chord_list, melody_list):
    """
    检查两小节的和弦是否存在和弦与同期旋律完全不同的情况 如果存在次数超过1次 则认为是和弦不合格
    :param chord_list: 两小节和弦列表
    :param melody_list: 同时期的主旋律列表（非pattern）
    :return: 两小节的和弦的离调比例是否符合要求
    """
    # 还要考虑这段时间没有主旋律/那个音符持续时间过短等情况
    abnm_chord_num = 0
    flag_last_step_abnm = False  # 上一个时间区段是否是异常 如果是异常 则这个值为True 否则为False
    for chord_pat_step_it in range(0, len(chord_list), 2):  # 时间步长为2拍
        melody_set = set()  # 这段时间内的主旋律列表
        for note_it in range(chord_pat_step_it * 8, (chord_pat_step_it + 2) * 8):
            if melody_list[note_it] != 0:
                try:
                    if note_it % 8 == 0 or (melody_list[note_it + 1] == 0 and melody_list[note_it + 2] == 0):  # 时间过短的音符不要
                        melody_set.add(melody_list[note_it] % 12)
                except IndexError:
                    pass
        if len(melody_set) != 0 and len(CHORD_LIST[chord_list[chord_pat_step_it]] | melody_set) == len(CHORD_LIST[chord_list[chord_pat_step_it]]) + len(melody_set):
            abnm_chord_num += 1
            flag_last_step_abnm = True
        elif len(melody_set) == 0 and flag_last_step_abnm is True:  # 如果这段时间内没有主旋律 则看上个时间区段内是否和弦不符合要求 如果上个时间区段内和弦不符合要求 则这个时间区段的和弦也同样不符合要求
            abnm_chord_num += 1
        else:
            flag_last_step_abnm = False
    if abnm_chord_num > 1:
        return False
    return True


class ChordConfidenceCheck:
    """由于该类计算ConfidenceLevel方法的特殊性，它不继承自BaseConfidenceLevelCheck"""

    def __init__(self, transfer_count, real_transfer_count):

        self.real_transfer_count = real_transfer_count
        self.transfer_mat = np.zeros([COMMON_CORE_NOTE_PAT_NUM * 2 + 2, len(CHORD_LIST) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦。这个变量是上个变量进行了反softmax变换得到的
        self.transfer_prob_real = np.zeros([COMMON_CORE_NOTE_PAT_NUM * 2 + 2, len(CHORD_LIST) + 1], dtype=np.float32)  # 真实的主旋律/调式与同时期和弦 概率取对数后的转移矩阵 这个转移矩阵的数字精度必须是float32的

        # 1.将频率转化为概率 将频率归一化之后进行反softmax变换
        for core_note_pat_dx in range(0, COMMON_CORE_NOTE_PAT_NUM * 2 + 2):
            self.transfer_mat[core_note_pat_dx, :] = transfer_count[core_note_pat_dx, :] / sum(transfer_count[core_note_pat_dx, :])
            self.transfer_mat[core_note_pat_dx, :] = np.log(self.transfer_mat[core_note_pat_dx, :])
            if sum(real_transfer_count[core_note_pat_dx, :]) != 0:  # 记录真实情况下的转移概率。如果计数全部为零的话，就把那一列直接记零
                self.transfer_prob_real[core_note_pat_dx, :] = real_transfer_count[core_note_pat_dx, :] / sum(real_transfer_count[core_note_pat_dx, :])

        # 2.定义交叉熵的计算方法
        self.core_note_pat_dx = tf.placeholder(tf.int32, [])
        self.cross_entropy_lost = self.loss_func(self.core_note_pat_dx)  # 各个和弦对应的损失函数

        # 3.定义1个和弦的交叉熵的计算方法
        self.core_note_pat_dx_1chord = tf.placeholder(tf.int32, [])
        self.chord_dx = tf.placeholder(tf.int32, [])
        self.cross_entropy_lost_1chord = self.loss_func_1chord(self.core_note_pat_dx_1chord, self.chord_dx)

    def loss_func(self, core_note_pat_dx):
        """定义一个主旋律下,各个和弦对应的损失函数的计算方法"""
        transfer_vec = tf.gather(self.transfer_mat, core_note_pat_dx)  # 这个主旋律下出现各个和弦的概率
        transfer_vec = tf.tile(tf.expand_dims(transfer_vec, 0), [len(CHORD_LIST) + 1, 1])  # 把它复刻若干遍 便于进行交叉熵的计算
        chord_one_hot = tf.constant(np.eye(len(CHORD_LIST) + 1))  # 所有和弦的独热编码
        cross_entropy_lost = tf.nn.softmax_cross_entropy_with_logits(labels=chord_one_hot, logits=transfer_vec)  # 这个主旋律下每个和弦对应的交叉熵损失函数
        # softmax_cross_entropy_with_logits首先将logits进行softmax变换，然后用labels*ln(1/logits)来计算
        # 如当labels为[1,0,0,0] logits为ln([0.5,0.25,0.125,0.125])时，交叉熵为1*ln(2)=0.693
        return cross_entropy_lost

    def loss_func_1chord(self, core_note_pat_dx, chord_dx):
        """定义一个主旋律下，一个和弦对应的损失函数的计算方法"""
        transfer_vec = tf.gather(self.transfer_mat, core_note_pat_dx)  # 这个主旋律下出现各个和弦的概率
        chord_one_hot = tf.one_hot(chord_dx, depth=len(CHORD_LIST) + 1)  # 这个和弦的独热编码
        cross_entropy_lost = tf.nn.softmax_cross_entropy_with_logits(labels=chord_one_hot, logits=transfer_vec)  # 这个主旋律下每个和弦对应的交叉熵损失函数
        return cross_entropy_lost

    def calc_confidence_level(self, session, core_note_pat_ary):
        """计算一个主旋律进行下的损失函数0.9置信区间"""
        section_prob = dict()  # 区间概率。存储方式为值:概率 如{1.5:0.03,2.3:0.05,...}
        for pat_step_dx in range(len(core_note_pat_ary)):
            if core_note_pat_ary[pat_step_dx] not in [0, COMMON_CORE_NOTE_PAT_NUM + 1]:  # 只在这个根音不为空 不为罕见根音组合时才计算
                # 1.1.计算当前步骤的每个和弦及其损失函数的关系
                lost_each_chord = session.run(self.cross_entropy_lost, feed_dict={self.core_note_pat_dx: core_note_pat_ary[pat_step_dx]})  # 计算这个主旋律下 每个和弦对应的交叉熵损失函数
                for chord_it in range(len(lost_each_chord)):
                    lost_each_chord[chord_it] = round(lost_each_chord[chord_it], 2)
                step_trans_vector = self.transfer_prob_real[core_note_pat_ary[pat_step_dx], :]  # 这个主旋律对应的各个和弦的比利
                step_section_prob = dict()
                for chord_it in range(len(lost_each_chord)):  # 写成{交叉熵损失: 对应比例}的结构
                    if step_trans_vector[chord_it] != 0:
                        if lost_each_chord[chord_it] not in step_section_prob:
                            step_section_prob[lost_each_chord[chord_it]] = step_trans_vector[chord_it]
                        else:
                            step_section_prob[lost_each_chord[chord_it]] += step_trans_vector[chord_it]
                # 1.2.计算几步的损失函数总和
                if section_prob == dict():  # 这段校验的第一步
                    section_prob = step_section_prob
                else:  # 不是这段校验的第一步 公式是{过去的损失+当前步的损失: 过去的概率×当前步的概率}
                    section_prob_bak = dict()
                    for loss_old in section_prob:
                        for loss_step in step_section_prob:
                            prob = round(loss_old + loss_step, 2)
                            if prob in section_prob_bak:
                                section_prob_bak[prob] += section_prob[loss_old] * step_section_prob[loss_step]
                            else:
                                section_prob_bak[prob] = section_prob[loss_old] * step_section_prob[loss_step]
                    section_prob = section_prob_bak

        # 2.获取交叉熵误差的90%阈值
        accumulate_prob = 0
        sorted_section_prob = sorted(section_prob.items(), key=lambda asd: asd[0], reverse=False)
        for prob_tuple in sorted_section_prob:
            accumulate_prob += prob_tuple[1]
            if accumulate_prob >= 0.9:
                loss09 = prob_tuple[0]
                self.confidence_level = loss09
                return
        self.confidence_level = np.inf

    @staticmethod
    def chord_check_1step(chord_dx, melody_list, last_step_level):
        """
        检查1个步长的和弦和同期主旋律的匹配程度
        :param last_step_level: 上一拍的和弦和谐程度判定等级
        :param melody_list: 同时期的主旋律列表（非pattern）
        :param chord_dx: 这两拍的和弦
        :return: 分为三个等级 2为和弦符合预期 1为和弦与同期主旋律有重叠音但不是特别和谐 0为和弦与同期主旋律完全不重叠
        """
        melody_set = set()  # 这段时间内的主旋律列表
        for note_it in range(len(melody_list)):
            if melody_list[note_it] != 0:
                try:
                    if note_it % 8 == 0 or (melody_list[note_it + 1] == 0 and melody_list[note_it + 2] == 0):  # 时间过短的音符不要
                        melody_set.add(melody_list[note_it] % 12)
                except IndexError:
                    pass
        if len(melody_set) != 0 and len(CHORD_LIST[chord_dx] | melody_set) == len(CHORD_LIST[chord_dx]) + len(melody_set):
            return 0
        elif len(melody_set) == 0:  # 如果这段时间内没有主旋律 则看上个时间区段内是否和弦不符合要求 如果上个时间区段内和弦不符合要求 则这个时间区段的和弦也同样不符合要求
            return last_step_level
        elif len(melody_set) == 1:  # 主旋律的长度为1 则只要有重叠音就返回2
            return 2
        else:
            # 主旋律与同期和弦有两个重叠音 或一个重叠音一个七音 则返回2 其他情况返回1
            if len(CHORD_LIST[chord_dx]) + len(melody_set) - len(CHORD_LIST[chord_dx] | melody_set) >= 2:
                return 2
            elif 1 <= chord_dx <= 72 and chord_dx % 6 == 1 and ((chord_dx // 6 + 10) % 12 in melody_set or (chord_dx // 6 + 11) % 12 in melody_set):  # 大三和弦 且其大七度或小七度在主旋律中
                return 2
            elif 1 <= chord_dx <= 72 and chord_dx % 6 == 2 and (chord_dx // 6 + 10) % 12 in melody_set:  # 大三和弦 且其小七度在主旋律中
                return 2
            else:
                return 1

    def check_chord_ary(self, session, melody_list, core_note_pat_list, chord_list):
        """
        计算这个和弦对应的交叉熵损失函数 如果超过了 则认为是不正确的
        （如果出现一个主旋律根音组合对应的和弦频数过少的情况 则以启发式方法替代）
        :param session:
        :param melody_list: 一段主旋律进行
        :param core_note_pat_list: 这段时期内的根音进行
        :param chord_list: 这段时期内的和弦进行
        """
        lost_sum = 0  # 四拍的损失函数的总和
        last_step_level = 2  # 上一拍的和弦和谐程度判定等级
        # 2.计算四个主旋律对应的和弦损失函数之和
        for pat_it in range(len(core_note_pat_list)):
            flag_use_loss = True  # 计算交叉熵损失函数还是用乐理来替代
            # 2.1.这两拍的主旋律是否可以使用交叉熵损失的方法来计算 判定条件为：主旋律不为空/不为罕见根音组合，该根音组合在训练集中至少出现100次，至少有3种输出和弦出现了十次
            if core_note_pat_list[pat_it] in [0, COMMON_CORE_NOTE_PAT_NUM + 1]:  # 只在这个根音不为空 不为罕见根音组合时才计算
                flag_use_loss = False
            if flag_use_loss is True:
                real_count_ary = self.real_transfer_count[core_note_pat_list[pat_it], :]
                if real_count_ary.sum() < 100:  # 该根音组合在训练集中应该至少出现100次
                    flag_use_loss = False
                if len(real_count_ary[real_count_ary >= 10]) < 3:  # 该根音与至少3种输出和弦在训练集中出现了十次
                    flag_use_loss = False
            # 2.2.如果符合计算交叉熵的标准则用交叉熵方法计算 否则用启发式方法估算
            if flag_use_loss is True:
                lost_1step = session.run(self.cross_entropy_lost_1chord, feed_dict={self.core_note_pat_dx_1chord: core_note_pat_list[pat_it], self.chord_dx: chord_list[pat_it]})  # 每一拍的交叉熵损失函数
                lost_sum += lost_1step
                last_step_level = self.chord_check_1step(chord_list[pat_it], melody_list[pat_it * 16: (pat_it + 1) * 16], last_step_level)  # 更新这一拍的和弦是否离调的情况
            else:
                last_step_level = self.chord_check_1step(chord_list[pat_it], melody_list[pat_it * 16: (pat_it + 1) * 16], last_step_level)
                if last_step_level == 0:
                    lost_sum += self.confidence_level * 0.7  # 如果完全离调 我们认为它等价与0.9置信区间的70%的交叉熵损失
                elif last_step_level == 1:
                    lost_sum += self.confidence_level * 0.35  # 如果不完全和谐 我们认为它等价与0.9置信区间的35%的交叉熵损失
        # 3.检查这个损失函数值是否在0.9置信区间内 如果在则返回true 否则返回false
        if lost_sum <= self.confidence_level:
            return True, lost_sum
        else:
            return False, lost_sum
