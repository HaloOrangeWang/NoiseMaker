from settings import *
from pipelines.functions import BaseLstmPipeline, pat_predict_addcode
from datainputs.chord import ChordTrainData, ChordTestData
from models.configs import ChordConfig
from validations.chord import ChordConfidenceCheck, chord_check
from interfaces.utils import DiaryLog
import numpy as np
import copy


def get_1chord_2steps(chord_list, melody_list, all_cc_pats):
    """
    当两拍的和弦不同时，从两个候选和弦中选择一个匹配此主旋律最佳的。选择标准为和弦的常见程度以及和主旋律的匹配程度
    :param chord_list: 候选和弦列表
    :param melody_list: 同期主旋律数据（绝对音高形式）
    :param all_cc_pats: 和弦-和弦组合列表
    :return: 返回整合成一个的和弦 及其在cc_pattern_list中的位置
    """
    fix_chord_list = [1, 14, 26, 31, 43, 56]  # 常见的和弦列表
    # 1.筛掉重复音不在cc_pat_count中的 如果两拍都不在则报错
    chord_ary = [t for t in chord_list if [t, t] in all_cc_pats]
    if len(chord_ary) == 0:
        raise RuntimeError
    # 2.剔掉不常见的和弦 如果都是不常见的则不提取
    fix_in_ary = np.array(chord_ary)[np.in1d(chord_ary, fix_chord_list)]
    if len(fix_in_ary) != 0:
        chord_choice_list = fix_in_ary
    else:
        chord_choice_list = chord_ary
    # 3.从候选的和弦组合中跳出一个最匹配的 作为这两拍的和弦
    max_match_score = 0
    max_match_dx = -1
    for chord_it in range(len(chord_choice_list)):
        contain_count = 0  # 有多少个步长的音符包含在和弦里
        diff_count = 0  # 有多少个步长的音符不包含在和弦里
        last_note_contain = -1  # 上一个主旋律音符是否在此和弦内 1表示在 0表示不在 -1表示不清楚
        chord_set = copy.deepcopy(CHORD_LIST[chord_choice_list[chord_it]])
        if 1 <= chord_choice_list[chord_it] <= 72 and chord_choice_list[chord_it] % 6 == 1:  # 大三和弦 chord_set增加大七度和小七度
            chord_set.add((chord_choice_list[chord_it] // 6 + 10) % 12)
            chord_set.add((chord_choice_list[chord_it] // 6 + 11) % 12)
        if 1 <= chord_choice_list[chord_it] <= 72 and chord_choice_list[chord_it] % 6 == 2:  # 小三和弦 chord_set增加小七度
            chord_set.add((chord_choice_list[chord_it] // 6 + 10) % 12)
        for note_it in range(len(melody_list)):
            if melody_list[note_it] != 0:  # 当前步长有主旋律
                if melody_list[note_it] % 12 in chord_set:  # 在和弦里
                    contain_count += 1
                    last_note_contain = 1
                else:
                    diff_count += 1
                    last_note_contain = 0
            else:
                if last_note_contain == 1:
                    contain_count += 1
                elif last_note_contain == 0:
                    diff_count += 1
        match_score = contain_count / (contain_count + diff_count)
        if match_score > max_match_score:
            max_match_score = match_score
            max_match_dx = chord_it
    return [chord_choice_list[max_match_dx], chord_choice_list[max_match_dx]], all_cc_pats.index([chord_choice_list[max_match_dx], chord_choice_list[max_match_dx]])


class ChordPipeline(BaseLstmPipeline):

    def __init__(self, is_train, *args):
        if is_train:
            # 训练时的情况
            # args = (melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data)
            self.train_data = ChordTrainData(*args)
        else:
            self.train_data = ChordTestData()
        self.confidence_cls = ChordConfidenceCheck(self.train_data.transfer_count, self.train_data.real_transfer_count)
        super().__init__()

    # noinspection PyAttributeOutsideInit
    def prepare(self):
        self.config = ChordConfig(self.train_data.cc_pat_num)
        self.test_config = ChordConfig(self.train_data.cc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'ChordModel'

    # noinspection PyAttributeOutsideInit
    def generate_init(self, session, melody_out_notes, melody_out_pats, common_corenote_pats, core_note_pat_list, melody_beat_num, end_check_beats, tone_restrict=DEF_TONE_MAJOR):
        self.melody_out_notes = melody_out_notes
        self.melody_out_pats = melody_out_pats
        self.common_corenote_pats = common_corenote_pats
        self.core_note_pat_list = core_note_pat_list
        self.melody_beat_num = melody_beat_num
        self.tone_restrict = tone_restrict
        self.end_check_beats = end_check_beats

        # 生成过程中的常量和类型
        self.melody1_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
        self.melody2_code_add_base = 4 + COMMON_MELODY_PAT_NUM + 2  # 主旋律第二拍数据编码增加的基数
        self.chord_code_add_base = 4 + (COMMON_MELODY_PAT_NUM + 2) * 2  # 和弦数据编码增加的基数

        # 生成过程中的一些变量
        self.rollback_times = 0  # 被打回重新生成的次数
        bar_num = len(melody_out_notes) // 32  # 主旋律一共有多少个小节
        self.confidence_back_times = [0 for t in range(bar_num + 1)]
        self.chord_choose_bak = [[1, 1, 1, 1] for t in range(bar_num + 1)]  # 备选和弦。为防止死循环，如果连续十次验证失败，则使用备选和弦
        self.cc_pat_choose_bak = [[1, 1, 1, 1] for t in range(bar_num + 1)]
        self.loss_bak = [np.inf for t in range(bar_num + 1)]  # 备选方案对应的损失函数
        self.beat_dx = 0  # 生成完第几个pattern了

        # 生成结果
        self.chord_out = []
        self.cc_pat_out = []

    def rollback(self, step_num):
        self.beat_dx -= step_num * 2  # 重新生成这一小节的音乐
        self.chord_out = self.chord_out[:len(self.chord_out) - step_num * 2]
        self.cc_pat_out = self.cc_pat_out[:len(self.cc_pat_out) - step_num]

    def generate_by_step(self, session):
        # 1.如果这两拍没有主旋律 和弦直接沿用之前的
        if self.melody_out_pats[self.beat_dx: self.beat_dx + 2] == [0, 0]:  # 最近这2拍没有主旋律 则和弦沿用之前的
            if self.beat_dx == 0:
                raise ValueError  # 前两拍的主旋律不能为空
            else:
                self.chord_out.extend([self.chord_out[-1], self.chord_out[-1]])
                self.cc_pat_out.append(self.cc_pat_out[-1])
            self.beat_dx += 2
            return

        # 2.使用LstmModel生成一个步长的音符
        # 2.1.逐时间步长生成test model输入数据
        chord_prediction_input = list()
        for backward_beat_it in range(self.beat_dx - 8, self.beat_dx + 2, 2):
            cur_step = backward_beat_it // 2  # 第几个bass的步长
            if backward_beat_it < 0:
                chord_prediction_input.append([cur_step % 2, self.melody1_code_add_base, self.melody2_code_add_base, self.chord_code_add_base])  # 这一拍的时间编码 这一拍的主旋律 下一拍的主旋律
            elif backward_beat_it < 2:
                chord_prediction_input.append([cur_step % 4, self.melody_out_pats[backward_beat_it] + self.melody1_code_add_base, self.melody_out_pats[backward_beat_it + 1] + self.melody2_code_add_base, self.chord_code_add_base])
            else:
                chord_prediction_input.append([cur_step % 4, self.melody_out_pats[backward_beat_it] + self.melody1_code_add_base, self.melody_out_pats[backward_beat_it + 1] + self.melody2_code_add_base, self.cc_pat_out[cur_step - 1] + self.chord_code_add_base])
        # 2.2.生成输出数据
        chord_predict = self.predict(session, [chord_prediction_input])  # LSTM预测 得到二维数组predict
        out_pat_dx = pat_predict_addcode(chord_predict, self.chord_code_add_base, 1, self.train_data.cc_pat_num)  # 将二维数组predict通过概率随机生成一维数组chord_out_vector，这个数组就是这两小节的和弦。每两小节生成一次和弦
        self.cc_pat_out.append(out_pat_dx)  # 和弦-和弦编码的输出
        if self.train_data.all_cc_pats[out_pat_dx][0] != self.train_data.all_cc_pats[out_pat_dx][1]:
            DiaryLog.warn('在第%d拍, 预测方法选择了两个不同的和弦，编码分别是%d和%d' % (self.beat_dx, self.train_data.all_cc_pats[out_pat_dx][0], self.train_data.all_cc_pats[out_pat_dx][1]))
            try:
                new_chord, out_pat_dx = get_1chord_2steps(self.train_data.all_cc_pats[out_pat_dx], self.melody_out_notes[self.beat_dx * 8: (self.beat_dx + 2) * 8], self.train_data.all_cc_pats)
                DiaryLog.warn('在第%d拍, 和弦已经被替换为编码%d, 它在数组中的位置为%d' % (self.beat_dx, new_chord[0], out_pat_dx))
            except RuntimeError:
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，两拍和弦不相同且没法找到替代' % (self.beat_dx, self.rollback_times))
                self.beat_dx -= 2
                self.cc_pat_out = self.cc_pat_out[:(-1)]
                self.rollback_times += 1
                return
        self.cc_pat_out[-1] = out_pat_dx
        self.chord_out.extend(self.train_data.all_cc_pats[out_pat_dx])  # 添加到最终的和弦输出列表中
        self.beat_dx += 2  # 步长是2拍

    def check_1step(self, session):
        # 1.检查两小节内和弦的离调情况。在每两小节的末尾进行检验
        if self.beat_dx >= 8 and self.beat_dx % 4 == 0 and not chord_check(self.chord_out[-8:], self.melody_out_notes[(self.beat_dx - 8) * 8: self.beat_dx * 8]):  # 离调和弦的比例过高
            DiaryLog.warn('在第%d拍, 和弦第%02d次打回，离调和弦比例过高, 最后八拍和弦为%s' % (self.beat_dx, self.rollback_times, repr(self.chord_out[-8:])))
            self.rollback(4)
            self.rollback_times += 1
            return

        # 2.和弦是否连续三小节不变化。在每小节的末尾检验
        if self.beat_dx >= 12 and len(set(self.chord_out[-12:])) == 1:
            DiaryLog.warn('在第%d拍, 和弦第%02d次打回，连续3小节和弦未变化, 始终为%d' % (self.beat_dx, self.rollback_times, self.chord_out[-1]))
            self.rollback(6)
            self.rollback_times += 1
            return

        # 3.检查两小节内的主旋律对应和弦在训练集中出现的频率是否过低。每2小节检验一次
        if self.beat_dx >= 8 and self.beat_dx % 8 == 0:
            self.confidence_cls.calc_confidence_level(session, self.core_note_pat_list)  # 计算这两拍的主旋律骨干音对应的各类和弦的转化频率
            chord_per_2beats = [self.chord_out[-8:][t] for t in range(0, len(self.chord_out[-8:]), 2)]
            cc_pat_per_2beats = self.cc_pat_out[-4:]  # 生成的和弦从每1拍一个转化为每两拍一个
            check_res, loss_value = self.confidence_cls.check_chord_ary(session, self.melody_out_notes[(self.beat_dx - 8) * 8: self.beat_dx * 8], self.core_note_pat_list[self.beat_dx // 2 - 4: self.beat_dx // 2], chord_per_2beats)  # 检查此时的主旋律对应同期的和弦的转化频率是否过低
            if not check_res:
                DiaryLog.warn('在第%d拍, 和弦第%d次未通过同时期主旋律转化频率验证。和弦的损失函数值为%.4f，而临界值为%.4f' % (self.beat_dx, self.confidence_back_times[self.beat_dx // 4], loss_value, self.confidence_cls.confidence_level))
                self.rollback(4)
                self.confidence_back_times[(self.beat_dx + 8) // 4] += 1
                if loss_value < self.loss_bak[(self.beat_dx + 8) // 4]:  # 前面减了8之后 这里beat_dx都应该变成beat_dx+8
                    self.chord_choose_bak[(self.beat_dx + 8) // 4] = chord_per_2beats
                    self.cc_pat_choose_bak[(self.beat_dx + 8) // 4] = cc_pat_per_2beats
                    self.loss_bak[(self.beat_dx + 8) // 4] = loss_value
                if self.confidence_back_times[(self.beat_dx + 8) // 4] >= 10:  # 为避免连续过多次的回滚，当连续回滚次数达到10次时，使用前10次中相对最佳的和弦组合
                    DiaryLog.warn('在第%d拍，由于连续10次未通过同时期主旋律转化频率的验证，和弦使用备选方案, 损失为%.4f' % (self.beat_dx, self.loss_bak[(self.beat_dx + 8) // 4]))
                    chord_part_list = list(np.repeat(self.chord_choose_bak[(self.beat_dx + 8) // 4], 2))
                    for chord_it in range(len(chord_part_list)):
                        chord_part_list[chord_it] = int(chord_part_list[chord_it])
                    self.chord_out.extend(chord_part_list)
                    self.cc_pat_out.extend(self.cc_pat_choose_bak[(self.beat_dx + 8) // 4])
                    self.beat_dx += 8
                else:
                    return

        # 4.和弦结束的校验。最后一拍的和弦必须为1级大和弦（大调）或6级小和弦。在最后一个步长执行检验
        if self.beat_dx in self.end_check_beats:
            if (self.tone_restrict == DEF_TONE_MAJOR and self.chord_out[-1] != 1) or (self.tone_restrict == DEF_TONE_MINOR and self.chord_out[-1] != 56):
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，收束不是1级大和弦或6级小和弦, 最后八拍和弦为%s' % (self.beat_dx, self.rollback_times, repr(self.chord_out[-8:])))
                self.rollback(4)
                self.rollback_times += 1

    def generate(self, session, melody_out_notes, melody_out_pats, common_corenote_pats, core_note_pat_list, melody_beat_num, end_check_beats, tone_restrict=DEF_TONE_MAJOR):
        self.generate_init(session, melody_out_notes, melody_out_pats, common_corenote_pats, core_note_pat_list, melody_beat_num, end_check_beats, tone_restrict)
        while True:
            self.generate_by_step(session)
            self.check_1step(session)

            if self.rollback_times >= MAX_GEN_CHORD_FAIL_TIME:
                DiaryLog.warn('和弦被打回次数超过%d次,重新生成。\n\n\n' % self.rollback_times)
                raise RuntimeError
            if self.beat_dx == self.melody_beat_num:
                assert len(self.chord_out) == self.melody_beat_num
                break

        DiaryLog.warn('和弦的输出: ' + repr(self.chord_out) + '\n\n\n')
        return self.chord_out
