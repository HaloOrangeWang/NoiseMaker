from settings import *
from pipelines.functions import BaseLstmPipeline, root_chord_encode, pat_predict_addcode
from datainputs.strings import StringTrainData, StringTestData
from models.configs import StringConfig
from interfaces.note_format import get_abs_notelist_chord
from interfaces.chord_parse import chord_row_in_list
from interfaces.utils import DiaryLog
from validations.strings import string_chord_check, string_end_check
import numpy as np


class StringPipeline(BaseLstmPipeline):

    def __init__(self, is_train, *args):
        if is_train:
            # 训练时的情况
            # args = (melody_pat_data, continuous_bar_data, corenote_pat_data, common_corenote_pats, chord_cls)
            self.train_data = StringTrainData(*args)
        else:
            self.train_data = StringTestData()
        super().__init__()

    # noinspection PyAttributeOutsideInit
    def prepare(self):
        self.config = StringConfig(self.train_data.rc_pat_num)
        self.test_config = StringConfig(self.train_data.rc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'StringModel'

    # noinspection PyAttributeOutsideInit
    def generate_init(self, session, melody_out_notes, chord_out, corenote_out, melody_beat_num, end_check_beats):
        self.melody_out_notes = melody_out_notes
        self.chord_out = chord_out
        self.corenote_out = corenote_out
        self.melody_beat_num = melody_beat_num
        self.end_check_beats = end_check_beats

        # 生成过程中的常量和类型
        self.corenote_code_add_base = 4  # 主旋律骨干音数据编码增加的基数
        self.rc1_code_add_base = 4 + (COMMON_CORE_NOTE_PAT_NUM + 2)  # 和弦第一拍数据编码增加的基数
        self.rc2_code_add_base = 4 + (COMMON_CORE_NOTE_PAT_NUM + 2) + self.train_data.rc_pat_num  # 和弦第二拍数据编码增加的基数
        self.string_code_add_base = 4 + (COMMON_CORE_NOTE_PAT_NUM + 2) + self.train_data.rc_pat_num * 2  # string数据编码增加的基数
        self.root_data, self.rc_pat_output = root_chord_encode(chord_out, self.train_data.all_rc_pats, self.train_data.string_avr_root)

        # 生成过程中的一些变量
        self.rollback_times = 0  # 被打回重新生成的次数
        bar_num = len(melody_out_notes) // 32  # 主旋律一共有多少个小节
        self.confidence_back_times = [0 for t in range(bar_num + 1)]  # 如果自我检查连续失败十次 则直接继续
        self.string_choose_bak = [None for t in range(bar_num + 1)]  # 备选string。为防止死循环，如果连续十次验证失败，则使用备选string
        self.string_abs_note_bak = [[0 for t0 in range(32)] for t in range(bar_num + 1)]
        self.diff_score_bak = [np.inf for t in range(bar_num + 1)]  # 备选方案对应的差异函数
        self.beat_dx = 0  # 生成完第几个pattern了

        # 生成结果
        self.string_out_notes = []
        self.string_out_pats = []

    def rollback(self, step_num):
        self.beat_dx -= 2 * step_num  # 重新生成这一小节的音乐
        self.string_out_notes = self.string_out_notes[:len(self.string_out_notes) - step_num * 8]
        self.string_out_pats = self.string_out_pats[:len(self.string_out_pats) - step_num]

    def generate_by_step(self, session):
        # 1.生成输入数据。生成当前时间的编码,过去10拍的主旋律骨干音和过去10拍的和弦和过去10拍的string
        string_prediction_input = list()
        for backward_beat_it in range(self.beat_dx - 8, self.beat_dx + 2, 2):
            cur_step = backward_beat_it // 2  # 第几个string的步长
            if cur_step < 0:
                string_prediction_input.append([cur_step % 2, self.corenote_code_add_base, self.rc1_code_add_base, self.rc2_code_add_base, self.string_code_add_base])
            elif cur_step < 1:
                string_prediction_input.append([cur_step % 4, self.corenote_out[cur_step] + self.corenote_code_add_base, self.rc_pat_output[backward_beat_it] + self.rc1_code_add_base, self.rc_pat_output[backward_beat_it + 1] + self.rc2_code_add_base, self.string_code_add_base])
            else:
                string_prediction_input.append([cur_step % 4, self.corenote_out[cur_step] + self.corenote_code_add_base, self.rc_pat_output[backward_beat_it] + self.rc1_code_add_base, self.rc_pat_output[backward_beat_it + 1] + self.rc2_code_add_base, self.string_out_pats[cur_step - 1] + self.string_code_add_base])
        # 2.确定即将生成的两拍音符能否为空
        flag_allow_empty = True  # 这两拍是否允许为空
        if self.beat_dx == 0:  # 首拍的string不能为空
            flag_allow_empty = False
        elif self.beat_dx >= 4 and self.string_out_pats[-2:] == [0, 0]:  # 不能连续六拍为空
            flag_allow_empty = False
        elif self.beat_dx >= 2 and chord_row_in_list(self.chord_out[self.beat_dx + 1]) != chord_row_in_list(self.chord_out[self.beat_dx - 1]) and chord_row_in_list(self.chord_out[self.beat_dx]) != chord_row_in_list(self.chord_out[self.beat_dx - 2]):  # 和弦发生变化后 string不能为空
            flag_allow_empty = False
        # 3.生成输出数据
        string_predict = self.predict(session, [string_prediction_input])  # LSTM预测 得到二维数组predict
        if flag_allow_empty is True:
            out_pat_dx = pat_predict_addcode(string_predict, self.string_code_add_base, 0, COMMON_STRING_PAT_NUM)  # 将二维数组predict通过概率随机生成这两拍的string组合out_pat_dx
        else:
            out_pat_dx = pat_predict_addcode(string_predict, self.string_code_add_base, 1, COMMON_STRING_PAT_NUM)  # 将二维数组predict通过概率随机生成这两拍的string组合out_pat_dx
        self.string_out_pats.append(out_pat_dx)  # 添加到最终的string输出列表中
        rel_note_list = self.train_data.common_string_pats[out_pat_dx]  # 将新生成的string组合变为相对音高列表
        # 4.将新生成的string组合变为绝对音高列表
        for rel_note_group in rel_note_list:
            if rel_note_group == 0:
                self.string_out_notes.append(0)
            else:
                self.string_out_notes.append(get_abs_notelist_chord(rel_note_group, self.root_data[self.beat_dx]))
        self.beat_dx += 2

    def check_1step(self, session):
        # 1.检查string与同时期的和弦差异是否过大，在每小节的末尾进行检验
        if self.beat_dx >= 8 and self.beat_dx % 4 == 0 and not string_chord_check(self.string_out_notes[-32:], self.chord_out[(self.beat_dx - 8): self.beat_dx]):  # string与同时期的和弦差异过大
            DiaryLog.warn('在第%d拍, string第%02d次打回，与同时期和弦差异太大, 最后八拍音符为%s' % (self.beat_dx, self.rollback_times, repr(self.string_out_notes[-32:])))
            self.rollback(4)
            self.rollback_times += 1
            return

        # 2.连续八拍伴奏的偏离程度(包括按键/音高差异/和同时期和弦的差异综合评定)。每生成了奇数小节之后进行校验
        if self.beat_dx >= 12 and self.beat_dx % 8 == 4:  # 每生成了奇数小节之后进行校验
            total_diff_score = self.train_data.StringConfidence.evaluate(note_out=self.string_out_notes[-48:], chord_out=self.chord_out[(self.beat_dx - 8): self.beat_dx])  # 根据训练集90%bass差异分判断的校验法
            if not self.train_data.StringConfidence.compare(total_diff_score):
                bar_dx = self.beat_dx // 4 - 1  # 当前小节 减一
                self.confidence_back_times[bar_dx] += 1
                DiaryLog.warn('第%d拍, string的误差分数为%.4f, 高于临界值%.4f' % (self.beat_dx, total_diff_score, self.train_data.StringConfidence.confidence_level))
                if total_diff_score <= self.diff_score_bak[bar_dx]:
                    self.string_abs_note_bak[bar_dx] = self.string_out_notes[-32:]
                    self.string_choose_bak[bar_dx] = self.string_out_pats[-4:]
                    self.diff_score_bak[bar_dx] = total_diff_score
                self.rollback(4)
                if self.confidence_back_times[bar_dx] >= 10:
                    DiaryLog.warn('第%d拍, string使用备选方案, 误差函数值为%.4f, 这八拍的string为%s' % (self.beat_dx, self.diff_score_bak[bar_dx], repr(self.string_out_notes[bar_dx])))
                    self.string_out_notes.extend(self.string_abs_note_bak[bar_dx])
                    self.string_out_pats.extend(self.string_choose_bak[bar_dx])
                    self.beat_dx += 8
                else:
                    return

        # 3.string结束阶段的检验: 必须全部在有1级大和弦或6级小和弦内。在最后一个步长执行检验
        if self.beat_dx in self.end_check_beats and not string_end_check(self.string_out_notes):
            DiaryLog.warn('在%d拍, string第%02d次打回，最后一个音是弦外音, 最后八拍音符为%s' % (self.beat_dx, self.rollback_times, repr(self.string_out_notes[-32:])))
            self.rollback(4)
            self.rollback_times += 1

    def generate(self, session, melody_out_notes, chord_out, corenote_out, melody_beat_num, end_check_beats):
        self.generate_init(session, melody_out_notes, chord_out, corenote_out, melody_beat_num, end_check_beats)
        while True:
            self.generate_by_step(session)
            self.check_1step(session)

            if self.rollback_times >= MAX_GEN_STRING_FAIL_TIME:
                DiaryLog.warn('string被打回次数超过%d次,重新生成。\n\n\n' % self.rollback_times)
                raise RuntimeError
            if self.beat_dx == self.melody_beat_num:
                assert self.beat_dx == len(self.string_out_notes) // 4
                break

        DiaryLog.warn('string的输出: ' + repr(self.string_out_notes))
        return self.string_out_notes
