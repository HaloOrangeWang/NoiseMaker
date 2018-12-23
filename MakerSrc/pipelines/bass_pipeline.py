from settings import *
from pipelines.functions import BaseLstmPipeline, root_chord_encode, pat_predict_addcode
from datainputs.bass import BassTrainData, BassTestData
from models.configs import BassConfig
from interfaces.note_format import get_abs_notelist_chord
from interfaces.utils import DiaryLog
from validations.bass import bass_check, bass_end_check
import numpy as np


class BassPipeline(BaseLstmPipeline):

    def __init__(self, is_train, *args):
        if is_train is True:
            # 训练时的情况
            # args = (melody_pat_data, continuous_bar_data, keypress_pat_data, all_keypress_pats, chord_cls)
            self.train_data = BassTrainData(*args)
        else:
            # 生成时的情况
            # args = (all_keypress_pats)
            self.train_data = BassTestData(*args)
        super().__init__()

    # noinspection PyAttributeOutsideInit
    def prepare(self):
        self.config = BassConfig(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config = BassConfig(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'BassModel'

    # noinspection PyAttributeOutsideInit
    def generate_init(self, session, melody_out_notes, keypress_out, chord_out, melody_beat_num, end_check_beats):
        self.melody_out_notes = melody_out_notes
        self.keypress_out = keypress_out
        self.chord_out = chord_out
        self.melody_beat_num = melody_beat_num
        self.end_check_beats = end_check_beats

        # 生成过程中的常量和类型
        self.keypress_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
        self.rc1_code_add_base = 4 + self.train_data.keypress_pat_num  # 和弦第一拍数据编码增加的基数
        self.rc2_code_add_base = 4 + self.train_data.keypress_pat_num + self.train_data.rc_pat_num  # 和弦第二拍数据编码增加的基数
        self.bass_code_add_base = 4 + self.train_data.keypress_pat_num + self.train_data.rc_pat_num * 2  # bass数据编码增加的基数
        self.root_data, self.rc_pat_output = root_chord_encode(chord_out, self.train_data.all_rc_pats, self.train_data.bass_avr_root)

        # 生成过程中的一些变量
        self.rollback_times = 0  # 被打回重新生成的次数
        bar_num = len(melody_out_notes) // 32  # 主旋律一共有多少个小节
        self.confidence_back_times = [0 for t in range(bar_num + 1)]  # 如果自我检查连续失败十次 则直接继续
        self.bass_choose_bak = [None for t in range(bar_num + 1)]  # 备选bass。为防止死循环，如果连续十次验证失败，则使用备选bass
        self.bass_abs_note_bak = [[0 for t0 in range(64)] for t in range(bar_num + 1)]
        self.diff_score_bak = [np.inf for t in range(bar_num + 1)]  # 备选方案对应的差异函数
        self.beat_dx = 0  # 生成完第几个pattern了

        # 生成结果
        self.bass_out_notes = []
        self.bass_out_pats = []

    def rollback(self, step_num):
        self.beat_dx -= 2 * step_num  # 重新生成这一小节的音乐
        self.bass_out_notes = self.bass_out_notes[:len(self.bass_out_notes) - step_num * 16]
        self.bass_out_pats = self.bass_out_pats[:len(self.bass_out_pats) - step_num]

    def generate_by_step(self, session):
        # 1.逐时间步长生成test model输入数据
        bass_prediction_input = list()
        for backward_beat_it in range(self.beat_dx - 8, self.beat_dx + 2, 2):
            cur_step = backward_beat_it // 2
            if cur_step < 0:
                bass_prediction_input.append([cur_step % 2, self.keypress_code_add_base, self.rc1_code_add_base, self.rc2_code_add_base, self.bass_code_add_base])
            elif cur_step < 1:
                bass_prediction_input.append([cur_step % 4, self.keypress_out[cur_step] + self.keypress_code_add_base, self.rc_pat_output[backward_beat_it] + self.rc1_code_add_base, self.rc_pat_output[backward_beat_it + 1] + self.rc2_code_add_base, self.bass_code_add_base])
            else:
                bass_prediction_input.append([cur_step % 4, self.keypress_out[cur_step] + self.keypress_code_add_base, self.rc_pat_output[backward_beat_it] + self.rc1_code_add_base, self.rc_pat_output[backward_beat_it + 1] + self.rc2_code_add_base, self.bass_out_pats[cur_step - 1] + self.bass_code_add_base])
        # 2.生成输出数据
        bass_predict = self.predict(session, [bass_prediction_input])  # LSTM预测 得到二维数组predict
        if self.beat_dx % 8 == 0:  # 每两小节的第一拍不能为空
            out_pat_dx = pat_predict_addcode(bass_predict, self.bass_code_add_base, 1, COMMON_BASS_PAT_NUM)  # 将二维数组predict通过概率随机生成这两拍的bass组合out_pat_dx
        else:
            out_pat_dx = pat_predict_addcode(bass_predict, self.bass_code_add_base, 0, COMMON_BASS_PAT_NUM)
        self.bass_out_pats.append(out_pat_dx)  # 添加到最终的bass输出列表中
        rel_note_list = self.train_data.common_bass_pats[out_pat_dx]  # 将新生成的bass组合变为相对音高列表
        # 3.将新生成的bass组合变为绝对音高列表
        for note_it in range(len(rel_note_list)):
            if rel_note_list[note_it] == 0:
                self.bass_out_notes.append(0)
            else:
                self.bass_out_notes.append(get_abs_notelist_chord(rel_note_list[note_it], self.root_data[self.beat_dx]))
        self.beat_dx += 2

    def check_1step(self, session):
        # 1.检查bass与同时期的和弦差异是否过大，在每小节的末尾进行检验
        if self.beat_dx >= 8 and self.beat_dx % 4 == 0 and not bass_check(self.bass_out_notes[-64:], self.chord_out[(self.beat_dx - 8): self.beat_dx]):
            DiaryLog.warn('在第%d拍, bass第%02d次打回，与同时期和弦差异太大, 最后八拍音符为%s' % (self.beat_dx, self.rollback_times, repr(self.bass_out_notes[-64:])))
            self.rollback(4)
            self.rollback_times += 1
            return

        # 2.连续八拍伴奏的偏离程度(包括按键/音高差异/和同时期和弦的差异综合评定)。每生成了奇数小节之后进行校验
        if self.beat_dx >= 12 and self.beat_dx % 8 == 4:
            total_diff_score = self.train_data.BassConfidence.evaluate(note_out=self.bass_out_notes[-96:], chord_out=self.chord_out[(self.beat_dx - 8): self.beat_dx])  # 根据训练集90%bass差异分判断的校验法
            if not self.train_data.BassConfidence.compare(total_diff_score):
                bar_dx = self.beat_dx // 4 - 1
                self.confidence_back_times[bar_dx] += 1
                DiaryLog.warn('第%d拍, bass的误差分数为%.4f, 高于临界值%.4f' % (self.beat_dx, total_diff_score, self.train_data.BassConfidence.confidence_level))
                if total_diff_score < self.diff_score_bak[bar_dx]:
                    self.bass_abs_note_bak[bar_dx] = self.bass_out_notes[-64:]
                    self.bass_choose_bak[bar_dx] = self.bass_out_pats[-4:]
                    self.diff_score_bak[bar_dx] = total_diff_score
                self.rollback(4)  # 当前小节 减二 重新生成这两小节的bass
                if self.confidence_back_times[bar_dx] >= 10:
                    DiaryLog.warn('第%d拍, bass使用备选方案, 误差函数值为%.4f, 这八拍的bass为%s' % (self.beat_dx, self.diff_score_bak[bar_dx], repr(self.bass_abs_note_bak[bar_dx])))
                    self.bass_out_notes.extend(self.bass_abs_note_bak[bar_dx])
                    self.bass_out_pats.extend(self.bass_choose_bak[bar_dx])
                    self.beat_dx += 8
                else:
                    return

        # 3.bass结束阶段的检验: 必须有1级大和弦或6级小和弦内的音。在最后一个步长执行检验
        if self.beat_dx in self.end_check_beats and not bass_end_check(self.bass_out_notes):
            DiaryLog.warn('在%d拍, bass第%02d次打回，最后一个音是弦外音, 最后八拍音符为%s' % (self.beat_dx, self.rollback_times, repr(self.bass_out_notes[-64:])))
            self.rollback(4)
            self.rollback_times += 1

    def generate(self, session, melody_out_notes, keypress_out, chord_out, melody_beat_num, end_check_beats):
        self.generate_init(session, melody_out_notes, keypress_out, chord_out, melody_beat_num, end_check_beats)
        while True:
            self.generate_by_step(session)
            self.check_1step(session)

            if self.rollback_times >= MAX_GEN_BASS_FAIL_TIME:
                DiaryLog.warn('bass被打回次数超过%d次,重新生成。\n\n\n' % self.rollback_times)
                raise RuntimeError
            if self.beat_dx == self.melody_beat_num:
                assert self.beat_dx == len(self.bass_out_notes) // 8
                break

        DiaryLog.warn('bass的输出: ' + repr(self.bass_out_notes))
        return self.bass_out_notes
