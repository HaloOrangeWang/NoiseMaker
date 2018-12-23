from settings import *
from pipelines.functions import BaseLstmPipeline, root_chord_encode, pat_predict_addcode
from datainputs.piano_guitar import PianoGuitarTrainData, PianoGuitarTestData
from models.configs import PianoGuitarConfig
from interfaces.note_format import get_abs_notelist_chord
from interfaces.utils import DiaryLog
from validations.piano_guitar import pg_chord_check, pg_end_check
import numpy as np


class PianoGuitarPipeline(BaseLstmPipeline):

    def __init__(self, is_train, *args):
        if is_train:
            # 训练时的情况
            # args = (melody_pat_data, continuous_bar_data, keypress_pat_data, all_keypress_pats, chord_cls)
            self.train_data = PianoGuitarTrainData(*args)
        else:
            # 生成时的情况
            # args = (all_keypress_pats)
            self.train_data = PianoGuitarTestData(*args)
        super().__init__()

    # noinspection PyAttributeOutsideInit
    def prepare(self):
        self.config = PianoGuitarConfig(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config = PianoGuitarConfig(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'PianoGuitarModel'

    # noinspection PyAttributeOutsideInit
    def generate_init(self, session, melody_out_notes, keypress_out, chord_out, melody_beat_num, end_check_beats):
        self.melody_out_notes = melody_out_notes
        self.keypress_out = keypress_out
        self.chord_out = chord_out
        self.melody_beat_num = melody_beat_num
        self.end_check_beats = end_check_beats

        # 生成过程中的常量和类型
        self.keypress_code_add_base = 8  # 主旋律第一拍数据编码增加的基数
        self.rc_code_add_base = 8 + self.train_data.keypress_pat_num  # 和弦第一拍数据编码增加的基数
        self.pg_code_add_base = 8 + self.train_data.keypress_pat_num + self.train_data.rc_pat_num  # piano_guitar数据编码增加的基数
        self.root_data, self.rc_pat_output = root_chord_encode(chord_out, self.train_data.all_rc_pats, self.train_data.pg_avr_root)

        # 生成过程中的一些变量
        self.rollback_times = 0  # 被打回重新生成的次数
        bar_num = len(melody_out_notes) // 32  # 主旋律一共有多少个小节
        self.confidence_back_times = [0 for t in range(bar_num + 1)]  # 如果自我检查连续失败十次 则直接继续
        self.pg_choose_bak = [None for t in range(bar_num + 1)]  # 备选piano_guitar。为防止死循环，如果连续十次验证失败，则使用备选piano_guitar
        self.pg_abs_note_bak = [[0 for t0 in range(32)] for t in range(bar_num + 1)]
        self.pg_score_bak = [np.inf for t in range(bar_num + 1)]  # 备选方案对应的差异函数
        self.beat_dx = 0  # 生成完第几个pattern了

        # 生成结果
        self.pg_out_notes = []
        self.pg_out_pats = []

    def rollback(self, step_num):
        self.beat_dx -= step_num  # 重新生成这一小节的音乐
        self.pg_out_notes = self.pg_out_notes[:len(self.pg_out_notes) - step_num * 4]
        self.pg_out_pats = self.pg_out_pats[:len(self.pg_out_pats) - step_num]

    def generate_by_step(self, session):
        # 1.逐时间步长生成test model输入数据
        pg_prediction_input = list()
        for backward_beat_it in range(self.beat_dx - 8, self.beat_dx + 1):
            cur_keypress_step = backward_beat_it // 2  # 第几个keypress的步长(keypress的步长是2拍)
            if backward_beat_it < 0:
                pg_prediction_input.append([backward_beat_it % 4, self.keypress_code_add_base, self.rc_code_add_base, self.pg_code_add_base])
            elif backward_beat_it < 1:
                pg_prediction_input.append([backward_beat_it % 8, self.keypress_out[cur_keypress_step] + self.keypress_code_add_base, self.rc_pat_output[backward_beat_it] + self.rc_code_add_base, self.pg_code_add_base])
            else:
                pg_prediction_input.append([backward_beat_it % 8, self.keypress_out[cur_keypress_step] + self.keypress_code_add_base, self.rc_pat_output[backward_beat_it] + self.rc_code_add_base, self.pg_out_pats[backward_beat_it - 1] + self.pg_code_add_base])
        # 2.生成输出数据
        pg_predict = self.predict(session, [pg_prediction_input])  # LSTM预测 得到二维数组predict
        out_pat_dx = pat_predict_addcode(pg_predict, self.pg_code_add_base, 0, COMMON_PG_PAT_NUM)  # 将二维数组predict通过概率随机生成这拍的piano_guitar组合pg_out_pattern
        self.pg_out_pats.append(out_pat_dx)  # 添加到最终的piano_guitar输出列表中
        rel_note_list = self.train_data.common_pg_pats[out_pat_dx]  # 将新生成的piano_guitar组合变为相对音高列表
        # 3.将新生成的piano_guitar组合变为绝对音高列表
        for rel_notes in rel_note_list:
            if rel_notes == 0:
                self.pg_out_notes.append(0)
            else:
                self.pg_out_notes.append(get_abs_notelist_chord(rel_notes, self.root_data[self.beat_dx]))
        self.beat_dx += 1

    def check_1step(self, session):
        # 1.检查piano_guitar与同时期的和弦差异是否过大，在每小节的末尾进行检验
        if self.beat_dx >= 8 and self.beat_dx % 4 == 0 and not pg_chord_check(self.pg_out_notes[-32:], self.chord_out[(self.beat_dx - 8): self.beat_dx]):
            DiaryLog.warn('在第%d拍, piano_guitar第%02d次打回，与同时期和弦差异太大, 最后八拍音符为%s' % (self.beat_dx, self.rollback_times, repr(self.pg_out_notes[-32:])))
            self.rollback(8)
            self.rollback_times += 1
            return

        # 2.连续八拍伴奏的偏离程度(包括按键/音高差异/和同时期和弦的差异综合评定)。每生成了奇数小节之后进行校验
        if self.beat_dx >= 12 and self.beat_dx % 8 == 4:  # 每生成了奇数小节之后进行校验
            total_diff_score = self.train_data.PgConfidence.evaluate(note_out=self.pg_out_notes[-48:], chord_out=self.chord_out[(self.beat_dx - 8): self.beat_dx])  # 根据训练集90%bass差异分判断的校验法
            if not self.train_data.PgConfidence.compare(total_diff_score):
                bar_dx = self.beat_dx // 4 - 1
                self.confidence_back_times[bar_dx] += 1
                DiaryLog.warn('第%d拍, piano_guitar的误差分数为%.4f, 高于临界值%.4f' % (self.beat_dx, total_diff_score, self.train_data.PgConfidence.confidence_level))
                if total_diff_score < self.pg_score_bak[bar_dx]:
                    self.pg_abs_note_bak[bar_dx] = self.pg_out_notes[-32:]
                    self.pg_choose_bak[bar_dx] = self.pg_out_pats[-8:]
                    self.pg_score_bak[bar_dx] = total_diff_score
                self.rollback(8)  # 当前小节 减二 重新生成这两小节的piano_guitar
                if self.confidence_back_times[bar_dx] >= 10:
                    DiaryLog.warn('第%d拍, piano_guitar使用备选方案, 误差函数值为%.4f, 这八拍的piano_guitar为%s' % (self.beat_dx, self.pg_score_bak[bar_dx], repr(self.pg_abs_note_bak[bar_dx])))
                    self.pg_out_notes.extend(self.pg_abs_note_bak[bar_dx])
                    self.pg_out_pats.extend(self.pg_choose_bak[bar_dx])
                    self.beat_dx += 8
                else:
                    return

        # 3.piano_guitar结束阶段的检验: 必须全部在有1级大和弦或6级小和弦内。在最后一个步长执行检验
        if self.beat_dx in self.end_check_beats and not pg_end_check(self.pg_out_notes):
            DiaryLog.warn('在%d拍, piano_guitar第%02d次打回，最后一个音是弦外音, 最后八拍音符为%s' % (self.beat_dx, self.rollback_times, repr(self.pg_out_notes[-32:])))
            self.rollback(8)
            self.rollback_times += 1

    def generate(self, session, melody_out_notes, keypress_out, chord_out, melody_beat_num, end_check_beats):
        self.generate_init(session, melody_out_notes, keypress_out, chord_out, melody_beat_num, end_check_beats)
        while True:
            self.generate_by_step(session)
            self.check_1step(session)

            if self.rollback_times >= MAX_GEN_PG_FAIL_TIME:
                DiaryLog.warn('piano_guitar被打回次数超过%d次,重新生成。\n\n\n' % self.rollback_times)
                raise RuntimeError
            if self.beat_dx == self.melody_beat_num:
                assert self.beat_dx == len(self.pg_out_notes) // 4
                break

        DiaryLog.warn('piano_guitar的输出: ' + repr(self.pg_out_notes))
        return self.pg_out_notes
