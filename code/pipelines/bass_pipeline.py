from dataoutputs.validation import bass_check, bass_end_check, bass_confidence_check
from models.configs import BassConfig, BassConfig2, BassConfig3, BassConfig4
from datainputs.bass import BassTrainData, BassTrainDataCheck, BassTrainData2, BassTrainData3, BassTrainData4
from datainputs.melody import MelodyPatternEncode
from settings import *
from pipelines.functions import music_pattern_prediction, keypress_encode, root_chord_encode, BaseLstmPipeline, BaseLstmPipelineMultiCode, music_pattern_prediction, pat_predict_addcode
from interfaces.chord_parse import chord_rootnote
from interfaces.utils import DiaryLog
from interfaces.note_format import get_abs_notelist_chord
import numpy as np


class BassPipeline(BaseLstmPipeline):

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_cls):
        self.train_data = BassTrainData(melody_pattern_data, continuous_bar_number_data, chord_cls)
        super().__init__()

    def prepare(self):
        self.config = BassConfig()
        self.test_config = BassConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'BassModel'

    def generate(self, session, melody_output, chord_output, common_melody_pats):
        bass_output = []  # bass输出
        bass_pattern_output = []  # bass输出的pattern形式
        # 1.数据预处理 将melody_output转化为pattern
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dic = MelodyPatternEncode(common_melody_pats, melody_pattern_input, 0.125, 1).music_pattern_dic
        melody_output_pattern = []  # 把melody_output编码为melody_pattern的形式
        generate_fail_time = 0  # 本次bass的生成一共被打回去多少次
        for key in range(round(len(melody_output) / 32)):  # 遍历这首歌的所有小节 把melody_pattern由dict形式转成list形式
            melody_output_pattern.extend(melody_pattern_dic[key])
        # 2.通过chord_output找到输出的根音列表
        root_data = []
        for chord_it in range(len(chord_output)):
            if chord_it == 0:
                root_data.append(chord_rootnote(chord_output[0], 0, BASS_AVERAGE_ROOT))
            else:
                root_data.append(chord_rootnote(chord_output[chord_it], root_data[chord_it - 1], BASS_AVERAGE_ROOT))
        # print(root_data)
        # 3.逐拍生成数据
        beat_dx = 2  # 准备生成第几个拍的数据
        while beat_dx <= len(melody_output_pattern):  # 遍历整个主旋律 步长是1拍
            # 3.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_BASS_GENERATE_FAIL_TIME:
                DiaryLog.warn('bass被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 3.2.生成输入数据
            # 3.2.1.生成当前时间的编码,过去10拍的主旋律和过去10拍的和弦
            bass_prediction_input = [((beat_dx - 2) % 8) // 2]  # 先保存当前时间的编码
            if beat_dx < 10:  # 还没有到第10拍 旋律输入在前面补0 否则使用对应位置的旋律和和弦
                last_bars_melody = [0 for t in range(10 - beat_dx)] + melody_output_pattern[:beat_dx]
                last_bars_chord = [0 for t in range(10 - beat_dx)] + chord_output[:beat_dx]
            else:
                last_bars_melody = melody_output_pattern[(beat_dx - 10): beat_dx]
                last_bars_chord = chord_output[(beat_dx - 10): beat_dx]
            bass_prediction_input += last_bars_melody + last_bars_chord
            # 3.2.2.生成过去8拍的bass 如果不足8拍则补0
            required_bass_length = 2 * TRAIN_BASS_IO_BARS  # 需要
            if len(bass_pattern_output) < required_bass_length:
                last_bars_bass = [0 for t in range(required_bass_length - len(bass_pattern_output))] + bass_pattern_output
            else:
                last_bars_bass = bass_pattern_output[-required_bass_length:]
            bass_prediction_input += last_bars_bass
            # print('          input', beat_dx, bass_prediction_input)
            # 3.3.生成输出数据
            bass_predict = self.predict(session, [bass_prediction_input])  # LSTM预测 得到二维数组predict
            if beat_dx % 8 == 2:  # 每两小节的第一拍不能为空
                bass_out_pattern = music_pattern_prediction(bass_predict, 1, COMMON_BASS_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的bass组合bass_out_pattern
            else:
                bass_out_pattern = music_pattern_prediction(bass_predict, 0, COMMON_BASS_PATTERN_NUMBER)
            bass_pattern_output.append(bass_out_pattern)  # 添加到最终的bass输出列表中
            rel_note_list = self.train_data.common_bass_pats[bass_out_pattern + 1]  # 将新生成的bass组合变为相对音高列表
            # print('     rnote_output', rel_note_list)
            for rel_note_group in rel_note_list:
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    bass_output.append(0)
                else:
                    bass_output.append(get_abs_notelist_chord(rel_note_group, root_data[beat_dx - 2]))
            beat_dx += 2
            # 3.4.检查生成的bass
            if beat_dx >= 10 and beat_dx % 4 == 2 and not bass_check(bass_output[-64:], chord_output[(beat_dx - 10): (beat_dx - 2)]):  # bass与同时期的和弦差异过大
                # print('在第%d拍, 鼓点第%02d次打回，第一拍为空拍' % (beat_dx, generate_fail_time), drum_output[-16:])
                DiaryLog.warn('在第%d拍, bass第%02d次打回，与同时期和弦差异太大' % (beat_dx, generate_fail_time) + repr(bass_output[-64:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-8)]
                generate_fail_time += 1
                continue
            if beat_dx > len(melody_output_pattern) and not bass_end_check(bass_output):  # 最后一个和弦不是1级大和弦
                DiaryLog.warn('在%d拍, bass第%02d次打回，最后一个音是弦外音' % (beat_dx, generate_fail_time) + repr(bass_output[-64:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-8)]
                generate_fail_time += 1
                continue
        DiaryLog.warn('bass的组合输出: ' + repr(bass_pattern_output))
        DiaryLog.warn('bass的最终输出: ' + repr(bass_output))
        return bass_output


class BassPipelineCheck(BaseLstmPipeline):

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_cls):
        self.train_data = BassTrainDataCheck(melody_pattern_data, continuous_bar_number_data, chord_cls)
        super().__init__()

    def prepare(self):
        self.config = BassConfig()
        self.test_config = BassConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'BassModel'

    def generate_check(self, session):  # , common_melody_pats, common_corenote_pats):
        """多步预测的校验方式"""
        print('校验开始')
        right_value = [0 for t in range(4)]
        wrong_value = [0 for t in range(4)]
        sample_amount = 250
        # 1.从用于校验的数据中随机选取250组作为校验用例
        rand_dx = np.random.permutation(len(self.train_data.check_melody_data))
        rand_dx = rand_dx[:sample_amount]
        check_melody_pat_ary = np.array(self.train_data.check_melody_data)[rand_dx]
        check_chord_data = np.array(self.train_data.check_chord_data)[rand_dx]
        check_bass_input_ary = np.array(self.train_data.check_bass_input_data)[rand_dx]
        check_bass_output_ary = np.array(self.train_data.check_bass_output_data)[rand_dx]
        rd_time_add_ary = np.array(self.train_data.time_add_data)[rand_dx]
        # 2.对这250组进行校验
        for check_it in range(sample_amount):
            real_bass_output_ary = check_bass_input_ary[check_it]  # 模型的bass输出情况
            generate_fail_time = 0  # 如果自我检查连续失败十次 则直接继续

            bass_choose_bk = [1, 1, 1, 1]  # 备选bass。为防止死循环，如果连续十次验证失败，则使用备选和弦
            diff_score_bk = np.inf  # 备选方案对应的差异函数
            while True:
                for step_it in range(0, 4):  # 逐2拍生成数据
                    # 2.1.生成输入数据
                    beat_dx = step_it * 2  # 当前拍是第几拍
                    bass_prediction_input = [(rd_time_add_ary[check_it] + step_it) % 4]  # 先保存当前时间的编码
                    last_bars_bass = real_bass_output_ary[-4:]
                    last_bars_chord = check_chord_data[check_it][beat_dx: beat_dx + 4 * TRAIN_BASS_IO_BARS + 2]
                    last_bars_melody = check_melody_pat_ary[check_it][beat_dx: beat_dx + 4 * TRAIN_BASS_IO_BARS + 2]
                    bass_prediction_input = np.append(bass_prediction_input, np.append(last_bars_melody, np.append(last_bars_chord, last_bars_bass)))  # 校验时输入到model中的数据
                    # print('          input', melody_iterator, chord_prediction_input)
                    # 2.2.生成输出数据
                    bass_predict = self.predict(session, [bass_prediction_input])  # LSTM预测 得到二维数组predict
                    if beat_dx % 8 == 0:  # 每两小节的第一拍不能为空
                        bass_out_pattern = music_pattern_prediction(bass_predict, 1, COMMON_BASS_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的bass组合bass_out_pattern
                    else:
                        bass_out_pattern = music_pattern_prediction(bass_predict, 0, COMMON_BASS_PATTERN_NUMBER)
                    real_bass_output_ary = np.append(real_bass_output_ary, bass_out_pattern)  # 添加到最终的bass输出列表中
                # 2.3.根据一些数学方法和乐理对输出的和弦进行自我检查。如果检查不合格则打回重新生成
                root_data = []
                for chord_it in range(len(check_chord_data[check_it])):
                    if chord_it == 0:
                        root_data.append(chord_rootnote(check_chord_data[check_it][0], 0, BASS_AVERAGE_ROOT))
                    else:
                        root_data.append(chord_rootnote(check_chord_data[check_it][chord_it], root_data[chord_it - 1], BASS_AVERAGE_ROOT))
                bass_output = []
                for step_it in range(6):  # (4):
                    beat_dx = step_it * 2  # 当前拍是第几拍
                    pat_dx = real_bass_output_ary[step_it - 6]  # 4]
                    if pat_dx == COMMON_BASS_PATTERN_NUMBER + 1:  # 罕见bass组合的特殊处理
                        rel_note_list = [0] * 16
                    else:
                        rel_note_list = self.train_data.common_bass_pats[pat_dx]  # + 1]  # 将新生成的bass组合变为相对音高列表
                    # print('     rnote_output', rel_note_list)
                    for rel_note_group in rel_note_list:
                        # print(pattern_iterator, note_iterator, rel_note_group)
                        if rel_note_group == 0:
                            bass_output.append(0)
                        else:
                            bass_output.append(get_abs_notelist_chord(rel_note_group, int(root_data[beat_dx - 12])))
                # print(bass_output[-64:])
                # print(check_chord_data[check_it][-8:])
                if generate_fail_time <= 10:
                    total_diff_score = bass_confidence_check(bass_output, check_chord_data[check_it][-8:])
                    if total_diff_score >= self.train_data.ConfidenceLevel:
                        # if bass_prob_log < self.train_data.ConfidenceLevel:
                        # if not bass_check(bass_output[-64:], check_chord_data[check_it][-8:]):  # bass与同时期的和弦差异过大
                        # print('在第%d拍, 鼓点第%02d次打回，第一拍为空拍' % (beat_dx, generate_fail_time), drum_output[-16:])
                        # DiaryLog.warn('bass被打回，与同时期和弦差异太大' + repr(bass_output[-64:]) + ', 对数为' + repr(bass_prob_log))
                        print('        第%d个检查上,bass的误差分为%.4f' % (check_it, total_diff_score))
                        generate_fail_time += 1
                        if total_diff_score <= diff_score_bk:
                            bass_choose_bk = real_bass_output_ary[-4:]
                            diff_score_bk = total_diff_score
                        real_bass_output_ary = real_bass_output_ary[:(-4)]  # 检查不合格 重新生成这两小节的bass
                        if generate_fail_time >= 10:
                            print('        bass使用备选方案,误差函数值为', diff_score_bk)
                            real_bass_output_ary = np.append(real_bass_output_ary, bass_choose_bk)
                            break
                    else:
                        print('第%d个检查上,bass的误差分为%.4f' % (check_it, total_diff_score))
                        break
                else:  # 如果自我检查失败超过10次 就直接返回 避免循环过长
                    break
            # 2.4.对输出数据逐两拍进行校验 只要符合真实这两拍和弦的其一即可
            for step_it in range(0, 4):  # 逐2拍进行校验
                if check_bass_output_ary[check_it][step_it] not in [0, COMMON_BASS_PATTERN_NUMBER + 1]:
                    if real_bass_output_ary[step_it + 2 * TRAIN_BASS_IO_BARS] == check_bass_output_ary[check_it][step_it]:
                        right_value[step_it] += 1
                    else:
                        wrong_value[step_it] += 1
            # if check_it < 10:
            #     print('melody input', check_melody_pat_ary[check_it])
            #     print('chord input', check_chord_data[check_it])
            #     print('bass input', np.append(check_bass_input_ary[check_it], check_bass_output_ary[check_it]))
            #     print('real out', real_bass_output_ary)
            #     print('\n')
        # 3.输出
        for step_it in range(4):
            right_ratio = right_value[step_it] / (right_value[step_it] + wrong_value[step_it])
            DiaryLog.warn('第%d个两拍正确的bass数量为%d,错误的bass数量为%d,正确率为%.4f.\n' % (step_it, right_value[step_it], wrong_value[step_it], right_ratio))

    def generate(self, session, melody_output, chord_output, common_melody_pats):
        bass_output = []  # bass输出
        bass_pattern_output = []  # bass输出的pattern形式
        # 1.数据预处理 将melody_output转化为pattern
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dic = MelodyPatternEncode(common_melody_pats, melody_pattern_input, 0.125, 1).music_pattern_dic
        melody_output_pattern = []  # 把melody_output编码为melody_pattern的形式
        generate_fail_time = 0  # 本次bass的生成一共被打回去多少次
        for key in range(round(len(melody_output) / 32)):  # 遍历这首歌的所有小节 把melody_pattern由dict形式转成list形式
            melody_output_pattern.extend(melody_pattern_dic[key])
        # 定义90%bass差异分判断的相关参数 包括次验证失败时的备选数据
        confidence_fail_time = [0] * MAX_GENERATE_BAR_NUMBER  # 如果自我检查连续失败十次 则直接继续
        bass_choose_bk = [[1, 1, 1, 1] for t in range(MAX_GENERATE_BAR_NUMBER)]  # 备选bass。为防止死循环，如果连续十次验证失败，则使用备选bass
        bass_abs_note_bk = [[0] * 64 for t in range(MAX_GENERATE_BAR_NUMBER)]
        diff_score_bk = [np.inf] * MAX_GENERATE_BAR_NUMBER  # 备选方案对应的差异函数
        # 2.通过chord_output找到输出的根音列表
        root_data = []
        for chord_it in range(len(chord_output)):
            if chord_it == 0:
                root_data.append(chord_rootnote(chord_output[0], 0, BASS_AVERAGE_ROOT))
            else:
                root_data.append(chord_rootnote(chord_output[chord_it], root_data[chord_it - 1], BASS_AVERAGE_ROOT))
        # print(root_data)
        # 3.逐拍生成数据
        beat_dx = 2  # 准备生成第几个拍的数据
        while beat_dx <= len(melody_output_pattern):  # 遍历整个主旋律 步长是1拍
            # 3.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_BASS_GENERATE_FAIL_TIME:
                DiaryLog.warn('bass被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 3.2.生成输入数据
            # 3.2.1.生成当前时间的编码,过去10拍的主旋律和过去10拍的和弦
            bass_prediction_input = [((beat_dx - 2) % 8) // 2]  # 先保存当前时间的编码
            if beat_dx < 10:  # 还没有到第10拍 旋律输入在前面补0 否则使用对应位置的旋律和和弦
                last_bars_melody = [0 for t in range(10 - beat_dx)] + melody_output_pattern[:beat_dx]
                last_bars_chord = [0 for t in range(10 - beat_dx)] + chord_output[:beat_dx]
            else:
                last_bars_melody = melody_output_pattern[(beat_dx - 10): beat_dx]
                last_bars_chord = chord_output[(beat_dx - 10): beat_dx]
            bass_prediction_input += last_bars_melody + last_bars_chord
            # 3.2.2.生成过去8拍的bass 如果不足8拍则补0
            required_bass_length = 2 * TRAIN_BASS_IO_BARS  # 需要
            if len(bass_pattern_output) < required_bass_length:
                last_bars_bass = [0 for t in range(required_bass_length - len(bass_pattern_output))] + bass_pattern_output
            else:
                last_bars_bass = bass_pattern_output[-required_bass_length:]
            bass_prediction_input += last_bars_bass
            # print('          input', beat_dx, bass_prediction_input)
            # 3.3.生成输出数据
            bass_predict = self.predict(session, [bass_prediction_input])  # LSTM预测 得到二维数组predict
            if beat_dx % 8 == 2:  # 每两小节的第一拍不能为空
                bass_out_pattern = music_pattern_prediction(bass_predict, 1, COMMON_BASS_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的bass组合bass_out_pattern
            else:
                bass_out_pattern = music_pattern_prediction(bass_predict, 0, COMMON_BASS_PATTERN_NUMBER)
            bass_pattern_output.append(bass_out_pattern)  # 添加到最终的bass输出列表中
            rel_note_list = self.train_data.common_bass_pats[bass_out_pattern + 1]  # 将新生成的bass组合变为相对音高列表
            # print('     rnote_output', rel_note_list)
            for rel_note_group in rel_note_list:
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    bass_output.append(0)
                else:
                    bass_output.append(get_abs_notelist_chord(rel_note_group, root_data[beat_dx - 2]))
            beat_dx += 2
            # 3.4.检查生成的bass
            if beat_dx >= 10 and beat_dx % 4 == 2 and not bass_check(bass_output[-64:], chord_output[(beat_dx - 10): (beat_dx - 2)]):  # bass与同时期的和弦差异过大
                # print('在第%d拍, 鼓点第%02d次打回，第一拍为空拍' % (beat_dx, generate_fail_time), drum_output[-16:])
                DiaryLog.warn('在第%d拍, bass第%02d次打回，与同时期和弦差异太大' % (beat_dx, generate_fail_time) + repr(bass_output[-64:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-4)]
                generate_fail_time += 1
                continue
            if beat_dx >= 14 and beat_dx % 8 == 6:  # 每生成了奇数小节之后进行校验
                total_diff_score = bass_confidence_check(bass_output[-96:], chord_output[(beat_dx - 10): (beat_dx - 2)])  # 根据训练集90%bass差异分判断的校验法
                if total_diff_score >= self.train_data.ConfidenceLevel:
                    DiaryLog.warn('第%d拍,bass的误差分数为%.4f,高于临界值%.4f' % (beat_dx, total_diff_score, self.train_data.ConfidenceLevel))
                    curbar = (beat_dx - 2) // 4 - 1  # 当前小节 减一
                    confidence_fail_time[curbar] += 1
                    if total_diff_score <= diff_score_bk[curbar]:
                        bass_abs_note_bk[curbar] = bass_output[-64:]
                        bass_choose_bk[curbar] = bass_pattern_output[-4:]
                        diff_score_bk[curbar] = total_diff_score
                    beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                    bass_output = bass_output[:(-64)]
                    bass_pattern_output = bass_pattern_output[:(-4)]  # 检查不合格 重新生成这两小节的bass
                    if confidence_fail_time[curbar] >= 10:
                        DiaryLog.warn('第%d拍,bass使用备选方案,误差函数值为%.4f' % (beat_dx, diff_score_bk[curbar]))
                        bass_output = bass_output + bass_abs_note_bk[curbar]
                        bass_pattern_output = bass_pattern_output + bass_choose_bk[curbar]
                        beat_dx += 8
                    else:
                        continue
            if beat_dx > len(melody_output_pattern) and not bass_end_check(bass_output):  # 最后一个和弦不是1级大和弦
                DiaryLog.warn('在%d拍, bass第%02d次打回，最后一个音是弦外音' % (beat_dx, generate_fail_time) + repr(bass_output[-64:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-8)]
                generate_fail_time += 1
                continue
        DiaryLog.warn('bass的组合输出: ' + repr(bass_pattern_output))
        DiaryLog.warn('bass的最终输出: ' + repr(bass_output))
        return bass_output


class BassPipeline2(BaseLstmPipelineMultiCode):

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_cls):
        self.train_data = BassTrainData2(melody_pattern_data, continuous_bar_number_data, chord_cls)
        super().__init__(5)  # chord输入数据的编码为二重编码

    def prepare(self):
        self.config = BassConfig2()
        self.test_config = BassConfig2()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'BassModel'


class BassPipeline3(BaseLstmPipelineMultiCode):

    def __init__(self, melody_pat_data, continuous_bar_data, keypress_pat_data, keypress_pat_ary, chord_cls):
        self.train_data = BassTrainData3(melody_pat_data, continuous_bar_data, keypress_pat_data, keypress_pat_ary, chord_cls)
        super().__init__(4)  # chord输入数据的编码为二重编码

    def prepare(self):
        self.config = BassConfig3(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config = BassConfig3(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'BassModel'

    def generate(self, session, melody_output, chord_output, keypress_pats):
        bass_output = []  # bass输出
        bass_pattern_output = []  # bass输出的pattern形式
        # 1.数据预处理 将melody_output转化为pattern
        keypress_pat_output = keypress_encode(melody_output, keypress_pats)
        root_data, rc_pat_output = root_chord_encode(chord_output, self.train_data.rc_pattern_count, BASS_AVERAGE_ROOT)
        generate_fail_time = 0  # 本次bass的生成一共被打回去多少次
        # 2.定义90%bass差异分判断的相关参数 包括次验证失败时的备选数据
        confidence_fail_time = [0] * MAX_GENERATE_BAR_NUMBER  # 如果自我检查连续失败十次 则直接继续
        bass_choose_bk = [[1, 1, 1, 1] for t in range(MAX_GENERATE_BAR_NUMBER)]  # 备选bass。为防止死循环，如果连续十次验证失败，则使用备选bass
        bass_abs_note_bk = [[0] * 64 for t in range(MAX_GENERATE_BAR_NUMBER)]
        diff_score_bk = [np.inf] * MAX_GENERATE_BAR_NUMBER  # 备选方案对应的差异函数
        keypress_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
        rc1_code_add_base = 4 + self.train_data.keypress_pat_num  # 和弦第一拍数据编码增加的基数
        rc2_code_add_base = 4 + self.train_data.keypress_pat_num + self.train_data.rc_pat_num  # 和弦第二拍数据编码增加的基数
        bass_code_add_base = 4 + self.train_data.keypress_pat_num + self.train_data.rc_pat_num * 2  # bass数据编码增加的基数
        # 3.逐拍生成数据
        beat_dx = 2  # 准备生成第几个拍的数据
        while beat_dx <= len(melody_output) // 8:  # 遍历整个主旋律 步长是2拍
            # 3.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_BASS_GENERATE_FAIL_TIME:
                DiaryLog.warn('bass被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 3.2.生成输入数据。生成当前时间的编码,过去10拍的主旋律和过去10拍的和弦
            bass_prediction_input = list()
            for backward_beat_dx in range(beat_dx - 10, beat_dx, 2):
                cur_step = backward_beat_dx // 2  # 第几个bass的步长
                if cur_step < 0:
                    bass_prediction_input.append([cur_step % 2, keypress_code_add_base, rc1_code_add_base, rc2_code_add_base])
                else:
                    bass_prediction_input.append([cur_step % 4, keypress_pat_output[cur_step] + keypress_code_add_base, rc_pat_output[backward_beat_dx] + rc1_code_add_base, rc_pat_output[backward_beat_dx + 1] + rc2_code_add_base])
            # 3.3.生成输出数据
            bass_predict = self.predict(session, [bass_prediction_input])  # LSTM预测 得到二维数组predict
            if beat_dx % 8 == 2:  # 每两小节的第一拍不能为空
                bass_out_pattern = pat_predict_addcode(bass_predict, bass_code_add_base, 1, COMMON_BASS_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的bass组合bass_out_pattern
            else:
                bass_out_pattern = pat_predict_addcode(bass_predict, bass_code_add_base, 0, COMMON_BASS_PATTERN_NUMBER)
            bass_pattern_output.append(bass_out_pattern)  # 添加到最终的bass输出列表中
            rel_note_list = self.train_data.common_bass_pats[bass_out_pattern]  # 将新生成的bass组合变为相对音高列表
            # print('     rnote_output', rel_note_list)
            for rel_note_group in rel_note_list:
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_group == 0:
                    bass_output.append(0)
                else:
                    bass_output.append(get_abs_notelist_chord(rel_note_group, root_data[beat_dx - 2]))
            beat_dx += 2
            # 3.4.检查生成的bass
            if beat_dx >= 10 and beat_dx % 4 == 2 and not bass_check(bass_output[-64:], chord_output[(beat_dx - 10): (beat_dx - 2)]):  # bass与同时期的和弦差异过大
                # print('在第%d拍, 鼓点第%02d次打回，第一拍为空拍' % (beat_dx, generate_fail_time), drum_output[-16:])
                DiaryLog.warn('在第%d拍, bass第%02d次打回，与同时期和弦差异太大' % (beat_dx, generate_fail_time) + repr(bass_output[-64:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-4)]
                generate_fail_time += 1
                continue
            if beat_dx >= 14 and beat_dx % 8 == 6:  # 每生成了奇数小节之后进行校验
                total_diff_score = bass_confidence_check(bass_output[-96:], chord_output[(beat_dx - 10): (beat_dx - 2)])  # 根据训练集90%bass差异分判断的校验法
                if total_diff_score >= self.train_data.ConfidenceLevel:
                    DiaryLog.warn('第%d拍,bass的误差分数为%.4f,高于临界值%.4f' % (beat_dx, total_diff_score, self.train_data.ConfidenceLevel))
                    curbar = (beat_dx - 2) // 4 - 1  # 当前小节 减一
                    confidence_fail_time[curbar] += 1
                    if total_diff_score <= diff_score_bk[curbar]:
                        bass_abs_note_bk[curbar] = bass_output[-64:]
                        bass_choose_bk[curbar] = bass_pattern_output[-4:]
                        diff_score_bk[curbar] = total_diff_score
                    beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                    bass_output = bass_output[:(-64)]
                    bass_pattern_output = bass_pattern_output[:(-4)]  # 检查不合格 重新生成这两小节的bass
                    if confidence_fail_time[curbar] >= 10:
                        DiaryLog.warn('第%d拍,bass使用备选方案,误差函数值为%.4f' % (beat_dx, diff_score_bk[curbar]))
                        bass_output = bass_output + bass_abs_note_bk[curbar]
                        bass_pattern_output = bass_pattern_output + bass_choose_bk[curbar]
                        beat_dx += 8
                    else:
                        continue
            if beat_dx > len(melody_output) // 8 and not bass_end_check(bass_output):  # 最后一个和弦不是1级大和弦
                DiaryLog.warn('在%d拍, bass第%02d次打回，最后一个音是弦外音' % (beat_dx, generate_fail_time) + repr(bass_output[-64:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-8)]
                generate_fail_time += 1
                continue
        DiaryLog.warn('bass的组合输出: ' + repr(bass_pattern_output))
        DiaryLog.warn('bass的最终输出: ' + repr(bass_output))
        return bass_output


class BassPipeline4(BaseLstmPipelineMultiCode):

    def __init__(self, melody_pat_data, continuous_bar_data, keypress_pat_data, keypress_pat_ary, chord_cls):
        self.train_data = BassTrainData4(melody_pat_data, continuous_bar_data, keypress_pat_data, keypress_pat_ary, chord_cls)
        super().__init__(5)  # bass输入数据的编码为五重编码

    def prepare(self):
        self.config = BassConfig4(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config = BassConfig4(self.train_data.keypress_pat_num, self.train_data.rc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'BassModel'

    def generate(self, session, melody_output, chord_output, keypress_pats, melody_beat_num):
        bass_output = []  # bass输出
        bass_pattern_output = []  # bass输出的pattern形式
        # 1.数据预处理 将melody_output转化为pattern
        keypress_pat_output = keypress_encode(melody_output, keypress_pats)
        root_data, rc_pat_output = root_chord_encode(chord_output, self.train_data.rc_pattern_count, BASS_AVERAGE_ROOT)
        generate_fail_time = 0  # 本次bass的生成一共被打回去多少次
        bar_num = len(melody_output) // 32  # 主旋律一共有多少个小节
        # 2.定义90%bass差异分判断的相关参数 包括次验证失败时的备选数据
        confidence_fail_time = [0] * (bar_num + 1)  # 如果自我检查连续失败十次 则直接继续
        bass_choose_bk = [[1, 1, 1, 1] for t in range(bar_num + 1)]  # 备选bass。为防止死循环，如果连续十次验证失败，则使用备选bass
        bass_abs_note_bk = [[0] * 64 for t in range(bar_num + 1)]
        diff_score_bk = [np.inf] * (bar_num + 1)  # 备选方案对应的差异函数
        keypress_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
        rc1_code_add_base = 4 + self.train_data.keypress_pat_num  # 和弦第一拍数据编码增加的基数
        rc2_code_add_base = 4 + self.train_data.keypress_pat_num + self.train_data.rc_pat_num  # 和弦第二拍数据编码增加的基数
        bass_code_add_base = 4 + self.train_data.keypress_pat_num + self.train_data.rc_pat_num * 2  # bass数据编码增加的基数
        # 3.逐拍生成数据
        beat_dx = 2  # 准备生成第几个拍的数据
        while beat_dx <= len(melody_output) // 8:  # 遍历整个主旋律 步长是2拍
            # 3.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_BASS_GENERATE_FAIL_TIME:
                DiaryLog.warn('bass被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 3.2.生成输入数据。生成当前时间的编码,过去10拍的主旋律和过去10拍的和弦
            bass_prediction_input = list()
            for backward_beat_dx in range(beat_dx - 10, beat_dx, 2):
                cur_step = backward_beat_dx // 2  # 第几个bass的步长
                if cur_step < 0:
                    bass_prediction_input.append([cur_step % 2, keypress_code_add_base, rc1_code_add_base, rc2_code_add_base, bass_code_add_base])
                elif cur_step < 1:
                    bass_prediction_input.append([cur_step % 4, keypress_pat_output[cur_step] + keypress_code_add_base, rc_pat_output[backward_beat_dx] + rc1_code_add_base, rc_pat_output[backward_beat_dx + 1] + rc2_code_add_base, bass_code_add_base])
                else:
                    bass_prediction_input.append([cur_step % 4, keypress_pat_output[cur_step] + keypress_code_add_base, rc_pat_output[backward_beat_dx] + rc1_code_add_base, rc_pat_output[backward_beat_dx + 1] + rc2_code_add_base, bass_pattern_output[cur_step - 1] + bass_code_add_base])
            # 3.3.生成输出数据
            bass_predict = self.predict(session, [bass_prediction_input])  # LSTM预测 得到二维数组predict
            if beat_dx % 8 == 2:  # 每两小节的第一拍不能为空
                bass_out_pattern = pat_predict_addcode(bass_predict, bass_code_add_base, 1, COMMON_BASS_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的bass组合bass_out_pattern
            else:
                bass_out_pattern = pat_predict_addcode(bass_predict, bass_code_add_base, 0, COMMON_BASS_PATTERN_NUMBER)
            bass_pattern_output.append(bass_out_pattern)  # 添加到最终的bass输出列表中
            rel_note_list = self.train_data.common_bass_pats[bass_out_pattern]  # 将新生成的bass组合变为相对音高列表
            # print('     rnote_output', rel_note_list)
            for note_it in range(len(rel_note_list)):
                # print(pattern_iterator, note_iterator, rel_note_group)
                if rel_note_list[note_it] == 0:
                    bass_output.append(0)
                else:
                    try:
                        bass_output.append(get_abs_notelist_chord(rel_note_list[note_it], root_data[beat_dx - 2]))
                    except TypeError:
                        pass
            beat_dx += 2
            # 3.4.检查生成的bass
            if beat_dx >= 10 and beat_dx % 4 == 2 and not bass_check(bass_output[-64:], chord_output[(beat_dx - 10): (beat_dx - 2)]):  # bass与同时期的和弦差异过大
                # print('在第%d拍, 鼓点第%02d次打回，第一拍为空拍' % (beat_dx, generate_fail_time), drum_output[-16:])
                DiaryLog.warn('在第%d拍, bass第%02d次打回，与同时期和弦差异太大' % (beat_dx, generate_fail_time) + repr(bass_output[-64:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-4)]
                generate_fail_time += 1
                continue
            if beat_dx >= 14 and beat_dx % 8 == 6:  # 每生成了奇数小节之后进行校验
                total_diff_score = bass_confidence_check(bass_output[-96:], chord_output[(beat_dx - 10): (beat_dx - 2)])  # 根据训练集90%bass差异分判断的校验法
                if total_diff_score >= self.train_data.ConfidenceLevel:
                    DiaryLog.warn('第%d拍,bass的误差分数为%.4f,高于临界值%.4f' % (beat_dx, total_diff_score, self.train_data.ConfidenceLevel))
                    curbar = (beat_dx - 2) // 4 - 1  # 当前小节 减一
                    confidence_fail_time[curbar] += 1
                    if total_diff_score <= diff_score_bk[curbar]:
                        bass_abs_note_bk[curbar] = bass_output[-64:]
                        bass_choose_bk[curbar] = bass_pattern_output[-4:]
                        diff_score_bk[curbar] = total_diff_score
                    beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                    bass_output = bass_output[:(-64)]
                    bass_pattern_output = bass_pattern_output[:(-4)]  # 检查不合格 重新生成这两小节的bass
                    if confidence_fail_time[curbar] >= 10:
                        DiaryLog.warn('第%d拍,bass使用备选方案,误差函数值为%.4f' % (beat_dx, diff_score_bk[curbar]))
                        bass_output = bass_output + bass_abs_note_bk[curbar]
                        bass_pattern_output = bass_pattern_output + bass_choose_bk[curbar]
                        beat_dx += 8
                    else:
                        continue
            if beat_dx in [melody_beat_num + 2, len(melody_output) // 8 + 2] and not bass_end_check(bass_output):  # 最后一个和弦不是1级大和弦
                DiaryLog.warn('在%d拍, bass第%02d次打回，最后一个音是弦外音' % (beat_dx, generate_fail_time) + repr(bass_output[-64:]))
                beat_dx -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-8)]
                generate_fail_time += 1
                continue
        DiaryLog.warn('bass的组合输出: ' + repr(bass_pattern_output))
        DiaryLog.warn('bass的最终输出: ' + repr(bass_output))
        return bass_output
