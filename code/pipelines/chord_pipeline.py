from dataoutputs.validation import chord_check, GetChordConfidenceLevel
from models.configs import ChordConfig, ChordConfig2, ChordConfig3, ChordConfig4
from datainputs.chord import ChordTrainData, ChordTrainDataCheck, ChordTrainData2, ChordTrainData3, ChordTrainData4
from datainputs.melody import MelodyPatternEncode, melody_core_note_for_chord, CoreNotePatternEncode
from settings import *
from pipelines.functions import BaseLstmPipeline, BaseLstmPipelineMultiCode, chord_prediction, chord_prediction_3, pat_predict_addcode
from interfaces.utils import DiaryLog
from interfaces.music_patterns import music_pattern_decode
import tensorflow as tf
import numpy as np
import copy


class ChordPipeline(BaseLstmPipeline):

    def __init__(self, melody_pattern_data, continuous_bar_data):
        self.train_data = ChordTrainData(melody_pattern_data, continuous_bar_data)
        super().__init__()

    def prepare(self):
        self.config = ChordConfig()
        self.test_config = ChordConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'ChordModel'

    def valid(self, predict_matrix, batches_output, pattern_number):
        """和弦的验证方法是最后两个 而不是最后一个。另外，pattern_number这个参数没有用"""
        predict_value = predict_matrix[:, -2:]
        right_value = batches_output[:, -2:]
        right_num = sum((predict_value == right_value).flatten())
        wrong_num = sum((predict_value != right_value).flatten())
        return right_num, wrong_num

    def generate(self, session, melody_output, common_melody_pats, tone_restrict=TONE_MAJOR):
        # 1.数据预处理 将melody_output转化为pattern
        chord_output = []  # 和弦输出
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dic = MelodyPatternEncode(common_melody_pats, melody_pattern_input, 0.125, 1).music_pattern_dic
        melody_output_pattern = []  # 把melody_output编码为melody_pattern的形式
        generate_fail_time = 0  # 本次和弦的生成一共被打回去多少次
        for key in range(round(len(melody_output) / 32)):  # 遍历这首歌的所有小节
            melody_output_pattern.extend(melody_pattern_dic[key])
        # 2.逐2拍生成数据
        beat_dx = 2
        while beat_dx <= len(melody_output_pattern):  # 遍历整个主旋律 步长是2拍
            # 2.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_CHORD_GENERATE_FAIL_TIME:
                DiaryLog.warn('和弦被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 2.2.如果这两拍没有主旋律 和弦直接沿用之前的
            if melody_output_pattern[(beat_dx - 2): beat_dx] == [0, 0]:  # 最近这2拍没有主旋律 则和弦沿用之前的
                if len(chord_output) == 0:
                    chord_output.extend([0, 0])
                else:
                    chord_output.extend([chord_output[-1] for t in range(2)])
            else:
                # 2.3.生成输入数据
                chord_prediction_input = [((beat_dx - 2) % 8) // 2]  # 先保存当前时间的编码
                required_chord_length = 4 * TRAIN_CHORD_IO_BARS
                if len(chord_output) < required_chord_length:
                    last_bars_chord = [0 for t in range(required_chord_length - len(chord_output))] + chord_output
                else:
                    last_bars_chord = chord_output[-required_chord_length:]
                if beat_dx < 10:  # 还没有到第十拍 旋律输入在前面补0 否则使用对应位置的旋律
                    last_bars_melody = [0 for t in range(10 - beat_dx)] + melody_output_pattern[:beat_dx]
                else:
                    last_bars_melody = melody_output_pattern[(beat_dx - 10): beat_dx]
                chord_prediction_input += last_bars_melody + last_bars_chord
                # print('          input', melody_iterator, chord_prediction_input)
                # 2.4.生成输出数据
                chord_predict = self.predict(session, [chord_prediction_input])  # LSTM预测 得到二维数组predict
                chord_out = chord_prediction(chord_predict)  # 将二维数组predict通过概率随机生成一维数组chord_out_vector，这个数组就是这两小节的和弦。每两小节生成一次和弦
                chord_output.extend([chord_out for t in range(2)])  # 添加到最终的和弦输出列表中
                # print(list(chord_predict[-2]))
                # print(list(chord_predict[-1]))
                # print(chord_output)
                # print('\n\n')
            beat_dx += 2  # 步长是2拍
            # 2.5.检查生成的和弦
            if beat_dx >= 10 and beat_dx % 4 == 2 and not chord_check(chord_output[-8:], melody_output[(beat_dx - 10) * 8: (beat_dx - 2) * 8]):  # 离调和弦的比例过高
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，离调和弦比例过高' % (beat_dx, generate_fail_time) + repr(chord_output))
                beat_dx -= 8
                chord_output = chord_output[:(-8)]
                generate_fail_time += 1
                continue
            if beat_dx >= 14 and len(set(chord_output[-12:])) == 1:  # 连续三小节和弦不变化
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，连续3小节和弦未变化' % (beat_dx, generate_fail_time) + repr(chord_output))
                beat_dx -= 12
                chord_output = chord_output[:(-12)]
                generate_fail_time += 1
                continue
            if beat_dx > len(melody_output_pattern) and chord_output[-1] != 1:  # 最后一个和弦不是1级大和弦
                if (tone_restrict == TONE_MAJOR and chord_output[-1] != 1) or (tone_restrict == TONE_MINOR and chord_output[-1] != 56):
                    DiaryLog.warn('在第%d拍, 和弦第%02d次打回，收束不是1级大和弦或6级小和弦' % (beat_dx, generate_fail_time) + repr(chord_output))
                    beat_dx -= 8
                    chord_output = chord_output[:(-8)]
                    generate_fail_time += 1
                    continue
        DiaryLog.warn('和弦的输出: ' + repr(chord_output) + '\n\n\n')
        return chord_output


class ChordPipelineCheck(BaseLstmPipeline):

    def __init__(self, melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord):
        self.train_data = ChordTrainDataCheck(melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord)
        super(ChordPipelineCheck, self).__init__()

    def prepare(self):
        self.config = ChordConfig()
        self.test_config = ChordConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'ChordModel'

    def valid(self, predict_matrix, batches_output, pattern_number):
        """和弦的验证方法是最后两个 而不是最后一个。另外，pattern_number这个参数没有用"""
        predict_value = predict_matrix[:, -2:]
        right_value = batches_output[:, -2:]
        right_num = sum((predict_value == right_value).flatten())
        wrong_num = sum((predict_value != right_value).flatten())
        return right_num, wrong_num

    def generate_check(self, session, common_melody_pats, common_corenote_pats):
        """多步预测的校验方式"""
        print('校验开始')
        right_value = [0 for t in range(4)]
        wrong_value = [0 for t in range(4)]
        sample_amount = 250
        # 1.从用于校验的数据中随机选取250组作为校验用例
        rand_dx = np.random.permutation(len(self.train_data.check_melody_data))
        rand_dx = rand_dx[:sample_amount]
        check_melody_pat_ary = np.array(self.train_data.check_melody_data)[rand_dx]
        check_raw_melody_ary = np.array(self.train_data.check_raw_melody_data)[rand_dx]
        check_chord_input_ary = np.array(self.train_data.check_chord_input_data)[rand_dx]
        check_chord_output_ary = np.array(self.train_data.check_chord_output_data)[rand_dx]
        rd_time_add_ary = np.array(self.train_data.time_add_data)[rand_dx]
        # 2.定义计算0.9置信区间和检验的方法
        confidence_level = GetChordConfidenceLevel(self.train_data.transfer_count, self.train_data.real_transfer_count)
        # 3.对这250组进行校验
        for check_it in range(sample_amount):
            print(check_it)
            real_chord_output_ary = check_chord_input_ary[check_it]  # 模型的和弦输出情况

            core_note_ary = melody_core_note_for_chord({t // 32: check_raw_melody_ary[check_it][-64:][t: t + 32] for t in range(0, 64, 32)})  # 提取出这几拍主旋律的骨干音
            core_note_pat_ary = CoreNotePatternEncode(common_corenote_pats, core_note_ary, 0.125, 2).music_pattern_ary
            confidence_level.get_loss09(session, core_note_pat_ary)  # 计算这段主旋律的和弦0.9置信区间
            generate_fail_time = 0  # 生成失败的次数
            chord_choose_bk = [1, 1, 1, 1]  # 备选和弦。为防止死循环，如果连续十次验证失败，则使用备选和弦
            loss_bk = np.inf  # 备选方案对应的损失函数
            while True:
                for beat_dx in range(0, 8, 2):  # 逐2拍生成数据
                    # 3.1.生成输入数据
                    chord_prediction_input = [rd_time_add_ary[check_it] + beat_dx // 2]  # 先保存当前时间的编码
                    last_bars_chord = real_chord_output_ary[-8:]
                    last_bars_melody = check_melody_pat_ary[check_it][beat_dx: beat_dx + 10]
                    chord_prediction_input = np.append(chord_prediction_input, np.append(last_bars_melody, last_bars_chord))
                    # print('          input', melody_iterator, chord_prediction_input)
                    # 3.2.生成输出数据
                    chord_predict = self.predict(session, [chord_prediction_input])  # LSTM预测 得到二维数组predict
                    chord_out = chord_prediction(chord_predict)  # 将二维数组predict通过概率随机生成一维数组chord_out_vector，这个数组就是这两小节的和弦。每两小节生成一次和弦
                    real_chord_output_ary = np.append(real_chord_output_ary, [chord_out for t in range(2)])  # 添加到最终的和弦输出列表中
                # 3.3.根据一些数学方法和乐理对输出的和弦进行自我检查。如果检查不合格则打回重新生成
                # if check_it <= 10:
                #     print('主旋律输入为', check_raw_melody_ary[check_it], '骨干音列表为', core_note_ary, '骨干音组合为', core_note_pat_ary)
                chord_per_2beats = [real_chord_output_ary[4 * TRAIN_CHORD_IO_BARS:][t] for t in range(0, len(real_chord_output_ary[4 * TRAIN_CHORD_IO_BARS:]), 2)]  # 生成的和弦从每1拍一个转化为每两拍一个
                # if check_it <= 10:
                #     print('90%置信区间为', confidence_level.loss09, '和弦的两拍输入为', chord_per_2beats)
                # if chord_check(real_chord_output_ary[4 * TRAIN_CHORD_IO_BARS:], check_molody_ary[-64:]):
                check_res, loss_value = confidence_level.check_chord_ary(session, check_raw_melody_ary[check_it][-64:], core_note_pat_ary, chord_per_2beats)
                # if check_it <= 10:
                #     print('该和弦组合的交叉熵损失函数为', x2, '校验结果为', x1)
                if check_res:
                    break
                else:
                    real_chord_output_ary = check_chord_input_ary[check_it]  # 重新生成时，需要把原先改变的一些变量改回去
                    generate_fail_time += 1
                    chord_choose_bk = chord_per_2beats
                    loss_bk = loss_value
                    if generate_fail_time >= 10:
                        print('和弦使用备选方案,损失为', loss_bk)
                        real_chord_output_ary = np.append(real_chord_output_ary, np.repeat(chord_choose_bk, 2))
                        break
            # 3.4.对输出数据逐两拍进行校验 只要符合真实这两拍和弦的其一即可
            for beat_dx in range(0, 8, 2):  # 逐2拍进行校验
                if check_chord_output_ary[check_it][beat_dx] != 0 or check_chord_output_ary[check_it][beat_dx + 1] != 0:
                    if real_chord_output_ary[beat_dx + 4 * TRAIN_CHORD_IO_BARS] == check_chord_output_ary[check_it][beat_dx] or real_chord_output_ary[beat_dx + 4 * TRAIN_CHORD_IO_BARS] == check_chord_output_ary[check_it][beat_dx + 1]:
                        right_value[beat_dx // 2] += 1
                    else:
                        wrong_value[beat_dx // 2] += 1
            # if check_it < 10:
            #     print('melody input', check_melody_pat_ary[check_it])
            #     print('chord input', np.append(check_chord_input_ary[check_it], check_chord_output_ary[check_it]))
            #     print('real out', real_chord_output_ary)
            #     print('\n')
        # 4.输出
        for step_it in range(4):
            right_ratio = right_value[step_it] / (right_value[step_it] + wrong_value[step_it])
            print('第%d个两拍正确的chord数量为%d,错误的chord数量为%d,正确率为%.4f.\n' % (step_it, right_value[step_it], wrong_value[step_it], right_ratio))

    def generate(self, session, melody_output, common_melody_pats, common_corenote_pats, tone_restrict=TONE_MAJOR):
        # 1.数据预处理 将melody_output转化为pattern
        chord_output = []  # 和弦输出
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dic = MelodyPatternEncode(common_melody_pats, melody_pattern_input, 0.125, 1).music_pattern_dic
        melody_output_pattern = []  # 把melody_output编码为melody_pattern的形式
        generate_fail_time = 0  # 本次和弦的生成一共被打回去多少次
        for key in range(round(len(melody_output) / 32)):  # 遍历这首歌的所有小节
            melody_output_pattern.extend(melody_pattern_dic[key])
        # 2.定义计算0.9置信区间和检验的方法和置信区间验证法的相关数据 包括多次验证失败时的备选数据
        confidence_level = GetChordConfidenceLevel(self.train_data.transfer_count, self.train_data.real_transfer_count)
        confidence_fail_time = [0 for t in range(MAX_GENERATE_BAR_NUMBER)]
        chord_choose_bk = [[1, 1, 1, 1] for t in range(MAX_GENERATE_BAR_NUMBER)]  # 备选和弦。为防止死循环，如果连续十次验证失败，则使用备选和弦
        loss_bk = [np.inf for t in range(MAX_GENERATE_BAR_NUMBER)]  # 备选方案对应的损失函数
        # 3.逐2拍生成数据
        beat_dx = 2
        while beat_dx <= len(melody_output_pattern):  # 遍历整个主旋律 步长是2拍
            # 3.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_CHORD_GENERATE_FAIL_TIME:
                DiaryLog.warn('和弦被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 3.2.如果这两拍没有主旋律 和弦直接沿用之前的
            if melody_output_pattern[(beat_dx - 2): beat_dx] == [0, 0]:  # 最近这2拍没有主旋律 则和弦沿用之前的
                if len(chord_output) == 0:
                    chord_output.extend([0, 0])
                else:
                    chord_output.extend([chord_output[-1] for t in range(2)])
            else:
                # 3.3.生成输入数据
                chord_prediction_input = [((beat_dx - 2) % 8) // 2]  # 先保存当前时间的编码
                required_chord_length = 4 * TRAIN_CHORD_IO_BARS
                if len(chord_output) < required_chord_length:
                    last_bars_chord = [0 for t in range(required_chord_length - len(chord_output))] + chord_output
                else:
                    last_bars_chord = chord_output[-required_chord_length:]
                if beat_dx < 10:  # 还没有到第十拍 旋律输入在前面补0 否则使用对应位置的旋律
                    last_bars_melody = [0 for t in range(10 - beat_dx)] + melody_output_pattern[:beat_dx]
                else:
                    last_bars_melody = melody_output_pattern[(beat_dx - 10): beat_dx]
                chord_prediction_input += last_bars_melody + last_bars_chord
                # print('          input', melody_iterator, chord_prediction_input)
                # 3.4.生成输出数据
                chord_predict = self.predict(session, [chord_prediction_input])  # LSTM预测 得到二维数组predict
                chord_out = chord_prediction(chord_predict)  # 将二维数组predict通过概率随机生成一维数组chord_out_vector，这个数组就是这两小节的和弦。每两小节生成一次和弦
                chord_output.extend([chord_out for t in range(2)])  # 添加到最终的和弦输出列表中
                # print(list(chord_predict[-2]))
                # print(list(chord_predict[-1]))
                # print(chord_output)
                # print('\n\n')
            beat_dx += 2  # 步长是2拍
            # 3.5.检查生成的和弦
            if beat_dx >= 10 and beat_dx % 4 == 2 and not chord_check(chord_output[-8:], melody_output[(beat_dx - 10) * 8: (beat_dx - 2) * 8]):  # 离调和弦的比例过高
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，离调和弦比例过高' % (beat_dx, generate_fail_time) + repr(chord_output))
                beat_dx -= 8
                chord_output = chord_output[:(-8)]
                generate_fail_time += 1
                continue
            if beat_dx >= 10 and beat_dx % 8 == 2:  # 置信区间校验法
                core_note_ary = melody_core_note_for_chord({(t + 80 - beat_dx * 8) // 32: melody_output[t: t + 32] for t in range(beat_dx * 8 - 80, beat_dx * 8 - 16, 32)})  # 提取出这几拍主旋律的骨干音
                core_note_pat_ary = CoreNotePatternEncode(common_corenote_pats, core_note_ary, 0.125, 2).music_pattern_ary
                confidence_level.get_loss09(session, core_note_pat_ary)  # 计算0.9置信区间
                chord_per_2beats = [chord_output[-8:][t] for t in range(0, len(chord_output[-8:]), 2)]  # 生成的和弦从每1拍一个转化为每两拍一个
                check_res, loss_value = confidence_level.check_chord_ary(session, melody_output[(beat_dx - 10) * 8: (beat_dx - 2) * 8], core_note_pat_ary, chord_per_2beats)
                if not check_res:  # 测试不通过
                    DiaryLog.warn('在第%d拍, 和弦第%d次未通过置信区间验证。和弦的损失函数值为%.4f，而置信区间为%.4f' % (beat_dx, confidence_fail_time[(beat_dx - 2) // 4], loss_value, confidence_level.loss09))
                    beat_dx -= 8
                    chord_output = chord_output[:(-8)]
                    confidence_fail_time[(beat_dx + 6) // 4] += 1  # 前面减了8之后 这里beat_dx-2都应该变成beat_dx+6
                    if loss_value < loss_bk[(beat_dx + 6) // 4]:
                        chord_choose_bk[(beat_dx + 6) // 4] = chord_per_2beats
                        loss_bk[(beat_dx + 6) // 4] = loss_value
                    if confidence_fail_time[(beat_dx + 6) // 4] >= 10:
                        print('在第%d拍，和弦使用备选方案,损失为%.4f' % (beat_dx, loss_bk[(beat_dx + 6) // 4]))
                        chord_output += list(np.repeat(chord_choose_bk[(beat_dx + 6) // 4], 2))
                        beat_dx += 8
                    else:
                        continue
            if beat_dx >= 14 and len(set(chord_output[-12:])) == 1:  # 连续三小节和弦不变化
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，连续3小节和弦未变化' % (beat_dx, generate_fail_time) + repr(chord_output))
                beat_dx -= 12
                chord_output = chord_output[:(-12)]
                generate_fail_time += 1
                continue
            if beat_dx > len(melody_output_pattern) and chord_output[-1] != 1:  # 最后一个和弦不是1级大和弦
                if (tone_restrict == TONE_MAJOR and chord_output[-1] != 1) or (tone_restrict == TONE_MINOR and chord_output[-1] != 56):
                    DiaryLog.warn('在第%d拍, 和弦第%02d次打回，收束不是1级大和弦或6级小和弦' % (beat_dx, generate_fail_time) + repr(chord_output))
                    beat_dx -= 8
                    chord_output = chord_output[:(-8)]
                    generate_fail_time += 1
                    continue
        DiaryLog.warn('和弦的输出: ' + repr(chord_output) + '\n\n\n')
        return chord_output


class ChordPipeline2(ChordPipelineCheck):

    def __init__(self, melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord):
        self.train_data = ChordTrainData2(melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord)
        super(ChordPipelineCheck, self).__init__()

    def prepare(self):
        self.config = ChordConfig2()
        self.test_config = ChordConfig2()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'ChordModel'


class ChordPipeline3(BaseLstmPipelineMultiCode):

    def __init__(self, melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord):
        self.train_data = ChordTrainData3(melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord)
        super().__init__(2)  # chord输入数据的编码为二重编码

    def prepare(self):
        self.config = ChordConfig3()
        self.test_config = ChordConfig3()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'ChordModel'

    def generate(self, session, melody_output, common_melody_pats, common_corenote_pats, tone_restrict=TONE_MAJOR):

        # 1.数据预处理 将melody_output转化为pattern
        chord_output = []  # 和弦输出
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dic = MelodyPatternEncode(common_melody_pats, melody_pattern_input, 0.125, 1).music_pattern_dic
        melody_output_pattern = []  # 把melody_output编码为melody_pattern的形式
        generate_fail_time = 0  # 本次和弦的生成一共被打回去多少次
        for key in range(round(len(melody_output) / 32)):  # 遍历这首歌的所有小节
            melody_output_pattern.extend(melody_pattern_dic[key])
        # 2.定义计算0.9置信区间和检验的方法和置信区间验证法的相关数据 包括多次验证失败时的备选数据
        confidence_level = GetChordConfidenceLevel(self.train_data.transfer_count, self.train_data.real_transfer_count)
        confidence_fail_time = [0 for t in range(MAX_GENERATE_BAR_NUMBER)]
        chord_choose_bk = [[1, 1, 1, 1] for t in range(MAX_GENERATE_BAR_NUMBER)]  # 备选和弦。为防止死循环，如果连续十次验证失败，则使用备选和弦
        loss_bk = [np.inf for t in range(MAX_GENERATE_BAR_NUMBER)]  # 备选方案对应的损失函数
        melody_code_add_base = 8
        chord_code_add_base = 8 + COMMON_MELODY_PATTERN_NUMBER + 2  # 和弦数据编码增加的基数
        # 3.逐2拍生成数据
        beat_dx = 2
        while beat_dx <= len(melody_output_pattern):  # 遍历整个主旋律 步长是2拍
            # 3.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_CHORD_GENERATE_FAIL_TIME:
                DiaryLog.warn('和弦被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 3.2.如果这两拍没有主旋律 和弦直接沿用之前的
            if melody_output_pattern[(beat_dx - 2): beat_dx] == [0, 0]:  # 最近这2拍没有主旋律 则和弦沿用之前的
                if len(chord_output) == 0:
                    chord_output.extend([0, 0])
                else:
                    chord_output.extend([chord_output[-1] for t in range(2)])
            else:
                # 3.3.生成输入数据
                chord_prediction_input = list()
                for backward_beat_dx in range(beat_dx - 8, beat_dx):
                    if backward_beat_dx < 0:
                        chord_prediction_input.append([backward_beat_dx % 4, melody_code_add_base])  # 这一拍的时间编码 这一拍的主旋律
                    else:
                        chord_prediction_input.append([backward_beat_dx % 8, melody_code_add_base + melody_output_pattern[backward_beat_dx]])
                # 3.4.生成输出数据
                chord_predict = self.predict(session, [chord_prediction_input])  # LSTM预测 得到二维数组predict
                chord_out = chord_prediction_3(chord_predict, chord_code_add_base)  # 将二维数组predict通过概率随机生成一维数组chord_out_vector，这个数组就是这两小节的和弦。每两小节生成一次和弦
                chord_output.extend([chord_out for t in range(2)])  # 添加到最终的和弦输出列表中
                # print(list(chord_predict[-2]))
                # print(list(chord_predict[-1]))
                # print(chord_output)
                # print('\n\n')
            beat_dx += 2  # 步长是2拍
            # 3.5.检查生成的和弦
            if beat_dx >= 10 and beat_dx % 4 == 2 and not chord_check(chord_output[-8:], melody_output[(beat_dx - 10) * 8: (beat_dx - 2) * 8]):  # 离调和弦的比例过高
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，离调和弦比例过高' % (beat_dx, generate_fail_time) + repr(chord_output))
                beat_dx -= 8
                chord_output = chord_output[:(-8)]
                generate_fail_time += 1
                continue
            if beat_dx >= 10 and beat_dx % 8 == 2:  # 置信区间校验法
                core_note_ary = melody_core_note_for_chord({(t + 80 - beat_dx * 8) // 32: melody_output[t: t + 32] for t in range(beat_dx * 8 - 80, beat_dx * 8 - 16, 32)})  # 提取出这几拍主旋律的骨干音
                core_note_pat_ary = CoreNotePatternEncode(common_corenote_pats, core_note_ary, 0.125, 2).music_pattern_ary
                confidence_level.get_loss09(session, core_note_pat_ary)  # 计算0.9置信区间
                chord_per_2beats = [chord_output[-8:][t] for t in range(0, len(chord_output[-8:]), 2)]  # 生成的和弦从每1拍一个转化为每两拍一个
                check_res, loss_value = confidence_level.check_chord_ary(session, melody_output[(beat_dx - 10) * 8: (beat_dx - 2) * 8], core_note_pat_ary, chord_per_2beats)
                if not check_res:  # 测试不通过
                    DiaryLog.warn('在第%d拍, 和弦第%d次未通过置信区间验证。和弦的损失函数值为%.4f，而置信区间为%.4f' % (beat_dx, confidence_fail_time[(beat_dx - 2) // 4], loss_value, confidence_level.loss09))
                    beat_dx -= 8
                    chord_output = chord_output[:(-8)]
                    confidence_fail_time[(beat_dx + 6) // 4] += 1  # 前面减了8之后 这里beat_dx-2都应该变成beat_dx+6
                    if loss_value < loss_bk[(beat_dx + 6) // 4]:
                        chord_choose_bk[(beat_dx + 6) // 4] = chord_per_2beats
                        loss_bk[(beat_dx + 6) // 4] = loss_value
                    if confidence_fail_time[(beat_dx + 6) // 4] >= 10:
                        print('在第%d拍，和弦使用备选方案,损失为%.4f' % (beat_dx, loss_bk[(beat_dx + 6) // 4]))
                        chord_output += list(np.repeat(chord_choose_bk[(beat_dx + 6) // 4], 2))
                        beat_dx += 8
                    else:
                        continue
            if beat_dx >= 14 and len(set(chord_output[-12:])) == 1:  # 连续三小节和弦不变化
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，连续3小节和弦未变化' % (beat_dx, generate_fail_time) + repr(chord_output))
                beat_dx -= 12
                chord_output = chord_output[:(-12)]
                generate_fail_time += 1
                continue
            if beat_dx > len(melody_output_pattern) and chord_output[-1] != 1:  # 最后一个和弦不是1级大和弦
                if (tone_restrict == TONE_MAJOR and chord_output[-1] != 1) or (tone_restrict == TONE_MINOR and chord_output[-1] != 56):
                    DiaryLog.warn('在第%d拍, 和弦第%02d次打回，收束不是1级大和弦或6级小和弦' % (beat_dx, generate_fail_time) + repr(chord_output))
                    beat_dx -= 8
                    chord_output = chord_output[:(-8)]
                    generate_fail_time += 1
                    continue
        DiaryLog.warn('和弦的输出: ' + repr(chord_output) + '\n\n\n')
        return chord_output


class ChordPipeline4(BaseLstmPipelineMultiCode):

    def __init__(self, melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord):
        self.train_data = ChordTrainData4(melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord)
        self.confidence_level = GetChordConfidenceLevel(self.train_data.transfer_count, self.train_data.real_transfer_count)
        super().__init__(4)  # chord输入数据的编码为四重编码

    def prepare(self):
        self.config = ChordConfig4(self.train_data.cc_pat_num)
        self.test_config = ChordConfig4(self.train_data.cc_pat_num)
        self.test_config.batch_size = 1
        self.variable_scope_name = 'ChordModel'

    def choose_best_chord(self, chord_ary, melody_list):
        """
        从多个候选和弦中选择一个匹配此主旋律最佳的。选择标准为和弦的常见程度以及和主旋律的匹配程度
        :param chord_ary: 一个和弦列表
        :param melody_list: 同期主旋律数据（绝对音高形式）
        :return: 返回整合成一个的和弦及其在cc_pattern_list中的位置
        """
        fix_chord_list = [1, 14, 26, 31, 43, 56]  # 常见的和弦列表
        chord_ary = [t for t in chord_ary if [t, t] in self.train_data.cc_pat_count]  # 筛掉重复音不在cc_pat_count中的 如果两拍都不在则报错
        if len(chord_ary) == 0:
            raise ValueError
        fix_in_ary = np.array(chord_ary)[np.in1d(chord_ary, fix_chord_list)]
        if len(fix_in_ary) != 0:  # 剔掉不常见的和弦 如果都是不常见的则不提取
            common_chord_ary = fix_in_ary
        else:
            common_chord_ary = chord_ary
        max_match_score = 0
        max_match_dx = -1
        for chord_it in range(len(common_chord_ary)):
            contain_count = 0  # 有多少歌步长的音符包含在和弦里
            diff_count = 0  # 有多少歌步长的音符不包含在和弦里
            last_note_contain = -1  # 上一个主旋律音符是否在此和弦内 1表示在 0表示不在 -1表示不清楚
            chord_set = copy.deepcopy(CHORD_DICT[common_chord_ary[chord_it]])
            if 1 <= common_chord_ary[chord_it] <= 72 and common_chord_ary[chord_it] % 6 == 1:  # 大三和弦 chord_set增加大七度和小七度
                chord_set.add((common_chord_ary[chord_it] // 6 + 10) % 12)
                chord_set.add((common_chord_ary[chord_it] // 6 + 11) % 12)
            if 1 <= common_chord_ary[chord_it] <= 72 and common_chord_ary[chord_it] % 6 == 2:  # 小三和弦 chord_set增加小七度
                chord_set.add((common_chord_ary[chord_it] // 6 + 10) % 12)
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
        return [common_chord_ary[max_match_dx], common_chord_ary[max_match_dx]], self.train_data.cc_pat_count.index([common_chord_ary[max_match_dx], common_chord_ary[max_match_dx]])

    def generate(self, session, melody_output, common_melody_pats, common_corenote_pats, melody_beat_num, tone_restrict=TONE_MAJOR):
        # 1.数据预处理 将melody_output转化为pattern
        chord_output = []  # 和弦输出
        cc_pat_output = []  # 和弦两拍编码的输出
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dic = MelodyPatternEncode(common_melody_pats, melody_pattern_input, 0.125, 1).music_pattern_dic
        melody_output_pattern = []  # 把melody_output编码为melody_pattern的形式
        generate_fail_time = 0  # 本次和弦的生成一共被打回去多少次
        bar_num = len(melody_output) // 32  # 主旋律一共有多少个小节
        for key in range(round(len(melody_output) / 32)):  # 遍历这首歌的所有小节
            melody_output_pattern.extend(melody_pattern_dic[key])
        # 2.定义计算0.9置信区间和检验的方法和置信区间验证法的相关数据 包括多次验证失败时的备选数据
        # confidence_?level = GetChordConfidenceLevel(self.train_data.transfer_count, self.train_data.real_transfer_count)
        confidence_fail_time = [0 for t in range(bar_num + 1)]
        chord_choose_bk = [[1, 1, 1, 1] for t in range(bar_num + 1)]  # 备选和弦。为防止死循环，如果连续十次验证失败，则使用备选和弦
        cc_pat_choose_bk = [[1, 1, 1, 1] for t in range(bar_num + 1)]
        loss_bk = [np.inf for t in range(bar_num + 1)]  # 备选方案对应的损失函数
        melody1_code_add_base = 4  # 主旋律第一拍数据编码增加的基数
        melody2_code_add_base = 4 + COMMON_MELODY_PATTERN_NUMBER + 2  # 主旋律第二拍数据编码增加的基数
        chord_code_add_base = 4 + (COMMON_MELODY_PATTERN_NUMBER + 2) * 2  # 和弦数据编码增加的基数
        # 3.逐2拍生成数据
        beat_dx = 2
        while beat_dx <= len(melody_output_pattern):  # 遍历整个主旋律 步长是2拍
            # 3.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_CHORD_GENERATE_FAIL_TIME:
                DiaryLog.warn('和弦被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 3.2.如果这两拍没有主旋律 和弦直接沿用之前的
            if melody_output_pattern[(beat_dx - 2): beat_dx] == [0, 0]:  # 最近这2拍没有主旋律 则和弦沿用之前的
                if len(chord_output) == 0:
                    chord_output.extend([0, 0])
                else:
                    chord_output.extend([chord_output[-1] for t in range(2)])
                    cc_pat_output.append(cc_pat_output[-1])
            else:
                # 3.3.生成输入数据
                chord_prediction_input = list()
                for backward_beat_dx in range(beat_dx - 10, beat_dx, 2):
                    cur_step = backward_beat_dx // 2  # 第几个bass的步长
                    if backward_beat_dx < 0:
                        chord_prediction_input.append([cur_step % 2, melody1_code_add_base, melody2_code_add_base, chord_code_add_base])  # 这一拍的时间编码 这一拍的主旋律 下一拍的主旋律
                    elif backward_beat_dx < 2:
                        chord_prediction_input.append([cur_step % 4, melody_output_pattern[backward_beat_dx] + melody1_code_add_base, melody_output_pattern[backward_beat_dx + 1] + melody2_code_add_base, chord_code_add_base])
                    else:
                        chord_prediction_input.append([cur_step % 4, melody_output_pattern[backward_beat_dx] + melody1_code_add_base, melody_output_pattern[backward_beat_dx + 1] + melody2_code_add_base, cc_pat_output[cur_step - 1] + chord_code_add_base])
                # 3.4.生成输出数据
                chord_predict = self.predict(session, [chord_prediction_input])  # LSTM预测 得到二维数组predict
                chord_out = pat_predict_addcode(chord_predict, chord_code_add_base, 1, self.train_data.cc_pat_num)  # 将二维数组predict通过概率随机生成一维数组chord_out_vector，这个数组就是这两小节的和弦。每两小节生成一次和弦
                cc_pat_output.append(chord_out)  # 和弦-和弦编码的输出
                if self.train_data.cc_pat_count[chord_out][0] != self.train_data.cc_pat_count[chord_out][1]:
                    DiaryLog.warn('在第%d拍, 预测方法选择了两个不同的和弦，编码分别是%d和%d' % (beat_dx, self.train_data.cc_pat_count[chord_out][0], self.train_data.cc_pat_count[chord_out][1]))
                    try:
                        new_chord, chord_out = self.choose_best_chord(self.train_data.cc_pat_count[chord_out], melody_output[(beat_dx - 2) * 8: beat_dx * 8])
                    except ValueError:
                        DiaryLog.warn('在第%d拍, 和弦第%02d次打回，两拍和弦不相同且没法找到替代' % (beat_dx, generate_fail_time) + repr(chord_output))
                        beat_dx -= 2
                        cc_pat_output = cc_pat_output[:(-1)]
                        generate_fail_time += 1
                        continue
                    DiaryLog.warn('在第%d拍, 和弦已改为[%d, %d]' % (beat_dx, new_chord[0], new_chord[1]))
                cc_pat_output[-1] = chord_out
                chord_output.extend(self.train_data.cc_pat_count[chord_out])  # 添加到最终的和弦输出列表中
            beat_dx += 2  # 步长是2拍
            # 3.5.检查生成的和弦
            if beat_dx >= 10 and beat_dx % 4 == 2 and not chord_check(chord_output[-8:], melody_output[(beat_dx - 10) * 8: (beat_dx - 2) * 8]):  # 离调和弦的比例过高
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，离调和弦比例过高' % (beat_dx, generate_fail_time) + repr(chord_output))
                beat_dx -= 8
                chord_output = chord_output[:(-8)]
                cc_pat_output = cc_pat_output[:(-4)]
                generate_fail_time += 1
                continue
            if beat_dx >= 10 and beat_dx % 8 == 2:  # 置信区间校验法
                core_note_ary = melody_core_note_for_chord({(t + 80 - beat_dx * 8) // 32: melody_output[t: t + 32] for t in range(beat_dx * 8 - 80, beat_dx * 8 - 16, 32)})  # 提取出这几拍主旋律的骨干音
                core_note_pat_ary = CoreNotePatternEncode(common_corenote_pats, core_note_ary, 0.125, 2).music_pattern_ary
                self.confidence_level.get_loss09(session, core_note_pat_ary)  # 计算0.9置信区间
                chord_per_2beats = [chord_output[-8:][t] for t in range(0, len(chord_output[-8:]), 2)]  # 生成的和弦从每1拍一个转化为每两拍一个
                cc_pat_per_2beats = cc_pat_output[-4:]
                check_res, loss_value = self.confidence_level.check_chord_ary(session, melody_output[(beat_dx - 10) * 8: (beat_dx - 2) * 8], core_note_pat_ary, chord_per_2beats)
                if not check_res:  # 测试不通过
                    DiaryLog.warn('在第%d拍, 和弦第%d次未通过置信区间验证。和弦的损失函数值为%.4f，而置信区间为%.4f' % (beat_dx, confidence_fail_time[(beat_dx - 2) // 4], loss_value, self.confidence_level.loss09))
                    beat_dx -= 8
                    chord_output = chord_output[:(-8)]
                    cc_pat_output = cc_pat_output[:(-4)]
                    confidence_fail_time[(beat_dx + 6) // 4] += 1  # 前面减了8之后 这里beat_dx-2都应该变成beat_dx+6
                    if loss_value < loss_bk[(beat_dx + 6) // 4]:
                        chord_choose_bk[(beat_dx + 6) // 4] = chord_per_2beats
                        cc_pat_choose_bk[(beat_dx + 6) // 4] = cc_pat_per_2beats
                        loss_bk[(beat_dx + 6) // 4] = loss_value
                    if confidence_fail_time[(beat_dx + 6) // 4] >= 10:
                        print('在第%d拍，和弦使用备选方案,损失为%.4f' % (beat_dx, loss_bk[(beat_dx + 6) // 4]))
                        chord_part_ary = list(np.repeat(chord_choose_bk[(beat_dx + 6) // 4], 2))
                        for chord_it in range(len(chord_part_ary)):  # np.repeat生成的东西是int64型的 把它转化成int型
                            chord_part_ary[chord_it] = int(chord_part_ary[chord_it])
                        chord_output += chord_part_ary
                        cc_pat_output += cc_pat_choose_bk[(beat_dx + 6) // 4]
                        beat_dx += 8
                    else:
                        continue
            if beat_dx >= 14 and len(set(chord_output[-12:])) == 1:  # 连续三小节和弦不变化
                DiaryLog.warn('在第%d拍, 和弦第%02d次打回，连续3小节和弦未变化' % (beat_dx, generate_fail_time) + repr(chord_output))
                beat_dx -= 12
                chord_output = chord_output[:(-12)]
                cc_pat_output = cc_pat_output[:(-6)]
                generate_fail_time += 1
                continue
            if beat_dx in [melody_beat_num + 2, len(melody_output_pattern) + 2] and chord_output[-1] != 1:  # 最后一个和弦不是1级大和弦
                if (tone_restrict == TONE_MAJOR and chord_output[-1] != 1) or (tone_restrict == TONE_MINOR and chord_output[-1] != 56):
                    DiaryLog.warn('在第%d拍, 和弦第%02d次打回，收束不是1级大和弦或6级小和弦' % (beat_dx, generate_fail_time) + repr(chord_output))
                    beat_dx -= 8
                    chord_output = chord_output[:(-8)]
                    cc_pat_output = cc_pat_output[:(-4)]
                    generate_fail_time += 1
                    continue
        DiaryLog.warn('和弦的输出: ' + repr(chord_output) + '\n\n\n')
        return chord_output
