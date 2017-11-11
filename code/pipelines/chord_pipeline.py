from dataoutputs.validation import ChordCheck
from models.configs import ChordConfig
from datainputs.chord import ChordTrainData
from datainputs.melody import MelodyPatternEncode
from settings import *
from dataoutputs.predictions import ChordPrediction
from interfaces.functions import MusicPatternDecode, LastNotZeroNumber
from models.LstmPipeline import BaseLstmPipeline


class ChordPipeline(BaseLstmPipeline):

    def __init__(self, melody_pattern_data, continuous_bar_number_data):
        self.train_data = ChordTrainData(melody_pattern_data, continuous_bar_number_data)
        super().__init__()

    def prepare(self):
        self.config = ChordConfig()
        self.test_config = ChordConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'ChordModel'

    def generate(self, session, melody_output, common_melody_patterns):
        # 1.数据预处理 将melody_output转化为pattern
        chord_output = []  # 和弦输出
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dict = MelodyPatternEncode(common_melody_patterns, melody_pattern_input, MELODY_TIME_STEP, MELODY_PATTERN_TIME_STEP).music_pattern_dict
        melody_output_pattern = []  # 把ｍｅｌｏｄｙ_ｏｕｔｐｕｔ编码为melody_patteｒn的形式
        generate_fail_time = 0  # 本次和弦的生成一共被打回去多少次
        for key in range(round(len(melody_output) / 32)):  # 遍历这首歌的所有小节
            melody_output_pattern += melody_pattern_dict[key]
        # 2.逐CHORD_GENERATE_TIME_STEP拍生成数据
        melody_iterator = 2
        while melody_iterator <= len(melody_output_pattern):  # 遍历整个主旋律 步长是2拍
            # 2.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_CHORD_GENERATE_FAIL_TIME:
                print('chord restart ', generate_fail_time)
                return None
            # 2.2.如果这两拍没有主旋律 和弦直接沿用之前的
            if melody_output_pattern[(melody_iterator - 2): melody_iterator] == [0, 0]:  # 最近这2拍没有主旋律 则和弦沿用之前的
                if len(chord_output) == 0:
                    chord_output = chord_output + [0, 0]
                else:
                    chord_output = chord_output + [chord_output[-1] for t in range(2)]
            else:
                # 2.3.生成输入数据
                chord_prediction_input = [((melody_iterator - 2) % 8) // 2]  # 先保存当前时间的编码
                required_chord_length = 4 * TRAIN_CHORD_IO_BARS
                if len(chord_output) < required_chord_length:
                    last_bars_chord = [0 for t in range(required_chord_length - len(chord_output))] + chord_output
                else:
                    last_bars_chord = chord_output[-required_chord_length:]
                if melody_iterator < 10:  # 还没有到第十拍 旋律输入在前面补0 否则使用对应位置的旋律
                    last_bars_melody = [0 for t in range(10 - melody_iterator)] + melody_output_pattern[:melody_iterator]
                else:
                    last_bars_melody = melody_output_pattern[(melody_iterator - 10): melody_iterator]
                chord_prediction_input += last_bars_melody + last_bars_chord
                # print('          input', melody_iterator, chord_prediction_input)
                # 2.4.生成输出数据
                chord_predict = self.predict(session, [chord_prediction_input])  # LSTM预测 得到二维数组predict
                chord_out = ChordPrediction(chord_predict)  # 将二维数组predict通过概率随机生成一维数组chord_out_vector，这个数组就是这两小节的和弦。每两小节生成一次和弦
                chord_output = chord_output + [chord_out for t in range(2)]  # 添加到最终的和弦输出列表中
                # print(list(chord_predict[-2][:73]))
                # print(list(chord_predict[-1][:73]))
                # print(chord_output)
                # print('\n\n')
            melody_iterator += 2  # 步长是2拍
            # 2.5.检查生成的和弦
            if melody_iterator >= 10 and melody_iterator % 4 == 2 and not ChordCheck(chord_output[-8:], melody_output[(melody_iterator - 10) * 8: (melody_iterator - 2) * 8]):  # 离调和弦的比例过高
                print(melody_iterator, '和弦第%02d次打回，离调和弦比例过高' % generate_fail_time, chord_output)
                melody_iterator -= 8
                chord_output = chord_output[:(-8)]
                generate_fail_time += 1
                continue
            if melody_iterator >= 14 and len(set(chord_output[-12:])) == 1:  # 连续三小节和弦不变化
                print(melody_iterator, '和弦第%02d次打回，连续3小节和弦未变化' % generate_fail_time, chord_output)
                melody_iterator -= 12
                chord_output = chord_output[:(-12)]
                generate_fail_time += 1
                continue
            if melody_iterator > len(melody_output_pattern) and chord_output[-1] != 1:  # 最后一个和弦不是1级大和弦
                print(melody_iterator, '和弦第%02d次打回，收束不是1级大和弦' % generate_fail_time, chord_output)
                melody_iterator -= 8
                chord_output = chord_output[:(-8)]
                generate_fail_time += 1
                continue
        print('chord_output', chord_output)
        return chord_output
