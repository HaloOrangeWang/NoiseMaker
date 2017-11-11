from dataoutputs.validation import BassCheck, BassEndCheck
from models.configs import BassConfig
from datainputs.bass import BassTrainData
from datainputs.melody import MelodyPatternEncode
from settings import *
from dataoutputs.predictions import MusicPatternPrediction
from interfaces.functions import MusicPatternDecode
from models.LstmPipeline import BaseLstmPipeline


class BassPipeline(BaseLstmPipeline):

    def __init__(self, melody_pattern_data, continuous_bar_number_data, chord_data):
        self.train_data = BassTrainData(melody_pattern_data, continuous_bar_number_data, chord_data)
        super().__init__()

    def prepare(self):
        self.config = BassConfig()
        self.test_config = BassConfig()
        self.test_config.batch_size = 1
        self.variable_scope_name = 'BassModel'

    def generate(self, session, melody_output, chord_output, common_melody_patterns):
        bass_output = []  # bass输出
        bass_pattern_output = []  # bass输出的pattern形式
        # 1.数据预处理 将melody_output转化为pattern
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dict = MelodyPatternEncode(common_melody_patterns, melody_pattern_input, 1 / 8, 1).music_pattern_dict
        melody_output_pattern = []  # 把melody_output编码为melody_pattern的形式
        generate_fail_time = 0  # 本次bass的生成一共被打回去多少次
        for key in range(round(len(melody_output) / 32)):  # 遍历这首歌的所有小节 把melody_pattern由dict形式转成list形式
            melody_output_pattern += melody_pattern_dict[key]
        # 2.逐拍生成数据
        generate_bass_iterator = 1
        while generate_bass_iterator <= len(melody_output_pattern):  # 遍历整个主旋律 步长是1拍
            # 2.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_BASS_GENERATE_FAIL_TIME:
                print('bass restart ', generate_fail_time)
                return None
            # 2.2.生成输入数据
            # 2.2.1.生成当前时间的编码,过去9拍的主旋律和过去9拍的和弦
            bass_prediction_input = [(generate_bass_iterator - 1) % 8]  # 先保存当前时间的编码
            if generate_bass_iterator < 9:  # 还没有到第9拍 旋律输入在前面补0 否则使用对应位置的旋律和和弦
                last_bars_melody = [0 for t in range(9 - generate_bass_iterator)] + melody_output_pattern[:generate_bass_iterator]
                last_bars_chord = [0 for t in range(9 - generate_bass_iterator)] + chord_output[:generate_bass_iterator]
            else:
                last_bars_melody = melody_output_pattern[(generate_bass_iterator - 9): generate_bass_iterator]
                last_bars_chord = chord_output[(generate_bass_iterator - 9): generate_bass_iterator]
            bass_prediction_input += last_bars_melody + last_bars_chord
            # 2.2.2.生成过去8拍的bass 如果不足8拍则补0
            required_bass_length = 4 * TRAIN_BASS_IO_BARS
            if len(bass_pattern_output) < required_bass_length:
                last_bars_bass = [0 for t in range(required_bass_length - len(bass_pattern_output))] + bass_pattern_output
            else:
                last_bars_bass = bass_pattern_output[-required_bass_length:]
            bass_prediction_input += last_bars_bass
            # print('          input', generate_bass_iterator, bass_prediction_input)
            # 2.3.生成输出数据
            bass_predict = self.predict(session, [bass_prediction_input])  # LSTM预测 得到二维数组predict
            if generate_bass_iterator % 8 == 1:  # 每两小节的第一拍不能为空
                bass_out_pattern = MusicPatternPrediction(bass_predict, 1, COMMON_BASS_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的bass组合bass_out_pattern
            else:
                bass_out_pattern = MusicPatternPrediction(bass_predict, 0, COMMON_BASS_PATTERN_NUMBER)
            bass_pattern_output.append(bass_out_pattern)  # 添加到最终的和弦输出列表中
            bass_output += MusicPatternDecode(self.train_data.common_bass_patterns, [bass_out_pattern], 1 / 8, 1)  # 将新生成的鼓点组合解码为音符列表
            # 添加到最终的和弦输出列表中
            # print(list(bass_predict[-1]))
            # print(bass_output)
            # print('\n\n')
            generate_bass_iterator += 1  # 步长是1拍
            # 2.4.检查生成的bass
            if generate_bass_iterator >= 9 and generate_bass_iterator % 4 == 1 and not BassCheck(bass_output[-64:], chord_output[(generate_bass_iterator - 9): (generate_bass_iterator - 1)]):  # bass与同时期的和弦差异过大
                print(generate_bass_iterator, 'bass第%02d次打回，与同时期和弦差异太大' % generate_fail_time, bass_output)
                generate_bass_iterator -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-8)]
                generate_fail_time += 1
                continue
            if generate_bass_iterator > len(melody_output_pattern) and not BassEndCheck(bass_output):  # 最后一个和弦不是1级大和弦
                print(generate_bass_iterator, 'bass第%02d次打回，最后一个音是弦外音' % generate_fail_time, bass_output)
                generate_bass_iterator -= 8  # 检查不合格 重新生成这两小节的bass
                bass_output = bass_output[:(-64)]
                bass_pattern_output = bass_pattern_output[:(-8)]
                generate_fail_time += 1
                continue
        print('bass_pattern_output', bass_pattern_output)
        print('bass_output', bass_output)
        return bass_output
