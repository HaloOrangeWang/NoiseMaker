from models.configs import DrumConfig
from datainputs.drum import DrumTrainData
from datainputs.melody import MelodyPatternEncode
from settings import *
from dataoutputs.predictions import MusicPatternPrediction
from interfaces.functions import MusicPatternDecode
from models.LstmPipeline import BaseLstmPipeline


class DrumPipeline(BaseLstmPipeline):

    def __init__(self, melody_pattern_data, continuous_bar_number_data):
        self.train_data = DrumTrainData(melody_pattern_data, continuous_bar_number_data)
        super().__init__()

    def prepare(self):
        self.config = DrumConfig()  # 训练和弦所用的训练模型配置
        self.test_config = DrumConfig()  # 测试生成的和弦所用的配置和训练的配置大体相同，但是batch_size为1
        self.test_config.batch_size = 1
        self.variable_scope_name = 'DrumModel'

    def generate(self, session, melody_output, common_melody_patterns, drum_test_input=None):
        # 1.数据预处理 将melody_output转化为pattern
        drum_output = []  # 鼓机输出
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dict = MelodyPatternEncode(common_melody_patterns, melody_pattern_input, MELODY_TIME_STEP, MELODY_PATTERN_TIME_STEP).music_pattern_dict
        melody_output_pattern = []
        for key in range(round(len(melody_output) / 32)):
            melody_output_pattern += melody_pattern_dict[key]
        # 2.逐两拍生成数据
        for melody_iterator in range(2, len(melody_output_pattern) + 1, 2):  # 遍历整个主旋律 步长是2拍
            drum_prediction_input = [((melody_iterator - 2) % 8) // 2]
            required_drum_length = round(4 * TRAIN_DRUM_IO_BARS / DRUM_PATTERN_TIME_STEP)
            if melody_output_pattern[(melody_iterator - 2): melody_iterator] == [0, 0]:  # 最近这2拍没有主旋律 则鼓机沿用之前的
                if len(drum_output) == 0:
                    drum_output = drum_output + [0]
                else:
                    drum_output = drum_output + [drum_output[-1]]
            else:
                # 2.1.生成输入数据
                if len(drum_output) < required_drum_length:
                    last_bars_drum = [0 for t in range(required_drum_length - len(drum_output))] + drum_output
                else:
                    last_bars_drum = drum_output[-required_drum_length:]
                if melody_iterator < 10:  # 还没有到第十拍 旋律输入在前面补0 否则使用对应位置的旋律
                    last_bars_melody = [0 for t in range(10 - melody_iterator)] + melody_output_pattern[:melody_iterator]
                else:
                    last_bars_melody = melody_output_pattern[(melody_iterator - 10): melody_iterator]
                drum_prediction_input += last_bars_melody + last_bars_drum
                # print('          input', melody_iterator, drum_prediction_input)
                # 2.2.生成输出数据
                drum_predict = self.predict(session, [drum_prediction_input])
                drum_out = MusicPatternPrediction(drum_predict, 1, COMMON_DRUM_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成一维数组chord_out_vector，这个数组就是这两小节的和弦。每两小节生成一次和弦
                drum_output.append(drum_out)  # 添加到最终的和弦输出列表中
                # print(list(drum_predict[-1]))
                # print(drum_output)
                # print('\n\n')
        if drum_test_input:
            drum_output = drum_output[len(drum_test_input):]
        drum_final_output = MusicPatternDecode(self.train_data.common_drum_patterns, drum_output, DRUM_TIME_STEP, DRUM_PATTERN_TIME_STEP)
        # print(self.common_drum_patterns, '\n\n\n', drum_output, '\n\n\n')
        print(drum_output)
        print(drum_final_output)
        return drum_final_output
