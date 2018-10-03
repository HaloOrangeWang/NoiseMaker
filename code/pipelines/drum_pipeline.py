from models.configs import DrumConfig
from datainputs.drum import DrumTrainData
from datainputs.melody import MelodyPatternEncode
from settings import *
from pipelines.functions import music_pattern_prediction, BaseLstmPipeline
from interfaces.music_patterns import music_pattern_decode
from interfaces.utils import DiaryLog


class DrumPipeline(BaseLstmPipeline):

    def __init__(self, melody_pattern_data, continuous_bar_data):
        self.train_data = DrumTrainData(melody_pattern_data, continuous_bar_data)
        super().__init__()

    def prepare(self):
        self.config = DrumConfig()  # 训练和弦所用的训练模型配置
        self.test_config = DrumConfig()  # 测试生成的和弦所用的配置和训练的配置大体相同，但是batch_size为1
        self.test_config.batch_size = 1
        self.variable_scope_name = 'DrumModel'

    def generate(self, session, melody_output, common_melody_patterns, drum_test_input=None):
        # 1.数据预处理 将melody_output转化为pattern
        drum_pattern_output = []  # 鼓机输出的pattern形式
        drum_output = []  # 鼓机输出的note list形式
        melody_pattern_input = {int(t / 32): melody_output[t:t + 32] for t in range(0, len(melody_output), 32)}
        melody_pattern_dict = MelodyPatternEncode(common_melody_patterns, melody_pattern_input, 0.125, 1).music_pattern_dic
        melody_output_pattern = []
        generate_fail_time = 0  # 鼓机被打回重新生成的次数
        for key in range(round(len(melody_output) / 32)):
            melody_output_pattern.extend(melody_pattern_dict[key])
        # 2.逐两拍生成数据
        beat_dx = 2
        while beat_dx <= len(melody_output_pattern):  # 遍历整个主旋律 步长是2拍
            drum_prediction_input = [((beat_dx - 2) % 8) // 2]
            required_drum_length = 2 * TRAIN_DRUM_IO_BARS
            # 2.1.如果被打回重新生成的次数超过上限 视作歌曲生成失败 整体重新生成
            if generate_fail_time > MAX_DRUM_GENERATE_FAIL_TIME:
                DiaryLog.warn('鼓机被打回次数超过%d次,重新生成。\n\n\n' % generate_fail_time)
                return None
            # 2.2.如果这两拍没有主旋律 鼓点直接沿用之前的鼓点
            if melody_output_pattern[(beat_dx - 2): beat_dx] == [0, 0]:  # 最近这2拍没有主旋律 则鼓机沿用之前的
                if len(drum_pattern_output) == 0:
                    drum_pattern_output.append(0)
                    drum_output.extend([0 for t in range(16)])
                else:
                    drum_pattern_output.append(drum_pattern_output[-1])
                    drum_output.extend(drum_output[-16:])
            else:
                # 2.3.生成输入数据
                if len(drum_pattern_output) < required_drum_length:
                    last_bars_drum = [0 for t in range(required_drum_length - len(drum_pattern_output))] + drum_pattern_output
                else:
                    last_bars_drum = drum_pattern_output[-required_drum_length:]
                if beat_dx < 10:  # 还没有到第十拍 旋律输入在前面补0 否则使用对应位置的旋律
                    last_bars_melody = [0 for t in range(10 - beat_dx)] + melody_output_pattern[:beat_dx]
                else:
                    last_bars_melody = melody_output_pattern[(beat_dx - 10): beat_dx]
                drum_prediction_input += last_bars_melody + last_bars_drum
                # print('          input', melody_iterator, drum_prediction_input)
                # 2.4.生成输出数据
                drum_predict = self.predict(session, [drum_prediction_input])
                drum_out = music_pattern_prediction(drum_predict, 1, COMMON_DRUM_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的鼓点组合drum_out
                drum_pattern_output.append(drum_out)  # 添加到最终的鼓点输出列表中
                drum_output.extend(music_pattern_decode(self.train_data.common_drum_pats, [drum_out], 0.125, 2))  # 将新生成的鼓点组合解码为音符列表
                # print(list(drum_predict[-1]))
                # print(drum_output)
                # print('\n\n')
                # 2.5.对生成的数据进行检查 检查内容为鼓机的第一拍不能为空
                if beat_dx % 4 == 2:
                    if drum_output[-16] == 0:  # 鼓点的第一拍为空 这是不符合要求的
                        DiaryLog.warn('在第%d拍, 鼓点第%02d次打回，第一拍为空拍' % (beat_dx, generate_fail_time) + repr(drum_output[-16:]))
                        beat_dx -= 2  # 回退2拍 重新生成
                        generate_fail_time += 1  # 被打回重新生成了一次
                        drum_output = drum_output[:len(drum_output) - 16]
                        drum_pattern_output = drum_pattern_output[:len(drum_pattern_output) - 1]
            beat_dx += 2  # 继续下两拍
        # if drum_test_input:  # 如果事先有输入的鼓点，把他们
        #     drum_pattern_output = drum_pattern_output[len(drum_test_input):]
        # print(self.common_drum_patterns, '\n\n\n', drum_output, '\n\n\n')
        DiaryLog.warn('鼓点组合的输出: ' + repr(drum_pattern_output))
        DiaryLog.warn('鼓点的输出: ' + repr(drum_output) + '\n\n\n')
        return drum_output
