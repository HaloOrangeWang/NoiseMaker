from dataoutputs.validation import MelodyCheck, MelodyClusterCheck
from models.configs import MelodyConfig
from datainputs.melody import GetMelodyCluster, GetMelodyProfileBySong, MelodyTrainData, MelodyPatternEncode
from settings import *
from dataoutputs.predictions import MusicPatternPrediction
from interfaces.functions import MusicPatternDecode, LastNotZeroNumber
import random
from models.LstmPipeline import BaseLstmPipeline


class MelodyPipeline(BaseLstmPipeline):

    def prepare(self):
        self.config = MelodyConfig()
        self.test_config = MelodyConfig()
        self.test_config.batch_size = 1
        self.kmeans_sess, self.kmeans_model, self.cluster_center_points = GetMelodyCluster()
        self.train_data = MelodyTrainData(self.kmeans_sess, self.kmeans_model, self.cluster_center_points, tone_restrict=TONE_MAJOR)
        self.variable_scope_name = 'MelodyModel'

    def generate(self, session, melody_generate_input=None, begin_empty_bar_number=0):
        # 1.数据预备处理
        no_input_melody = not melody_generate_input  # 是否没有预输入的音乐片段
        if no_input_melody:
            melody_generate_input = [0 for t in range(128)]  # 如果没有输入的旋律 前面补上4个空小节
            begin_empty_bar_number = 4  # 最开始的空小节数量
        assert len(melody_generate_input) >= round(128)  # melody_generate_input在4-6小节之间
        assert len(melody_generate_input) <= round(192)
        melody_output = [t for t in melody_generate_input]  # 初始输出 在最后会把最开始的空小节去掉
        iterator = 0
        # 2.获取melody_generate_input的melody profile
        if not no_input_melody:  # 有初始主旋律的情况
            melody_cluster_input = {int(t / 32): melody_generate_input[t:t + 32] for t in range(0, len(melody_generate_input), 32)}  # 获取歌曲的melody profile需要将主旋律转换成dict形式
            melody_profile = GetMelodyProfileBySong(self.kmeans_sess, self.kmeans_model, self.cluster_center_points, melody_cluster_input)
            initial_melody_profile_length = len(melody_profile)
            initial_melody_cluster = melody_profile[-1]  # 初始音乐的最后一小节的ｍｅｌｏｄｙ_cluster
            # melody_profile += [melody_profile[-1] for t in range(10)]
        else:  # 没有初始主旋律的情况
            initial_melody_profile_length = 4
            melody_profile = [0 for t in range(4)]  # 四个小节的0
        # 3.将melody_generate_input转化为常见的melody_pattern
        if not no_input_melody:  # 有初始主旋律的情况
            melody_pattern_input = {int(t / 32): melody_generate_input[t:t + 32] for t in range(0, len(melody_generate_input), 32)}  # 获取melody pattern 也需要将主旋律转换成dict的形式
            melody_pattern_dict = MelodyPatternEncode(self.train_data.common_melody_patterns, melody_pattern_input, MELODY_TIME_STEP, MELODY_PATTERN_TIME_STEP).music_pattern_dict
            melody_pattern_list = []  # 把melody_pattern_dict转换成list的形式
            for key in range(round(len(melody_generate_input) / 32)):  # 遍历这首歌的所有小节
                melody_pattern_list += melody_pattern_dict[key]
        # 4.生成这首歌曲的melody_profile 生成策略是 1-MIN_GENERATE_BAR_NUMBER小节由初始音符的cluster向中音dol所在的cluster渐变 MIN_GENERATE_BAR_NUMBER小节以后全部为中音dol所在的cluster
        mid_do_melody_cluster = GetMelodyProfileBySong(self.kmeans_sess, self.kmeans_model, self.cluster_center_points, {0: [72 for t in range(32)], 1: [72 for t in range(32)]})[1]  # 求全部是中音dol的情况下的melody_cluster
        min_generate_cluster_number = int((MIN_GENERATE_BAR_NUMBER - len(melody_generate_input) / 32 + begin_empty_bar_number) / 2)  # 最小生成的cluster数量
        if not no_input_melody:
            melody_profile += [round((initial_melody_cluster * ((min_generate_cluster_number + 1) - t) + mid_do_melody_cluster * t) / (min_generate_cluster_number + 1)) for t in range(1, (min_generate_cluster_number + 1)) for t0 in range(2)]
            melody_profile += [mid_do_melody_cluster for t in range(MAX_GENERATE_BAR_NUMBER - MIN_GENERATE_BAR_NUMBER)]
        # 5.逐时间步长生成数据
        while True:
            # 5.1.如果输入数据为空 随机生成一个起始音符(dol mi sol中间选一个)作为乐曲的开始
            if no_input_melody and iterator == 0:
                melody_pattern_list = [0 for t in range(16)]  # 前四小节的melody_pattern肯定都是0
                start_note = [72, 76, 79][random.randint(0, 2)]  # 从 72 76 79中随机选取一个
                start_pattern = self.train_data.common_melody_patterns.index([start_note] + [0 for t in range(7)])  # 找到这个pattern的代码 并添加进去
                melody_pattern_list.append(start_pattern)
                melody_output += [start_note] + [0 for t in range(7)]
                # 由于开始音符可能会发生变化 因此，每生成一个开始音符都要重新生成一遍melody_cluster
                start_note_cluster = GetMelodyProfileBySong(self.kmeans_sess, self.kmeans_model, self.cluster_center_points, {0: [start_note for t in range(32)], 1: [start_note for t in range(32)]})[1]  # 初始音符所在的cluster
                min_generate_cluster_number = int(MIN_GENERATE_BAR_NUMBER / 2)
                melody_profile = melody_profile[:4] + [round((start_note_cluster * (min_generate_cluster_number - t) + mid_do_melody_cluster * t) / min_generate_cluster_number) for t in range(min_generate_cluster_number) for t0 in range(2)]  # melody_profile为从开始音符所在的cluster到中音dol所在的cluster的为期八小节的过度，加上4小节的中音dol所在的cluster
                melody_profile += [mid_do_melody_cluster for t in range(MAX_GENERATE_BAR_NUMBER - MIN_GENERATE_BAR_NUMBER)]
                print('start_note', start_note, start_pattern)
                print(iterator, 'Profile', melody_profile)
                iterator = 1
            # 5.2.逐时间步长生成test model输入数据
            # melody_input = [[round(iterator % 4)]]
            # melody_input[0] += melody_pattern_list[-16:]  # 最近4小节
            bar_add = int(iterator % 8 >= 4)
            melody_input = [[round(iterator % 4) + bar_add * 4]]  # 当前时间的编码
            melody_input[0] += melody_profile[int(iterator / 4) + initial_melody_profile_length - 4: int(iterator / 4) + initial_melody_profile_length + 1]  # 过去５小节的melody_profile
            melody_input[0] += melody_pattern_list[-16:]  # 最近4小节的旋律组合
            # print('          input', iterator, melody_input[0], end=' ')
            # 5.3.生成输出的音符
            melody_predict = self.predict(session, melody_input)  # LSTM预测 得到二维数组predict
            if iterator % 4 == 0:  # 每小节的第一拍不能为空 其余三拍可以为空
                output_melody_pattern = MusicPatternPrediction(melody_predict, 1, COMMON_MELODY_PATTERN_NUMBER)  # 随机生成一个输出旋律组合
            else:
                output_melody_pattern = MusicPatternPrediction(melody_predict, 0, COMMON_MELODY_PATTERN_NUMBER)  # 随机生成一个输出旋律组合
            melody_pattern_list.append(output_melody_pattern)
            melody_output = melody_output + MusicPatternDecode(self.train_data.common_melody_patterns, [output_melody_pattern], MELODY_TIME_STEP, MELODY_PATTERN_TIME_STEP)
            # print(list(melody_predict[-1]))
            # print(output_melody_pattern)
            # print('\n\n')
            iterator += 1
            # 5.4.检查生成的旋律
            if iterator % 4 == 0:
                if not MelodyCheck(melody_output[-32:]):  # 检查这一小节的音乐是否符合要求 如果不符合要求 则返工 重新生成这一小节的音乐
                    print(iterator, 'False', melody_output[-32:])
                    iterator -= 4  # 重新生成这一小节的音乐
                    melody_output = melody_output[:len(melody_output) - 32]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 4]
                    continue
            if iterator % 8 == 0:
                # print(MelodyClusterCheck(melody.kmeans_sess, melody.kmeans_model, melody.cluster_center_points, melody_cluster_check_input, 7, 0))
                if not MelodyClusterCheck(self.kmeans_sess, self.kmeans_model, self.cluster_center_points, melody_output[-64:], melody_profile[int(iterator / 4) + initial_melody_profile_length - 1], 2, train_pattern=True):  # 检查这两小节的音乐cluster是否符合要求 如果不符合要求 则返工 重新生成这两1小节的音乐
                    print(iterator, 'ClusterFalse', melody_output[-64:])
                    iterator -= 8
                    melody_output = melody_output[:len(melody_output) - 64]
                    melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 8]
                    continue
                else:
                    melody_cluster_input = {int(t / 32): melody_output[-64:][t:t + 32] for t in range(0, 64, 32)}
                    latest_melody_cluster = GetMelodyProfileBySong(self.kmeans_sess, self.kmeans_model, self.cluster_center_points, melody_cluster_input)
                    melody_profile = melody_profile[:(int(iterator / 4) + initial_melody_profile_length - 2)] + latest_melody_cluster + melody_profile[(int(iterator / 4) + initial_melody_profile_length):]  # 更新前一个区段的melody_profile 同时后一个区段的的melody_profile为两个6
                    # print(iterator, 'Profile', melody_profile)
            # 5.5.判断是否符合结束生成的条件 如果最后一个音是72或84，且总小节数在8-16之间的偶数 则结束生成。
            bar_generate = int(iterator / 4)
            if bar_generate + initial_melody_profile_length - begin_empty_bar_number >= MIN_GENERATE_BAR_NUMBER and iterator % 8 == 0:
                if LastNotZeroNumber(melody_output, reverse=True) == 72 or LastNotZeroNumber(melody_output, reverse=True) == 84:  # 结束条件是最后音符是ｄｏｌ
                    break
            if bar_generate + initial_melody_profile_length - begin_empty_bar_number >= MAX_GENERATE_BAR_NUMBER:  # 到max小节仍未结束 全部打回重新生成
                print('Restart')
                iterator = 0
                melody_output = melody_output[:len(melody_output) - 32 * bar_generate]
                melody_pattern_list = melody_pattern_list[:len(melody_pattern_list) - 4 * bar_generate]
        # 6.输出
        melody_output = melody_output[32 * begin_empty_bar_number:]  # 把前面的空小节去掉
        print(melody_output)
        return melody_output
