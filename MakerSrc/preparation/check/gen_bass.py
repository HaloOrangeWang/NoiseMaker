# TODO: 调整这里的code，让它能够运行

# from settings import *
# from datainputs.bass import BassTrainData
# from pipelines.functions import BaseLstmPipeline
# from interfaces.utils import DiaryLog
# import numpy as np


# class BassTrainDataCheck(BassTrainData):
#
#     def __init__(self, melody_pat_data, continuous_bar_data, keypress_pat_data, all_keypress_pats, chord_cls):
#
#         super().__init__(melody_pat_data, continuous_bar_data, keypress_pat_data, all_keypress_pats, chord_cls)
#         self.check_melody_data = []  # 用于check的主旋律组合数据(共16拍)
#         self.check_chord_data = []  # 用于check的和弦组合数据(共16拍)
#         self.check_bass_input_data = []  # 用于check的bass输入数据(共8拍)
#         self.check_bass_output_data = []  # 用于check的bass输出数据(共8拍)
#         self.time_add_data = []  # 用于check的时间编码数据
#         self.chord_data = chord_cls.chord_data
#         bass_prob_ary = []
#         # 1.获取用于验证的数据
#         for song_it in range(TRAIN_FILE_NUMBERS):
#             if chord_cls.chord_data[song_it] != {} and self.bass_pat_data[song_it] != {}:
#                 self.get_check_io_data(self.bass_pat_data[song_it], melody_pat_data[song_it], continuous_bar_data[song_it], chord_cls.chord_data[song_it])
#                 bass_prob_ary.extend(get_diff_value(flat_array(self.raw_bass_data[song_it]), self.chord_data[song_it]))
#         # 2.找出前90%所在位置
#         bass_prob_ary = sorted(bass_prob_ary)
#         prob_09_dx = int(len(bass_prob_ary) * 0.9 + 1)
#         self.ConfidenceLevel = bass_prob_ary[prob_09_dx]
#
#     def get_check_io_data(self, bass_pat_data, melody_pat_data, continuous_bar_data, chord_data):
#         """生成一首歌bass校验所需的数据"""
#         for step_it in range(len(bass_pat_data)):  # 这里bass_pat_data是以步长为单位的,而不是以小节为单位的
#             try:
#                 flag_drop = False  # 这组数据是否忽略
#                 melody_input_time_data = []
#                 chord_input_time_data = []
#                 bass_input_time_data = []
#                 bass_output_time_data = []
#                 # 1.获取当前小节号添加当前时间的编码（0-3）
#                 cur_bar = step_it // 2
#                 pat_step_in_bar = step_it % 2
#                 time_add = (1 - continuous_bar_data[cur_bar + TRAIN_BASS_IO_BARS] % 2) * 2 + pat_step_in_bar
#                 # 2.添加最近4小节(TRAIN_BASS_IO_BARS+2)的主旋律
#                 melody_input_time_data = melody_input_time_data + melody_pat_data[cur_bar][pat_step_in_bar * 2:]
#                 for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS + 2):
#                     melody_input_time_data = melody_input_time_data + melody_pat_data[bar_it]
#                     if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节
#                         if bar_it >= cur_bar + TRAIN_BASS_IO_BARS:  # 位于输入和弦与输出和弦分界处之后出现了空小节 则直接忽略这组数据
#                             flag_drop = True
#                             break
#                         else:
#                             melody_input_time_data = [0 for t in range(len(melody_input_time_data))]
#                 if flag_drop is True:
#                     continue
#                 melody_input_time_data = melody_input_time_data + melody_pat_data[cur_bar + TRAIN_BASS_IO_BARS + 2][:2 * pat_step_in_bar]
#                 # 3.添加最近4小节(TRAIN_BASS_IO_BARS+2)的和弦
#                 chord_input_time_data = chord_input_time_data + chord_data[cur_bar][pat_step_in_bar * 2:]
#                 for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS + 2):
#                     chord_input_time_data = chord_input_time_data + chord_data[bar_it]
#                     if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节
#                         chord_input_time_data = [0 for t in range(len(chord_input_time_data))]
#                 chord_input_time_data = chord_input_time_data + chord_data[cur_bar + TRAIN_BASS_IO_BARS + 2][:2 * pat_step_in_bar]
#                 # 4.添加过去2小节的bass
#                 if melody_pat_data[cur_bar] == [0 for t in range(4)]:
#                     bass_input_time_data = bass_input_time_data + [0 for t in range(2 - pat_step_in_bar)]
#                 else:
#                     bass_input_time_data = bass_input_time_data + bass_pat_data[cur_bar * 2 + pat_step_in_bar: (cur_bar + 1) * 2]
#                 for bar_it in range(cur_bar + 1, cur_bar + TRAIN_BASS_IO_BARS):
#                     bass_input_time_data = bass_input_time_data + bass_pat_data[bar_it * 2: bar_it * 2 + 2]
#                     if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
#                         bass_input_time_data = [0 for t in range(len(bass_input_time_data))]
#                 bass_input_time_data = bass_input_time_data + bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2: (cur_bar + TRAIN_BASS_IO_BARS) * 2 + pat_step_in_bar]
#                 # 5.添加之后2小节的bass
#                 bass_output_time_data = bass_output_time_data + bass_pat_data[(cur_bar + TRAIN_BASS_IO_BARS) * 2 + pat_step_in_bar: (cur_bar + TRAIN_BASS_IO_BARS + 2) * 2 + pat_step_in_bar]
#                 if len(bass_output_time_data) < 4:  # 如果bass序列没那么长（到头了） 则舍弃这组序列
#                     flag_drop = True
#                 for bar_it in range(3):  # 检查该项数据是否可以用于训练。条件是最后三个小节的bass均不为空
#                     if bass_pat_data[cur_bar + TRAIN_BASS_IO_BARS + bar_it] == [0 for t in range(2)]:
#                         flag_drop = True
#                 if flag_drop is True:
#                     continue
#                 # 5.将这项数据添加进校验集中
#                 self.check_melody_data.append(melody_input_time_data)
#                 self.check_chord_data.append(chord_input_time_data)
#                 self.check_bass_input_data.append(bass_input_time_data)
#                 self.check_bass_output_data.append(bass_output_time_data)
#                 self.time_add_data.append(time_add)
#             except KeyError:
#                 pass
#             except IndexError:
#                 pass
#
#
# class BassPipelineCheck(BaseLstmPipeline):
#
#     def __init__(self, melody_pat_data, continuous_bar_data, keypress_pat_data, all_keypress_pats, chord_cls):
#         self.train_data = BassTrainDataCheck(melody_pat_data, continuous_bar_data, keypress_pat_data, all_keypress_pats, chord_cls)
#         super().__init__()
#
#     def generate_check(self, session):  # , common_melody_pats, common_corenote_pats):
#         """多步预测的校验方式"""
#         print('校验开始')
#         right_value = [0 for t in range(4)]
#         wrong_value = [0 for t in range(4)]
#         sample_amount = 250
#         # 1.从用于校验的数据中随机选取250组作为校验用例
#         rand_dx = np.random.permutation(len(self.train_data.check_melody_data))
#         rand_dx = rand_dx[:sample_amount]
#         check_melody_pat_ary = np.array(self.train_data.check_melody_data)[rand_dx]
#         check_chord_data = np.array(self.train_data.check_chord_data)[rand_dx]
#         check_bass_input_ary = np.array(self.train_data.check_bass_input_data)[rand_dx]
#         check_bass_output_ary = np.array(self.train_data.check_bass_output_data)[rand_dx]
#         rd_time_add_ary = np.array(self.train_data.time_add_data)[rand_dx]
#         # 2.对这250组进行校验
#         for check_it in range(sample_amount):
#             real_bass_output_ary = check_bass_input_ary[check_it]  # 模型的bass输出情况
#             generate_fail_time = 0  # 如果自我检查连续失败十次 则直接继续
#
#             bass_choose_bk = [1, 1, 1, 1]  # 备选bass。为防止死循环，如果连续十次验证失败，则使用备选和弦
#             diff_score_bk = np.inf  # 备选方案对应的差异函数
#             while True:
#                 for step_it in range(0, 4):  # 逐2拍生成数据
#                     # 2.1.生成输入数据
#                     beat_dx = step_it * 2  # 当前拍是第几拍
#                     bass_prediction_input = [(rd_time_add_ary[check_it] + step_it) % 4]  # 先保存当前时间的编码
#                     last_bars_bass = real_bass_output_ary[-4:]
#                     last_bars_chord = check_chord_data[check_it][beat_dx: beat_dx + 4 * TRAIN_BASS_IO_BARS + 2]
#                     last_bars_melody = check_melody_pat_ary[check_it][beat_dx: beat_dx + 4 * TRAIN_BASS_IO_BARS + 2]
#                     bass_prediction_input = np.append(bass_prediction_input, np.append(last_bars_melody, np.append(last_bars_chord, last_bars_bass)))  # 校验时输入到model中的数据
#                     # print('          input', melody_iterator, chord_prediction_input)
#                     # 2.2.生成输出数据
#                     bass_predict = self.predict(session, [bass_prediction_input])  # LSTM预测 得到二维数组predict
#                     if beat_dx % 8 == 0:  # 每两小节的第一拍不能为空
#                         bass_out_pattern = music_pattern_prediction(bass_predict, 1, COMMON_BASS_PATTERN_NUMBER)  # 将二维数组predict通过概率随机生成这两拍的bass组合bass_out_pattern
#                     else:
#                         bass_out_pattern = music_pattern_prediction(bass_predict, 0, COMMON_BASS_PATTERN_NUMBER)
#                     real_bass_output_ary = np.append(real_bass_output_ary, bass_out_pattern)  # 添加到最终的bass输出列表中
#                 # 2.3.根据一些数学方法和乐理对输出的和弦进行自我检查。如果检查不合格则打回重新生成
#                 root_data = []
#                 for chord_it in range(len(check_chord_data[check_it])):
#                     if chord_it == 0:
#                         root_data.append(chord_rootnote(check_chord_data[check_it][0], 0, BASS_AVERAGE_ROOT))
#                     else:
#                         root_data.append(chord_rootnote(check_chord_data[check_it][chord_it], root_data[chord_it - 1], BASS_AVERAGE_ROOT))
#                 bass_output = []
#                 for step_it in range(6):  # (4):
#                     beat_dx = step_it * 2  # 当前拍是第几拍
#                     pat_dx = real_bass_output_ary[step_it - 6]  # 4]
#                     if pat_dx == COMMON_BASS_PATTERN_NUMBER + 1:  # 罕见bass组合的特殊处理
#                         rel_note_list = [0] * 16
#                     else:
#                         rel_note_list = self.train_data.common_bass_pats[pat_dx]  # + 1]  # 将新生成的bass组合变为相对音高列表
#                     # print('     rnote_output', rel_note_list)
#                     for rel_note_group in rel_note_list:
#                         # print(pattern_iterator, note_iterator, rel_note_group)
#                         if rel_note_group == 0:
#                             bass_output.append(0)
#                         else:
#                             bass_output.append(get_abs_notelist_chord(rel_note_group, int(root_data[beat_dx - 12])))
#                 # print(bass_output[-64:])
#                 # print(check_chord_data[check_it][-8:])
#                 if generate_fail_time <= 10:
#                     total_diff_score = bass_confidence_check(bass_output, check_chord_data[check_it][-8:])
#                     if total_diff_score >= self.train_data.ConfidenceLevel:
#                         # if bass_prob_log < self.train_data.ConfidenceLevel:
#                         # if not bass_check(bass_output[-64:], check_chord_data[check_it][-8:]):  # bass与同时期的和弦差异过大
#                         # print('在第%d拍, 鼓点第%02d次打回，第一拍为空拍' % (beat_dx, generate_fail_time), drum_output[-16:])
#                         # DiaryLog.warn('bass被打回，与同时期和弦差异太大' + repr(bass_output[-64:]) + ', 对数为' + repr(bass_prob_log))
#                         print('        第%d个检查上,bass的误差分为%.4f' % (check_it, total_diff_score))
#                         generate_fail_time += 1
#                         if total_diff_score <= diff_score_bk:
#                             bass_choose_bk = real_bass_output_ary[-4:]
#                             diff_score_bk = total_diff_score
#                         real_bass_output_ary = real_bass_output_ary[:(-4)]  # 检查不合格 重新生成这两小节的bass
#                         if generate_fail_time >= 10:
#                             print('        bass使用备选方案,误差函数值为', diff_score_bk)
#                             real_bass_output_ary = np.append(real_bass_output_ary, bass_choose_bk)
#                             break
#                     else:
#                         print('第%d个检查上,bass的误差分为%.4f' % (check_it, total_diff_score))
#                         break
#                 else:  # 如果自我检查失败超过10次 就直接返回 避免循环过长
#                     break
#             # 2.4.对输出数据逐两拍进行校验 只要符合真实这两拍和弦的其一即可
#             for step_it in range(0, 4):  # 逐2拍进行校验
#                 if check_bass_output_ary[check_it][step_it] not in [0, COMMON_BASS_PATTERN_NUMBER + 1]:
#                     if real_bass_output_ary[step_it + 2 * TRAIN_BASS_IO_BARS] == check_bass_output_ary[check_it][step_it]:
#                         right_value[step_it] += 1
#                     else:
#                         wrong_value[step_it] += 1
#         # 3.输出
#         for step_it in range(4):
#             right_ratio = right_value[step_it] / (right_value[step_it] + wrong_value[step_it])
#             DiaryLog.warn('第%d个两拍正确的bass数量为%d,错误的bass数量为%d,正确率为%.4f.\n' % (step_it, right_value[step_it], wrong_value[step_it], right_ratio))
