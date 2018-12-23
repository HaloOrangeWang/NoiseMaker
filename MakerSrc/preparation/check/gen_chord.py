# TODO: 调整这里的code，让它能够运行

# class ChordTrainDataCheck(ChordTrainData):
#
#     def __init__(self, melody_pat_data, raw_melody_data, continuous_bar_data, core_note_pat_data):
#
#         super().__init__(melody_pat_data, continuous_bar_data)
#         self.check_melody_data = []  # 用于check的molody组合数据(共16拍)
#         self.check_raw_melody_data = []  # 用于check的molody原始数据(共16拍)
#         self.check_chord_input_data = []  # 用于check的chord输入数据(共8拍)
#         self.check_chord_output_data = []  # 用于check的chord输出数据(共8拍)
#         self.time_add_data = []  # 用于check的时间编码数据
#
#         self.transfer_count = np.zeros([COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2, len(CHORD_DICT) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦的转移矩阵 0是空主旋律 1-400是大调对应的主旋律 401-800是小调对应的主旋律 801对应的是罕见主旋律
#         self.real_transfer_count = np.zeros([COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2, len(CHORD_DICT) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦的转移矩阵 与上个变量的区别是不添加e**(-3)
#         self.transfer_count += np.e ** (-3)
#         self.transfer = np.zeros([COMMON_CORE_NOTE_PATTERN_NUMBER * 2 + 2, len(CHORD_DICT) + 1], dtype=np.float32)  # 主旋律/调式与同时期和弦 概率取对数后的转移矩阵 这个转移矩阵的数字精度必须是float32的
#         # self.confidence_level = 0  # 连续四步预测的90%置信水平
#         # 1.获取用于验证的数据
#         for song_it in range(TRAIN_FILE_NUMBERS):
#             if self.chord_data[song_it] != {} and melody_pat_data[song_it] != {}:
#                 self.get_check_io_data(self.chord_data[song_it], melody_pat_data[song_it], raw_melody_data[song_it], continuous_bar_data[song_it])
#         # print(len(self.check_melody_data))
#         # 2.获取训练用曲的旋律列表
#         tone_list = get_tone_list()
#         # 3.获取主旋律与同时期和弦的转移矩阵
#         for song_it in range(TRAIN_FILE_NUMBERS):
#             if self.chord_data[song_it] != {} and melody_pat_data[song_it] != {}:
#                 self.count(self.chord_data[song_it], core_note_pat_data[song_it], tone_list[song_it])
#
#     def get_check_io_data(self, chord_data, melody_pat_data, raw_melody_data, continuous_bar_data):
#         """生成一首歌校验所需的数据"""
#         for key in chord_data:
#             for time_in_bar in range(2):  # 和弦训练的步长为2拍
#                 try:
#                     flag_drop = False  # 这组数据是否忽略
#                     melody_input_time_data = []
#                     raw_melody_time_data = []
#                     chord_input_time_data = []
#                     chord_output_time_data = []
#                     # 1.添加当前时间的编码（0-3）
#                     time_add = (1 - continuous_bar_data[key + TRAIN_CHORD_IO_BARS] % 2) * 2
#                     # 2.添加最近4小节(TRAIN_CHORD_IO_BARS+2)的主旋律(包括原始的旋律和旋律组合)
#                     melody_input_time_data = melody_input_time_data + melody_pat_data[key][time_in_bar * 2:]
#                     raw_melody_time_data = raw_melody_time_data + raw_melody_data[key][time_in_bar * 16:]
#                     for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS + 2):
#                         melody_input_time_data = melody_input_time_data + melody_pat_data[bar_it]
#                         raw_melody_time_data = raw_melody_time_data + raw_melody_data[bar_it]
#                         if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节
#                             if bar_it >= key + TRAIN_CHORD_IO_BARS:  # 位于输入和弦与输出和弦分界处之后出现了空小节 则直接忽略这组数据
#                                 flag_drop = True
#                                 break
#                             else:
#                                 melody_input_time_data = [0 for t in range(len(melody_input_time_data))]
#                                 raw_melody_time_data = [0 for t in range(len(raw_melody_time_data))]
#                     if flag_drop is True:
#                         continue
#                     melody_input_time_data = melody_input_time_data + melody_pat_data[key + TRAIN_CHORD_IO_BARS + 2][:2 * time_in_bar]
#                     raw_melody_time_data = raw_melody_time_data + raw_melody_data[key + TRAIN_CHORD_IO_BARS + 2][:16 * time_in_bar]
#                     # 3.添加过去2小节的和弦
#                     if melody_pat_data[key] == [0 for t in range(4)]:
#                         chord_input_time_data = chord_input_time_data + [0 for t in range(4 - 2 * time_in_bar)]
#                     else:
#                         chord_input_time_data = chord_input_time_data + chord_data[key][2 * time_in_bar:]
#                     for bar_it in range(key + 1, key + TRAIN_CHORD_IO_BARS):
#                         chord_input_time_data = chord_input_time_data + chord_data[bar_it]
#                         if melody_pat_data[bar_it] == [0 for t in range(4)]:  # 如果训练集中出现了空小节 那么把它前面的也都置为空
#                             chord_input_time_data = [0 for t in range(len(chord_input_time_data))]
#                     chord_input_time_data = chord_input_time_data + chord_data[key + TRAIN_CHORD_IO_BARS][:2 * time_in_bar]
#                     # 4.添加之后2小节的和弦
#                     chord_output_time_data = chord_output_time_data + chord_data[key + TRAIN_CHORD_IO_BARS][2 * time_in_bar:]
#                     chord_output_time_data = chord_output_time_data + chord_data[key + TRAIN_CHORD_IO_BARS + 1]
#                     chord_output_time_data = chord_output_time_data + chord_data[key + TRAIN_CHORD_IO_BARS + 2][:2 * time_in_bar]
#                     for bar_it in range(3):  # 检查该项数据是否可以用于训练。条件是最后三个小节的和弦均不为空
#                         if chord_data[key + TRAIN_CHORD_IO_BARS + bar_it] == [0 for t in range(4)]:
#                             flag_drop = True
#                     if flag_drop is True:
#                         continue
#                     # 5.将这项数据添加进校验集中
#                     self.check_melody_data.append(melody_input_time_data)
#                     self.check_raw_melody_data.append(raw_melody_time_data)
#                     self.check_chord_input_data.append(chord_input_time_data)
#                     self.check_chord_output_data.append(chord_output_time_data)
#                     self.time_add_data.append(time_add)
#                 except KeyError:
#                     pass
#                 except IndexError:
#                     pass
#
#
# class ChordPipelineCheck(BaseLstmPipeline):
#
#     def __init__(self, melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord):
#         self.train_data = ChordTrainDataCheck(melody_pattern_data, raw_melody_data, continuous_bar_data, core_note_pat_nres_for_chord)
#         super(ChordPipelineCheck, self).__init__()
#
#     def prepare(self):
#         self.config = ChordConfig()
#         self.test_config = ChordConfig()
#         self.test_config.batch_size = 1
#         self.variable_scope_name = 'ChordModel'
#
#     def valid(self, predict_matrix, batches_output, pattern_number):
#         """和弦的验证方法是最后两个 而不是最后一个。另外，pattern_number这个参数没有用"""
#         predict_value = predict_matrix[:, -2:]
#         right_value = batches_output[:, -2:]
#         right_num = sum((predict_value == right_value).flatten())
#         wrong_num = sum((predict_value != right_value).flatten())
#         return right_num, wrong_num
#
#     def generate_check(self, session, common_melody_pats, common_corenote_pats):
#         """多步预测的校验方式"""
#         print('校验开始')
#         right_value = [0 for t in range(4)]
#         wrong_value = [0 for t in range(4)]
#         sample_amount = 250
#         # 1.从用于校验的数据中随机选取250组作为校验用例
#         rand_dx = np.random.permutation(len(self.train_data.check_melody_data))
#         rand_dx = rand_dx[:sample_amount]
#         check_melody_pat_ary = np.array(self.train_data.check_melody_data)[rand_dx]
#         check_raw_melody_ary = np.array(self.train_data.check_raw_melody_data)[rand_dx]
#         check_chord_input_ary = np.array(self.train_data.check_chord_input_data)[rand_dx]
#         check_chord_output_ary = np.array(self.train_data.check_chord_output_data)[rand_dx]
#         rd_time_add_ary = np.array(self.train_data.time_add_data)[rand_dx]
#         # 2.定义计算0.9置信区间和检验的方法
#         confidence_level = GetChordConfidenceLevel(self.train_data.transfer_count, self.train_data.real_transfer_count)
#         # 3.对这250组进行校验
#         for check_it in range(sample_amount):
#             print(check_it)
#             real_chord_output_ary = check_chord_input_ary[check_it]  # 模型的和弦输出情况
#
#             core_note_ary = melody_core_note_for_chord({t // 32: check_raw_melody_ary[check_it][-64:][t: t + 32] for t in range(0, 64, 32)})  # 提取出这几拍主旋律的骨干音
#             core_note_pat_ary = CoreNotePatternEncode(common_corenote_pats, core_note_ary, 0.125, 2).music_pattern_ary
#             confidence_level.get_loss09(session, core_note_pat_ary)  # 计算这段主旋律的和弦0.9置信区间
#             generate_fail_time = 0  # 生成失败的次数
#             chord_choose_bk = [1, 1, 1, 1]  # 备选和弦。为防止死循环，如果连续十次验证失败，则使用备选和弦
#             loss_bk = np.inf  # 备选方案对应的损失函数
#             while True:
#                 for beat_dx in range(0, 8, 2):  # 逐2拍生成数据
#                     # 3.1.生成输入数据
#                     chord_prediction_input = [rd_time_add_ary[check_it] + beat_dx // 2]  # 先保存当前时间的编码
#                     last_bars_chord = real_chord_output_ary[-8:]
#                     last_bars_melody = check_melody_pat_ary[check_it][beat_dx: beat_dx + 10]
#                     chord_prediction_input = np.append(chord_prediction_input, np.append(last_bars_melody, last_bars_chord))
#                     # print('          input', melody_iterator, chord_prediction_input)
#                     # 3.2.生成输出数据
#                     chord_predict = self.predict(session, [chord_prediction_input])  # LSTM预测 得到二维数组predict
#                     chord_out = chord_prediction(chord_predict)  # 将二维数组predict通过概率随机生成一维数组chord_out_vector，这个数组就是这两小节的和弦。每两小节生成一次和弦
#                     real_chord_output_ary = np.append(real_chord_output_ary, [chord_out for t in range(2)])  # 添加到最终的和弦输出列表中
#                 # 3.3.根据一些数学方法和乐理对输出的和弦进行自我检查。如果检查不合格则打回重新生成
#                 # if check_it <= 10:
#                 #     print('主旋律输入为', check_raw_melody_ary[check_it], '骨干音列表为', core_note_ary, '骨干音组合为', core_note_pat_ary)
#                 chord_per_2beats = [real_chord_output_ary[4 * TRAIN_CHORD_IO_BARS:][t] for t in range(0, len(real_chord_output_ary[4 * TRAIN_CHORD_IO_BARS:]), 2)]  # 生成的和弦从每1拍一个转化为每两拍一个
#                 # if check_it <= 10:
#                 #     print('90%置信区间为', confidence_level.loss09, '和弦的两拍输入为', chord_per_2beats)
#                 # if chord_check(real_chord_output_ary[4 * TRAIN_CHORD_IO_BARS:], check_molody_ary[-64:]):
#                 check_res, loss_value = confidence_level.check_chord_ary(session, check_raw_melody_ary[check_it][-64:], core_note_pat_ary, chord_per_2beats)
#                 # if check_it <= 10:
#                 #     print('该和弦组合的交叉熵损失函数为', x2, '校验结果为', x1)
#                 if check_res:
#                     break
#                 else:
#                     real_chord_output_ary = check_chord_input_ary[check_it]  # 重新生成时，需要把原先改变的一些变量改回去
#                     generate_fail_time += 1
#                     chord_choose_bk = chord_per_2beats
#                     loss_bk = loss_value
#                     if generate_fail_time >= 10:
#                         print('和弦使用备选方案,损失为', loss_bk)
#                         real_chord_output_ary = np.append(real_chord_output_ary, np.repeat(chord_choose_bk, 2))
#                         break
#             # 3.4.对输出数据逐两拍进行校验 只要符合真实这两拍和弦的其一即可
#             for beat_dx in range(0, 8, 2):  # 逐2拍进行校验
#                 if check_chord_output_ary[check_it][beat_dx] != 0 or check_chord_output_ary[check_it][beat_dx + 1] != 0:
#                     if real_chord_output_ary[beat_dx + 4 * TRAIN_CHORD_IO_BARS] == check_chord_output_ary[check_it][beat_dx] or real_chord_output_ary[beat_dx + 4 * TRAIN_CHORD_IO_BARS] == check_chord_output_ary[check_it][beat_dx + 1]:
#                         right_value[beat_dx // 2] += 1
#                     else:
#                         wrong_value[beat_dx // 2] += 1
#             # if check_it < 10:
#             #     print('melody input', check_melody_pat_ary[check_it])
#             #     print('chord input', np.append(check_chord_input_ary[check_it], check_chord_output_ary[check_it]))
#             #     print('real out', real_chord_output_ary)
#             #     print('\n')
#         # 4.输出
#         for step_it in range(4):
#             right_ratio = right_value[step_it] / (right_value[step_it] + wrong_value[step_it])
#             print('第%d个两拍正确的chord数量为%d,错误的chord数量为%d,正确率为%.4f.\n' % (step_it, right_value[step_it], wrong_value[step_it], right_ratio))
