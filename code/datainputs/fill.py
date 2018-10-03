from settings import *
import numpy as np
from interfaces.utils import get_first_index_bigger, get_last_index_smaller, get_dict_max_key
from interfaces.music_patterns import CommonMusicPatterns, MusicPatternEncodeStep
from interfaces.sql.sqlite import get_raw_song_data_from_dataset, get_tone_list, get_bpm_list, NoteDict
from datainputs.functions import one_song_rel_notelist_melody


def judge_imitation(input_melody_list, input_comp_list, speed_ratio_dict):
    """
    判断伴奏是否对主旋律存在某种模仿关系 以1拍为时间步长进行存储
    存储方式为[时间差别 音高差别 速度差别]。其中时间差别不能超过8拍（64），速度差别必须是1（一半） 2（相同） 3（两倍）之间选择
    为０则不模仿　为１则与前一拍的情况相同
    只有一个音符的情况下不能视作模仿
    :param speed_ratio_dict: 主旋律速度和伴奏速度比例的对照表
    :param input_melody_list: 主旋律列表 相对音高形式
    :param input_comp_list: 伴奏列表 相对音高形式
    :return:
    """
    # 1.首先将melody_list和comp_list中的零值都去掉 加上位置 并转成np.array形式
    melody_list = np.array([[t[0][0], t[0][1], i] for (i, t) in enumerate(input_melody_list) if t != 0])
    comp_list = np.array([[t[0][0], t[0][1], i] for (i, t) in enumerate(input_comp_list) if t != 0])
    imitation_comp_list = np.array([0 for t in range(len(input_comp_list) // 8)], dtype=object)
    # A.一整拍以内的所有音符必须都要符合此模仿关系
    # B.被模仿的主旋律音符相邻两个音符间隔不能超过1拍
    # C.相隔距离不能超过模仿长度的2倍+2拍
    for step_it in range(0, len(input_comp_list), 8):
        beat_comp = np.array(input_comp_list[step_it: step_it + 8], dtype=object)
        if imitation_comp_list[step_it // 8] != 0:
            continue
        if len(beat_comp[beat_comp != 0]) == 0:
            continue
        imit_range_first_dx, __ = get_first_index_bigger(melody_list[:, 2], step_it - 56)  # 在主旋律中模仿的区间 前７拍之内模仿
        imit_range_last_dx, __ = get_last_index_smaller(melody_list[:, 2], step_it + 1)
        part_comp_list = comp_list[comp_list[:, 2] >= step_it]  # 这一拍之后的伴奏列表
        longest = 1  # 最长的模仿序列长度
        longest_message = {'note': 0, 'time': 0, 'speed': 0}  # 最长的模仿序列在主旋律列表中的位置
        for startnote_it in range(imit_range_first_dx, imit_range_last_dx + 1):
            try:
                note_diff = part_comp_list[0, 0] - melody_list[startnote_it, 0]  # 音高差别
                step_diff = part_comp_list[0, 2] - melody_list[startnote_it, 2]  # 时间差别
                speed_ratio = round((part_comp_list[1, 2] - part_comp_list[0, 2]) / (melody_list[startnote_it + 1, 2] - melody_list[startnote_it, 2]), 1)  # 速度的比
                imitation_length = 1  # 模仿序列的长度
                melody_note_it = startnote_it + 1
                while melody_note_it <= len(melody_list) - 1:  # 判断模仿序列的长度
                    if (part_comp_list[imitation_length, 0] - melody_list[melody_note_it, 0] != note_diff) or (part_comp_list[imitation_length, 2] - melody_list[melody_note_it, 2] != step_diff):
                        break
                    if melody_list[melody_note_it, 2] - melody_list[melody_note_it - 1, 2] > 8:  # 被模仿的主旋律音符相邻两个音符间隔不能超过1拍
                        break
                    if round((part_comp_list[imitation_length, 2] - part_comp_list[imitation_length - 1, 2]) / (melody_list[melody_note_it, 2] - melody_list[melody_note_it - 1, 2]), 1) != speed_ratio:
                        break
                    imitation_length += 1
                    melody_note_it += 1
                if imitation_length >= 2 and imitation_length >= longest:  # 模仿序列的长度在２以上且大于等于当前的最长序列 则认为是
                    speed_ratio = speed_ratio_dict.get(speed_ratio, 0)
                    if speed_ratio == 0:
                        continue
                    if step_diff >= 16 + 2 * (part_comp_list[imitation_length - 1, 2] - part_comp_list[0, 2]):  # 相隔距离不能超过模仿长度的2倍+2拍
                        continue
                    step_lack = 8 - part_comp_list[imitation_length - 1, 2] + step_it
                    if step_lack >= 2 and input_comp_list[part_comp_list[imitation_length - 1, 2]: step_it + 8] != [0 for t in range(step_lack)]:  # 一整拍以内的所有音符必须都要符合此模仿关系
                        continue
                    longest = imitation_length
                    longest_message = {'note': note_diff, 'time': step_diff, 'speed': speed_ratio}
            except IndexError:
                pass
        if longest >= 2:
            imitation_comp_list[step_it // 8] = [longest_message['note'], longest_message['time'], longest_message['speed']]
            imitation_end_beat = part_comp_list[longest - 1, 2] // 8  # 模仿序列结束于哪拍
            imitation_comp_list[step_it // 8 + 1: imitation_end_beat + 1] = 1
    return imitation_comp_list


class FillClassifyAndPats:
    """生成加花的类型和组合"""

    def __init__(self, part_count):
        self.classify_data = [[[] for t1 in range(TRAIN_FILE_NUMBERS)] for t0 in range(part_count)]
        self.pats_data = [[], [], []]

    def get_fill_pat(self, fill_type, fill_data, start_step_dx, end_step_dx):
        """
        生成加花组合的数据
        :param fill_type: 加花的类型
        :param fill_data: 一首歌的加花数据
        :param start_step_dx: 加花开始于哪一个步长
        :param end_step_dx: 加花结束于哪一个步长
        """
        fill_1pat = [0 for t in range(start_step_dx % 8)] + fill_data[start_step_dx: end_step_dx]
        fill_1pat = [(NoteDict[t] if t != 0 else 0) for t in fill_1pat]  # 转换成绝对音高
        if len(fill_1pat) % 8 != 0:
            fill_1pat.extend([0] * (8 - len(fill_1pat) % 8))
        if fill_1pat not in self.pats_data[fill_type - 1]:
            self.pats_data[fill_type - 1].append(fill_1pat)

    def run_1song(self, part_dx, song_dx, fill_data, raw_melody_data, bpm):
        """
        对加花数据进行分类，分成过渡型、强调型、补充型三种
        :param part_dx: 加花的第几个部分
        :param song_dx: 歌曲编号
        :param fill_data: 一首歌的加花数据
        :param raw_melody_data: 这一首歌的主旋律数据
        :param bpm: 这一首歌的bpm数据
        :return: 分类数据
        """
        fill_note_count = 0  # 一段时间内加花的音符计数
        same_note_count = 0  # 一段时间内加花和主旋律音符同时出现的计数
        diff_note_count = 0  # 一段时间内加花和主旋律音符不同时出现的计数
        empty_note_count = 0  # 音符空白期的时间长度
        judge_start_step_dx = 0  # 这“一段时间”具体开始于哪一步
        self.classify_data[part_dx][song_dx] = [0 for t in range(len(fill_data))]  # 分类结果
        flag_no_fill = True  # 当前时间段没有加花

        def classify(step_dx2):
            """对一段加花进行分类"""
            if fill_note_count == 0:
                return 0
            if bpm >= 90 and fill_note_count >= 5 and (step_dx2 - judge_start_step_dx) / fill_note_count < 4:
                return 1  # 过渡型的加花
            elif fill_note_count >= 5 and (step_dx2 - judge_start_step_dx) / fill_note_count <= 2:
                return 1
            elif same_note_count / (same_note_count + diff_note_count) > 0.5:
                for step_it2 in range(judge_start_step_dx if judge_start_step_dx % 16 == 0 else 16 * (judge_start_step_dx // 16 + 1), 16 * (step_dx2 // 16) + 1, 16):
                    if fill_data[step_it2] != 0 and raw_melody_data[step_it2 // 32][step_it2 % 32] != 0:
                        return 2  # 强调型的加花
            return 3

        step_dx = 0
        while True:
            # for step_it in range(len(fill_data) + 1):  # 这里有个加一是为了处理加花最后部分
            if step_dx % 32 == 0:  # 处在整小节的位置
                if flag_no_fill is False and step_dx < len(fill_data) and fill_data[step_dx] != 0 and len(NoteDict[fill_data[step_dx]]) <= 1 and fill_data[step_dx + 1: step_dx + 9] == [0] * 8:  # 如果小节的第一个步长有一个音符，而之后的一整拍都没有音符，则说明这个音符属于上一个加花
                    fill_note_count += 1
                    if step_dx < len(raw_melody_data) * 32 and raw_melody_data[step_dx // 32][step_dx % 32] != 0:
                        same_note_count += 1
                    else:
                        diff_note_count += 1
                    filltype = classify(step_dx)
                    self.get_fill_pat(filltype, fill_data, judge_start_step_dx, step_dx)
                    for step_it3 in range(judge_start_step_dx, step_dx + 1):
                        self.classify_data[part_dx][song_dx][step_it3] = filltype
                    fill_note_count = 0
                    same_note_count = 0
                    diff_note_count = 0
                    empty_note_count = 0
                    judge_start_step_dx = step_dx
                    flag_no_fill = True
                    step_dx += 1
                    continue
                elif flag_no_fill is False:  # 结算上一个小节的加花的情况。这个条件不直接continue，因为这一拍的情况没有结算
                    filltype = classify(step_dx - empty_note_count)
                    self.get_fill_pat(filltype, fill_data, judge_start_step_dx, step_dx - empty_note_count)
                    for step_it3 in range(judge_start_step_dx, step_dx - empty_note_count):
                        self.classify_data[part_dx][song_dx][step_it3] = filltype
                    fill_note_count = 0
                    same_note_count = 0
                    diff_note_count = 0
                    empty_note_count = 0
                    judge_start_step_dx = step_dx
                    flag_no_fill = True
                if step_dx >= len(fill_data):
                    break  # 对于越界的一个步长 直接退出
            if fill_data[step_dx] == 0:
                if flag_no_fill:  # 当前时段没有加花
                    judge_start_step_dx = step_dx
                    step_dx += 1
                    continue
                empty_note_count += 1
                if step_dx < len(raw_melody_data) * 32 and raw_melody_data[step_dx // 32][step_dx % 32] != 0:
                    diff_note_count += 1
                if fill_data[step_dx: step_dx + 8] == [0] * 8:  # 超过1拍没有加花的音符 需要重新判断分类
                    filltype = classify(step_dx)
                    self.get_fill_pat(filltype, fill_data, judge_start_step_dx, step_dx)
                    for step_it3 in range(judge_start_step_dx, step_dx):
                        self.classify_data[part_dx][song_dx][step_it3] = filltype
                    fill_note_count = 0
                    same_note_count = 0
                    diff_note_count = 0
                    empty_note_count = 0
                    judge_start_step_dx = step_dx
                    flag_no_fill = True
                    step_dx += 8
                    continue
            else:
                if flag_no_fill is True:
                    judge_start_step_dx = step_dx
                    flag_no_fill = False
                fill_note_count += 1
                empty_note_count = 0
                if step_dx < len(raw_melody_data) * 32 and raw_melody_data[step_dx // 32][step_dx % 32] != 0:
                    same_note_count += 1
                else:
                    diff_note_count += 1
            step_dx += 1


class FillPatternEncode(MusicPatternEncodeStep):

    def handle_rare_pattern(self, pattern_dx, raw_note_list, common_patterns):
        # 在常见的加花列表里找不到某一个加花组合的处理方法：
        # a.寻找符合以下条件的加花组合
        # a1.整半拍处的休止情况完全相同 且没有模仿关系
        # a2.该组合的所有音符是待求组合所有音符的子集 或与待求组合的音符相差7的倍数
        # a3.满足上述两个条件的情况下，该旋律组合与待求旋律组合的差别尽量小
        # b.记为common_patterns+1
        if pattern_dx == -1:
            choose_pattern = 0  # 选取的pattern
            choose_pattern_diff_score = 75  # 两个旋律组合的差异程度
            for common_pat_it in range(1, len(common_patterns)):
                total_note_count = 0
                diff_note_count = 0
                # 1.1.检查该旋律组合是否符合要求
                if raw_note_list.count(1) > 0 or common_patterns[common_pat_it].count(1) > 0:  # 有模仿关系 则不近似
                    continue
                note_satisfactory = True
                for note_group_it in range(len(common_patterns[common_pat_it])):
                    # 1.1.1.检查有无模仿关系 且整半拍的休止情况是否相同
                    if note_group_it % 2 == 0:
                        if bool(common_patterns[common_pat_it][note_group_it]) ^ bool(raw_note_list[note_group_it]):  # 一个为休止符 一个不为休止符
                            note_satisfactory = False
                            break
                    if common_patterns[common_pat_it][note_group_it] == 0 and raw_note_list[note_group_it] == 0:  # 如果这个时间步长是休止符的话，直接进入下个时间步长
                        continue
                    elif common_patterns[common_pat_it][note_group_it] == 0 and raw_note_list[note_group_it] != 0:  # pattern是休止而待求片段不是休止 计入不同后进入下个时间步长
                        total_note_count += len(raw_note_list[note_group_it]) - 1
                        diff_note_count += (len(raw_note_list[note_group_it]) - 1) * 1.2
                        continue
                    elif common_patterns[common_pat_it][note_group_it] != 0 and raw_note_list[note_group_it] == 0:  # pattern不是休止而待求片段是休止 计入不同后进入下个时间步长
                        total_note_count += len(common_patterns[common_pat_it][note_group_it]) - 1
                        diff_note_count += (len(common_patterns[common_pat_it][note_group_it]) - 1) * 1.2
                        continue
                    # 1.1.2.求出相对音高组合并对7取余数 按升降号分开
                    cur_pattern_note_list = common_patterns[common_pat_it][note_group_it][1:]  # 这个时间步长中常见组合的真实音符组合
                    cur_pattern_note_list_div7 = [[t[0] % 7, t[1]] for t in cur_pattern_note_list]  # 对7取余数
                    cur_pattern_sj_set = {t[1] for t in cur_pattern_note_list}  # 升降关系的集合
                    cur_pattern_note_dict_div7 = {t0: {t1 for [t1, t2] in cur_pattern_note_list_div7 if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开
                    cur_pattern_note_dict = {t0: {t1 for [t1, t2] in cur_pattern_note_list if t2 == t0} for t0 in cur_pattern_sj_set}  # 按照升降号将音符分开

                    cur_step_note_list = raw_note_list[note_group_it][1:]  # 这个时间步长中待求组合的真实音符组合
                    cur_step_note_list_div7 = [[t[0] % 7, t[1]] for t in cur_step_note_list]  # 对7取余数
                    cur_step_sj_set = {t[1] for t in cur_step_note_list}  # 升降关系的集合
                    cur_step_note_dict_div7 = {t0: {t1 for [t1, t2] in cur_step_note_list_div7 if t2 == t0} for t0 in cur_step_sj_set}  # 按照升降号将音符分开
                    cur_step_note_dict = {t0: {t1 for [t1, t2] in cur_step_note_list if t2 == t0} for t0 in cur_step_sj_set}  # 按照升降号将音符分开
                    # 1.1.3.遍历升降号 如果组合音符列表是待求音符列表的子集的话则认为是可以替代的
                    for sj in cur_pattern_sj_set:
                        try:
                            if not cur_pattern_note_dict_div7[sj].issubset(cur_step_note_dict_div7[sj]):
                                note_satisfactory = False
                                break
                        except KeyError:
                            note_satisfactory = False
                            break
                    if not note_satisfactory:
                        break
                    # 1.1.4.计算该组合与待求组合之间的差异分
                    # print(bar_index, bar_pg_pattern_iterator, group_iterator, common_pg_iterator, cur_step_note_dict, cur_pattern_note_dict)
                    for sj in cur_step_note_dict:  # 遍历待求音符组合的所有升降号
                        total_note_count += len(cur_step_note_dict[sj])
                        if sj not in cur_pattern_note_dict:
                            diff_note_count += len(cur_step_note_dict[sj])
                            break
                        for rel_note in cur_step_note_dict[sj]:
                            if rel_note not in cur_pattern_note_dict[sj]:
                                diff_note_count += 1
                        for pattern_note in cur_pattern_note_dict[sj]:
                            if pattern_note not in cur_step_note_dict[sj]:
                                diff_note_count += 1.2  # 如果pattern中有而待求组合中没有，记为1.2分。（这里其实可以说是“宁缺毋滥”）
                if not note_satisfactory:
                    continue
                # 1.2.如果找到符合要求的组合 将其记录下来
                pattern_diff_score = (100 * diff_note_count) // total_note_count
                # print(bar_index, bar_pg_pattern_iterator, common_pg_iterator, pattern_diff_score)
                if pattern_diff_score < choose_pattern_diff_score:
                    choose_pattern = common_pat_it
                    choose_pattern_diff_score = pattern_diff_score
            # 2.如果有相似的旋律组合 则保存该旋律组合 否则保存为COMMON_PIANO_GUITAR_PATTERNS+1
            # print(bar_pg_pattern_iterator, raw_bar_pattern_list, choose_pattern, choose_pattern_diff_score)
            if choose_pattern != 0:
                pattern_dx = choose_pattern
            else:
                pattern_dx = COMMON_FILL_PATTERN_NUMBER + 1
        # print(bar_pattern_list, '\n')
        return pattern_dx


class FillTrainData:

    speed_ratio_dict = {0.5: 1, 1: 2}

    def __init__(self, rel_melody_data, core_note_ary, melody_pat_data, continuous_bar_data):

        self.input_data = []  # 输入model的数据
        self.output_data = []  # 从model输出的数据
        # 1.从数据集中读取歌的加花数据
        raw_fill_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        part_count = 1
        song_tone_list = get_tone_list()  # 歌曲的调式信息
        while True:
            fill_part_data = get_raw_song_data_from_dataset('fill' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if fill_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_fill_data.append(fill_part_data)
            # flatten_pg_data += pg_part_data
            part_count += 1
        # 2.将主旋律组合展成以组合步长为单位的列表
        flatten_melody_pat_data = [[] for t in range(TRAIN_FILE_NUMBERS)]
        for song_it in range(len(melody_pat_data)):
            for bar in range(get_dict_max_key(melody_pat_data[song_it]) + 1):
                flatten_melody_pat_data[song_it].extend(melody_pat_data[song_it][bar])
        # 3.将绝对音高转变为相对音高
        for fill_part_it, fill_part in enumerate(raw_fill_data):  # 将绝对音高转变为相对音高
            for song_it in range(TRAIN_FILE_NUMBERS):
                if fill_part[song_it] != {} and song_tone_list[song_it] is not None:
                    raw_fill_data[fill_part_it][song_it] = self.get_rel_fill_data(fill_part[song_it], rel_melody_data[song_it], core_note_ary[song_it], song_tone_list[song_it])  # 将加花数据变更为相对音高数据
        # 4.获取最常见的加花组合
        common_pattern_cls = CommonMusicPatterns(COMMON_FILL_PATTERN_NUMBER)  # 这个类可以获取常见的主旋律组合
        if FLAG_IS_TRAINING is True:  # 训练模式
            common_pattern_cls.train(raw_fill_data, 0.125, 1, multipart=True)
            common_pattern_cls.store('Fill')  # 存储在sqlite文件中
        else:
            common_pattern_cls.restore('Fill')  # 直接从sqlite文件中读取
        self.common_fill_pats = common_pattern_cls.common_pattern_list  # 常见的加花组合列表
        # 5.将其编码为组合并获取模型的输入输出数据
        for part_it in range(len(raw_fill_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                # print(song_iterator)
                if raw_fill_data[part_it][song_it] != {} and rel_melody_data[song_it] != {}:
                    fill_pat_list = FillPatternEncode(self.common_fill_pats, raw_fill_data[part_it][song_it], 0.125, 1).music_pattern_ary
                    self.get_model_io_data(fill_pat_list, flatten_melody_pat_data[song_it], continuous_bar_data[song_it])
        # print('\n\n\n\n\n')
        # for t in self.input_data[:50]:
        #     print(t)
        # print('\n\n\n')
        # for t in self.output_data[:50]:
        #     print(t)
        # print(len(self.input_data), len(self.output_data))

    def get_rel_fill_data(self, fill_data, rel_melody_data, core_note_ary, tone):
        # 1.首先将fill_Data和melody_data展成一维数组的形式
        flatten_fill_data = []
        for key in range(get_dict_max_key(fill_data) + 1):
            flatten_fill_data += fill_data[key]
        # 2.将fill_data转化成相对音高形式
        if tone == TONE_MAJOR:
            flatten_fill_data = one_song_rel_notelist_melody(flatten_fill_data, TONE_MAJOR, 72, use_note_dict=True)
        elif tone == TONE_MINOR:
            flatten_fill_data = one_song_rel_notelist_melody(flatten_fill_data, TONE_MINOR, 69, use_note_dict=True)
        else:
            raise ValueError
        # 3.寻找模仿结构
        imitate_fill_list = judge_imitation(rel_melody_data, flatten_fill_data, self.speed_ratio_dict)
        # 4.得到相对音高列表(相对于同期主旋律的骨干音)
        rel_fill_data = []
        for step_it in range(0, len(flatten_fill_data), 8):
            if imitate_fill_list[step_it // 8] not in [0, 1]:
                rel_fill_data.extend([[1, imitate_fill_list[step_it // 8]]] + [1 for t in range(7)])
            elif imitate_fill_list[step_it // 8] == 1:
                rel_fill_data.extend([1 for t in range(8)])
            elif step_it >= len(rel_melody_data):  # 这一拍没有主旋律
                rel_fill_data.extend([0, 0, 0, 0, 0, 0, 0, 0])
            else:  # 这一拍没有模仿结构
                for note_it in range(step_it, step_it + 8):
                    if flatten_fill_data[note_it] == 0 or core_note_ary[note_it] == 0:  # 这一拍没有加花或主旋律
                        rel_fill_data.append(0)
                    else:
                        rel_fill_data.append([0] + [[t[0] - core_note_ary[note_it][0][0], t[1] - core_note_ary[note_it][0][1]] for t in flatten_fill_data[note_it]])  # 添加加花数据和主旋律数据的音高差值
        return rel_fill_data

    def get_model_io_data(self, fill_data, melody_pat_data, continuous_bar_data):
        # 模型输入内容为当前时间的编码，前四拍+当拍+后两拍的主旋律，前三拍的加花，以及上一次的四拍加花和四拍主旋律 总计19
        if len(melody_pat_data) <= 6:
            return
        for fill_step_it in range(len(fill_data)):
            if fill_step_it >= len(melody_pat_data) + 4:
                break
            if fill_data[fill_step_it] != 0 and melody_pat_data[fill_step_it] != 0:
                time_in_bar = fill_step_it % 4
                time_add = (1 - continuous_bar_data[fill_step_it // 4] % 2) * 4
                input_time_data = [time_in_bar + time_add] + [0 for t in range(8)]
                output_time_data = [time_in_bar + time_add] + [0 for t in range(8)]

                if fill_step_it <= 3:
                    input_time_data = input_time_data + [0 for t in range(4 - fill_step_it)] + melody_pat_data[:(fill_step_it + 3)]
                    output_time_data = output_time_data + [0 for t in range(4 - fill_step_it)] + melody_pat_data[:(fill_step_it + 3)]
                elif fill_step_it >= len(melody_pat_data) - 2:
                    input_time_data = input_time_data + melody_pat_data[fill_step_it - 4:] + [0 for t in range(fill_step_it - len(melody_pat_data) + 3)]
                    output_time_data = output_time_data + melody_pat_data[fill_step_it - 4:] + [0 for t in range(fill_step_it - len(melody_pat_data) + 3)]
                else:
                    input_time_data = input_time_data + melody_pat_data[fill_step_it - 4: fill_step_it + 3]
                    output_time_data = output_time_data + melody_pat_data[fill_step_it - 4: fill_step_it + 3]
                if fill_step_it <= 2:
                    input_time_data = input_time_data + [0 for t in range(3 - fill_step_it)] + fill_data[:fill_step_it]
                    output_time_data = output_time_data + [0 for t in range(2 - fill_step_it)] + fill_data[:(fill_step_it + 1)]
                else:
                    input_time_data = input_time_data + fill_data[fill_step_it - 3: fill_step_it]
                    output_time_data = output_time_data + fill_data[fill_step_it - 2: fill_step_it + 1]

                for lookback_it in range(fill_step_it - 4, -1, -1):
                    if fill_data[lookback_it] != 0 and melody_pat_data[lookback_it] != 0:
                        if lookback_it <= 2:
                            input_time_data[4 - lookback_it: 5] = melody_pat_data[:lookback_it + 1]
                            input_time_data[8 - lookback_it: 9] = fill_data[:lookback_it + 1]
                            output_time_data[4 - lookback_it: 5] = melody_pat_data[:lookback_it + 1]
                            output_time_data[8 - lookback_it: 9] = fill_data[:lookback_it + 1]
                        else:
                            input_time_data[1: 5] = melody_pat_data[lookback_it - 3: lookback_it + 1]
                            input_time_data[5: 9] = fill_data[lookback_it - 3: lookback_it + 1]
                            output_time_data[1: 5] = melody_pat_data[lookback_it - 3: lookback_it + 1]
                            output_time_data[5: 9] = fill_data[lookback_it - 3: lookback_it + 1]
                        break
                self.input_data.append(input_time_data)
                self.output_data.append(output_time_data)


class FillTrainData2:

    def __init__(self, raw_melody_data, section_data, continuous_bar_data):  # rel_melody_data, core_note_ary, melody_pat_data, continuous_bar_data):
        # 1.从数据集中读取歌的加花数据
        raw_fill_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        part_count = 1
        song_bpm_list = get_bpm_list()  # 歌曲的调式信息
        while True:
            fill_part_data = get_raw_song_data_from_dataset('fill' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if fill_part_data == [{} for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_fill_data.append(fill_part_data)
            # flatten_pg_data += pg_part_data
            part_count += 1
        # 2.将加花的数据展开成以步长为单位的形式
        flatten_fill_data = [[[] for t1 in range(TRAIN_FILE_NUMBERS)] for t0 in range(part_count)]
        for fill_part_it, fill_part in enumerate(raw_fill_data):  # 将绝对音高转变为相对音高
            for song_it in range(TRAIN_FILE_NUMBERS):
                for bar in range(get_dict_max_key(fill_part[song_it]) + 1):
                    flatten_fill_data[fill_part_it][song_it].extend(fill_part[song_it][bar])
        # 3.对加花数据进行分类和组合，分成过渡型、强调型、补充型三种。组合取全部的，而不只取最常见的
        self.fill_type_pat_cls = FillClassifyAndPats(part_count)  # 记录加花数据的分类和组合
        # fill_type_data = [[[] for t1 in range(TRAIN_FILE_NUMBERS)] for t0 in range(part_count)]
        for fill_part_it, fill_part in enumerate(flatten_fill_data):  # 将绝对音高转变为相对音高
            for song_it in range(TRAIN_FILE_NUMBERS):
                self.fill_type_pat_cls.run_1song(fill_part_it, song_it, fill_part[song_it], raw_melody_data[song_it], song_bpm_list[song_it])
        # 4.对于前一拍无加花的情况，本拍是否加花
        self.all_fill_ary = [0 for t in range(4)]
        self.keypress_fill_ary = np.zeros((4, 16), dtype=np.int32)  # 主旋律按键情况对应加花的频数
        self.timecode_fill_ary = np.zeros((4, 8), dtype=np.int32)  # 时间编码对应加花的频数（最大值8。0-4为非过渡小节的时间编码 5-8为过渡小节的时间编码）
        self.sec_nfill_ary = np.zeros((3, 6), dtype=np.int32)  # 本段落此前的加花概率对应此拍不加花的频数
        self.sameinsec_fill_ary = np.zeros((3, 6), dtype=np.int32)  # 本段落此前的加花概率对应此拍同种类型加花的频数
        for fill_part_it, fill_part in enumerate(flatten_fill_data):  # 将绝对音高转变为相对音高
            for song_it in range(TRAIN_FILE_NUMBERS):
                self.cal_fill_freq(self.fill_type_pat_cls.classify_data[fill_part_it][song_it], raw_melody_data[song_it], section_data[song_it], continuous_bar_data[song_it])
        # 5.对于前一拍有加花的情况，本拍是否加花
        self.all_fill_rep_ary = [0 for t in range(6)]
        self.keypress_fill_rep_ary = np.zeros((6, 32), dtype=np.int32)  # 主旋律按键情况对应加花的频数
        self.timecode_fill_rep_ary = np.zeros((6, 32), dtype=np.int32)  # 时间编码对应加花的频数（最大值8。0-4为非过渡小节的时间编码 5-8为过渡小节的时间编码）
        for fill_part_it, fill_part in enumerate(flatten_fill_data):  # 将绝对音高转变为相对音高
            for song_it in range(TRAIN_FILE_NUMBERS):
                self.cal_fill_freq_repeat(self.fill_type_pat_cls.classify_data[fill_part_it][song_it], raw_melody_data[song_it], section_data[song_it], continuous_bar_data[song_it])

    def cal_fill_freq(self, fill_type_data, raw_melody_data, section_data, continuous_bar_data):
        """计算在前一拍没有同种加花的情况下，本拍加花的频率"""
        section_beats = 0
        section_fill_count = [0, 0, 0, 0]
        sec_dx_old = -1
        fill_type_1beat_old = 0  # 上一拍的加花情况，用于判断这一拍和上一拍的加花是否同类
        for beat_it in range(len(fill_type_data) // 8):
            # 1.获取此拍加花情况
            fill_type_1beat = 0
            for step_it in range(beat_it * 8, beat_it * 8 + 8):
                if fill_type_data[step_it] != 0:
                    fill_type_1beat = fill_type_data[step_it]
                    break
            # 2.获取当前拍所属段落
            if section_data:
                sec_dx = -1  # 记录这一拍属于这首歌的第几个乐段
                for sec_it in range(len(section_data)):
                    if section_data[sec_it][0] * 4 + section_data[sec_it][1] > beat_it:  # 将区段的起始拍和当前拍进行比较 如果起始拍在当前拍之前则说明是属于这个区段
                        sec_dx = sec_it - 1
                        break
                if sec_dx == -1:
                    sec_dx = len(section_data) - 1  # 属于这首歌的最后一个区段
            if (section_data and section_data[sec_dx][2] != SECTION_EMPTY) or ((not section_data) and beat_it // 4 < len(continuous_bar_data) and continuous_bar_data[beat_it // 4] != 0):  # 如果这一拍属于间奏 则统计加花时不计算在内
                flag_curstep_no_melody = False
            else:
                flag_curstep_no_melody = True
            # 3.判断此拍加花是否为前面加花的延续。判断方式为两拍加花内容相同，前一拍不为空，且fill_type_data中间不能有0
            flag_fill_rep = False
            if fill_type_1beat_old != 0 and fill_type_1beat == fill_type_1beat_old and beat_it >= 1:
                if fill_type_data[beat_it * 8 - 1] != 0:
                    flag_fill_rep = True
            if flag_fill_rep is False and flag_curstep_no_melody is False:
                self.all_fill_ary[fill_type_1beat] += 1
            # 4.计算主旋律的按键情况对应同时期的加花情况。按键情况的编码方式为Σ(1,4) pow(2, x-1)*（第x个半拍有无按键）
            if flag_fill_rep is False and flag_curstep_no_melody is False:
                if (section_data and section_data[sec_dx][1] != SECTION_EMPTY) or ((not section_data) and continuous_bar_data[beat_it // 4] != 0):  # 如果这一拍属于间奏 则不计算在内
                    try:
                        melody_mark = 0
                        for step_it in range(beat_it * 8 - 8, beat_it * 8 + 8, 4):
                            if step_it >= 0 and raw_melody_data[step_it // 32][step_it % 32: step_it % 32 + 4] != [0, 0, 0, 0]:
                                melody_mark += pow(2, (step_it - beat_it * 8 + 8) // 4)
                        self.keypress_fill_ary[fill_type_1beat][melody_mark] += 1
                    except KeyError:
                        pass
            # 5.计算当前时间的编码对应同时期加花的情况。0-4为非过渡小节的时间编码 5-8为过渡小节的时间编码
            try:
                if section_data:
                    sec_start_maj_beat = int(section_data[sec_dx][0] * 4)  # 这个乐段的起始重拍（即不计算前面的
                    if sec_dx == len(section_data) - 1:
                        sec_end_beat = len(raw_melody_data) * 4
                    else:
                        sec_end_beat = min(len(raw_melody_data) * 4, section_data[sec_dx + 1][0] * 4)  # 这个乐段的结尾重拍在第多少拍
                    time_add = (beat_it - sec_start_maj_beat) % 8
                    if sec_end_beat - beat_it + time_add % 4 <= 4:  # 这一拍处在距离乐段结束不足1小节的位置
                        timecode = 4 + time_add % 4
                    else:
                        timecode = time_add % 4
                else:
                    if beat_it // 4 < len(continuous_bar_data) and continuous_bar_data[beat_it // 4] != 0 and (beat_it // 4 == len(continuous_bar_data) - 1 or continuous_bar_data[beat_it // 4 + 1] == 0):  # 下一小节是否没有旋律
                        timecode = 4 + (beat_it % 4)
                    else:
                        timecode = beat_it % 4
                if flag_fill_rep is False and flag_curstep_no_melody is False:  # 上一拍有同种加花的情况下，只记录乐段的情况（用于后面给计数器赋值），而不记入对照表中
                    self.timecode_fill_ary[fill_type_1beat][timecode] += 1
            except KeyError:
                pass
            # 6.处理当前乐段的各类加花数量的计数器。如果变更乐段（有乐段的情况）或下一小节为空小节（无乐段的情况）则计数器清零
            if section_data:  # 有乐段的情况
                if sec_dx != sec_dx_old:
                    sec_dx_old = sec_dx
                    section_fill_count = [0, 0, 0, 0]
                    section_beats = 0
            else:  # 无乐段的情况
                if beat_it // 4 >= len(continuous_bar_data) or continuous_bar_data[beat_it // 4] == 0:
                    section_fill_count = [0, 0, 0, 0]
                    section_beats = 0
            # 7.计算本段落此前的加花频率对应当前加花的情况。
            if flag_fill_rep is False and flag_curstep_no_melody is False:
                if fill_type_1beat != 0:
                    fill_freq = (20 * section_fill_count[fill_type_1beat] + 1) / (20 * (section_beats + 1))
                    fill_index = self.index_by_freq(fill_freq)
                    self.sameinsec_fill_ary[fill_type_1beat - 1][fill_index] += 1
                else:
                    for count_it in range(1, 4):
                        fill_freq = (25 * section_fill_count[count_it] + 1) / (25 * (section_beats + 1))
                        fill_index = self.index_by_freq(fill_freq)
                        self.sec_nfill_ary[count_it - 1][fill_index] += 1
            # 8.一次循环的收尾部分
            section_fill_count[fill_type_1beat] += 1  # 本段落内该种类型加花计数器加一
            fill_type_1beat_old = fill_type_1beat
            section_beats += 1

    def cal_fill_freq_repeat(self, fill_type_data, raw_melody_data, section_data, continuous_bar_data):
        """计算在前一拍有同种加花的情况下，本拍加花的频率"""
        fill_type_1beat_old = 0  # 上一拍的加花情况，用于判断这一拍和上一拍的加花是否同类
        fill_rep_beats = 0  # 此前已经有多少拍的同类型加花
        for beat_it in range(len(fill_type_data) // 8):
            # 1.获取此拍加花情况
            fill_type_1beat = 0
            for step_it in range(beat_it * 8, beat_it * 8 + 8):
                if fill_type_data[step_it] != 0:
                    fill_type_1beat = fill_type_data[step_it]
                    break
            # 2.获取当前拍所属段落
            if section_data:
                sec_dx = -1  # 记录这一拍属于这首歌的第几个乐段
                for sec_it in range(len(section_data)):
                    if section_data[sec_it][0] * 4 + section_data[sec_it][1] > beat_it:  # 将区段的起始拍和当前拍进行比较 如果起始拍在当前拍之前则说明是属于这个区段
                        sec_dx = sec_it - 1
                        break
                if sec_dx == -1:
                    sec_dx = len(section_data) - 1  # 属于这首歌的最后一个区段
            # 3.判断此拍加花是否为前面加花的延续。判断方式为两拍加花内容相同，前一拍不为空，且fill_type_data中间不能有0
            flag_fill_rep = False
            if fill_type_1beat_old != 0 and fill_type_1beat == fill_type_1beat_old and beat_it >= 1:
                if fill_type_data[beat_it * 8 - 1] != 0:
                    flag_fill_rep = True
            if fill_type_1beat_old != 0:  # 前一拍的加花不为空 才进入下面的这个判断
                if flag_fill_rep is True:  # 只有在加花内容为延续前一拍内容，或加花变更为0时才插入到数组中
                    self.all_fill_rep_ary[fill_type_1beat + 2] += 1
                elif fill_type_1beat == 0:
                    self.all_fill_rep_ary[fill_type_1beat_old - 1] += 1
            # 4.计算主旋律的按键情况对应同时期的加花情况。按键情况的编码方式为Σ(1,4) pow(2, x-1)*（第x个半拍有无按键）
            # if flag_fill_rep is True:
            if fill_type_1beat_old != 0:  # 前一拍的加花不为空 才进入下面的这个判断
                try:
                    melody_mark = 0
                    for step_it in range(beat_it * 8 - 8, beat_it * 8 + 8, 4):
                        if raw_melody_data[step_it // 32][step_it % 32: step_it % 32 + 4] != [0, 0, 0, 0]:
                            melody_mark += pow(2, (step_it - beat_it * 8 + 8) // 4)
                    if flag_fill_rep is True:  # 只有在加花内容为延续前一拍内容，或加花变更为0时才插入到数组中
                        self.keypress_fill_rep_ary[fill_type_1beat + 2][melody_mark] += 1
                    elif fill_type_1beat == 0:
                        self.keypress_fill_rep_ary[fill_type_1beat_old - 1][melody_mark] += 1
                except KeyError:
                    pass
            # 5.计算当前时间的编码对应同时期加花的情况。0-4为非过渡小节的时间编码 5-8为过渡小节的时间编码
            if fill_type_1beat_old != 0:
                if section_data:
                    sec_start_maj_beat = int(section_data[sec_dx][0] * 4)  # 这个乐段的起始重拍（即不计算前面的
                    if sec_dx == len(section_data) - 1:
                        sec_end_beat = len(raw_melody_data) * 4
                    else:
                        sec_end_beat = min(len(raw_melody_data) * 4, section_data[sec_dx + 1][0] * 4)  # 这个乐段的结尾重拍在第多少拍
                    time_add = (beat_it - sec_start_maj_beat) % 8
                    if sec_end_beat - beat_it + time_add % 4 <= 4:  # 这一拍处在距离乐段结束不足1小节的位置
                        timecode = 4 + time_add % 4
                    else:
                        timecode = time_add % 4
                else:
                    if beat_it // 4 < len(continuous_bar_data) and continuous_bar_data[beat_it // 4] != 0 and (beat_it // 4 == len(continuous_bar_data) - 1 or continuous_bar_data[beat_it // 4 + 1] == 0):  # 下一小节是否没有旋律
                        timecode = 4 + (beat_it % 4)
                    else:
                        timecode = beat_it % 4
                if flag_fill_rep is True:  # 只有在加花内容为延续前一拍内容，或加花变更为0时才插入到数组中
                    self.timecode_fill_rep_ary[fill_type_1beat + 2][(3 if fill_rep_beats >= 3 else fill_rep_beats) * 8 + timecode] += 1
                elif fill_type_1beat == 0:
                    self.timecode_fill_rep_ary[fill_type_1beat_old - 1][(3 if fill_rep_beats >= 3 else fill_rep_beats) * 8 + timecode] += 1
            # 6.一次循环的收尾部分
            if flag_fill_rep:
                fill_rep_beats += 1  # 记录一个加花持续多少拍的计数器
            else:
                fill_rep_beats = 0
            fill_type_1beat_old = fill_type_1beat

    @staticmethod
    def index_by_freq(freq):
        threshold_ary = [0.003, 0.008, 0.015, 0.045, 0.2, 100]
        for it in range(len(threshold_ary)):
            if freq < threshold_ary[it]:
                return it
        return len(threshold_ary) - 1
