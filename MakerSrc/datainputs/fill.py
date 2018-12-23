from settings import *
from interfaces.sql.sqlite import get_bpm_list, get_raw_song_data_from_dataset, NoteDict
from interfaces.music_patterns import BaseMusicPatterns
from interfaces.utils import flat_array, DiaryLog
import numpy as np
import copy
import os


def get_freq_dx(freq):
    threshold_list = [0.003, 0.008, 0.015, 0.045, 0.2, 100]
    for it in range(len(threshold_list)):
        if freq < threshold_list[it]:
            return it
    return len(threshold_list) - 1


class FillClassifyAndPats:
    """生成加花的类型和组合"""

    def __init__(self, part_count):
        self.classify_data = [[[] for t1 in range(TRAIN_FILE_NUMBERS)] for t0 in range(part_count)]  # 对所有歌的加花数据进行分类后的结果
        self.classified_fill_pats = [[], [], []]  # 每一个加花类别中的加花组合有哪些

    def get_fill_pat(self, fill_type, fill_data, start_step_dx, end_step_dx):
        """
        根据一组加花数据，将其组合添加到classified_fill_pats中
        :param fill_type: 加花的类型
        :param fill_data: 一首歌的加花数据
        :param start_step_dx: 加花开始于哪一个步长
        :param end_step_dx: 加花结束于哪一个步长
        """
        fill_1pat = [0 for t in range(start_step_dx % 8)] + fill_data[start_step_dx: end_step_dx]
        fill_1pat = copy.deepcopy([(NoteDict.nd[t] if t != 0 else 0) for t in fill_1pat])  # 转换成绝对音高
        if len(fill_1pat) % 8 != 0:
            fill_1pat.extend([0] * (8 - len(fill_1pat) % 8))
        if fill_1pat not in self.classified_fill_pats[fill_type - 1]:
            self.classified_fill_pats[fill_type - 1].append(fill_1pat)

    def run_1song(self, part_dx, song_dx, fill_data, raw_melody_data, bpm):
        """
        对加花数据进行分类，分成过渡型、强调型、补充型三种
        :param part_dx: 加花的第几个部分
        :param song_dx: 歌曲编号
        :param fill_data: 一首歌的加花数据
        :param raw_melody_data: 这一首歌的主旋律数据
        :param bpm: 这一首歌的bpm数据（每分钟有多少拍）
        """
        fill_note_count = 0  # 一段时间内加花的音符计数
        same_keypress_count = 0  # 一段时间内加花和主旋律音符同时出现的计数
        diff_keypress_count = 0  # 一段时间内加花和主旋律音符不同时出现的计数
        empty_note_count = 0  # 音符空白期的时间长度
        judge_start_step_dx = 0  # 这“一段时间”具体开始于哪一步
        self.classify_data[part_dx][song_dx] = [0 for t in range(len(fill_data))]  # 分类结果
        flag_no_fill = True  # 当前时间段没有加花

        def classify(judge_end_step_dx):
            """对一段加花进行分类"""
            if fill_note_count == 0:
                return 0
            if bpm >= 90 and fill_note_count >= 5 and (judge_end_step_dx - judge_start_step_dx) / fill_note_count < 4:
                return 1  # 过渡型的加花：音符比较密集且至少有5个
            elif fill_note_count >= 5 and (judge_end_step_dx - judge_start_step_dx) / fill_note_count <= 2:
                return 1
            elif same_keypress_count / (same_keypress_count + diff_keypress_count) > 0.5:
                for step_it2 in range(judge_start_step_dx if judge_start_step_dx % 16 == 0 else 16 * (judge_start_step_dx // 16 + 1), 16 * (judge_end_step_dx // 16) + 1, 16):
                    if fill_data[step_it2] != 0 and raw_melody_data[step_it2] != 0:
                        return 2  # 强调型的加花：和同时期主旋律按键的重叠程度较高，且出现在整两拍处
            return 3  # 其余：补充型加花

        def reset_variables():
            """进行完一次加花内容的判断之后，重置一些计数和flag的变量"""
            nonlocal fill_note_count
            nonlocal same_keypress_count
            nonlocal diff_keypress_count
            nonlocal empty_note_count
            nonlocal flag_no_fill
            nonlocal judge_start_step_dx
            fill_note_count = 0
            same_keypress_count = 0
            diff_keypress_count = 0
            empty_note_count = 0
            judge_start_step_dx = step_dx
            flag_no_fill = True

        step_dx = 0
        while True:
            # for step_it in range(len(fill_data) + 1):  # 这里有个加一是为了处理加花最后部分
            if step_dx % 32 == 0:  # 处在整小节的位置
                if flag_no_fill is False and step_dx < len(fill_data) and fill_data[step_dx] != 0 and len(NoteDict.nd[fill_data[step_dx]]) <= 1 and fill_data[step_dx + 1: step_dx + 9] == [0] * 8:  # 如果小节的第一个步长有一个音符，而之后的一整拍都没有音符，则说明这个音符属于上一个加花
                    fill_note_count += 1
                    if step_dx < len(raw_melody_data) and raw_melody_data[step_dx] != 0:
                        same_keypress_count += 1
                    else:
                        diff_keypress_count += 1
                    filltype = classify(step_dx)  # 获取这段加花数据的类型
                    self.get_fill_pat(filltype, fill_data, judge_start_step_dx, step_dx)  # 将这段加花数据记录在pat data中
                    for backward_step_it in range(judge_start_step_dx, step_dx + 1):  # 记录这首歌这段时期的加花内容
                        self.classify_data[part_dx][song_dx][backward_step_it] = filltype
                    reset_variables()  # 重置变量值，便于下一轮循环
                    step_dx += 1
                    continue
                elif flag_no_fill is False:  # 结算上一个小节的加花的情况。这个条件不直接continue，因为这一拍的情况没有结算
                    filltype = classify(step_dx - empty_note_count)
                    self.get_fill_pat(filltype, fill_data, judge_start_step_dx, step_dx - empty_note_count)
                    for backward_step_it in range(judge_start_step_dx, step_dx - empty_note_count):
                        self.classify_data[part_dx][song_dx][backward_step_it] = filltype
                    reset_variables()
                if step_dx >= len(fill_data):
                    break  # 对于越界的一个步长 直接退出
            if fill_data[step_dx] == 0:
                if flag_no_fill:  # 当前时段没有加花
                    judge_start_step_dx = step_dx
                    step_dx += 1
                    continue
                empty_note_count += 1  # 记录这一个步长没有加花数据
                if step_dx < len(raw_melody_data) and raw_melody_data[step_dx] != 0:
                    diff_keypress_count += 1  # 这一个步长有主旋律数据，但是没有加花数据
                if fill_data[step_dx: step_dx + 8] == [0 for t in range(8)]:  # 超过1拍没有加花的音符 需要重新判断分类
                    filltype = classify(step_dx)
                    self.get_fill_pat(filltype, fill_data, judge_start_step_dx, step_dx)
                    for backward_step_it in range(judge_start_step_dx, step_dx):
                        self.classify_data[part_dx][song_dx][backward_step_it] = filltype
                    reset_variables()
                    step_dx += 8
                    continue
            else:  # 这个步长有加花数据
                if flag_no_fill is True:
                    judge_start_step_dx = step_dx
                    flag_no_fill = False
                fill_note_count += 1
                empty_note_count = 0
                if step_dx < len(raw_melody_data) and raw_melody_data[step_dx] != 0:
                    same_keypress_count += 1
                else:
                    diff_keypress_count += 1
            step_dx += 1

    def store(self):
        """存储加花的组合数据"""
        for type_it in range(3):
            fill_pattern_cls = BaseMusicPatterns(store_count=False)
            fill_pattern_cls.common_pattern_list = self.classified_fill_pats[type_it]
            fill_pattern_cls.store('Fill%d' % type_it)

    def restore(self):
        """读取加花的组合数据"""
        for type_it in range(3):
            fill_pattern_cls = BaseMusicPatterns(store_count=False)
            fill_pattern_cls.restore('Fill%d' % type_it)
            self.classified_fill_pats[type_it] = fill_pattern_cls.common_pattern_list


class FillTrainData:

    def __init__(self, raw_melody_data, section_data, continuous_bar_data):

        # 1.从数据集中读取歌的加花数据，并变更为以音符步长为单位的列表
        raw_fill_data = []  # 四维数组 第一维是piano_guitar的编号 第二维是歌曲id 第三维是小节列表(dict) 第四维是小节内容
        part_count = 1
        song_bpm_list = get_bpm_list()  # 歌曲的调式信息
        while True:
            fill_part_data = get_raw_song_data_from_dataset('fill' + str(part_count), None)  # 从sqlite文件中读取的数据。三维数组，第一维是歌的id，第二维是小节的id，第三维是小节内容
            if fill_part_data == [dict() for t in range(TRAIN_FILE_NUMBERS)]:
                break
            raw_fill_data.append(fill_part_data)
            part_count += 1
        del part_count
        for part_it in range(len(raw_fill_data)):
            for song_it in range(TRAIN_FILE_NUMBERS):
                if raw_fill_data[part_it][song_it] != dict():
                    raw_fill_data[part_it][song_it] = flat_array(raw_fill_data[part_it][song_it])
                else:
                    raw_fill_data[part_it][song_it] = []  # 对于没有和弦的歌曲，将格式转化为list格式

        # 2.对加花数据进行分类和组合，分成过渡型、强调型、补充型三种。组合取全部的，而不只取最常见的
        self.fill_type_pat_cls = FillClassifyAndPats(len(raw_fill_data))  # 记录加花数据的分类和组合
        # fill_type_data = [[[] for t1 in range(TRAIN_FILE_NUMBERS)] for t0 in range(part_count)]
        for fill_part_it, fill_part in enumerate(raw_fill_data):  # 将绝对音高转变为相对音高
            for song_it in range(TRAIN_FILE_NUMBERS):
                self.fill_type_pat_cls.run_1song(fill_part_it, song_it, fill_part[song_it], raw_melody_data[song_it], song_bpm_list[song_it])
        self.fill_type_pat_cls.store()

        # 3.对于前一拍无加花的情况，本拍是否加花
        self.all_fill_ary = [0 for t in range(4)]
        self.keypress_fill_ary = np.zeros((4, 16), dtype=np.int32)  # 主旋律按键情况对应加花的频数
        self.timecode_fill_ary = np.zeros((4, 8), dtype=np.int32)  # 时间编码对应加花的频数（最大值8。0-4为非过渡小节的时间编码 5-8为过渡小节的时间编码）
        self.sec_nfill_ary = np.zeros((3, 6), dtype=np.int32)  # 本段落此前的加花概率对应此拍不加花的频数
        self.sameinsec_fill_ary = np.zeros((3, 6), dtype=np.int32)  # 本段落此前的加花概率对应此拍同种类型加花的频数
        for fill_part_it, fill_part in enumerate(raw_fill_data):  # 将绝对音高转变为相对音高
            for song_it in range(TRAIN_FILE_NUMBERS):
                self.cal_fill_freq(self.fill_type_pat_cls.classify_data[fill_part_it][song_it], raw_melody_data[song_it], section_data[song_it], continuous_bar_data[song_it])
        np.save(os.path.join(PATH_PATTERNLOG, 'FillTotalCount.npy'), self.all_fill_ary)
        np.save(os.path.join(PATH_PATTERNLOG, 'FillKeypressCount.npy'), self.keypress_fill_ary)
        np.save(os.path.join(PATH_PATTERNLOG, 'FillTimecodeCount.npy'), self.timecode_fill_ary)
        np.save(os.path.join(PATH_PATTERNLOG, 'FillSecNCount.npy'), self.sec_nfill_ary)
        np.save(os.path.join(PATH_PATTERNLOG, 'FillSecYCount.npy'), self.sameinsec_fill_ary)

        # 5.对于前一拍有加花的情况，本拍是否加花
        self.all_fill_rep_ary = [0 for t in range(6)]  # 六个数分别是上一拍的加花内容为第一类/第二类/第三类，本拍不延续；上一拍的加花内容为第一类/第二类/第三类，本拍延续
        self.keypress_fill_rep_ary = np.zeros((6, 16), dtype=np.int32)  # 主旋律按键情况对应加花的频数
        self.timecode_fill_rep_ary = np.zeros((6, 32), dtype=np.int32)  # 加花已延续拍数和时间编码对应加花的频数（高两位为加花已延续拍数-1，低三位为时间编码）
        for fill_part_it, fill_part in enumerate(raw_fill_data):  # 将绝对音高转变为相对音高
            for song_it in range(TRAIN_FILE_NUMBERS):
                self.cal_fill_freq_repeat(self.fill_type_pat_cls.classify_data[fill_part_it][song_it], raw_melody_data[song_it], section_data[song_it], continuous_bar_data[song_it])
        np.save(os.path.join(PATH_PATTERNLOG, 'FillRepTotalCount.npy'), self.all_fill_rep_ary)
        np.save(os.path.join(PATH_PATTERNLOG, 'FillRepKeypressCount.npy'), self.keypress_fill_rep_ary)
        np.save(os.path.join(PATH_PATTERNLOG, 'FillRepTimecodeCount.npy'), self.timecode_fill_rep_ary)

        DiaryLog.warn('Generation of fill train data has finished!')

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
            if (section_data and section_data[sec_dx][2] != DEF_SEC_EMPTY) or ((not section_data) and beat_it // 4 < len(continuous_bar_data) and continuous_bar_data[beat_it // 4] != 0):  # 如果这一拍属于间奏 则统计加花时不计算在内
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
                if (section_data and section_data[sec_dx][1] != DEF_SEC_EMPTY) or ((not section_data) and continuous_bar_data[beat_it // 4] != 0):  # 如果这一拍属于间奏 则不计算在内
                    try:
                        melody_mark = 0
                        for step_it in range(beat_it * 8 - 8, beat_it * 8 + 8, 4):
                            if step_it >= 0 and raw_melody_data[step_it: step_it + 4] != [0, 0, 0, 0]:
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
            # 默认在一个段落的第一拍时，加花的概率在0.04(1/25)-0.05(1/20)之间
            if flag_fill_rep is False and flag_curstep_no_melody is False:
                if fill_type_1beat != 0:  # 本段落此前某种加花的频率对应当前拍出现同种加花的频率（有一些平滑处理）
                    fill_freq = (20 * section_fill_count[fill_type_1beat] + 1) / (20 * (section_beats + 1))
                    freq_dx = get_freq_dx(fill_freq)
                    self.sameinsec_fill_ary[fill_type_1beat - 1][freq_dx] += 1
                else:
                    for count_it in range(1, 4):  # 本段落此前某种加花的频率对应当前拍不加花的频率
                        fill_freq = (25 * section_fill_count[count_it] + 1) / (25 * (section_beats + 1))
                        freq_dx = get_freq_dx(fill_freq)
                        self.sec_nfill_ary[count_it - 1][freq_dx] += 1
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
            if fill_type_1beat_old != 0:  # 前一拍的加花不为空 才进入下面的这个判断
                try:
                    melody_mark = 0
                    for step_it in range(beat_it * 8 - 8, beat_it * 8 + 8, 4):
                        if raw_melody_data[step_it: step_it + 4] != [0, 0, 0, 0]:
                            melody_mark += pow(2, (step_it - beat_it * 8 + 8) // 4)
                    if flag_fill_rep is True:  # 只有在加花内容为延续前一拍内容，或加花变更为0时才插入到数组中
                        self.keypress_fill_rep_ary[fill_type_1beat + 2][melody_mark] += 1
                    elif fill_type_1beat == 0:
                        self.keypress_fill_rep_ary[fill_type_1beat_old - 1][melody_mark] += 1
                except KeyError:
                    pass
            # 5.计算当前时间的编码及加花已延续拍数对应同时期加花的情况。0-4为非过渡小节的时间编码 5-8为过渡小节的时间编码
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


class FillTestData:

    def __init__(self):
        # 1.获取加花数据的分类和组合
        self.fill_type_pat_cls = FillClassifyAndPats(0)  # 记录加花数据的分类和组合
        self.fill_type_pat_cls.restore()

        # 2.获取前一拍无加花的情况，本拍是否加花的数据
        self.all_fill_ary = np.load(os.path.join(PATH_PATTERNLOG, 'FillTotalCount.npy'))
        self.keypress_fill_ary = np.load(os.path.join(PATH_PATTERNLOG, 'FillKeypressCount.npy'))
        self.timecode_fill_ary = np.load(os.path.join(PATH_PATTERNLOG, 'FillTimecodeCount.npy'))
        self.sec_nfill_ary = np.load(os.path.join(PATH_PATTERNLOG, 'FillSecNCount.npy'))
        self.sameinsec_fill_ary = np.load(os.path.join(PATH_PATTERNLOG, 'FillSecYCount.npy'))
        assert self.all_fill_ary.shape == (4, )
        assert self.keypress_fill_ary.shape == (4, 16)
        assert self.timecode_fill_ary.shape == (4, 8)
        assert self.sec_nfill_ary.shape == (3, 6)
        assert self.sameinsec_fill_ary.shape == (3, 6)

        # 3.获取前一拍有加花的情况，本拍是否加花的数据
        self.all_fill_rep_ary = np.load(os.path.join(PATH_PATTERNLOG, 'FillRepTotalCount.npy'))
        self.keypress_fill_rep_ary = np.load(os.path.join(PATH_PATTERNLOG, 'FillRepKeypressCount.npy'))
        self.timecode_fill_rep_ary = np.load(os.path.join(PATH_PATTERNLOG, 'FillRepTimecodeCount.npy'))
        assert self.all_fill_rep_ary.shape == (6, )  # 六个数分别是上一拍的加花内容为第一类/第二类/第三类，本拍不延续；上一拍的加花内容为第一类/第二类/第三类，本拍延续
        assert self.keypress_fill_rep_ary.shape == (6, 16)  # 主旋律按键情况对应加花的频数
        assert self.timecode_fill_rep_ary.shape == (6, 32)  # 加花已延续拍数和时间编码对应加花的频数（高两位为加花已延续拍数-1，低三位为时间编码）

        DiaryLog.warn('Restoring of fill associated data has finished!')
