from settings import *
from interfaces.sql.sqlite import NoteDict
from interfaces.chord_parse import noteset2chord
import abc
import copy
import sqlite3


class BaseConfidenceLevelCheck(object, metaclass=abc.ABCMeta):
    """
    这个虚基类描述了一类检查生成的数据是否合理的方法
    检查的过程分为如下四个步骤
    a.在训练阶段，对训练数据进行一些处理，获取每一个训练数据对应的一个评价数值.如生成数据为[0, 1, 2, ..., 19]
    b.将上述数组从小到大排列，以x%的数据所在的点作为判别生成数据是否合理的阈值
    c.在生成阶段，对生成数据进行类似的处理，获取一个评价值
    d.将这个评价值和b步骤的阈值相对比，如果比阈值小则通过，否则不通过
    """
    def __init__(self):
        self.evaluating_score_list = []  # 评价数值
        self.confidence_level = None  # 阈值

    @abc.abstractmethod
    def train_1song(self, **kwargs):
        pass

    def calc_confidence_level(self, ratio, reverse=False):
        """
        计算判别阈值
        :param ratio: 比例
        :param reverse: 判别的标准是大于(reverse=True)还是小于(reverse=False)
        """
        evaluating_score_list = sorted(self.evaluating_score_list, reverse=reverse)
        prob_dx = int(len(evaluating_score_list) * ratio + 1)
        self.confidence_level = evaluating_score_list[prob_dx]

    @abc.abstractmethod
    def evaluate(self, **kwargs):
        pass

    def compare(self, score, reverse=False):
        """
        检查生成数据对应的分数是否合理
        :param score: 生成数据的评价分数
        :param reverse: 判别的标准是大于(reverse=True)还是小于(reverse=False)
        """
        if reverse is True:
            if score <= self.confidence_level:
                return False
            else:
                return True
        elif reverse is False:
            if score >= self.confidence_level:
                return False
            else:
                return True
        else:
            raise ValueError

    def store(self, mark):
        """
        将常见组合的数据保存到sqlite文件中
        :param mark: 数据字段的标签
        """
        # 1.建表
        conn = sqlite3.connect(PATH_PAT_DATASET)
        conn.execute('create table if not exists ConfidenceLevel(id integer primary key autoincrement, mark varchar(30), confidence_level float)')
        conn.commit()
        # 2.保存数据
        conn.execute("insert into ConfidenceLevel(mark, confidence_level) values ('%s', %.4f)" % (mark, self.confidence_level))
        conn.commit()

    def restore(self, mark):
        """
        从sqlite文件中获取常见的confidence_level数据
        :param mark: 数据字段的标签
        """
        conn = sqlite3.connect(PATH_PAT_DATASET)
        rows = conn.execute("select confidence_level from ConfidenceLevel where mark='%s'" % mark)
        for row in rows:
            self.confidence_level = float(row[0])


class AccompanyConfidenceCheck(BaseConfidenceLevelCheck):

    def __init__(self, config):
        super().__init__()
        self.config = config

    def train_1song(self, **kwargs):
        """
        计算连续八拍伴奏的偏离程度（检查的步长为2拍）
        评定标准有三个:
        A.后四拍的伴奏的平均音高与四拍前的相减 之后除七
        B.后四拍的伴奏的按键与四拍前的进行比较 计算杰卡德距离
        C.每两拍的伴奏与同时期和弦的包含关系 计算方法是 不包含量/全部（如果两拍内有两个和弦 取前一个作为这两拍的和弦）
          - 如果伴奏可以构成一个和弦 但与当前和弦不能匹配的 额外增加0.25个差异分
        之后对上述量取平方和作为伴奏的误差
        舍弃包含了未知和弦的数据和连续四拍以上没有伴奏的数据
        kwargs['raw_data']: 这首歌的伴奏数据
        kwargs['chord_data']: 一首歌的和弦数据
        """
        raw_data = kwargs['raw_data']
        chord_data = kwargs['chord_data']

        time_step_ratio = round(2 / self.config.note_time)
        notes_in_beat = round(1 / self.config.note_time)
        check_num = len(raw_data) // time_step_ratio
        diff_score_list = []
        chord_diff_score_by_pat = [0 for t in range(check_num)]  # 每一个步长的伴奏与同时期和弦的差异分 如果同时期的和弦为未知和弦
        chord_temp = 0  # 用于存储上两拍的和弦
        accompany_temp = []  # 用于存储上两拍的最后一个伴奏音符
        accompany_temp_beat_dx = 0  # “上两拍的最后一个伴奏音符”是在哪一拍录得的

        # 1.计算整首歌中伴奏与同时期和弦的差异分
        for check_step_it in range(2, check_num):
            try:
                chord_diff_score_1step = 0  # 与同期和弦的差异分
                note_diff_count = 0  # 伴奏中不在和弦内的音符数
                abs_note_list = []
                for note_it in range(check_step_it * time_step_ratio, (check_step_it + 1) * time_step_ratio):  # 获取这两拍的伴奏音符的绝对音高
                    if raw_data[note_it] != 0:
                        abs_note_list.extend(NoteDict.nd[raw_data[note_it]])
                        accompany_temp = NoteDict.nd[raw_data[note_it]]
                        accompany_temp_beat_dx = note_it // notes_in_beat
                if chord_data[check_step_it * 2] == 0 and chord_data[check_step_it * 2 + 1] == 0:
                    chord_diff_score_by_pat[check_step_it] = -1  # 出现了未知和弦 差异分标记为-1
                    continue
                accompany_chord = noteset2chord(set(abs_note_list))  # 这个伴奏是否可以组成一个特定的和弦
                div12_note_list = []  # 把所有音符对12取余数的结果 用于分析伴奏和和弦之间的重复音关系
                for note in abs_note_list:
                    div12_note_list.append(note % 12)
                if accompany_chord != 0:
                    if accompany_chord != chord_data[check_step_it * 2] and accompany_chord != chord_data[check_step_it * 2 + 1]:
                        chord_diff_score_1step += self.config.chord_diff_score  # 这个伴奏组合与当前和弦不能匹配 额外增加0.5个差异分
                if chord_data[check_step_it * 2] != 0:
                    chord_set = copy.deepcopy(CHORD_LIST[chord_data[check_step_it * 2]])  # 和弦的音符列表(已对12取余数)
                    chord_dx = chord_data[check_step_it * 2]
                else:
                    chord_set = copy.deepcopy(CHORD_LIST[chord_data[check_step_it * 2 + 1]])
                    chord_dx = chord_data[check_step_it * 2 + 1]
                if len(abs_note_list) == 0:  # 这两拍的伴奏没有音符。如果前两拍也没有则给值-1。如果和弦没变则赋前值 如果和弦变了则以前两拍最后一个音作为判断依据
                    if len(accompany_temp) == 0 or check_step_it * 2 - accompany_temp_beat_dx >= 3:  # 前两拍也没有
                        chord_diff_score_by_pat[check_step_it] = -1
                        continue
                    elif chord_dx == chord_temp:  # 和弦没变 赋前值
                        chord_diff_score_by_pat[check_step_it] = chord_diff_score_by_pat[check_step_it - 1]
                        continue
                    else:  # 用上两拍的最后一个伴奏组合来进行计算
                        abs_note_list = accompany_temp
                        div12_note_list = [note % 12 for note in abs_note_list]
                chord_temp = chord_dx
                if 1 <= chord_dx <= 72 and chord_dx % 6 == 1:  # 大三和弦 chord_set增加大七度和小七度
                    chord_set.add((chord_dx // 6 + 10) % 12)
                    chord_set.add((chord_dx // 6 + 11) % 12)
                if 1 <= chord_dx <= 72 and chord_dx % 6 == 2:  # 小三和弦 chord_set增加小七度
                    chord_set.add((chord_dx // 6 + 10) % 12)
                for note in div12_note_list:
                    if note not in chord_set:
                        note_diff_count += 1
                chord_diff_score_1step += note_diff_count / len(abs_note_list) * self.config.chord_note_diff_score  # 伴奏与同期和弦的差异分
                chord_diff_score_by_pat[check_step_it] = chord_diff_score_1step
            except IndexError:
                chord_diff_score_by_pat[check_step_it] = -1

        # 2.获取一首歌中各个步长的伴奏的差异得分
        for pat_step_it in range(2, check_num - 3):  # 前两个步长不看 因为没有两个步长前的数据
            note_dx = pat_step_it * time_step_ratio
            # 2.1.计算平均音高之间的差异
            note_list = []  # 这四拍中有几个音符 它们的音高分别是多少
            note_list_old = []
            note_count = 0  # 八拍之前的四拍中有几个音符 它们的音高分别是多少
            note_count_old = 0
            for note_it in range(note_dx + notes_in_beat * 4, note_dx + notes_in_beat * 8):
                if raw_data[note_it] != 0:
                    note_list.extend(NoteDict.nd[raw_data[note_it]])
                    note_count += len(NoteDict.nd[raw_data[note_it]])
                if raw_data[note_it - notes_in_beat * 8] != 0:
                    note_list_old.extend(NoteDict.nd[raw_data[note_it - notes_in_beat * 8]])
                    note_count_old += len(NoteDict.nd[raw_data[note_it - notes_in_beat * 8]])
            if note_count == 0 or note_count_old == 0:
                note_diff_score = 0
            else:
                avr_note = sum(note_list) / note_count  # 四拍所有音符的平均音高
                avr_note_old = sum(note_list_old) / note_count_old  # 八拍之前的四拍所有音符的平均音高
                note_diff_score = abs(avr_note - avr_note_old) * self.config.pitch_diff_ratio  # 音高的差异（除以7的考虑因素是如果是176543的话差异分刚好为1分）
            # 2.2.计算按键的差异
            note_same_count = 0  # 四拍中有几个相同按键位置
            note_diff_count = 0  # 四拍中有几个不同的按键位置
            for note_it in range(note_dx + notes_in_beat * 4, note_dx + notes_in_beat * 8):
                if bool(raw_data[note_it]) ^ bool(raw_data[note_it - notes_in_beat * 8]):
                    note_diff_count += 1
                elif raw_data[note_it] != 0 and raw_data[note_it - notes_in_beat * 8] != 0:
                    note_same_count += 1
            if note_same_count == 0 and note_diff_count == 0:
                keypress_diff_score = 0  # 按键的差异分
            else:
                keypress_diff_score = note_diff_count / (note_same_count + note_diff_count) * self.config.keypress_diff_ratio
            # 2.3.计算与同时期和弦的差异分
            if -1 in chord_diff_score_by_pat[pat_step_it: pat_step_it + 4]:
                continue  # 中间有未知和弦 直接
            if raw_data[note_dx: note_dx + notes_in_beat * 4] == [0 for t in range(notes_in_beat * 4)] or raw_data[note_dx + (notes_in_beat * 4): note_dx + (notes_in_beat * 8)] == [0 for t in range(notes_in_beat * 4)]:
                continue  # 连续四拍没有伴奏的情况
            chord_diff_score = chord_diff_score_by_pat[pat_step_it: pat_step_it + 4]  # 这四拍的和弦差异分
            # 2.4.计算总差异分
            total_diff_score = note_diff_score * note_diff_score + keypress_diff_score * keypress_diff_score + sum([t * t for t in chord_diff_score])
            diff_score_list.append(total_diff_score)

        self.evaluating_score_list.extend(diff_score_list)

    def evaluate(self, **kwargs):
        note_out = kwargs['note_out']
        chord_out = kwargs['chord_out']

        time_step_ratio = round(2 / self.config.note_time)
        notes_in_beat = round(1 / self.config.note_time)
        note_list = []
        note_list_old = []
        note_count = 0  # 八拍之前的四拍中有几个音符 它们的音高分别是多少
        note_count_old = 0
        for note_it in range(notes_in_beat * 8, notes_in_beat * 12):
            if note_out[note_it] != 0:
                note_list.extend(note_out[note_it])
                note_count += len(note_out[note_it])
            if note_out[note_it - notes_in_beat * 8] != 0:
                note_list_old.extend(note_out[note_it - notes_in_beat * 8])
                note_count_old += len(note_out[note_it - notes_in_beat * 8])
        if note_count == 0 or note_count_old == 0:
            note_diff_score = 0
        else:
            avr_note = sum(note_list) / note_count  # 四拍所有音符的平均音高
            avr_note_old = sum(note_list_old) / note_count_old  # 八拍之前的四拍所有音符的平均音高
            note_diff_score = abs(avr_note - avr_note_old) * self.config.pitch_diff_ratio  # 音高的差异（如果是176543的话差异分刚好为1分）

        # 2.计算按键的差异
        note_same_count = 0  # 四拍中有几个相同按键位置
        note_diff_count = 0  # 四拍中有几个不同的按键位置
        for note_it in range(notes_in_beat * 8, notes_in_beat * 12):
            if bool(note_out[note_it]) ^ bool(note_out[note_it - 64]):
                note_diff_count += 1
            elif note_out[note_it] != 0 and note_out[note_it - notes_in_beat * 8] != 0:
                note_same_count += 1
        if note_same_count == 0 and note_diff_count == 0:
            keypress_diff_score = 0  # 按键的差异分
        else:
            keypress_diff_score = note_diff_count / (note_same_count + note_diff_count) * self.config.keypress_diff_ratio

        # 3.计算与同时期和弦的差异分
        chord_bak = 0  # 用于存储上两拍的和弦
        accompany_bak = []  # 用于存储上两拍的最后一个bass音符
        chord_diff_score_by_pat = [0, 0, 0, 0]
        for step_it in range(2, 6):
            chord_step_dx = (step_it - 2) * 2  # 这个bass的位置对应和弦的第几个下标
            chord_diff_score_1step = 0  # 与同期和弦的差异分
            note_diff_count = 0  # 和弦内不包含的bass音符数
            if chord_out[chord_step_dx] == 0 and chord_out[chord_step_dx + 1] == 0:
                chord_diff_score_by_pat[step_it - 2] = -1  # 出现了未知和弦 差异分标记为-1
                continue
            abs_notelist = []
            for note_it in range(step_it * time_step_ratio, (step_it + 1) * time_step_ratio):
                if note_out[note_it] != 0:
                    abs_notelist.extend(note_out[note_it])
                    accompany_bak = note_out[note_it]
            accompany_chord = noteset2chord(set(abs_notelist))  # 这个bass是否可以找出一个特定的和弦
            div12_note_list = []  # 把所有音符对12取余数的结果 用于分析和和弦之间的重复音关系
            for note in abs_notelist:
                div12_note_list.append(note % 12)
            if accompany_chord != 0:
                if accompany_chord != chord_out[chord_step_dx] and accompany_chord != chord_out[chord_step_dx + 1]:  # 这个bass组合与当前和弦不能匹配 额外增加0.5个差异分
                    chord_diff_score_1step += self.config.chord_diff_score
            if chord_out[chord_step_dx] != 0:
                chord_set = copy.deepcopy(CHORD_LIST[chord_out[chord_step_dx]])  # 和弦的音符列表(已对１２取余数)
                chord_dx = chord_out[chord_step_dx]
            else:
                chord_set = copy.deepcopy(CHORD_LIST[chord_out[chord_step_dx + 1]])
                chord_dx = chord_out[chord_step_dx + 1]
            if len(abs_notelist) == 0:  # 这两拍的bass没有音符。如果前两拍也没有则给值-1。如果和弦没变则赋前值 如果和弦变了则以前两拍最后一个音作为判断依据
                if len(accompany_bak) == 0:  # 前两拍也没有
                    chord_diff_score_by_pat[step_it - 2] = -1
                    continue
                elif chord_dx == chord_bak:
                    chord_diff_score_by_pat[step_it - 2] = chord_diff_score_by_pat[step_it - 3]  # 和弦没变 赋前值
                    continue
                else:  # 用上两拍的最后一个bass组合来进行计算
                    abs_notelist = accompany_bak
                    div12_note_list = [note % 12 for note in abs_notelist]
            chord_bak = chord_dx
            if 1 <= chord_dx <= 72 and chord_dx % 6 == 1:  # 大三和弦 chord_set增加大七度和小七度
                chord_set.add((chord_dx // 6 + 10) % 12)
                chord_set.add((chord_dx // 6 + 11) % 12)
            if 1 <= chord_dx <= 72 and chord_dx % 6 == 2:  # 小三和弦 chord_set增加小七度
                chord_set.add((chord_dx // 6 + 10) % 12)
            for note in div12_note_list:
                if note not in chord_set:
                    note_diff_count += 1
            chord_diff_score_1step += note_diff_count / len(abs_notelist) * self.config.chord_note_diff_score  # bass与同期和弦的差异分
            chord_diff_score_by_pat[step_it - 2] = chord_diff_score_1step
            if -1 in chord_diff_score_by_pat:  # 把-1替换成均值 如果全是-1则替换为全零
                score_sum = 0
                score_count = 0
                for score in chord_diff_score_by_pat:
                    if score != -1:
                        score_sum += score
                        score_count += 1
                avr_score = 0 if score_count == 0 else score_sum / score_count
                for score_it in range(len(chord_diff_score_by_pat)):
                    if chord_diff_score_by_pat[score_it] == -1:
                        chord_diff_score_by_pat[score_it] = avr_score
        total_diff_score = note_diff_score * note_diff_score + keypress_diff_score * keypress_diff_score + sum([t * t for t in chord_diff_score_by_pat])
        return total_diff_score
