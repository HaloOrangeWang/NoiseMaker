import copy
import argparse
from settings import *


def LastNotZeroNumber(array, reverse=False):
    """
    返回一个数组首个不为0的元素 如果这个数组的所有元素全部为0 则返回0
    :param reverse: 是否倒序
    :param array: 一个数组
    :return: 数组中首个不为0的元素
    """
    if reverse:
        array_temp = [array[-1-t] for t in range(len(array))]
    else:
        array_temp = array
    for t in array_temp:
        if t != 0:
            return t
    return 0


def MinNumberExcept_1(array):
    """
    找出一个数组中不是-1的最小数
    :param array: 数组
    :return: 这个最小数及其在数组中的位置（如果这个数组全部是-1则返回-1,len(array)）
    """
    min_number = max(array)
    min_at = len(array)
    for (i, t) in enumerate(array):
        if t != -1 and t <= min_number:
            min_number = t
            min_at = i
    return min_number, min_at


def GetDictMaxKey(dic):
    """
    找出一个dict中最大的key值。这个dict中的所有key必须都为数字 且至少有一个正数
    :param dic: 一个dict
    :return: 它的最大key
    """
    max_key = -1
    for key in dic:
        if key > max_key:
            max_key = key
    return max_key


def GetFirstIndexBigger(array, number):
    """
    找出一个数组中第一个大于number的元素
    :param array: 数组
    :param number:
    :return: 第一个大于number的元素的位置及其值
    """
    for (i, t) in enumerate(array):
        if t >= number:
            return i, t
    return -1, number - 1


def GetFirstIndexSmaller(array, number):
    """
    找出一个数组中最后一个小于number的元素
    :param array: 数组
    :param number:
    :return: 最后一个小于number的元素的位置及其值
    """
    for iterator in range(len(array) - 1, -1, -1):
        if array[iterator] <= number:
            return iterator, array[iterator]
    return -1, number + 1


def CommonMusicPatterns(raw_music_data, number, note_time_step, pattern_time_step):
    """
    获取最常见的number种旋律组合
    :param raw_music_data: 一首歌一个音轨的数据
    :param number: 保存最常见的多少种旋律组合
    :param note_time_step: 音符的时间步长
    :param pattern_time_step: 旋律组合的时间步长
    :return: 最常见的旋律组合
    """
    common_pattern_dict = {}
    time_step_ratio = round(pattern_time_step / note_time_step)
    for song_iterator in range(len(raw_music_data)):
        if raw_music_data[song_iterator] != {}:
            for key in raw_music_data[song_iterator]:
                try:
                    melody_pattern_list = [raw_music_data[song_iterator][key][time_step_ratio*t: time_step_ratio*(t+1)] for t in range(round(4 / pattern_time_step))]
                    for melody_pattern_iterator in melody_pattern_list:
                        try:
                            common_pattern_dict[str(melody_pattern_iterator)] += 1
                        except KeyError:
                            common_pattern_dict[str(melody_pattern_iterator)] = 1
                except KeyError:
                    pass
    common_pattern_list_temp = sorted(common_pattern_dict.items(), key=lambda asd: asd[1], reverse=True)  # 按照次数由高到底排序
    # print(len(common_pattern_list_temp))
    # sum1 = 0
    # sum2 = 0
    # for t in common_pattern_list_temp[1:]:
    #     sum1 += t[1]
    # for t in common_pattern_list_temp[1: (number + 1)]:
    #     sum2 += t[1]
    # print(sum2 / sum1)
    common_pattern_list = []  # 最常见的number种组合
    pattern_number_list = []  # 这些组合出现的次数
    for pattern_tuple in common_pattern_list_temp[:(number + 1)]:
        # print(drum_pattern)
        common_pattern_list.append(eval(pattern_tuple[0]))
        pattern_number_list.append(pattern_tuple[1])
    # print(common_pattern_list)
    # print(pattern_number_list)
    return common_pattern_list, pattern_number_list


class MusicPatternEncode(object):

    def __init__(self, common_patterns, music_data_dict, note_time_step, pattern_time_step):
        self.music_pattern_dict = copy.deepcopy(music_data_dict)  # 按照最常见的音符组合编码之后的组合列表（dict形式）
        time_step_ratio = round(pattern_time_step / note_time_step)
        music_data_list = sorted(music_data_dict.items(), key=lambda asd: asd[0], reverse=False)  # 把ｄｉｃｔ形式的音符列表转化成list形式
        for bar_data in music_data_list:
            # print(bar_iterator[0])
            bar_index = bar_data[0]  # 这个小节是这首歌的第几小节
            bar_note_list = [bar_data[1][time_step_ratio * t: time_step_ratio * (t + 1)] for t in range(round(4 / pattern_time_step))]  # 将音符列表按照pattern_time_step进行分割 使其变成二维数组
            bar_pattern_list = self.handle_common_patterns(bar_note_list, common_patterns)
            bar_pattern_list = self.handle_rare_pattern(bar_index, bar_pattern_list, bar_note_list, common_patterns)
            self.music_pattern_dict[bar_data[0]] = bar_pattern_list  # 将编码后的pattern list保存在新的pattern dict中

    def handle_common_patterns(self, raw_bar_pattern_list, common_patterns):
        bar_pattern_list = []
        for bar_pattern_iterator in range(len(raw_bar_pattern_list)):
            try:
                pattern_code = common_patterns.index(raw_bar_pattern_list[bar_pattern_iterator])  # 在常见的组合列表中找到这个音符组合
                bar_pattern_list.append(pattern_code)
            except ValueError:  # 找不到
                bar_pattern_list.append(-1)
        return bar_pattern_list

    def handle_rare_pattern(self, bar_index, bar_pattern_list, raw_bar_pattern_list, common_patterns):
        for bar_pattern_iterator in range(len(bar_pattern_list)):
            if bar_pattern_list[bar_pattern_iterator] == -1:
                bar_pattern_list[bar_pattern_iterator] = len(common_patterns)
        return bar_pattern_list


def MusicPatternDecode(common_patterns, pattern_list, note_time_step, pattern_time_step):
    """
    音符组合解码。将音符组合解码为音符列表，时间步长为pattern_time_step
    :param pattern_time_step: 音符组合的时间步长
    :param note_time_step: 音符的时间步长
    :param pattern_list: 音符组合列表
    :param common_patterns: 常见的音符组合
    :return: 以note_time_step为时间步长的音符列表
    """
    note_list = []
    time_step_radio = round(pattern_time_step / note_time_step)
    for pattern in pattern_list:
        if pattern == 0 or pattern >= len(common_patterns):
            note_list = note_list + [0 for t in range(time_step_radio)]
        else:
            note_list = note_list + common_patterns[pattern]
    return note_list


def getsysargs():
    parser = argparse.ArgumentParser()
    for arg in SYSARGS:
        parser.add_argument('--' + arg['name'], type=arg['type'], default=arg['default'])
    args = parser.parse_args()
    return args
