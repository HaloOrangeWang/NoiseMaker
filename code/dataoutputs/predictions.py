import random
from settings import *


def MusicPatternPrediction(predict_matrix, min_pattern_number, max_pattern_number):
    while True:
        row_sum = sum(predict_matrix[-1][min_pattern_number: (max_pattern_number + 1)])
        random_value = random.random() * row_sum
        for element in range(len(predict_matrix[-1][min_pattern_number: (max_pattern_number + 1)])):  # 最多只能到common_melody_pattern
            random_value -= predict_matrix[-1][min_pattern_number:][element]
            if random_value <= 0:
                return element + min_pattern_number


def ChordPrediction(predict_matrix):
    """
    生成和弦的方法。
    :param predict_matrix: 和弦预测的二维矩阵
    :param chord_time_step: 每多少拍一个和弦
    :return: 和弦预测的一维向量
    """
    time_step_ratio = round(CHORD_GENERATE_TIME_STEP / CHORD_TIME_STEP)
    chord_number = len(CHORD_DICT) - 1  # 和弦词典中一共有多少种和弦
    predict_matrix = predict_matrix[-round(time_step_ratio):]
    row_sum_array = []
    for note_iterator in range(1, (1 + chord_number)):
        row_sum_array.append(sum(predict_matrix[:, note_iterator]))
    while True:
        row_sum = sum(row_sum_array)
        random_value = random.random() * row_sum
        for element in range(len(CHORD_DICT) - 1):
            random_value -= row_sum_array[element]
            if random_value < 0:
                return element + 1


def GetFirstMelodyPattern(pattern_number_list, min_pattern_number, max_pattern_number):
    """

    :param pattern_number_list: 各种pattern在训练集中出现的次数
    :param min_pattern_number: 选取的pattern在序列中的最小值
    :param max_pattern_number: 选取的pattern在序列中的最大值
    :return: 选取的pattern的代码
    """
    row_sum = sum(pattern_number_list[min_pattern_number: (max_pattern_number + 1)])
    random_value = random.random() * row_sum
    for element in range(len(pattern_number_list[min_pattern_number: (max_pattern_number + 1)])):
        random_value -= pattern_number_list[min_pattern_number: (max_pattern_number + 1)][element]
        if random_value < 0:
            return element + min_pattern_number
    return max_pattern_number
