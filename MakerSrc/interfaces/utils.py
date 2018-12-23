from settings import *
import argparse
import os
import datetime
import traceback
import logging
import sys


def last_not0_number_in_array(array, reverse=False):
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


def min_number_except_1(array):
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


def get_dict_max_key(dic):
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


def get_first_index_bigger(array, number):
    """
    找出一个数组中第一个大于number的元素
    :param array: 数组
    :param number:
    :return: 第一个大于number的元素的位置及其值
    """
    for (iterator, value) in enumerate(array):
        if value >= number:
            return iterator, value
    return -1, number - 1


def get_last_index_smaller(array, number):
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


def flat_array(ary):
    """
    把数组将一维
    :param ary: 原数组
    :return: 降１维之后的数组
    """
    flatten_ary = []
    if type(ary) is list:
        for item in ary:
            flatten_ary.extend(item)
    elif type(ary) is dict:
        for key in range(0, get_dict_max_key(ary) + 1):
            if key in ary:
                flatten_ary.extend(ary[key])
    return flatten_ary


def split_by_number(ary, number):
    """
    将数组根据一个数进行分割。如[0,1,0,0,4,5,7,0],0会生成两个返回值，分别是[[1],[4,5,7]]和[[1,1],[4,6]]
    :param number: 进行分割的标志数
    :param ary: 数组
    :return:
    """
    split_ary = []
    split_dx = []
    start_dx = -1
    for it in range(len(ary)):
        if ary[it] != number:
            if start_dx == -1:
                split_ary.append([ary[it]])
                start_dx = it
            else:
                split_ary[-1].append(ary[it])
        elif ary[it] == number and start_dx != -1:
            end_dx = it
            split_dx.append([start_dx, end_dx])
            start_dx = -1
    if start_dx != -1:
        end_dx = len(ary)
        split_dx.append([start_dx, end_dx])
    return split_ary, split_dx


def get_dict_key_to_key(dic, start_key, end_key):
    """
    获取一个字典从start_key到end_key的值 如{0:1,1:[3,4],2:'3',3:4},0,2 => [1,[3,4],'3']
    如果中间出现了不存在的键则抛出KeyError
    :param dic: 一个字典
    :param start_key: 开始的键
    :param end_key: 结束的健
    :return: list
    """
    result = []
    for key in range(start_key, end_key + 1):
        if key in dic:
            result.append(dic[key])
        else:
            raise KeyError
    return result


def get_nearest_number_multiple(x, k):
    """找出最接近x的一个为k的整数倍的数"""
    assert k >= 1 and type(k) is int
    if x % k <= k / 2:
        return x - x % k
    else:
        return x - x % k + k


def remove_files_in_dir(dirname):
    """
    删除一个文件夹及其所有子文件夹中所有的文件（但不删除文件夹）
    :param dirname: 根目录名
    """
    for root, __, filelist in os.walk(dirname):
        for filename in filelist:
            full_filename = os.path.join(root, filename)
            os.remove(full_filename)


def init_folder():
    # 1.新建日志存储的文件夹
    if not os.path.exists('../Diary/Train'):
        os.makedirs('../Diary/Train')
    if not os.path.exists('../Diary/Generate'):
        os.makedirs('../Diary/Generate')
    # 2.新建训练数据存储的文件夹
    if not os.path.exists(PATH_TFLOG):
        os.makedirs(PATH_TFLOG)
    if not os.path.exists(PATH_PATTERNLOG):
        os.makedirs(PATH_PATTERNLOG)
    # 3.新建输出文件的文件夹
    if not os.path.exists('../Outputs'):
        os.makedirs('../Outputs')


def get_sys_args():
    parser = argparse.ArgumentParser()
    for arg in PROGRAM_ARGS:
        parser.add_argument('--' + arg['name'], type=arg['type'], default=arg['default'])
    args = parser.parse_args()
    return args


def logger(diary_id):
    # 1.获取日志输出文件的地址
    curdate = datetime.datetime.now()
    if FLAG_TRAINING is True:
        status = 'Train'
    else:
        status = 'Generate'
    if diary_id == -1:
        diary_id = 1
        while os.path.exists(PATH_DIARY % (status, curdate.year % 100, curdate.month, curdate.day, diary_id)):
            diary_id += 1
    diary_path = PATH_DIARY % (status, curdate.year % 100, curdate.month, curdate.day, diary_id)
    # 2.定义一个logger
    log = logging.getLogger()
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    log.addHandler(ch)
    fh = logging.FileHandler(filename=diary_path)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    return log


def run_with_exc(f):
    """运行时捕捉异常并再次抛出"""
    def call(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            exc_info = traceback.format_exc()
            DiaryLog.critical(exc_info)
            raise e
    return call


# 后面的内容是定义一些全局变量。但这些全局变量并不像settings里面那样是固定的，而是根据前面的一些函数生成的
# 包含了新建文件夹/读取系统参数 定义log输出等

init_folder()
SystemArgs = get_sys_args()  # 系统参数
DiaryLog = logger(diary_id=SystemArgs.diaryId)  # 日志输出
