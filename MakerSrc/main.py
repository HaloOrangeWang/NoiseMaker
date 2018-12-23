from settings import *
from interfaces.midi import multi_pianoroll_to_midi, midi2wav
from interfaces.utils import remove_files_in_dir, SystemArgs, run_with_exc, DiaryLog
from interfaces.sql.sqlite import NoteDict
from pipelines.bass_pipeline import BassPipeline
from pipelines.chord_pipeline import ChordPipeline
from pipelines.drum_pipeline import DrumPipeline
from pipelines.fill_pipeline import FillPipeline
from pipelines.intro_pipeline import IntroPipeline
from pipelines.melody_pipeline import MelodyPipeline, MelodyPipelineGen1Sec
from pipelines.piano_guitar_pipeline import PianoGuitarPipeline
from pipelines.string_pipeline import StringPipeline
from preparation.data_manifest import Manifest
from preparation.store_raw_data import SaveMidiData
from preparation.check.check_data import run_validation
import dataoutputs.musicout as out
import tensorflow as tf
import traceback
import copy
import os
import sys


def init_train_model():
    """
    训练的情况下，获取模型的输入,并定义pipeline
    :return: stream_out是很多的pipeline(尚未调整参数的)
    """
    stream_out = {'melody': None, 'intro': None, 'chord': None, 'drum': None, 'bass': None, 'fill': None, 'pg': None, 'string': None}
    if SystemArgs.MelodyMethod == 'full':  # melody的完整输出
        stream_out['melody'] = MelodyPipeline(True, GENERATE_TONE)
    elif SystemArgs.MelodyMethod == 'short':  # melody的短输出
        stream_out['melody'] = MelodyPipelineGen1Sec(True, GENERATE_TONE)
    else:
        raise ValueError
    melody_cls = stream_out['melody'].train_data
    stream_out['intro'] = IntroPipeline(True, stream_out['melody'], GENERATE_TONE)
    stream_out['chord'] = ChordPipeline(True, melody_cls.melody_pat_data_nres, melody_cls.raw_melody_data, melody_cls.continuous_bar_data_nres, melody_cls.core_note_pat_nres)  # 在训练和弦和打击乐的时候使用无调式限制的主旋律数据
    chord_cls = stream_out['chord'].train_data
    stream_out['drum'] = DrumPipeline(True, melody_cls.melody_pat_data_nres, melody_cls.continuous_bar_data_nres)
    stream_out['bass'] = BassPipeline(True, melody_cls.melody_pat_data_nres, melody_cls.continuous_bar_data_nres, melody_cls.keypress_pat_data, melody_cls.all_keypress_pats, chord_cls)
    stream_out['pg'] = PianoGuitarPipeline(True, melody_cls.melody_pat_data_nres, melody_cls.continuous_bar_data_nres, melody_cls.keypress_pat_data, melody_cls.all_keypress_pats, chord_cls)
    stream_out['string'] = StringPipeline(True, melody_cls.melody_pat_data_nres, melody_cls.continuous_bar_data_nres, melody_cls.core_note_pat_nres, melody_cls.common_corenote_pats, chord_cls)
    stream_out['fill'] = FillPipeline(True, melody_cls.raw_melody_data, melody_cls.section_data, melody_cls.continuous_bar_data_nres)
    return stream_out


def init_test_model():
    """
    生成的情况下，获取模型的输入,并定义pipeline
    :return: stream_out是很多的pipeline(尚未调整参数的)
    """
    stream_out = {'melody': None, 'intro': None, 'chord': None, 'drum': None, 'bass': None, 'fill': None, 'pg': None, 'string': None}
    if SystemArgs.MelodyMethod == 'full':  # melody的完整输出
        stream_out['melody'] = MelodyPipeline(False, GENERATE_TONE)
    elif SystemArgs.MelodyMethod == 'short':  # melody的短输出
        stream_out['melody'] = MelodyPipelineGen1Sec(False, GENERATE_TONE)
    else:
        raise ValueError
    stream_out['intro'] = IntroPipeline(False, stream_out['melody'], GENERATE_TONE)
    stream_out['chord'] = ChordPipeline(False)  # 在训练和弦和打击乐的时候使用无调式限制的主旋律数据
    stream_out['drum'] = DrumPipeline(False)
    stream_out['bass'] = BassPipeline(False, stream_out['melody'].train_data.all_keypress_pats)
    stream_out['pg'] = PianoGuitarPipeline(False, stream_out['melody'].train_data.all_keypress_pats)
    stream_out['string'] = StringPipeline(False)
    stream_out['fill'] = FillPipeline(False)
    return stream_out


def train(stream_in):
    """
    使用LSTM训练,并生成很多段从模型输出的数据
    :param stream_in: 前面model_definition的输出
    :return: 生成的模型输出数据
    """
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sv = tf.train.Supervisor(logdir=PATH_TFLOG)
    with sv.managed_session() as session:
        session.run(init_op)
        stream_in['melody'].get_center_points(session, is_train=True)
        # 2.训练
        stream_in['melody'].run_epoch(session, COMMON_MELODY_PAT_NUM)
        stream_in['intro'].run_epoch(session, COMMON_MELODY_PAT_NUM)
        stream_in['chord'].run_epoch(session)
        stream_in['drum'].run_epoch(session, COMMON_DRUM_PAT_NUM)
        stream_in['bass'].run_epoch(session, COMMON_BASS_PAT_NUM)
        stream_in['pg'].run_epoch(session, COMMON_PG_PAT_NUM)
        stream_in['string'].run_epoch(session, COMMON_STRING_PAT_NUM)
        sv.saver.save(session, PATH_TFLOG, global_step=sv.global_step)
        # 3.输出
        stream_out = output_rawdata(session, stream_in)
    return stream_out


def restore(stream_input):
    """
    从文件中获取参数来代替训练过程 并生成很多段从模型输出的数据
    :param stream_input: 前面model_definition的输出
    :return: 生成的模型输出数据
    """
    sv = tf.train.Supervisor(logdir=PATH_TFLOG)
    with sv.managed_session() as session:
        sv.saver.restore(session, PATH_TFLOG)
        stream_input['melody'].get_center_points(session, is_train=False)
        stream_output = output_rawdata(session, stream_input)
    return stream_output


def generate_1track(track_in, mark, *args):
    """
    根据track_in类型的generate方法来生成一个音轨。
    如果连续生成失败超过max_fail_time次则抛出ValueError, 否则反复生成直到成功为止
    :param mark: 音轨的标注
    :param track_in: 输入的音轨生成方式类
    :param args: 这个类的generate方法的相关参数
    :return: 生成的音轨数据
    """
    for fail_time in range(MAX_GEN_1SONG_FAIL_TIME):
        try:
            track_out = track_in.generate(*args)
            return track_out
        except RuntimeError:
            DiaryLog.warn(traceback.format_exc())
            DiaryLog.warn('%s音轨已连续生成失败%d次' % (mark, fail_time))
        except IndexError:
            DiaryLog.warn(traceback.format_exc())
            DiaryLog.warn('%s音轨已连续生成失败%d次' % (mark, fail_time))
    raise RuntimeError


def output_rawdata(session, stream_in):
    """
    输出使用算法生成的音符数据 输出很多段从模型输出的数据
    :param session:
    :param stream_in: 经训练调整参数(或从文件中获取参数)之后的模型
    :return: 很多段从模型输出的数据
    """
    stream_out = dict()
    while True:
        melody_fail_time = 0
        intro_fail_time = 0
        # 1.生成主旋律
        while True:
            try:
                stream_out['melody'], melody_out_pats, core_note_out_pats, keypress_out_pats = stream_in['melody'].generate(session)
                break
            except RuntimeError:
                DiaryLog.warn('主旋律已连续生成失败%d次' % melody_fail_time)
                melody_fail_time += 1
        stream_out['section'] = copy.deepcopy(stream_in['melody'].section_data)  # 主旋律的乐段数据
        # 2.生成前奏（melody短生成的情况下没有前奏和间奏）
        if SystemArgs.MelodyMethod != 'short':
            while True:
                try:
                    stream_out['intro'], intro_out_pats, intro_cn_out_pats, intro_kp_out_pats = stream_in['intro'].generate(session, stream_out['melody'], melody_out_pats)
                    break
                except RuntimeError:
                    DiaryLog.warn('前奏已连续生成失败%d次' % intro_fail_time)
                    intro_fail_time += 1
        else:
            stream_out['intro'] = []
            intro_out_pats = []
            intro_cn_out_pats = []
            intro_kp_out_pats = []
        melody_intro_out = copy.deepcopy(stream_out['melody'])
        melody_intro_out.extend(stream_out['intro'])  # 结合了主旋律和前奏的输出（注意是主旋律在前 前奏在后）
        melody_out_pats.extend(intro_out_pats)
        core_note_out_pats.extend(intro_cn_out_pats)
        keypress_out_pats.extend(intro_kp_out_pats)
        melody_beat_num = (len(stream_out['melody']) + len(stream_out['intro'])) // 8  # 主旋律一共有多少拍
        end_check_beats = [len(stream_out['melody']) // 8]  # 在哪几拍执行end_check
        # 3.生成其他的几个音轨（除了加花以外）
        try:
            stream_out['chord'] = generate_1track(stream_in['chord'], 'chord', session, melody_intro_out, melody_out_pats, stream_in['melody'].train_data.common_corenote_pats, core_note_out_pats, melody_beat_num, end_check_beats)  # 在生成和弦和打击乐的时候也使用没有调式限制的主旋律pattern数据
            stream_out['drum'] = generate_1track(stream_in['drum'], 'drum', session, melody_intro_out, melody_out_pats)
            stream_out['bass'] = generate_1track(stream_in['bass'], 'bass', session, melody_intro_out, keypress_out_pats, stream_out['chord'], melody_beat_num,end_check_beats)
            stream_out['pg'] = generate_1track(stream_in['pg'], 'piano_guitar', session, melody_intro_out, keypress_out_pats, stream_out['chord'], melody_beat_num, end_check_beats)
            stream_out['string'] = generate_1track(stream_in['string'], 'string', session, melody_intro_out, stream_out['chord'], core_note_out_pats, melody_beat_num, end_check_beats)
        except RuntimeError:
            continue
        # 4.生成加花
        fill_fail_time = 0
        while True:
            try:
                judge_fill_list = stream_in['fill'].judge_fill(stream_out['melody'], stream_in['melody'].section_data)
                if sum(judge_fill_list):  # 加花不能为空
                    stream_out['fill'] = stream_in['fill'].generate(stream_out['melody'], stream_out['chord'][:len(stream_out['melody']) // 8], judge_fill_list)  # 生成加花的具体内容
                    break
                else:
                    fill_fail_time += 1
                    DiaryLog.warn('加花数据为空，重新判断，已经重试了%d次' % fill_fail_time)
            except IndexError:
                DiaryLog.warn('加花已连续生成失败%d次' % fill_fail_time)
                fill_fail_time += 1
        return stream_out


def tracks2song(stream_input, output_dir=None):
    """
    根据模型的输出数据生成音频文件
    :param stream_input: 模型的输出数据
    :param output_dir: 输出文件的存放地址
    """
    if SystemArgs.MelodyMethod != 'short':  # melody短生成的情况下没有promote，直接一遍结束
        stream_output = out.music_promote(stream_input)
    else:
        stream_output = stream_input
    melody_pianoroll = out.melodylist2pianoroll(stream_output['melody'], 100, 0.9)  # 分别生成主旋律 和弦 鼓对应的piano roll
    intro_pianoroll = out.melodylist2pianoroll(stream_output['intro'], 92, 0.9)  # 前奏
    drum_pianoroll = out.drumlist2pianoroll(stream_output['drum'], 85, 0.6)
    bass_pianoroll = out.basslist2pianoroll(stream_output['bass'], 90, 0.9)
    fill_pianoroll = out.filllist2pianoroll(stream_output['fill'], 85, 0.25)
    pg_pianoroll = out.pglist2pianoroll(stream_output['pg'], 85, 0.6)
    string_pianoroll = out.stringlist2pianoroll(stream_output['string'], 67, 1)
    if output_dir is None:
        output_dir = PATH_GENERATE_MIDIFILE
    multi_pianoroll_to_midi(output_dir, 100,
                            {0: {'name': 'Main', 'program': 26, 'note': melody_pianoroll},
                             1: {'name': 'Intro', 'program': 5, 'note': intro_pianoroll},
                             2: {'name': 'Bass', 'program': 33, 'note': bass_pianoroll},
                             3: {'name': 'PG', 'program': 1, 'note': pg_pianoroll},
                             4: {'name': 'String', 'program': 48, 'note': string_pianoroll},
                             5: {'name': 'Fill', 'program': 10, 'note': fill_pianoroll},
                             9: {'name': 'Drum', 'program': 0, 'note': drum_pianoroll}})
    if FLAG_GEN_WAV is True:  # 除了输出mid文件以外 还输出一个同名的wav文件
        filename_dx = output_dir.rfind('.')
        wav_filename = output_dir[:filename_dx] + '.wav'
        midi2wav(output_dir, wav_filename)


@run_with_exc
def generate_1song(output_dir=None):
    """
    依次调用上述函数 生成一段音乐
    :param output_dir: 输出文件的存放地址
    """
    NoteDict.read_note_dict()
    if FLAG_TRAINING is True:
        remove_files_in_dir(os.path.dirname(PATH_TFLOG))  # 在训练状态 需要先删掉log文件 防止冲突
    with tf.Graph().as_default():
        if FLAG_TRAINING:
            stream = init_train_model()
            stream = train(stream)  # 训练
        else:
            stream = init_test_model()
            stream = restore(stream)  # 从log中获取训练数据
    # with tf.Graph().as_default():
    #     stream = output_by_hmm(stream)  # 生成使用HMM模型得到的数据
    tracks2song(stream, output_dir)  # 生成midi文件


@run_with_exc
def run_manifest():
    manifest = Manifest()
    manifest.store()
    SaveMidiData(manifest.song_info_list, manifest.skiplist)


if __name__ == '__main__':
    if ACTIVE_MODULE == PROGRAM_MODULES["Manifest"]:  # 如果这个值为True的话，程序将率先从midi文件中读取音乐信息存储到数据库中。这个过程不需要在每次运行程序时都执行。
        run_manifest()
    elif ACTIVE_MODULE == PROGRAM_MODULES["Check"]:
        run_validation()  # 运行验证内容
    elif ACTIVE_MODULE == PROGRAM_MODULES["Main"]:  # 运行主程序
        os.chdir(sys.path[0])
        generate_1song(output_dir=SystemArgs.outputpath)
    else:
        raise ValueError("Active Module cannot be %d." % ACTIVE_MODULE)
