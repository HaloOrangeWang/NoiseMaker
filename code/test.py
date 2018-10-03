import os
import sys
import traceback
import copy
import time

import tensorflow as tf

import dataoutputs.musicout as out
from interfaces.midi.midi import multi_pianoroll_to_midi, midi2wav
from interfaces.music_patterns import MusicPatternEncodeStep
from interfaces.utils import remove_files_in_dir, SystemArgs, run_with_exc, DiaryLog
from pipelines.bass_pipeline import BassPipeline, BassPipeline4
from pipelines.chord_pipeline import ChordPipeline, ChordPipeline4
from pipelines.drum_pipeline import DrumPipeline
from pipelines.fill_pipeline import FillPipeline2
from pipelines.intro_pipeline import IntroPipeline
from pipelines.melody_pipeline import MelodyPipelineNoProfile
from pipelines.piano_guitar_pipeline import PianoGuitarPipeline, PianoGuitarPipeline2
from pipelines.string_pipeline import StringPipeline, StringPipeline2
from preparation.store_raw_data import store_song_info, SaveMidiData
from preparation.validations import run_validation
from settings import *


def model_definition():
    """
    获取模型的输入,并定义pipeline
    :return: stream_output是很多的pipeline(尚未调整参数的)
    """
    stream_output = {'melody': None, 'intro': None, 'chord': None, 'drum': None, 'bass': None, 'fill': None, 'piano_guitar': None, 'string': None}
    stream_output['melody'] = MelodyPipelineNoProfile(GENERATE_TONE, train=FLAG_IS_TRAINING)
    melody_data = stream_output['melody'].train_data
    stream_output['intro'] = IntroPipeline(stream_output['melody'], GENERATE_TONE)
    stream_output['chord'] = ChordPipeline4(melody_data.melody_pat_data_nres, melody_data.raw_melody_data, melody_data.continuous_bar_data_nres, melody_data.core_note_pat_nres_for_chord)  # 在训练和弦和打击乐的时候使用无调式限制的主旋律数据
    chord_data = stream_output['chord'].train_data
    stream_output['drum'] = DrumPipeline(melody_data.melody_pat_data_nres, melody_data.continuous_bar_data_nres)
    stream_output['bass'] = BassPipeline4(melody_data.melody_pat_data_nres, melody_data.continuous_bar_data_nres, melody_data.keypress_pat_data, melody_data.keypress_pat_ary, chord_data)
    stream_output['piano_guitar'] = PianoGuitarPipeline2(melody_data.melody_pat_data_nres, melody_data.continuous_bar_data_nres, melody_data.keypress_pat_data, melody_data.keypress_pat_ary, chord_data)
    stream_output['string'] = StringPipeline2(melody_data.melody_pat_data_nres, melody_data.continuous_bar_data_nres, melody_data.core_note_pat_nres_for_chord, melody_data.common_corenote_pats, chord_data)
    stream_output['fill'] = FillPipeline2(melody_data.raw_melody_data, melody_data.section_data, melody_data.continuous_bar_data_nres)
    return stream_output


def train(stream_input):
    """
    使用LSTM训练,并生成很多段从模型输出的数据
    :param stream_input: 前面model_definition的输出
    :return: 生成的模型输出数据
    """
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sv = tf.train.Supervisor(logdir=LOGDIR)
    with sv.managed_session() as session:
        session.run(init_op)
        stream_input['melody'].get_center_points(session, train=True)  # .get_train_data(session)
        # 2.训练
        stream_input['melody'].run_epoch(session, COMMON_MELODY_PATTERN_NUMBER)
        stream_input['intro'].run_epoch(session, COMMON_MELODY_PATTERN_NUMBER)
        stream_input['chord'].run_epoch(session)
        stream_input['drum'].run_epoch(session, COMMON_DRUM_PATTERN_NUMBER)
        stream_input['bass'].run_epoch(session, COMMON_BASS_PATTERN_NUMBER)
        stream_input['piano_guitar'].run_epoch(session, COMMON_PIANO_GUITAR_PATTERN_NUMBER)
        stream_input['string'].run_epoch(session, COMMON_STRING_PATTERN_NUMBER)
        sv.saver.save(session, LOGDIR, global_step=sv.global_step)
        # 3.输出
        stream_output = output_rawdata(session, stream_input)
    return stream_output


def restore(stream_input):
    """
    从文件中获取参数来代替训练过程 并生成很多段从模型输出的数据
    :param stream_input: 前面model_definition的输出
    :return: 生成的模型输出数据
    """
    sv = tf.train.Supervisor(logdir=LOGDIR)
    with sv.managed_session() as session:
        sv.saver.restore(session, LOGDIR)
        stream_input['melody'].get_train_data(session, train=FLAG_IS_TRAINING)
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
    for fail_time in range(MAX_1SONG_GENERATE_FAIL_TIME):
        try:
            track_out = track_in.generate(*args)
            if track_out is not None:
                return track_out
            else:
                DiaryLog.warn('%s音轨已连续生成失败%d次' % (mark, fail_time))
        except IndexError:
            DiaryLog.warn(traceback.format_exc())
            DiaryLog.warn('%s音轨已连续生成失败%d次' % (mark, fail_time))
    raise ValueError


def output_rawdata(session, stream_input):
    """
    输出使用算法生成的音符数据 输出很多段从模型输出的数据
    :param session:
    :param stream_input: 经训练调整参数(或从文件中获取参数)之后的模型
    :return: 很多段从模型输出的数据
    """
    stream_output = {}
    while True:
        melody_fail_time = 0
        intro_fail_time = 0
        # 1.生成主旋律
        while True:
            stream_output['melody'] = stream_input['melody'].generate(session)
            if stream_output['melody']:
                break
            else:
                DiaryLog.warn('主旋律已连续生成失败%d次' % melody_fail_time)
                melody_fail_time += 1
        melody_pat_out = MusicPatternEncodeStep(stream_input['melody'].train_data.common_melody_pats, stream_output['melody'], 0.125, 1).music_pattern_ary  # 对主旋律的输出进行编码
        melody_beat_num = len(stream_output['melody']) // 8  # 主旋律一共有多少拍
        stream_output['section'] = copy.deepcopy(stream_input['melody'].section_data)  # 主旋律的乐段数据
        # 2.生成前奏
        while True:
            stream_output['intro'] = stream_input['intro'].generate(session, stream_output['melody'], melody_pat_out)
            if stream_output['intro']:
                break
            else:
                DiaryLog.warn('前奏已连续生成失败%d次' % intro_fail_time)
                intro_fail_time += 1
        melody_intro_out = copy.deepcopy(stream_output['melody'])
        melody_intro_out.extend(stream_output['intro'])  # 结合了主旋律和前奏的输出（注意是主旋律在前 前奏在后）
        # 3.生成其他的几个音轨（除了加花以外）
        try:
            stream_output['chord'] = generate_1track(stream_input['chord'], 'chord', session, melody_intro_out, stream_input['melody'].train_data.common_melody_pats, stream_input['melody'].train_data.common_corenote_pats, melody_beat_num)  # 在生成和弦和打击乐的时候也使用没有调式限制的主旋律pattern数据
            stream_output['drum'] = generate_1track(stream_input['drum'], 'drum', session, melody_intro_out, stream_input['melody'].train_data.common_melody_pats)
            stream_output['bass'] = generate_1track(stream_input['bass'], 'bass', session, melody_intro_out, stream_output['chord'], stream_input['melody'].train_data.keypress_pat_ary, melody_beat_num)
            stream_output['piano_guitar'] = generate_1track(stream_input['piano_guitar'], 'piano_guitar', session, melody_intro_out, stream_output['chord'], stream_input['melody'].train_data.keypress_pat_ary, melody_beat_num)
            stream_output['string'] = generate_1track(stream_input['string'], 'string', session, melody_intro_out, stream_output['chord'], stream_input['melody'].train_data.common_corenote_pats, melody_beat_num)
        except ValueError:
            continue
        # 4.生成加花
        fill_fail_time = 0
        while True:
            try:
                judge_fill_ary = stream_input['fill'].judge_fill(stream_output['melody'], stream_input['melody'].section_data)
                if sum(judge_fill_ary):  # 加花不能为空
                    stream_output['fill'] = stream_input['fill'].generate(stream_output['melody'], stream_output['chord'][:len(stream_output['melody']) // 8], judge_fill_ary)  # 生成加花的具体内容
                    break
                else:
                    fill_fail_time += 1
                    DiaryLog.warn('加花数据为空，重新判断，已经重试了%d次' % fill_fail_time)
            except IndexError:
                DiaryLog.warn('加花已连续生成失败%d次' % fill_fail_time)
                fill_fail_time += 1
        return stream_output


def tracks2song(stream_input, output_dir=None):
    """
    根据模型的输出数据生成音频文件
    :param stream_input: 模型的输出数据
    :param output_dir: 输出文件的存放地址
    """
    stream_output = out.music_promote_3(stream_input)
    melody_pianoroll = out.melodylist2pianoroll(stream_output['melody'], 100, 0.9)  # 分别生成主旋律 和弦 鼓对应的piano roll
    intro_pianoroll = out.melodylist2pianoroll(stream_output['intro'], 92, 0.9)  # 前奏
    drum_pianoroll = out.drumlist2pianoroll(stream_output['drum'], 85, 0.6)
    bass_pianoroll = out.basslist2pianoroll(stream_output['bass'], 90, 0.9)
    fill_pianoroll = out.filllist2pianoroll(stream_output['fill'], 85, 0.25)
    pg_pianoroll = out.pglist2pianoroll(stream_output['piano_guitar'], 85, 0.6, scale_adjust=12)
    string_pianoroll = out.stringlist2pianoroll(stream_output['string'], 67, 1, scale_adjust=-12)
    if output_dir is None:
        output_dir = GENERATE_MIDIFILE_PATH
    multi_pianoroll_to_midi(output_dir, 100,
                            {0: {'name': 'Main', 'program': 26, 'note': melody_pianoroll},
                             1: {'name': 'Intro', 'program': 5, 'note': intro_pianoroll},
                             2: {'name': 'Bass', 'program': 33, 'note': bass_pianoroll},
                             3: {'name': 'PG', 'program': 1, 'note': pg_pianoroll},
                             4: {'name': 'String', 'program': 48, 'note': string_pianoroll},
                             5: {'name': 'Fill', 'program': 10, 'note': fill_pianoroll},
                             9: {'name': 'Drum', 'program': 0, 'note': drum_pianoroll}})
    if GENERATE_WAV is True:  # 除了输出mid文件以外 还输出一个同名的wav文件
        filename_dx = output_dir.rfind('.')
        wav_filename = output_dir[:filename_dx] + '.wav'
        midi2wav(output_dir, wav_filename)


@run_with_exc
def generate_1song():
    """
    依次调用上述函数 生成一段音乐
    """
    if FLAG_IS_TRAINING is True:
        remove_files_in_dir(os.path.dirname(LOGDIR))  # 在训练状态 需要先删掉log文件 防止冲突
    with tf.Graph().as_default():
        stream = model_definition()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sv = tf.train.Supervisor(logdir=LOGDIR)
        with sv.managed_session() as session:
            session.run(init_op)
            stream['melody'].get_center_points(session, train=True)  # .get_train_data(session)
            # 2.训练
            stream['melody'].run_epoch(session, COMMON_MELODY_PATTERN_NUMBER)
            stream['intro'].run_epoch(session, COMMON_MELODY_PATTERN_NUMBER)
            stream['chord'].run_epoch(session)
            stream['drum'].run_epoch(session, COMMON_DRUM_PATTERN_NUMBER)
            stream['bass'].run_epoch(session, COMMON_BASS_PATTERN_NUMBER)
            stream['piano_guitar'].run_epoch(session, COMMON_PIANO_GUITAR_PATTERN_NUMBER)
            stream['string'].run_epoch(session, COMMON_STRING_PATTERN_NUMBER)
            sv.saver.save(session, LOGDIR, global_step=sv.global_step)
            # 3.输出

            for tt in range(1, 5):
                stream_output = output_rawdata(session, stream)
                outtt = '../Outputs/test096_%d.mid' % tt
                tracks2song(stream_output, outtt)  # 生成midi文件
                time.sleep(30)


if __name__ == '__main__':
        generate_1song()
