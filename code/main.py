from dataoutputs.musicout import MusicPromote, MelodyList2PianoRoll, ChordList2PianoRoll, NoteList2PianoRoll, \
    BassList2PianoRoll, PgList2PianoRoll, FillList2PianoRoll, StringList2PianoRoll
from interfaces.midi.midi import MultiPianoRoll2Midi
from others.validation import RunValidation
from pipelines.bass_pipeline import BassPipeline
from pipelines.chord_pipeline import ChordPipeline
from pipelines.drum_pipeline import DrumPipeline
from pipelines.melody_pipeline import MelodyPipeline
from pipelines.string_pipeline import StringPipeline
from pipelines.piano_guitar_pipeline import PianoGuitarPipeline_3
from pipelines.fill_pipeline import FillPipeline
from settings import *
from preparation.store_dataset import StoreMidiFileInfo, SaveMidiData
from interfaces.functions import getsysargs
import tensorflow as tf
import os
import sys


def train(melody, chord, drum, bass, fill):
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sv = tf.train.Supervisor(logdir=LOGDIR)
    with sv.managed_session() as session:
        session.run(init_op)
        melody.get_train_data(session)
        # 2.训练
        melody.run_epoch(session)
        chord.run_epoch(session)
        drum.run_epoch(session)
        bass.run_epoch(session)
        fill.run_epoch(session)
        sv.saver.save(session, LOGDIR, global_step=sv.global_step)
        melody_output, chord_output, drum_output, bass_output, fill_output = output_by_lstm(session, melody, chord, drum, bass, fill)
    return melody_output, chord_output, drum_output, bass_output, fill_output


def restore(melody, chord, drum, bass, fill):
    sv = tf.train.Supervisor(logdir=LOGDIR)
    with sv.managed_session() as session:
        sv.saver.restore(session, LOGDIR)
        melody.get_train_data(session, train=FLAG_IS_TRAINING)
        melody_output, chord_output, drum_output, bass_output, fill_output = output_by_lstm(session, melody, chord, drum, bass, fill)
    return melody_output, chord_output, drum_output, bass_output, fill_output


def output_by_lstm(session, melody, chord, drum, bass, fill):
    while True:
        melody_output = melody.generate(session, None)
        if not melody_output:
            continue
        chord_output = chord.generate(session, melody_output, melody.train_data.common_melody_patterns)  # 在生成和弦和打击乐的时候也使用没有调式限制的主旋律ｐａｔｔｅｒｎ数据
        if not chord_output:
            continue
        drum_output = drum.generate(session, melody_output, melody.train_data.common_melody_patterns)
        if not drum_output:
            continue
        bass_output = bass.generate(session, melody_output, chord_output, melody.train_data.common_melody_patterns)
        if not bass_output:
            continue
        fill_output = fill.generate(session, melody_output, melody.train_data.common_melody_patterns, TONE_MAJOR)
        if not fill_output:
            continue
        return melody_output, chord_output, drum_output, bass_output, fill_output


def output_by_hmm(melody, chord, melody_output, chord_output):
    pg = PianoGuitarPipeline_3(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data, melody.train_data.keypress_pattern_data, melody.train_data.keypress_pattern_dict)
    st = StringPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data)
    pg.rhy_model_definition(melody_output)
    pg.final_model_definition(chord_output)
    st.model_definition(chord_output)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as hmm_session:
        hmm_session.run(init_op)
        rhy_out = pg.rhythm_generate(hmm_session)
        pg_output = pg.final_generate(hmm_session, rhy_out)
        string_output = st.generate(hmm_session)
    return pg_output, string_output


def tracks2song(melody_output, chord_output, drum_output, bass_output, fill_output, pg_output, string_output, output_dir=None):
    intro, split, melody_output, chord_output, drum_output, bass_output, fill_output, pg_output, string_output = MusicPromote(melody_output, chord_output, drum_output, bass_output, fill_output, pg_output, string_output)
    melody_pianoroll = MelodyList2PianoRoll(melody_output, 100, 0.9, split=split)  # 分别生成主旋律 和弦 鼓对应的piano roll
    intro_pianoroll = MelodyList2PianoRoll(intro, 90, 0.9)  # 前奏
    drum_pianoroll = NoteList2PianoRoll(drum_output, DRUM_TIME_STEP, 85, 0.6)
    bass_pianoroll = BassList2PianoRoll(bass_output, 90, 0.9, split=split)
    fill_pianoroll = FillList2PianoRoll(fill_output, 85, 0.25)
    pg_pianoroll = PgList2PianoRoll(pg_output, 85, 0.6, scale_adjust=12, split=split)
    string_pianoroll = StringList2PianoRoll(string_output, 70, 1, scale_adjust=-12, split=split)
    if output_dir is None:
        output_dir = GENERATE_MIDIFILE_PATH
    MultiPianoRoll2Midi(output_dir, 100,
                        {0: {'name': 'Main', 'program': 26, 'note': melody_pianoroll},
                         1: {'name': 'Intro', 'program': 5, 'note': intro_pianoroll},
                         2: {'name': 'Bass', 'program': 33, 'note': bass_pianoroll},
                         3: {'name': 'PG', 'program': 1, 'note': pg_pianoroll},
                         4: {'name': 'String', 'program': 48, 'note': string_pianoroll},
                         5: {'name': 'Fill', 'program': 10, 'note': fill_pianoroll},
                         9: {'name': 'Drum', 'program': 0, 'note': drum_pianoroll}})


def generate_1song(output_dir=None):
    with tf.Graph().as_default():
        melody = MelodyPipeline(TONE_MAJOR, train=FLAG_IS_TRAINING)
        chord = ChordPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)  # 在训练和弦和打击乐的时候使用无调式限制的主旋律数据
        drum = DrumPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)
        bass = BassPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data)
        fill = FillPipeline(melody.train_data.raw_melody_data, melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)
        if FLAG_IS_TRAINING:
            melody_output, chord_output, drum_output, bass_output, fill_output = train(melody, chord, drum, bass, fill)
        else:
            melody_output, chord_output, drum_output, bass_output, fill_output = restore(melody, chord, drum, bass, fill)
    with tf.Graph().as_default():
        pg_output, string_output = output_by_hmm(melody, chord, melody_output, chord_output)
    tracks2song(melody_output, chord_output, drum_output, bass_output, fill_output, pg_output, string_output, output_dir=output_dir)


if __name__ == '__main__':
    if FLAG_READ_MIDI_FILES:  # 如果这个值为True的话，程序将率先从midi文件中读取音乐信息存储到数据库中。这个过程不需要在每次运行程序时都执行。
        StoreMidiFileInfo()
        smd = SaveMidiData()
    if FLAG_RUN_VALIDATION:
        RunValidation()  # 运行验证内容
    if FLAG_RUN_MAIN:
        os.chdir(sys.path[0])
        outputpath = getsysargs().outputpath
        generate_1song(output_dir=outputpath)
