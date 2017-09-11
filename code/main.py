from dataoutputs.musicout import MusicPromote, MelodyList2PianoRoll, ChordList2PianoRoll, NoteList2PianoRoll
from interfaces.midi.midi import MultiPianoRoll2Midi
from others.validation import RunValidation
from pipelines.chord_pipeline import ChordPipeline
from pipelines.drum_pipeline import DrumPipeline
from pipelines.melody_pipeline import MelodyPipeline
from settings import *
from preparation.store_dataset import StoreMidiFileInfo, SaveMidiData
from test import TestCase
import tensorflow as tf

if __name__ == '__main__':
    if FLAG_READ_MIDI_FILES:  # 如果这个值为True的话，程序将率先从midi文件中读取音乐信息存储到数据库中。这个过程不需要在每次运行程序时都执行。
        StoreMidiFileInfo()
        smd = SaveMidiData()
    if FLAG_RUN_TEST_CASE:
        testcase = TestCase()
        testcase.run()
    if FLAG_RUN_VALIDATION:
        RunValidation()  # 运行验证内容
    if FLAG_RUN_MAIN:
        melody = MelodyPipeline()
        chord = ChordPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)  # 在训练和弦和打击乐的时候使用无调式限制的主旋律数据
        drum = DrumPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)  # 初始化
            # 2.训练
            melody.run_epoch(sess)
            chord.run_epoch(sess)
            drum.run_epoch(sess)
            # 3.生成主旋律
            melody_output = melody.generate(sess, None)
            chord_output = chord.generate(sess, melody_output, melody.train_data.common_melody_patterns)  # 在生成和弦和打击乐的时候也使用没有调式限制的主旋律ｐａｔｔｅｒｎ数据
            drum_output = drum.generate(sess, melody_output, melody.train_data.common_melody_patterns)
            # 4.这个函数是让生成的音乐更丰满一些
            melody_output, chord_output, drum_output = MusicPromote(melody_output, chord_output, drum_output)
            # 5.旋律解码，得到midi文件
            melody_pianoroll = MelodyList2PianoRoll(melody_output, 100, 0.9)  # 分别生成主旋律 和弦 鼓对应的piano roll
            chord_pianoroll = ChordList2PianoRoll(chord_output, 85, 0.6)
            drum_pianoroll = NoteList2PianoRoll(drum_output, DRUM_TIME_STEP, 85, 0.6)
            MultiPianoRoll2Midi(GENERATE_MIDIFILE_PATH, 100, {0: {'name': 'Main', 'program': 5, 'note': melody_pianoroll}, 1: {'name': 'Chord', 'program': 1, 'note': chord_pianoroll}, 9: {'name': 'Drum', 'program': 0, 'note': drum_pianoroll}})
