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
import tensorflow as tf

if __name__ == '__main__':
    if FLAG_READ_MIDI_FILES:  # 如果这个值为True的话，程序将率先从midi文件中读取音乐信息存储到数据库中。这个过程不需要在每次运行程序时都执行。
        StoreMidiFileInfo()
        smd = SaveMidiData()
    if FLAG_RUN_VALIDATION:
        RunValidation()  # 运行验证内容
    if FLAG_RUN_MAIN:
        melody = MelodyPipeline(TONE_MAJOR)
        chord = ChordPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)  # 在训练和弦和打击乐的时候使用无调式限制的主旋律数据
        drum = DrumPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)
        bass = BassPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data)
        pg = PianoGuitarPipeline_3(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data, melody.train_data.keypress_pattern_data, melody.train_data.keypress_pattern_dict)
        st = StringPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data)
        fill = FillPipeline(melody.train_data.raw_melody_data, melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as lstm_session:
            lstm_session.run(init_op)  # 初始化
            # 2.训练
            melody.run_epoch(lstm_session)
            chord.run_epoch(lstm_session)
            drum.run_epoch(lstm_session)
            bass.run_epoch(lstm_session)
            fill.run_epoch(lstm_session)
            # 3.生成主旋律 和弦 鼓点和bass
            while True:
                melody_output = melody.generate(lstm_session, None)
                if not melody_output:
                    continue
                chord_output = chord.generate(lstm_session, melody_output, melody.train_data.common_melody_patterns)  # 在生成和弦和打击乐的时候也使用没有调式限制的主旋律ｐａｔｔｅｒｎ数据
                if not chord_output:
                    continue
                drum_output = drum.generate(lstm_session, melody_output, melody.train_data.common_melody_patterns)
                if not chord_output:
                    continue
                bass_output = bass.generate(lstm_session, melody_output, chord_output, melody.train_data.common_melody_patterns)
                if not bass_output:
                    continue
                fill_output = fill.generate(lstm_session, melody_output, melody.train_data.common_melody_patterns, TONE_MAJOR)
                if not fill_output:
                    break
                if melody_output and chord_output and drum_output and bass_output and fill_output:  # 当所有的音乐元素都生成成功之后 就退出循环
                    break
        # 4.生成钢琴 吉他 弦乐伴奏
        pg.rhy_model_definition(melody_output)
        pg.final_model_definition(chord_output)
        st.model_definition(chord_output)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as hmm_session:
            hmm_session.run(init_op)
            rhy_out = pg.rhythm_generate(hmm_session)
            pg_output = pg.final_generate(hmm_session, rhy_out)
            string_output = st.generate(hmm_session)
        # 5.这个函数是让生成的音乐更丰满一些
        intro, split, melody_output, chord_output, drum_output, bass_output, fill_output, pg_output, string_output = MusicPromote(melody_output, chord_output, drum_output, bass_output, fill_output, pg_output, string_output)
        # 6.旋律解码，得到midi文件
        melody_pianoroll = MelodyList2PianoRoll(melody_output, 100, 0.9, split=split)  # 分别生成主旋律 和弦 鼓对应的piano roll
        intro_pianoroll = MelodyList2PianoRoll(intro, 90, 0.9)  # 前奏
        chord_pianoroll = ChordList2PianoRoll(chord_output, 85, 0.6)
        drum_pianoroll = NoteList2PianoRoll(drum_output, DRUM_TIME_STEP, 85, 0.6)
        bass_pianoroll = BassList2PianoRoll(bass_output, 90, 0.9, split=split)
        fill_pianoroll = FillList2PianoRoll(fill_output, 85, 0.25)
        pg_pianoroll = PgList2PianoRoll(pg_output, 85, 0.6, scale_adjust=12, split=split)
        string_pianoroll = StringList2PianoRoll(string_output, 70, 1, scale_adjust=-12, split=split)
        MultiPianoRoll2Midi(GENERATE_MIDIFILE_PATH, 100,
                            {0: {'name': 'Main', 'program': 26, 'note': melody_pianoroll},
                             1: {'name': 'Intro', 'program': 5, 'note': intro_pianoroll},
                             2: {'name': 'Bass', 'program': 33, 'note': bass_pianoroll},
                             3: {'name': 'PG', 'program': 1, 'note': pg_pianoroll},
                             4: {'name': 'String', 'program': 48, 'note': string_pianoroll},
                             5: {'name': 'Fill', 'program': 10, 'note': fill_pianoroll},
                             9: {'name': 'Drum', 'program': 0, 'note': drum_pianoroll}})
