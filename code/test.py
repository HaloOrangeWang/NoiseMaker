import copy

from settings import *


class TestCase:

    def G1(self):
        from pipelines.melody_pipeline import MelodyPipeline
        from pipelines.chord_pipeline import ChordPipeline
        import tensorflow as tf
        melody = MelodyPipeline(TONE_MAJOR)
        chord = ChordPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)  # 在训练和弦和打击乐的时候使用无调式限制的主旋律数据
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)  # 初始化
            # 2.训练
            melody.run_epoch(sess)
            chord.run_epoch(sess)
            # 3.生成主旋律 和弦 鼓点和bass
            while True:
                melody_output = melody.generate(sess, None)
                if not melody_output:
                    continue
                chord_output = chord.generate(sess, melody_output, melody.train_data.common_melody_patterns)  # 在生成和弦和打击乐的时候也使用没有调式限制的主旋律ｐａｔｔｅｒｎ数据
                if not chord_output:
                    continue
                if melody_output and chord_output:  # 当所有的音乐元素都生成成功之后 就退出循环
                    break
            from dataoutputs.musicout import MelodyList2PianoRoll
            melody_pianoroll = MelodyList2PianoRoll(melody_output, 100, 0.9)  # 分别生成主旋律 和弦 鼓对应的piano roll
            from dataoutputs.musicout import ChordList2PianoRoll
            chord_pianoroll = ChordList2PianoRoll(chord_output, 85, 0.6)
            from interfaces.midi.midi import MultiPianoRoll2Midi
            MultiPianoRoll2Midi(GENERATE_MIDIFILE_PATH, 100,
                                {0: {'name': 'Main', 'program': 5, 'note': melody_pianoroll},
                                 1: {'name': 'Chord', 'program': 1, 'note': chord_pianoroll}})

            print('melody_output', melody_output)
            print('chord_output', chord_output)

    def G2(self):
        from interfaces.sql.sqlite import NoteDict
        print(NoteDict)
        from pipelines.piano_guitar_pipeline import PianoGuitarPipeline
        from datainputs.melody import MelodyTrainData
        from datainputs.fill import FillTrainData
        from interfaces.sql.sqlite import GetRawSongDataFromDataset
        from pipelines.fill_pipeline import FillPipeline
        from pipelines.melody_pipeline import MelodyPipeline
        import tensorflow as tf
        melody = MelodyPipeline(TONE_MAJOR)
        melody_output = [84, 0, 0, 0, 0, 0, 81, 0, 81, 0, 0, 0, 81, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 79, 0, 74, 0, 0, 0, 0, 0, 74, 0, 81, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 0, 0, 0, 83, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0]
        # no_tone_restrict_melody_data = GetRawSongDataFromDataset('main', None)  # 没有旋律限制的主旋律数据　用于训练其他数据
        # raw_melody_data = copy.deepcopy(no_tone_restrict_melody_data)  # 最原始的主旋律数据
        #  pg = PianoGuitarPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data, TONE_MAJOR)

        # FillTrainData(melody.train_data.raw_melody_data, melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)
        fill = FillPipeline(melody.train_data.raw_melody_data, melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)
        # fill = FillPipeline(None, None, None)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            fill.run_epoch(sess)
            print(fill.generate(sess, melody_output, melody.train_data.common_melody_patterns, TONE_MAJOR))
        # fill.generate(None, melody_output, cm, TONE_MAJOR)

    def G3(self):
        from interfaces.sql.sqlite import NoteDict
        print(NoteDict)
        from pipelines.piano_guitar_pipeline import PianoGuitarPipeline
        import tensorflow as tf
        from pipelines.melody_pipeline import MelodyPipeline
        from pipelines.chord_pipeline import ChordPipeline
        melody = MelodyPipeline(TONE_MAJOR)
        chord = ChordPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)  # 在训练和弦和打击乐的时候使用无调式限制的主旋律数据
        pg = PianoGuitarPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data, TONE_MAJOR)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        melody_output = [84, 0, 0, 0, 0, 0, 81, 0, 81, 0, 0, 0, 81, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 79, 0, 74, 0, 0, 0, 0, 0, 74, 0, 81, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 0, 0, 0, 83, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0]
        chord_output = [56, 56, 43, 43, 31, 31, 31, 31, 31, 31, 43, 43, 43, 43, 1, 1, 43, 43, 1, 1, 31, 31, 31, 31, 1, 1, 43, 43, 1, 1, 1, 1]
        melody_output2 = [0, 0, 0, 0, 76, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 81, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 74, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0]
        chord_output2 = [1, 1, 43, 43, 1, 1, 31, 31, 1, 1, 31, 31, 43, 43, 1, 1, 101, 101, 14, 14, 26, 26, 43, 43, 43, 43, 43, 43, 43, 43, 1, 1]
        melody_output3 = [74, 0, 0, 0, 72, 0, 0, 0, 67, 0, 0, 0, 72, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 71, 0, 0, 0, 72, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 74, 0, 76, 0, 72, 0, 0, 0, 74, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 74, 0, 0, 0, 76, 0, 0, 0, 77, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 77, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        chord_output3 = [1, 1, 1, 1, 31, 31, 31, 31, 56, 56, 56, 56, 56, 56, 14, 14, 43, 43, 43, 43, 1, 1, 56, 56, 43, 43, 1, 1, 1, 1, 1, 1]
        melody_output4 = [81, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 74, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 0, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 69, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 69, 0, 0, 0, 69, 0, 0, 0, 65, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 72, 0, 0, 0, 74, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 72, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0]
        chord_output4 = [1, 1, 1, 1, 1, 1, 1, 1, 56, 56, 43, 43, 56, 56, 43, 43, 43, 43, 1, 1, 31, 31, 1, 1, 1, 1, 1, 1, 1, 1, 43, 43, 1, 1, 1, 1, 31, 31, 1, 1, 1, 1, 43, 43, 1, 1, 1, 1]
        melody_output5 = [69, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 71, 0, 0, 0, 69, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0, 0, 60, 0, 0, 0, 69, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 72, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0]
        chord_output5 = [56, 56, 1, 1, 1, 1, 31, 31, 1, 1, 43, 43, 1, 1, 1, 1, 1, 1, 31, 31, 31, 31, 31, 31, 43, 43, 43, 43, 43, 43, 1, 1, 14, 14, 13, 13, 43, 43, 1, 1]

        with tf.Session() as sess:
            sess.run(init_op)  # 初始化
            # 2.训练
            pg.run_epoch(sess)
            # 3.生成主旋律
            for t in range(100):
                print(t)
                pg.generate(sess, melody_output, chord_output, melody.train_data.common_melody_patterns)
                pg.generate(sess, melody_output2, chord_output2, melody.train_data.common_melody_patterns)
                pg.generate(sess, melody_output3, chord_output3, melody.train_data.common_melody_patterns)
                pg.generate(sess, melody_output4, chord_output4, melody.train_data.common_melody_patterns)
                pg.generate(sess, melody_output5, chord_output5, melody.train_data.common_melody_patterns)

            print(pg.x1, pg.x2)
            # from dataoutputs.musicout import MelodyList2PianoRoll
            # melody_pianoroll = MelodyList2PianoRoll(melody_output, 100, 0.9)  # 分别生成主旋律 和弦 鼓对应的piano roll
            # from dataoutputs.musicout import ChordList2PianoRoll
            # chord_pianoroll = ChordList2PianoRoll(chord_output, 85, 0.6)
            # from dataoutputs.musicout import NoteList2PianoRoll
            # pg_pianoroll = NoteList2PianoRoll(pg_output, 0.25, 85, 0.6)
            # from interfaces.midi.midi import MultiPianoRoll2Midi
            # MultiPianoRoll2Midi(GENERATE_MIDIFILE_PATH, 100,
            #                     {0: {'name': 'Main', 'program': 5, 'note': melody_pianoroll},
            #                      1: {'name': 'Chord', 'program': 1, 'note': chord_pianoroll},
            #                      3: {'name': 'PG', 'program': 5, 'note': pg_pianoroll}})

    def G4(self):

        melody_output = [84, 0, 0, 0, 0, 0, 81, 0, 81, 0, 0, 0, 81, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 79, 0, 74, 0, 0, 0, 0, 0, 74, 0, 81, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 0, 0, 0, 83, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0]
        chord_output = [56, 56, 43, 43, 31, 31, 31, 31, 31, 31, 43, 43, 43, 43, 1, 1, 43, 43, 1, 1, 31, 31, 31, 31, 1, 1, 43, 43, 1, 1, 1, 1]

        from pipelines.melody_pipeline import MelodyPipeline
        from pipelines.chord_pipeline import ChordPipeline
        melody = MelodyPipeline(TONE_MAJOR)
        chord = ChordPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data)  # 在训练和弦和打击乐的时候使用无调式限制的主旋律数据
        from pipelines.string_pipeline import StringPipeline
        import tensorflow as tf
        # pg = PianoGuitarPipeline_3(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data, melody.train_data.keypress_pattern_data, melody.train_data.keypress_pattern_dict)
        # # PianoGuitarTrainData_2(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data)
        # pg.rhy_model_definition(melody_output)
        # pg.final_model_definition(chord_output)
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # hmm_sess = tf.Session()
        # hmm_sess.run(init_op)
        # rhy_out = pg.rhythm_generate(hmm_sess)
        # pg.final_generate(hmm_sess, rhy_out)
        st = StringPipeline(melody.train_data.no_tone_restrict_melody_pattern_data, melody.train_data.no_tone_restrict_continuous_bar_number_data, chord.train_data.chord_data)
        st.model_definition(chord_output)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        hmm_sess = tf.Session()
        hmm_sess.run(init_op)
        st.generate(hmm_sess)

    def run(self):
        from dataoutputs.musicout import GetIntro
        melody_output = [84, 0, 0, 0, 0, 0, 81, 0, 81, 0, 0, 0, 81, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 77, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 81, 0, 0, 0, 0, 0, 79, 0, 74, 0, 0, 0, 0, 0, 74, 0, 81, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 0, 0, 0, 83, 0, 0, 0, 81, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 76, 0, 0, 0, 74, 0, 0, 0, 74, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0]
        x = GetIntro(melody_output)
        print(x)


if __name__ == '__main__':
    test_case = TestCase()
    test_case.run()
