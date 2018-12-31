import mido
import random


pitch_list = [[] for t0 in range(32)]  # 每个半拍所有音符的音高
vel_list = [[] for t1 in range(32)]  # 每个半拍所有音符的音量

# 随机确定每个半拍的音量和音高
for i in range(32):
    note_num = random.randint(0, 3)
    while True:
        pitch_list[i] = [random.randint(60, 84) for t2 in range(note_num)]
        vel_list[i] = [random.randint(64, 100) for t3 in range(note_num)]
        if len(pitch_list[i]) == len(set(pitch_list[i])):  # 同一时间音符的音高不能重合
            break


mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)

track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
track.append(mido.MetaMessage('track_name', name='Piano 1', time=0))
track.append(mido.Message('program_change', program=1, time=0))  # 这个音轨使用的乐器

current_beat = 0
for i in range(32):
    # 添加这个半拍所有音符的开始
    for note in range(len(pitch_list[i])):
        note_beat = i * 0.5
        track.append(mido.Message('note_on', note=pitch_list[i][note], velocity=vel_list[i][note],
                                  time=round(480 * (note_beat - current_beat))))
        current_beat = note_beat
    # 添加这个半拍所有音符的终止
    for note in range(len(pitch_list[i])):
        note_beat = i * 0.5 + 0.4
        track.append(mido.Message('note_off', note=pitch_list[i][note], velocity=vel_list[i][note],
                                  time=round(480 * (note_beat - current_beat))))
        current_beat = note_beat

mid.save('test.mid')
