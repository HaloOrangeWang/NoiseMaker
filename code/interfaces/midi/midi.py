import mido
from settings import *


class GenerateDataFromMidiFile:

    marked_track_list = {}
    channel_list = []  # 音轨编号 在note_list中的位置

    def __init__(self, file_name, bias_time=0, scale=0):
        self.marked_track_list = {}
        self.channel_list = []
        tracks = self.readfile(file_name)  # 1.获取一首歌的各个音轨
        marked_note_list = self.get_marked_note_list(tracks, bias_time)  # 2.获取这首歌的音符列表 分音轨保存
        pianoroll_list = self.generate_multi_pianoroll(marked_note_list)
        self.pianoroll_list = self.adjust_scale(pianoroll_list, scale)

    def readfile(self, file_name):
        mid = mido.MidiFile(file_name)
        self.ticks_per_beat = mid.ticks_per_beat  # 每拍多少个ticks
        return mid.tracks

    def get_note_list(self, track, bias_time):
        """
        获取一首歌的一个一个音轨的音符列表。音符列表为二维数组 第一维为音符 第二维为音符属性（时间+音高+音量+持续时间）
        :param track: 一首歌的音轨
        :param bias_time: 偏移时间（即对整个歌曲进行平移，将重音移到每小节的第一拍）
        :return: 音符列表
        """
        curr_beat = 0  # 当前拍子
        note_list = []  # 音符列表
        # 1.保存这个音轨中的所有音符
        for msg in track:
            curr_beat += msg.time / self.ticks_per_beat
            if msg.type == 'note_on':
                note_list.append([curr_beat, msg.note, msg.velocity, -1])
            elif msg.type == 'note_off':
                for note in note_list:
                    if note[1] == msg.note and note[3] == -1:
                        note[3] = curr_beat - note[0]
        if not note_list:  # 空的音符列表直接返回None
            return None
        # 2.后续工作：对整个音轨进行平移，如果有音符的持续时间不确定则记为1拍
        for note in note_list:
            note[0] -= bias_time
            if note[3] == -1:
                note[3] = 1
        note_iterator = 0
        # 3.由于平移之后可能会有音符的时间为负数，因此去掉所有位置为负数的音符
        while 1:
            if note_list[note_iterator][0] < 0:
                note_list.pop(note_iterator)  # 去掉音符位置为负数的音符
            else:
                note_iterator += 1
            if note_iterator >= len(note_list):
                break
        return note_list

    def get_marked_note_list(self, tracks, bias_time, eliminate=None):
        marked_note_list = {'others': []}  # marked_note_list是四维列表。第一维是音轨标注，第二维是音轨，第三维是音轨中的音符，第四维是音符属性
        for track in tracks:
            key = track.name
            if key.startswith('Main'):
                if eliminate is None or 'main' not in eliminate:
                    if 'main' not in marked_note_list:
                        marked_note_list['main'] = []
                    note_list = self.get_note_list(track, bias_time)
                    if note_list:
                        marked_note_list['main'].append(note_list)
            elif key.startswith('Chord'):
                if eliminate is None or 'chord' not in eliminate:
                    if 'chord' not in marked_note_list:
                        marked_note_list['chord'] = []
                    note_list = self.get_note_list(track, bias_time)
                    if note_list:
                        marked_note_list['chord'].append(note_list)
            elif key.startswith('Drum'):
                if eliminate is None or 'drum' not in eliminate:
                    if 'drum' not in marked_note_list:
                        marked_note_list['drum'] = []
                    note_list = self.get_note_list(track, bias_time)
                    if note_list:
                        marked_note_list['drum'].append(note_list)
            else:
                note_list = self.get_note_list(track, bias_time)
                if note_list:
                    marked_note_list['others'].append(note_list)
        return marked_note_list

    def generate_pianoroll(self, note_list):
        """
        :param note_list: 音符列表 上面一个函数返回的
        :return: 转为piano roll形式的音符列表 二维数组 音符按时间排序 忽略了channel
        """
        pianoroll_list = []
        track_note_number_list = []  # 这个二维列表的用途是存储每个track中已经存储了多少个音符
        curr_beat = 0  # 当前的拍数
        for track_iterator in range(len(note_list)):  # 初始化channel_list
            track_note_number_list.append(0)
        while 1:
            for track_iterator in range(len(track_note_number_list)):
                if track_note_number_list[track_iterator] == len(note_list[track_iterator]):
                    continue
                while 1:  # 加一个while处理逻辑是为了防止两个音符处在同一个time_step区间导致bug
                    if track_note_number_list[track_iterator] == len(note_list[track_iterator]):
                        break
                    note_beat = note_list[track_iterator][track_note_number_list[track_iterator]][0]  # 音符所在的时间位置
                    # print(channel_iterator, channel_list[channel_iterator], note_beat)
                    if note_beat - curr_beat > 1 / 2 * MELODY_TIME_STEP:  # 这个音符的时间超过了当前时间 直接break
                        break
                    if abs(note_beat - curr_beat) <= 1 / 2 * MELODY_TIME_STEP:  # 这个音符所在时间为当前时间
                        if note_list[track_iterator][track_note_number_list[track_iterator]][2] > 0:  # 不要音量为0的音
                            pianoroll_list.append(
                                [curr_beat, note_list[track_iterator][track_note_number_list[track_iterator]][1],
                                 note_list[track_iterator][track_note_number_list[track_iterator]][2],
                                 note_list[track_iterator][track_note_number_list[track_iterator]][3]])  # 把这个音添加到pianoroll_list中 但是这个音的时间并不是note_list中存储的时间而是curr_beat
                            track_note_number_list[track_iterator] += 1
            curr_beat += MELODY_TIME_STEP  # 这个时间步长处理完毕 进入下一个时间步长
            # print(curr_beat)
            # 跳出循环的条件是note_list的所有音都进入了pianoroll_list中
            flag_break = True
            for track_iterator in range(len(track_note_number_list)):
                if track_note_number_list[track_iterator] != len(note_list[track_iterator]):
                    flag_break = False
                    break
            if flag_break or curr_beat >= 1000:
                break
        return pianoroll_list

    def generate_multi_pianoroll(self, marked_note_list):
        pianoroll_list = {}
        for key in marked_note_list:
            # print(key)
            pianoroll_list[key] = self.generate_pianoroll(note_list=marked_note_list[key])
        return pianoroll_list

    def adjust_scale(self, pianoroll_list, scale):
        """
        对歌曲进行转调使得歌曲为C大调/A小调
        :param pianoroll_list: piano roll的列表
        :param scale: 要转的调数（要降多少个半音）
        :return: 转调之后的piano roll列表
        """
        for key in pianoroll_list:
            # 主旋律变调scale，打击乐不变调，其他的变调scale2
            if key == 'main':
                for pianoroll_iterator in range(len(pianoroll_list[key])):
                    pianoroll_list[key][pianoroll_iterator][1] -= scale  # 因为要转成C大调/A小调 这里是减而不是加
            elif key != 'drum':
                scale2 = scale - 12 * round(scale / 12)
                for pianoroll_iterator in range(len(pianoroll_list[key])):
                    pianoroll_list[key][pianoroll_iterator][1] -= scale2  # 因为要转成C大调/A小调 这里是减而不是加
        return pianoroll_list


def MultiPianoRoll2Midi(file_name, bpm, pianoroll_dict):
    # 1.初始化
    mid = mido.MidiFile()
    tracks = {}  # 要保存的音轨信息
    first_track = True
    midi_tempo = round(60000000 / bpm)  # 这首歌的速度（每一拍多少微秒）
    # 2.保存音符
    for key in pianoroll_dict:
        # print(key)
        # 2.1.定义音轨名称/使用乐器等
        tracks[key] = mido.MidiTrack()  # 定义新的音轨
        mid.tracks.append(tracks[key])  # 在midi中添加这个音轨

        if first_track:
            tracks[key].append(mido.MetaMessage('set_tempo', tempo=midi_tempo, time=0))  # 设置歌曲的速度
            first_track = False
        tracks[key].append(mido.MetaMessage('track_name', name=pianoroll_dict[key]['name'], time=0))  # 这个音轨的名称
        tracks[key].append(mido.Message('program_change', program=pianoroll_dict[key]['program'], time=0, channel=key))  # 这个音轨使用的乐器
        # 2.2.从piano_dict中获取音符列表并转化为midi message的形式
        note_list = []
        for note_iterator in pianoroll_dict[key]['note']:
            note_list.append(['on', note_iterator[0], note_iterator[1], note_iterator[2]])
            note_list.append(['off', note_iterator[0] + note_iterator[3], note_iterator[1], note_iterator[2]])
        note_list = sorted(note_list, key=lambda item:item[1])  # 按照音符的时间排序
        # 2.3.往tracks中保存这些音符
        current_note_time = 0
        for note_iterator in note_list:
            if note_iterator[0] == 'on':
                tracks[key].append(mido.Message('note_on', note=note_iterator[2], velocity=note_iterator[3], time=round(480 * (note_iterator[1] - current_note_time)), channel=key))
            elif note_iterator[0] == 'off':
                tracks[key].append(mido.Message('note_off', note=note_iterator[2], velocity=note_iterator[3], time=round(480 * (note_iterator[1] - current_note_time)), channel=key))
            current_note_time = note_iterator[1]
    # 3.保存这个midi文件
    mid.save(file_name)
