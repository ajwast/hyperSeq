import rtmidi
from rtmidi.midiconstants import TIMING_CLOCK

class Sequencer:
    def __init__(self, pitches1, rhythm1, channel1, 
                       pitches2, rhythm2, channel2,
                       duration=5, clock_in=0, port_out=0):
        # Sequence data
        self.pitches1 = pitches1
        self.rhythm1 = rhythm1
        self.channel1 = max(1, min(channel1, 16)) - 1

        self.pitches2 = pitches2
        self.rhythm2 = rhythm2
        self.channel2 = max(1, min(channel2, 16)) - 1

        self.duration = duration

        # MIDI setup
        self.midiin = rtmidi.MidiIn()
        self.midiin.ignore_types(timing=False)
        self.midiin.open_port(clock_in)

        self.midi_out = rtmidi.MidiOut()
        self.midi_out.open_port(port_out)

        # Global timing
        self.ticks = 0
        self.current_tick = 0
        self.playstate = False

        # Sequencer 1 state
        self.sx1_n = 0
        self.note1_playing = False
        self.note1 = 0
        self.gate1_off = 0

        # Sequencer 2 state
        self.sx2_n = 0
        self.note2_playing = False
        self.note2 = 0
        self.gate2_off = 0

        print("MIDI ports opened successfully.")

    def start(self):
        print("Sequencer started.")
        while True:
            msg = self.midiin.get_message()
            if msg:
                data = msg[0]
                if data == [250]:  # Start
                    print('MIDI Start received.')
                    self.reset()
                elif data == [252]:  # Stop
                    print('MIDI Stop received.')
                    self.stop_note(1)
                    self.stop_note(2)
                    self.playstate = False
                elif data == [TIMING_CLOCK] and self.playstate:
                    self.current_tick += 1
                    self.ticks = (self.ticks + 1) % 6
                    if self.ticks == 0:
                        self.process_step(1)
                        self.process_step(2)
                    self.check_note_off(1)
                    self.check_note_off(2)

    def reset(self):
        self.ticks = 0
        self.current_tick = 0
        self.sx1_n = 0
        self.sx2_n = 0
        self.note1_playing = False
        self.note2_playing = False
        self.stop_note(1)
        self.stop_note(2)
        self.playstate = True

    def stop_note(self, seq_id):
        if seq_id == 1 and self.note1_playing:
            self.midi_out.send_message([0x80 | self.channel1, self.note1, 0])
            self.note1_playing = False
        elif seq_id == 2 and self.note2_playing:
            self.midi_out.send_message([0x80 | self.channel2, self.note2, 0])
            self.note2_playing = False

    def process_step(self, seq_id):
        if seq_id == 1:
            step = self.rhythm1[self.sx1_n]
            if step == 1:
                self.stop_note(1)
                self.note1 = self.pitches1[self.sx1_n]
                print("Note 1 on:", self.note1)
                self.midi_out.send_message([0x90 | self.channel1, self.note1, 100])
                self.note1_playing = True
                self.gate1_off = self.current_tick + self.duration
            self.sx1_n = (self.sx1_n + 1) % len(self.rhythm1)

        elif seq_id == 2:
            step = self.rhythm2[self.sx2_n]
            if step == 1:
                self.stop_note(2)
                self.note2 = self.pitches2[self.sx2_n]
                print("Note 2 on:", self.note2)
                self.midi_out.send_message([0x90 | self.channel2, self.note2, 100])
                self.note2_playing = True
                self.gate2_off = self.current_tick + self.duration
            self.sx2_n = (self.sx2_n + 1) % len(self.rhythm2)

    def check_note_off(self, seq_id):
        if seq_id == 1 and self.note1_playing and self.current_tick >= self.gate1_off:
            self.stop_note(1)
        elif seq_id == 2 and self.note2_playing and self.current_tick >= self.gate2_off:
            self.stop_note(2)
