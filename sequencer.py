import rtmidi
from rtmidi.midiconstants import TIMING_CLOCK

class Sequencer:
    def __init__(self, pitches, rhythm, channel, duration=5, p_start = 0,  seq_len = 16,  clock_in=0, port_out=0):
        # Sequence data
        self.pitches = pitches
        self.rhythm = rhythm
        self.channel = max(1, min(channel, 16)) - 1
        self.seq_len = seq_len
        self.duration = duration
        self.p_start = p_start

        # MIDI setup
        self.midiin = rtmidi.MidiIn()
        self.midiin.ignore_types(timing=False)
        self.midiin.open_port(clock_in)

        self.midi_out = rtmidi.MidiOut()
        self.midi_out.open_port(port_out)

        # Timing state
        self.ticks = 0
        self.current_tick = 0
        self.playstate = False

        # Sequence state
        self.step_index = 0
        self.note_i = p_start
        self.note_playing = False
        self.current_note = 0
        self.gate_off_tick = 0

        print("MIDI ports opened successfully.")

    def start(self):
        msg = self.midiin.get_message()
        if msg:
            data = msg[0]
            if data == [250]:  # Start
                print('MIDI Start received.')
                self.reset()
            elif data == [252]:  # Stop
                print('MIDI Stop received.')
                self.stop_note()
                self.playstate = False
            elif data == [TIMING_CLOCK] and self.playstate:
                self.current_tick += 1
                self.ticks = (self.ticks + 1) % 6
                if self.ticks == 0:
                    self.process_step()
                self.check_note_off()

    def reset(self):
        self.ticks = 0
        self.current_tick = 0
        self.step_index = 0
        self.note_i = 0
        self.note_playing = False
        self.stop_note()
        self.playstate = True

    def stop_note(self):
        if self.note_playing:
            self.midi_out.send_message([0x80 | self.channel, self.current_note, 0])
            self.note_playing = False

    def process_step(self):
        step = self.rhythm[self.step_index]
        if step == 0:
            self.note_i = self.p_start
        if step == 1:
            self.stop_note()
            self.current_note = self.pitches[self.note_i]
          #  print("Note on:", self.current_note)
            self.midi_out.send_message([0x90 | self.channel, self.current_note, 100])
            self.note_playing = True
            self.gate_off_tick = self.current_tick + self.duration
            self.note_i = self.note_i + 1
        self.step_index = (self.step_index + 1) % self.seq_len

    def check_note_off(self):
        if self.note_playing and self.current_tick >= self.gate_off_tick:
            self.stop_note()

