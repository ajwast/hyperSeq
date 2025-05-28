import rtmidi
from rtmidi.midiconstants import TIMING_CLOCK

class Sequencer:
    def __init__(self, pitches, rhythm, duration=5, clock_in=0, port_out=0, channel=1):
        self.pitches = pitches
        self.rhythm = rhythm
        self.duration = duration

        # MIDI setup
        self.midiin = rtmidi.MidiIn()
        self.midiin.ignore_types(timing=False)
        self.midiin.open_port(clock_in)

        self.midi_out = rtmidi.MidiOut()
        self.midi_out.open_port(port_out)

        # MIDI channel (convert from 1–16 to 0–15)
        self.channel = max(1, min(channel, 16)) - 1

        # State variables
        self.ticks = 0
        self.current_tick = 0
        self.sx_n = 0
        self.playstate = False
        self.note_playing = False
        self.midi_note = 0
        self.gate_off = 0

        print(f"MIDI ports opened successfully. Using channel {self.channel + 1}")

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
        self.sx_n = 0
        self.playstate = True
        self.note_playing = False
        self.stop_note()

    def stop_note(self):
        if self.note_playing:
            note_off = 0x80 | self.channel
            self.midi_out.send_message([note_off, self.midi_note, 0])
            self.note_playing = False

    def process_step(self):
        step = self.rhythm[self.sx_n]
        if step == 1:
            self.stop_note()
            self.midi_note = self.pitches[self.sx_n]
            print("Note on:", self.midi_note)
            note_on = 0x90 | self.channel
            self.midi_out.send_message([note_on, self.midi_note, 100])
            self.note_playing = True
            self.gate_off = self.current_tick + self.duration
        self.sx_n = (self.sx_n + 1) % 16

    def check_note_off(self):
        if self.note_playing and self.current_tick >= self.gate_off:
            self.stop_note()
