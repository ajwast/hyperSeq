from note_rnn import NoteRNN
from rhythm_autoencoder import RhythmAutoencoder
from trainer import Trainer
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import csv
from sequencer import Sequencer
import rtmidi
import tkinter as tk
from gui import SequencerGUI


print("Initialising...")

print("MIDI SETUP")

midiIn = rtmidi.MidiIn()
inPorts = midiIn.get_ports()
midiOut = rtmidi.MidiOut()
outPorts = midiOut.get_ports()

print("Available MIDI In ports:")

for i, port in enumerate(inPorts):
    print(port, "=", i)

midiClock = int(input("Choose MIDI clock input: "))

print("Available MIDI Out ports:")
for i, port in enumerate(outPorts):
    print(port, "=", i)

outputPort1 = int(input("Choose MIDI out port 1: "))
outputPort2 = int(input("Choose MIDI out port 2: "))

# === Data Loading Functions ===
def load_training_data(x_filename='data/X_data.csv', y_filename='data/y_data.csv', seq_len=16):
    X, y = [], []
    with open(x_filename, mode='r') as f_x:
        for row in csv.reader(f_x):
            seq = torch.tensor(list(map(float, row))).reshape(seq_len, 127)
            X.append(seq)
    with open(y_filename, mode='r') as f_y:
        for row in csv.reader(f_y):
            y.append(torch.tensor(list(map(int, row))))
    return torch.stack(X), torch.stack(y)

def load_ae_data(dataset_file='data/dataset.csv'):
    d_data = torch.tensor(np.loadtxt(dataset_file, delimiter=','), dtype=torch.float32)
    return DataLoader(TensorDataset(d_data, d_data), batch_size=16, shuffle=True)

# # === Training / Loading ===
print ("Loading data...")
X, y = load_training_data()
ae_loader = load_ae_data()


rnn_gen1 = NoteRNN()
ae_gen1 = RhythmAutoencoder()
print("Models defined. Beginning training")
Trainer.train_rnn(rnn_gen1, X, y, epochs=100)
Trainer.train_autoencoder(ae_gen1, ae_loader, epochs=50)


# Generate initial sequences
p1 = rnn_gen1.generate([62], 16, temperature=0.5)[:16]
p1 = [max(50, min(p, 80)) for p in p1]

p2 = rnn_gen1.generate(start_sequence=[65], length=16)
p2 = [max(50, min(p, 80)) for p in p2]
r1, _ = ae_gen1.generate()
r2, _ = ae_gen1.generate()
print (p1, r1, p2, r2)



# Launch sequencer
seq1 = Sequencer(
    pitches=p1,
    rhythm=r1,
    channel=1,
    duration=5,
    seq_len = 16,
    clock_in=midiClock,
    port_out=outputPort1
    )

seq2 = Sequencer(
        pitches=p2,
        rhythm=r2,
        channel=2,
        duration=5,
        clock_in=midiClock,
        port_out=outputPort2
        )

root = tk.Tk()
root.title("Sequencer Visualiser")

gui1 = SequencerGUI(root, r1, label="Seq 1")
gui2 = SequencerGUI(root, r2, label="Seq 2")

# === Button Callbacks ===
def gen_pitches1():
    global p1
    p1 = rnn_gen1.generate([62], 16, temperature=0.5)[:16]
    p1 = [max(50, min(p, 80)) for p in p1]
    seq1.pitches = p1
    print("Updated Seq1 Pitches:", p1)

def gen_rhythm1():
    global r1
    r1, _ = ae_gen1.generate()
    seq1.rhythm = r1
    gui1.update(seq1.step_index, r1)
    print("Updated Seq1 Rhythm:", r1)

def gen_pitches2():
    global p2
    p2 = rnn_gen1.generate([62], 16, temperature=0.5)[:16]
    p2 = [max(50, min(p, 80)) for p in p2]
    seq2.pitches = p2
    print("Updated Seq2 Pitches:", p2)

def gen_rhythm2():
    global r2
    r2, _ = ae_gen1.generate()
    seq2.rhythm = r2
    gui2.update(seq2.step_index, r2)
    print("Updated Seq2 Rhythm:", r2)

# Hook GUI buttons
gui1.pitch_btn.config(command=gen_pitches1)
gui1.rhythm_btn.config(command=gen_rhythm1)
gui2.pitch_btn.config(command=gen_pitches2)
gui2.rhythm_btn.config(command=gen_rhythm2)

def run_loop():
    seq1.start()
    seq2.start()
    gui1.update(seq1.step_index, seq1.rhythm)
    gui2.update(seq2.step_index, seq2.rhythm)
    root.after(10, run_loop)

run_loop()
root.mainloop()


