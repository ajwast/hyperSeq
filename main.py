from note_rnn import NoteRNN
from rhythm_autoencoder import RhythmAutoencoder
from trainer import Trainer
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import csv
from sequencer import Sequencer
import sys
import select
import rtmidi


print("Initialising...")

# === Data Loading Functions ===
def load_training_data(x_filename='X_data.csv', y_filename='y_data.csv', seq_len=16):
    X, y = [], []
    with open(x_filename, mode='r') as f_x:
        for row in csv.reader(f_x):
            seq = torch.tensor(list(map(float, row))).reshape(seq_len, 127)
            X.append(seq)
    with open(y_filename, mode='r') as f_y:
        for row in csv.reader(f_y):
            y.append(torch.tensor(list(map(int, row))))
    return torch.stack(X), torch.stack(y)

def load_ae_data(dataset_file='dataset.csv'):
    d_data = torch.tensor(np.loadtxt(dataset_file, delimiter=','), dtype=torch.float32)
    return DataLoader(TensorDataset(d_data, d_data), batch_size=16, shuffle=True)

# # === Training / Loading ===
# X, y = load_training_data()
# ae_loader = load_ae_data()

# rnn_gen1 = NoteRNN()
# ae_gen1 = RhythmAutoencoder()
# Trainer.train_rnn(rnn_gen1, X, y, epochs=60)
# Trainer.train_autoencoder(ae_gen1, ae_loader, epochs=50)


# # Generate initial sequences
# p1 = rnn_gen1.generate([62], 16, temperature=0.5)[:16]
# p1 = [max(50, min(p, 80)) for p in p1]

# p2 = rnn_gen1.generate(start_sequence=[65], length=16)
# p2 = [max(50, min(p, 80)) for p in p2]
# r1, _ = ae_gen1.generate()
# r2, _ = ae_gen1.generate()
# print (p1, r1, p2, r2)

midiIn = rtmidi.MidiIn()
inPorts = midiIn.get_ports()
midiOut = rtmidi.MidiOut()
outPorts = midiOut.get_ports()

print("Available MIDI In ports:")
print(inPorts)
midiClock = input("Choose MIDI clock input")

print("Available MIDI Out ports:")
print(outPorts)

outputPort1 = input("Choose MIDI out port 1")
outputPort2 = input("Choose MIDI out port 2")

# Launch sequencer
seq1 = Sequencer(
    pitches=p1,
    rhythm=r1,
    channel=1,
    duration=5,
    seq_len = 4,
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

while True:
	seq1.start()
	seq2.start()
 #  if select.select([sys.stdin], [], [], 0)[0]:
     #   user_input = sys.stdin.readline().strip()
	#	if user_input == '1':
	#		p1 = rnn_gen1.generate([62], 16, temperature=0.5)[:16]
	#		p1 = [max(50, min(p, 80)) for p in p1]
	#		seq.pitches1 = p1
	#	if user_input == '2':
	#		r1, _ = ae_gen1.generate()
	#		seq.rhythm1 = r1
	#	if user_input == '3':
	#		p2 = rnn_gen2.generate([62], 16, temperature=0.5)[:16]
	#		p2 = [max(50, min(p, 80)) for p in p2]
	#		seq.pitches2 = p2
	#	if user_input == '4':
	#		r2, _ = ae_gen2.generate()
	#		seq.rhythm2 = r2

