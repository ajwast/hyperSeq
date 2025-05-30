from note_rnn import NoteRNN
from rhythm_autoencoder import RhythmAutoencoder
from trainer import Trainer
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import csv
from sequencer import Sequencer

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

# === Training / Loading ===
X, y = load_training_data()
ae_loader = load_ae_data()

rnn_gen1 = NoteRNN()
rnn_gen2 = NoteRNN()
ae_gen1 = RhythmAutoencoder()
ae_gen2 = RhythmAutoencoder()
Trainer.train_rnn(rnn_gen1, X, y, epochs=60)
Trainer.train_autoencoder(ae_gen1, ae_loader, epochs=50)

Trainer.train_rnn(rnn_gen2, X, y, epochs=60)
Trainer.train_autoencoder(ae_gen2, ae_loader, epochs=50)


#rnn.save("note_rnn.pth")
#ae.save("rhythm_ae.pth")

# === Generate example sequence ===
#pitches = rnn.generate([62], 16, temperature=0.5)[:16]
#pitches = [max(50, min(p, 80)) for p in pitches]
#rhythm, _ = ae.generate()

#print("Generated Pitches:", pitches)
#print("Generated Rhythm:", rhythm)


# Generate initial sequences
p1 = rnn_gen1.generate([62], 16, temperature=0.5)[:16]
p1 = [max(50, min(p, 80)) for p in p1]

p2 = rnn_gen2.generate(start_sequence=[65], length=16)
p2 = [max(50, min(p, 80)) for p in p2]
r1 = ae_gen1.generate()
r2 = ae_gen2.generate()

# Launch sequencer
seq = Sequencer(p1, r1, channel1=1, 
                pitches2=p2, rhythm2=r2, channel2=2, 
                duration=5, clock_in=1, port_out=3)

seq.start()
