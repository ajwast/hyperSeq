from note_rnn import NoteRNN
from rhythm_autoencoder import RhythmAutoencoder
from trainer import Trainer
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import csv

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

rnn = NoteRNN()
ae = RhythmAutoencoder()

Trainer.train_rnn(rnn, X, y, epochs=60)
Trainer.train_autoencoder(ae, ae_loader, epochs=50)

rnn.save("note_rnn.pth")
ae.save("rhythm_ae.pth")

# === Generate example sequence ===
pitches = rnn.generate([62], 16, temperature=0.5)[:16]
pitches = [max(50, min(p, 80)) for p in pitches]
rhythm, _ = ae.generate()

print("Generated Pitches:", pitches)
print("Generated Rhythm:", rhythm)
