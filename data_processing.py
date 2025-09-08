import torch
import aubio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

os.chdir('../../../ML Corpus 2/')
cd = os.getcwd()      


filenames = []

for (root,dirs,files) in os.walk(cd, topdown=True): 
    for name in files:
          if name.endswith('.'+'wav'):
            filenames.append(os.path.join(root, name))

print("Found", len(filenames), "files.")


class AudioRNNDataset(Dataset):
    def __init__(self, filenames, tonic=60, num_degrees=7, num_octaves=5, 
                 num_velocities=32, seq_len=16, downsample=1):
        """
        Dataset for extracting note sequences from audio and preparing 
        conditioning features for an RNN.

        Args:
            filenames (list): List of audio file paths (.wav).
            tonic (int): MIDI note number of the tonic (default: 60 = C4).
            num_degrees (int): Number of scale degrees in conditioning histogram.
            num_octaves (int): Number of octave bins.
            num_velocities (int): Number of velocity bins.
            seq_len (int): Sequence length for training.
            downsample (int): Downsample factor for aubio analysis.
        """
        self.seq_len = seq_len
        self.num_degrees = num_degrees
        self.num_octaves = num_octaves
        self.num_velocities = num_velocities
        self.tonic = tonic

        # Extract notes from all files in one pass
        self.notes = self._extract_notes(filenames, downsample)
        
        # Convert notes into training sequences
        self.X, self.Y = self._prepare_sequences()

    def _extract_notes(self, filenames, downsample):
        samplerate = 44100 // downsample
        win_s = 512 // downsample
        hop_s = 256 // downsample
        notes = []

        for file in filenames:
            s = aubio.source(file, samplerate, hop_s)
            samplerate = s.samplerate
            notes_o = aubio.notes("default", win_s, hop_s, samplerate)

            while True:
                samples, read = s()
                new_note = notes_o(samples)
                if new_note[0] != 0:  # only valid notes
                    notes.append((int(new_note[0]), int(new_note[1])))  # (midi_pitch, velocity)
                if read < hop_s:
                    break
        print(f"Extracted {len(notes)} notes from {len(filenames)} files.")
        return notes

    def _prepare_sequences(self):
        notes = self.notes
        if len(notes) <= self.seq_len:
            raise ValueError(f"Not enough notes in dataset ({len(notes)}) for seq_len={self.seq_len}.")

        # Split into sequences
        X, Y = [], []
        for i in range(len(notes) - self.seq_len):
            input_seq = notes[i:i+self.seq_len]
            target_seq = notes[i+1:i+self.seq_len+1]

            # Decompose into features
            degrees, octaves, velocities = [], [], []
            for pitch, vel in input_seq:
                degree = (pitch - self.tonic) % self.num_degrees
                octave = (pitch - self.tonic) // self.num_degrees
                velocity_bin = min(vel * self.num_velocities // 128, self.num_velocities - 1)

                degrees.append(degree)
                octaves.append(octave)
                velocities.append(velocity_bin)

            # Pitch histogram (scale degrees only)
            hist = np.bincount(degrees, minlength=self.num_degrees).astype(np.float32)
            hist = hist / (hist.sum() + 1e-6)  # normalize

            # Store input & target
            X.append({
                "degrees": torch.tensor(degrees, dtype=torch.long),
                "octaves": torch.tensor(octaves, dtype=torch.long),
                "velocities": torch.tensor(velocities, dtype=torch.long),
                "histograms": torch.tensor(hist, dtype=torch.float32),
            })

            # Targets = next-step values
            tgt_degrees, tgt_octaves, tgt_velocities = [], [], []
            for pitch, vel in target_seq:
                tgt_degrees.append((pitch - self.tonic) % self.num_degrees)
                tgt_octaves.append((pitch - self.tonic) // self.num_degrees)
                tgt_velocities.append(min(vel * self.num_velocities // 128, self.num_velocities - 1))

            Y.append({
                "degrees": torch.tensor(tgt_degrees, dtype=torch.long),
                "octaves": torch.tensor(tgt_octaves, dtype=torch.long),
                "velocities": torch.tensor(tgt_velocities, dtype=torch.long),
            })

        return X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    


dataset = AudioRNNDataset(filenames, seq_len=16)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_X, batch_Y in loader:
    print(batch_X["degrees"].shape)    # (B, seq_len)
    print(batch_X["histograms"].shape) # (B, num_degrees)
    print(batch_Y["degrees"].shape)    # (B, seq_len)
    break
