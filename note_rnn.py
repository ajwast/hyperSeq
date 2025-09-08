import torch
import torch.nn as nn
import torch.optim as optim

class NoteRNN(nn.Module):
    def __init__(self, input_size=127, hidden_size=20, output_size=127):
        super(NoteRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out)

    def generate(self, start_sequence, length, temperature=1.0, device="cpu"):
        self.eval()
        seq_len = len(start_sequence)
        input_seq = torch.zeros(1, seq_len, 127).to(device)
        for t, note in enumerate(start_sequence):
            input_seq[0, t, note] = 1
        generated = start_sequence[:]

        with torch.no_grad():
            for _ in range(length):
                output = self(input_seq)
                logits = output[0, -1]
                probs = torch.softmax(logits / temperature, dim=0)
                next_note = torch.multinomial(probs, 1).item()
                generated.append(next_note)
                new_input = torch.zeros(1, 1, 127).to(device)
                new_input[0, 0, next_note] = 1
                input_seq = torch.cat((input_seq[:, 1:, :], new_input), dim=1)
        return generated

    def save(self, path="note_rnn.pth"):
        torch.save(self.state_dict(), path)

    def load(self, path="note_rnn.pth"):
        self.load_state_dict(torch.load(path))



class PitchVelRNN(nn.Module):
    def __init__(self, num_degrees=7, num_octaves=5, num_velocities=32, hidden_size=256):
        super().__init__()
        
        # Embeddings
        self.degree_emb = nn.Embedding(num_degrees, 32)
        self.octave_emb = nn.Embedding(num_octaves, 16)
        self.velocity_emb = nn.Embedding(num_velocities, 8)
        
        # LSTM core
        self.lstm = nn.LSTM(
            input_size=32 + 16 + 8 + num_degrees,  # embeddings + histogram
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output heads
        self.fc_pitch = nn.Linear(hidden_size, num_degrees * num_octaves)
        self.fc_vel = nn.Linear(hidden_size, num_velocities)
        
        self.num_degrees = num_degrees
        self.num_octaves = num_octaves
        self.num_velocities = num_velocities

    def forward(self, degrees, octaves, velocities, histograms, hidden=None):
        """
        Args:
            degrees: (batch, seq_len) tensor of degree indices
            octaves: (batch, seq_len) tensor of octave indices
            velocities: (batch, seq_len) tensor of velocity indices
            histograms: (batch, seq_len, num_degrees) conditioning vector
            hidden: (h, c) tuple for LSTM
        Returns:
            pitch_logits: (batch, seq_len, num_degrees * num_octaves)
            vel_logits: (batch, seq_len, num_velocities)
            hidden: updated hidden state
        """
        deg_emb = self.degree_emb(degrees)       # (B, T, 32)
        oct_emb = self.octave_emb(octaves)       # (B, T, 16)
        vel_emb = self.velocity_emb(velocities)  # (B, T, 8)
        
        x = torch.cat([deg_emb, oct_emb, vel_emb, histograms], dim=-1)
        
        out, hidden = self.lstm(x, hidden)  # (B, T, H)
        
        pitch_logits = self.fc_pitch(out)   # (B, T, D*O)
        vel_logits = self.fc_vel(out)       # (B, T, V)
        
        return pitch_logits, vel_logits, hidden

    def sample(self, start_degree, start_octave, start_velocity, histogram, 
               length=32, temperature=1.0):
        """
        Autoregressive sampling of pitch + velocity sequence.
        """
        self.eval()
        
        degree = torch.tensor([[start_degree]], dtype=torch.long)
        octave = torch.tensor([[start_octave]], dtype=torch.long)
        velocity = torch.tensor([[start_velocity]], dtype=torch.long)
        hist = torch.tensor(histogram, dtype=torch.float).unsqueeze(0).unsqueeze(1)
        
        hidden = None
        seq = []
        
        for _ in range(length):
            pitch_logits, vel_logits, hidden = self.forward(degree, octave, velocity, hist, hidden)
            
            # --- Pitch sample ---
            pitch_probs = F.softmax(pitch_logits[:, -1, :] / temperature, dim=-1)
            pitch_idx = torch.multinomial(pitch_probs, num_samples=1).item()
            next_degree = pitch_idx % self.num_degrees
            next_octave = pitch_idx // self.num_degrees
            
            # --- Velocity sample ---
            vel_probs = F.softmax(vel_logits[:, -1, :] / temperature, dim=-1)
            next_velocity = torch.multinomial(vel_probs, num_samples=1).item()
            
            seq.append((next_degree, next_octave, next_velocity))
            
            # Prepare next step
            degree = torch.tensor([[next_degree]], dtype=torch.long)
            octave = torch.tensor([[next_octave]], dtype=torch.long)
            velocity = torch.tensor([[next_velocity]], dtype=torch.long)
        
        return seq
