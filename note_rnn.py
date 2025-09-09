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



class PitchTransitionRNN(nn.Module):
    def __init__(self, num_degrees, num_octaves, num_velocities,
                 embedding_dim=32, hidden_dim=128, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Embeddings for discrete inputs
        self.degree_emb = nn.Embedding(num_degrees, embedding_dim)
        self.octave_emb = nn.Embedding(num_octaves, embedding_dim)
        self.velocity_emb = nn.Embedding(num_velocities, embedding_dim)

        # LSTM input size = embeddings + histogram conditioning
        input_dim = embedding_dim * 3 + num_degrees  

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Output heads
        self.degree_out = nn.Linear(hidden_dim, num_degrees)
        self.octave_out = nn.Linear(hidden_dim, num_octaves)
        self.velocity_out = nn.Linear(hidden_dim, num_velocities)

    def forward(self, degrees, octaves, velocities, histograms, hidden=None):
        """
        Args:
            degrees, octaves, velocities: (B, T) long
            histograms: (B, T, num_degrees) float
        """
        d_emb = self.degree_emb(degrees)      # (B, T, E)
        o_emb = self.octave_emb(octaves)      # (B, T, E)
        v_emb = self.velocity_emb(velocities) # (B, T, E)

        x = torch.cat([d_emb, o_emb, v_emb, histograms], dim=-1)  # (B, T, D)

        out, hidden = self.lstm(x, hidden)  # (B, T, H)

        return {
            "degrees": self.degree_out(out),      # (B, T, num_degrees)
            "octaves": self.octave_out(out),      # (B, T, num_octaves)
            "velocities": self.velocity_out(out)  # (B, T, num_velocities)
        }