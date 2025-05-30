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
