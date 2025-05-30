import torch
import torch.nn as nn
import torch.optim as optim

class RhythmAutoencoder(nn.Module):
    def __init__(self):
        super(RhythmAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def decode(self, latent):
        return self.decoder(latent)

    def generate(self, tolerance=0.5, transit=0.35):
        latent = torch.rand((1, 4))
        decoded = self.decode(latent).squeeze()
        steps, substeps = torch.split(decoded, 16)
        step_out = [int(v > tolerance) for v in steps]
        trans_out = [int(v > transit) for v in substeps]
        return step_out, trans_out

    def save(self, path="rhythm_ae.pth"):
        torch.save(self.state_dict(), path)

    def load(self, path="rhythm_ae.pth"):
        self.load_state_dict(torch.load(path))
