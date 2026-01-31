import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RhythmAutoencoder(nn.Module):
    def __init__(self):
        super(RhythmAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def decode(self, latent, use_dropout=True):
        """
        Decode from latent with optional dropout.
        Dropout stays active if use_dropout=True.
        """
        # Enable dropout during inference if requested
        if use_dropout:
            self.train()
        else:
            self.eval()

        with torch.no_grad():
            return self.decoder(latent)

    def generate(self):
        """
        Sample from latent space, decode with dropout active,
        and treat output as Bernoulli probabilities.
        """
        latent = torch.rand((1, 4))
        decoded = self.decode(latent, use_dropout=True).squeeze()

        # Sample from Bernoulli distribution
        steps, substeps = torch.split(decoded, 16)
        step_out = torch.bernoulli(steps).int().tolist()
        trans_out = torch.bernoulli(substeps).int().tolist()
        return step_out, trans_out

    def save(self, path="rhythm_ae.pth"):
        torch.save(self.state_dict(), path)

    def load(self, path="rhythm_ae.pth"):
        self.load_state_dict(torch.load(path))
