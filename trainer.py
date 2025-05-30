import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    @staticmethod
    def train_rnn(model, X, y, epochs=50, lr=0.001):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        losses = []

        for epoch in range(epochs):
            output = model(X).transpose(1, 2)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if epoch % 10 == 0:
                print(f"[RNN] Epoch {epoch}, Loss: {loss.item():.4f}")

    @staticmethod
    def train_autoencoder(model, dataloader, epochs=50, lr=0.001):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            for batch, _ in dataloader:
                out = model(batch)
                loss = criterion(out, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
            if epoch % 5 == 0:
                print(f"[AE] Epoch {epoch}, Loss: {avg_loss:.4f}")
