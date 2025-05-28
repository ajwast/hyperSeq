
print("Initialising...")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import csv


            
def load_training_data(x_filename='X_data.csv', y_filename='y_data.csv', seq_len=16):
    """
    Load input and target tensors from CSV files.

    Args:
        x_filename (str): Filename for input CSV.
        y_filename (str): Filename for target CSV.
        seq_len (int): Length of each sequence.

    Returns:
        torch.Tensor: Input tensor of shape (num_sequences, seq_len, 127).
        torch.Tensor: Target tensor of shape (num_sequences, seq_len).
    """
    X, y = [], []

    # Load X
    with open(x_filename, mode='r') as f_x:
        reader = csv.reader(f_x)
        for row in reader:
            flat_seq = list(map(float, row))
            seq_tensor = torch.tensor(flat_seq).reshape(seq_len, 127)
            X.append(seq_tensor)

    # Load y
    with open(y_filename, mode='r') as f_y:
        reader = csv.reader(f_y)
        for row in reader:
            seq_tensor = torch.tensor(list(map(int, row)))
            y.append(seq_tensor)

    X = torch.stack(X)
    y = torch.stack(y)
    return X, y





class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out)  # Output shape: (batch_size, seq_len, output_size)
        return out

input_size = 127
hidden_size = 20
output_size = 127
rnn = SimpleRNN(input_size, hidden_size, output_size)
rnn2 = SimpleRNN(input_size, hidden_size, output_size)

def generate_note_sequence(model, start_sequence, length, temperature=1.0, device="cpu"):
    """
    Generates a new MIDI note sequence using a trained RNN with stochastic sampling.

    Args:
        model (nn.Module): Trained RNN model.
        start_sequence (list): List of initial MIDI notes to start generation.
        length (int): Total length of the sequence to generate.
        temperature (float): Temperature for softmax scaling (controls randomness).
        device (str): Device to run the model on ("cpu" or "cuda").

    Returns:
        list: Generated MIDI note sequence.
    """
    model.eval()  # Set the model to evaluation mode

    # One-hot encode the start sequence
    seq_len = len(start_sequence)
    input_seq = torch.zeros(1, seq_len, 127).to(device)  # Shape: (batch_size=1, seq_len, input_size)
    for t, note in enumerate(start_sequence):
        input_seq[0, t, note] = 1

    # Initialize the generated sequence with the start sequence
    generated_sequence = start_sequence[:]

    with torch.no_grad():
        for _ in range(length):
            # Pass the current input sequence to the model
            output = model(input_seq)  # Shape: (1, seq_len, 127)

            # Get the last time step's output (logits for next note)
            logits = output[0, -1]  # Shape: (127,)

            # Apply temperature scaling to logits
            scaled_logits = logits / temperature

            # Convert logits to probabilities using softmax
            probabilities = torch.softmax(scaled_logits, dim=0)

            # Sample the next note using a multinomial distribution
            next_note = torch.multinomial(probabilities, num_samples=1).item()

            # Append the predicted note to the generated sequence
            generated_sequence.append(next_note)

            # Update the input sequence with the new note
            new_one_hot = torch.zeros(1, 1, 127).to(device)
            new_one_hot[0, 0, next_note] = 1
            input_seq = torch.cat((input_seq[:, 1:, :], new_one_hot), dim=1)  # Slide window

    return generated_sequence



def train_rnn(model, epochs):
# # Loss and optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = epochs
    losses = []
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        outputs = model(X)  # Shape: (batch_size, seq_len, output_size)
        outputs = outputs.transpose(1, 2)  # Shape: (batch_size, output_size, seq_len) for CrossEntropyLoss
        loss = criterion(outputs, y)  # Targets (y) are shape (batch_size, seq_len)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


X, y = load_training_data()

dataset = np.loadtxt('dataset.csv', delimiter=',')

d_data = torch.tensor(dataset, dtype=torch.float32)
t_dataset = TensorDataset(d_data, d_data)  # Input and target are the same
dataloader = DataLoader(t_dataset, batch_size=16, shuffle=True)



# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(8, 32),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def decode(self, latent):
        return self.decoder(latent)

#losses = []
# Train the autoencoder
def train_autoencoder(model, dataloader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch, _ in dataloader:
            # Forward pass
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(loss.item())
        if epoch % 1 == 0:
            
            print(f"Epoch [{epoch}/{num_epochs}] Loss: {total_loss / len(dataloader)}")


AE = Autoencoder()
AE2 = Autoencoder()



def Generate_rhythm(model, tolerance=0.5, transit = 0.35):
    latent_vector = torch.rand((1, 4))  # Random latent vector
    generated_rhythm = model.decode(latent_vector)
    
    split = torch.split(generated_rhythm[0],16)
    
    steps = split[0]
    
    substeps = split[1]
    
    output1 = []
    output2 = []
    
    for op in steps:
        if op > tolerance:
            output1.append(1)
        else: output1.append(0)
            
    for s in substeps:
        if s > transit:
            output2.append(1)
        else: output2.append(0)
        
    return output1, output2

print("training RNN")
train_rnn(rnn, 60)
train_rnn(rnn2, 60)
print("RNN ready")

print("training AE")
train_autoencoder(AE,dataloader,50)
train_autoencoder(AE2,dataloader,50)
print("AE ready")



from sequencer import Sequencer

# # Prepare the generated data
p1 = generate_note_sequence(rnn, [62], 16, 0.5)[:16]
p1 = [max(48, min(p, 90)) for p in p1]
r1, _ = Generate_rhythm(AE, 0.5)
r1 = r1[:16]
print(r1, p1)

p2 = generate_note_sequence(rnn2, [62], 16, 0.5)[:16]
p2 = [max(48, min(p, 90)) for p in p2]
r2, _ = Generate_rhythm(AE2, 0.5)
r2 = r2[:16]
print(r2, p2)

# Start the sequencer
seq = Sequencer(p1, r1, channel1=1, 
                pitches2=p2, rhythm2=r2, channel2=2, 
                duration=5, clock_in=1, port_out=1)

seq.start()
