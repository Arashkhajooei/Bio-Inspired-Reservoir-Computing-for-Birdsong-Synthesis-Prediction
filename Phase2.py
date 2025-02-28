import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.signal as signal

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate bird-like song (same as Phase 1)
def generate_realistic_birdsong(duration=2, fs=16000, noise_level=0.05, occlusion=False):
    t = np.linspace(0, duration, int(fs * duration))

    # Chirped fundamental frequency
    f_start, f_end = 800, 2500  
    fundamental = signal.chirp(t, f_start, duration, f_end, method='quadratic')

    # Harmonics
    harmonic1 = 0.6 * signal.chirp(t, f_start * 2, duration, f_end * 2, method='quadratic')
    harmonic2 = 0.3 * signal.chirp(t, f_start * 3, duration, f_end * 3, method='quadratic')

    # Combine harmonics
    birdsong = fundamental + harmonic1 + harmonic2

    # Apply amplitude modulation (AM)
    modulation = (1 + np.sin(2 * np.pi * 5 * t)) / 2  
    birdsong *= modulation

    # Add occlusion (simulate missing parts)
    if occlusion:
        birdsong[int(len(birdsong) * 0.4) : int(len(birdsong) * 0.5)] = 0  # 10% missing

    # Add background noise
    pink_noise = np.cumsum(np.random.normal(0, noise_level, size=t.shape))
    birdsong += pink_noise

    return t, birdsong

# Generate bird song with occlusions
t, birdsong = generate_realistic_birdsong(occlusion=True)

# Define Temporal Memory ESN with Dual Reservoirs
class TemporalMemoryESN(nn.Module):
    def __init__(
        self,
        input_size,
        reservoir_size,
        output_size,
        spectral_radius=1.1,
        leak_rate=0.3,
        sparsity=0.3,
        lstm_hidden=256
    ):
        super(TemporalMemoryESN, self).__init__()
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.lstm_hidden = lstm_hidden

        # Input-to-reservoir weights
        self.W_in = torch.rand(reservoir_size, input_size, device=device) * 2 - 1

        # Sparse reservoir recurrent weights
        W = torch.rand(reservoir_size, reservoir_size, device=device) * 2 - 1
        mask = torch.rand(reservoir_size, reservoir_size, device=device) < sparsity
        W *= mask.float()
        spectral_norm = torch.linalg.norm(W, 2)
        self.W = (W / spectral_norm) * spectral_radius  # Scale reservoir dynamics

        # Second reservoir for predictive coding
        self.W_pred = self.W.clone()

        # LSTM for integrating past/future context
        self.lstm = nn.LSTM(
            input_size=reservoir_size * 2, hidden_size=lstm_hidden, batch_first=True
        )

        # Readout layer (predicts next frames)
        self.W_out = nn.Linear(lstm_hidden, output_size).to(device)

    def forward(self, x, state, pred_state, lstm_hidden, lstm_cell):
        """
        x: single input sample (shape [1]),
        state, pred_state: reservoir states (shape [1, reservoir_size]),
        lstm_hidden, lstm_cell: LSTM hidden states (shape [1, 1, lstm_hidden]).
        Returns: (output, new_state, new_pred_state, new_lstm_hidden, new_lstm_cell)
        """
        x = x.to(device).unsqueeze(0)  # shape [1, 1]
        state = state.to(device)       # shape [1, reservoir_size]
        pred_state = pred_state.to(device)

        # Update both reservoirs
        new_state = (1 - self.leak_rate) * state + self.leak_rate * torch.tanh(
            self.W_in @ x.T + self.W @ state.T
        ).permute(1, 0)

        new_pred_state = (1 - self.leak_rate) * pred_state + self.leak_rate * torch.tanh(
            self.W_in @ x.T + self.W_pred @ pred_state.T
        ).permute(1, 0)

        # Error feedback (though not explicitly used as input this moment)
        error_feedback = new_state - new_pred_state

        # Concatenate both states => shape [1, reservoir_size*2]
        lstm_input = torch.cat([new_state, new_pred_state], dim=-1)

        # LSTM forward
        lstm_out, (new_lstm_hidden, new_lstm_cell) = self.lstm(
            lstm_input.unsqueeze(0), (lstm_hidden, lstm_cell)
        )

        # Predict next
        output = self.W_out(lstm_out.squeeze(0))  # shape [1, output_size]
        return output, new_state, new_pred_state, new_lstm_hidden, new_lstm_cell

# Initialize Model
lstm_esn = TemporalMemoryESN(1, 500, 1).to(device)

# Convert signals to PyTorch tensors
train_data = torch.tensor(birdsong[:-1], dtype=torch.float32).view(-1, 1).to(device)
train_target = torch.tensor(birdsong[1:], dtype=torch.float32).view(-1, 1).to(device)

# Optimizer & Loss
optimizer = optim.Adam(lstm_esn.W_out.parameters(), lr=0.002, weight_decay=1e-3)
criterion = nn.MSELoss()

# Training loop with truncated BPTT
epochs = 150
batch_size = 200

for epoch in range(epochs):
    # Re-initialize or keep states from previous epoch
    state = torch.zeros((1, 500), dtype=torch.float32).to(device)
    pred_state = torch.zeros((1, 500), dtype=torch.float32).to(device)
    lstm_hidden_state = torch.zeros((1, 1, 256), dtype=torch.float32).to(device)
    lstm_cell_state = torch.zeros((1, 1, 256), dtype=torch.float32).to(device)

    epoch_loss = 0.0
    count = 0

    for i in range(0, train_data.shape[0], batch_size):
        batch_x = train_data[i : i + batch_size]
        batch_y = train_target[i : i + batch_size]

        # We'll collect outputs for the batch
        outputs = []
        for j in range(batch_x.shape[0]):
            output, state, pred_state, lstm_hidden_state, lstm_cell_state = lstm_esn(
                batch_x[j], state, pred_state, lstm_hidden_state, lstm_cell_state
            )
            outputs.append(output)

        outputs = torch.stack(outputs).squeeze()  # shape [batch_size]
        loss = criterion(outputs, batch_y.squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # **Detach states** to prevent reusing the graph
        state = state.detach()
        pred_state = pred_state.detach()
        lstm_hidden_state = lstm_hidden_state.detach()
        lstm_cell_state = lstm_cell_state.detach()

        epoch_loss += loss.item()
        count += 1

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {epoch_loss / count:.5f}")

# Generate Predictions
# We can keep the final states from training, or re-init them
state = torch.zeros((1, 500), dtype=torch.float32).to(device)
pred_state = torch.zeros((1, 500), dtype=torch.float32).to(device)
lstm_hidden_state = torch.zeros((1, 1, 256), dtype=torch.float32).to(device)
lstm_cell_state = torch.zeros((1, 1, 256), dtype=torch.float32).to(device)

predictions = []
for i in range(len(train_data)):
    output, state, pred_state, lstm_hidden_state, lstm_cell_state = lstm_esn(
        train_data[i], state, pred_state, lstm_hidden_state, lstm_cell_state
    )
    predictions.append(output.item())

# Plot Results
plt.figure(figsize=(12, 4))
plt.plot(t[:-1], birdsong[:-1], label="Generated Birdsong")
plt.plot(t[:-1], predictions, label="Temporal Memory ESN Prediction", linestyle="dashed")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Temporal Memory ESN: Occlusion Handling & Next-Step Prediction")
plt.show()
