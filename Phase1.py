import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.signal as signal

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to generate realistic bird-like songs
def generate_realistic_birdsong(duration=2, fs=16000, noise_level=0.05):
    t = np.linspace(0, duration, int(fs * duration))

    # Chirped fundamental frequency (mimicking bird pitch changes)
    f_start = 800  
    f_end = 2500  
    fundamental = signal.chirp(t, f_start, duration, f_end, method='quadratic')

    # Harmonics
    harmonic1 = 0.6 * signal.chirp(t, f_start * 2, duration, f_end * 2, method='quadratic')
    harmonic2 = 0.3 * signal.chirp(t, f_start * 3, duration, f_end * 3, method='quadratic')

    # Combine harmonics
    birdsong = fundamental + harmonic1 + harmonic2

    # Apply amplitude modulation (AM) to mimic syllable envelope
    modulation = (1 + np.sin(2 * np.pi * 5 * t)) / 2  
    birdsong *= modulation

    # Repetitive motifs
    repeat_interval = int(0.4 * fs)  
    for i in range(0, len(birdsong), repeat_interval):
        birdsong[i:i+repeat_interval] *= np.sin(2 * np.pi * 2 * t[i:i+repeat_interval])  

    # Background noise
    pink_noise = np.cumsum(np.random.normal(0, noise_level, size=t.shape))
    birdsong += pink_noise

    return t, birdsong

# Generate synthetic birdsong
t, birdsong = generate_realistic_birdsong()

# Define Predictive Coding LSTM-Enhanced ESN
class PredictiveCodingLSTMESN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=1.1, sparsity=0.3, leak_rate=0.3, lstm_hidden=256):
        super(PredictiveCodingLSTMESN, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
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

        # LSTM Layer for processing reservoir outputs
        self.lstm = nn.LSTM(input_size=reservoir_size, hidden_size=lstm_hidden, batch_first=True)

        # Output layer (readout)
        self.W_out = nn.Linear(lstm_hidden, output_size).to(device)

    def forward(self, x, state, lstm_hidden, lstm_cell, error_feedback):
        x = x.to(device).unsqueeze(0)
        state = state.to(device)
        error_feedback = error_feedback.to(device)

        # Predictive coding: Adjust state with the error signal
        state = (1 - self.leak_rate) * state + self.leak_rate * torch.tanh(
            self.W_in @ x.T + self.W @ state.T + error_feedback.T
        ).permute(1, 0)

        # Process through LSTM
        state_lstm, (lstm_hidden, lstm_cell) = self.lstm(state.unsqueeze(0), (lstm_hidden, lstm_cell))

        # Predict next frame
        output = self.W_out(state_lstm.squeeze(0))

        return output, state, lstm_hidden, lstm_cell

# Hyperparameters
input_size = 1
reservoir_size = 500  # Increased for better peak tracking
output_size = 1
spectral_radius = 1.1  # Stronger temporal dynamics
sparsity = 0.3
leak_rate = 0.3
lstm_hidden = 256  # More LSTM capacity

# Initialize Model
lstm_esn = PredictiveCodingLSTMESN(input_size, reservoir_size, output_size, spectral_radius, sparsity, leak_rate, lstm_hidden).to(device)

# Convert signals to PyTorch tensors
train_data = torch.tensor(birdsong[:-5], dtype=torch.float32).view(-1, 1).to(device)
train_target = torch.tensor(birdsong[5:], dtype=torch.float32).view(-1, 1).to(device)  # 5-step ahead prediction

# Optimizer & Loss
optimizer = optim.Adam(lstm_esn.W_out.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.85)
criterion = nn.MSELoss()

# Initial States
state = torch.zeros((1, reservoir_size), dtype=torch.float32).to(device)
lstm_hidden_state = torch.zeros((1, 1, lstm_hidden), dtype=torch.float32).to(device)
lstm_cell_state = torch.zeros((1, 1, lstm_hidden), dtype=torch.float32).to(device)
error_feedback = torch.zeros((1, reservoir_size), dtype=torch.float32).to(device)

# Training loop
epochs = 200
batch_size = 200

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = []
    
    for i in range(0, train_data.shape[0], batch_size):
        batch_x = train_data[i:i+batch_size].to(device)
        batch_y = train_target[i:i+batch_size].to(device)

        batch_output = []
        lstm_hidden_state = lstm_hidden_state.detach()
        lstm_cell_state = lstm_cell_state.detach()

        for j in range(batch_x.shape[0]):
            output, state, lstm_hidden_state, lstm_cell_state = lstm_esn(
                batch_x[j], state, lstm_hidden_state, lstm_cell_state, error_feedback
            )

            # Compute predictive coding error
            prediction_error = batch_x[j] - output.detach()
            error_feedback = error_feedback * 0.8 + prediction_error * 0.2  # Stabilized feedback
            
            batch_output.append(output)

        batch_output = torch.stack(batch_output).squeeze()
        loss = criterion(batch_output, batch_y.squeeze())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_esn.parameters(), max_norm=1.0)
        optimizer.step()

        outputs.extend(batch_output.detach().cpu().numpy())

    scheduler.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.5f}")

#  **Final Step: Test & Plot Results**
plt.figure(figsize=(12, 4))
plt.plot(t[:-5], birdsong[:-5], label="Generated Birdsong")
plt.plot(t[:-5], outputs, label="Optimized Predictive Coding ESN", linestyle="dashed")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
