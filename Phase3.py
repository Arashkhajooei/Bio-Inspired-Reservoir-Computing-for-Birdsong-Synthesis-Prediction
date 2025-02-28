import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.signal as signal

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Data Generation ----------------
def generate_birdsong_with_colony_noise(duration=2, fs=16000, noise_level=0.05):
    """
    Generate a simple bird-like chirp plus colony noise.
    """
    t = np.linspace(0, duration, int(fs * duration))
    
    # Fundamental frequency sweep
    f_start, f_end = 800, 2500  
    fundamental = signal.chirp(t, f_start, duration, f_end, method='quadratic')

    # Harmonics
    harmonic1 = 0.6 * signal.chirp(t, f_start*2, duration, f_end*2, method='quadratic')
    harmonic2 = 0.3 * signal.chirp(t, f_start*3, duration, f_end*3, method='quadratic')

    # Combine
    birdsong = fundamental + harmonic1 + harmonic2

    # Amplitude modulation
    modulation = (1 + np.sin(2 * np.pi * 5 * t)) / 2  
    birdsong *= modulation

    # Colony noise (sim multiple birds)
    colony_noise = 0.3 * np.random.normal(0, noise_level, size=t.shape)
    birdsong += colony_noise

    return t, birdsong


# ---------------- Utility: Spectral Radius Scaling ----------------
def scale_to_spectral_radius(W, desired_radius=0.9):
    """
    Approximate the largest eigenvalue by the largest singular value,
    then scale W so that its spectral radius ~ desired_radius.
    """
    with torch.no_grad():
        W_cpu = W.to("cpu")
        U, S, V = torch.svd(W_cpu)
        max_singular = S[0].item()
        scale_factor = desired_radius / max_singular
    return W * scale_factor


# ---------------- Model: BioInspiredESN ----------------
class BioInspiredESN(nn.Module):
    """
    Simplified 'Bio-Inspired ESN' with:
    - Spatial reservoir zones (core, shell, feedback).
    - *Reduced* Harmonic oscillator dynamics.
    - *No random neuromodulation noise* (only dopamine scaling).
    - LSTM readout.
    """
    def __init__(
        self,
        input_size,
        reservoir_size,
        output_size,
        spectral_radius=0.9,  # smaller to avoid blow-up
        leak_rate=0.1,        # more conservative
        sparsity=0.3,
        lstm_hidden=256
    ):
        super(BioInspiredESN, self).__init__()
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.lstm_hidden = lstm_hidden

        # ---------------- Split reservoir into 3 zones ----------------
        self.core_size = int(reservoir_size * 0.6)   # 60%
        self.shell_size = int(reservoir_size * 0.3)  # 30%
        self.feedback_size = reservoir_size - (self.core_size + self.shell_size)  # 10%

        # ---------------- Input->Reservoir ----------------
        self.W_in = nn.Parameter(torch.rand(reservoir_size, input_size, device=device)*2 - 1)

        # ---------------- Internal submatrices ----------------
        W_core = torch.rand(self.core_size, self.core_size, device=device)*2 - 1
        W_shell = torch.rand(self.shell_size, self.shell_size, device=device)*2 - 1
        W_feedback = torch.rand(self.feedback_size, self.feedback_size, device=device)*2 - 1

        # Sparse connections in the shell
        mask_shell = (torch.rand(self.shell_size, self.shell_size, device=device) < sparsity).float()
        W_shell = W_shell * mask_shell

        # Cross-zone
        W_shell_core = torch.rand(self.core_size, self.shell_size, device=device)*2 - 1
        W_feedback_core = torch.rand(self.core_size, self.feedback_size, device=device)*2 - 1

        # ---------------- Enforce smaller spectral radius ----------------
        W_core = scale_to_spectral_radius(W_core, spectral_radius)
        W_shell = scale_to_spectral_radius(W_shell, spectral_radius)
        W_feedback = scale_to_spectral_radius(W_feedback, spectral_radius)

        # ---------------- Register as learnable parameters ----------------
        self.W_core = nn.Parameter(W_core)
        self.W_shell = nn.Parameter(W_shell)
        self.W_feedback = nn.Parameter(W_feedback)
        self.W_shell_core = nn.Parameter(W_shell_core)
        self.W_feedback_core = nn.Parameter(W_feedback_core)

        # ---------------- Neuromodulation scalars (only dopamine used) ----------------
        self.dopamine = nn.Parameter(torch.tensor(0.5, device=device))
        # We'll skip random noise from acetylcholine for now:
        # self.acetylcholine = nn.Parameter(torch.tensor(0.5, device=device))

        # ---------------- Harmonic oscillator parameters (reduced effect) ----------------
        self.omega = nn.Parameter(torch.ones(reservoir_size, device=device)*2.0)
        self.gamma = nn.Parameter(torch.ones(reservoir_size, device=device)*0.1)

        # ---------------- LSTM readout ----------------
        self.lstm = nn.LSTM(input_size=reservoir_size, hidden_size=lstm_hidden, batch_first=True)
        self.W_out = nn.Linear(lstm_hidden, output_size).to(device)


    def forward(self, x, state, lstm_hidden, lstm_cell):
        """
        x: shape (1,) => single time-step input
        state: shape (1, reservoir_size)
        lstm_hidden, lstm_cell: (1, 1, lstm_hidden)

        Returns:
          output       (1, output_size)
          new_state    (1, reservoir_size)
          new_hidden, new_cell for LSTM
        """
        # 1) Convert input x -> reservoir_in
        reservoir_in = self.W_in @ x  # shape = (reservoir_size,)

        # 2) Split previous reservoir state into zones
        core_part     = state[:, :self.core_size]
        shell_part    = state[:, self.core_size : self.core_size + self.shell_size]
        feedback_part = state[:, -self.feedback_size:]

        # 3) Predict next state for each zone
        core_pred = torch.tanh(
            (self.W_core @ core_part.T).T + reservoir_in[:self.core_size].unsqueeze(0)
        )
        shell_pred = torch.tanh(self.W_shell @ shell_part.T).T
        feedback_pred = torch.tanh(self.W_feedback @ feedback_part.T).T

        # 4) Cross-zone interactions (add them to the core zone)
        core_pred = core_pred + torch.tanh(self.W_shell_core @ shell_pred.T).T
        core_pred = core_pred + torch.tanh(self.W_feedback_core @ feedback_pred.T).T

        # 5) Leak-rate combination
        core_new = (1 - self.leak_rate)*core_part + self.leak_rate*core_pred
        shell_new = (1 - self.leak_rate)*shell_part + self.leak_rate*shell_pred
        feedback_new = (1 - self.leak_rate)*feedback_part + self.leak_rate*feedback_pred

        # 6) Recombine all zones
        combined_state = torch.cat([core_new, shell_new, feedback_new], dim=1)

        # 7) Reduced harmonic oscillator update (0.1 scale factor)
        ho_update = self.omega * combined_state - self.gamma * torch.tanh(combined_state)
        combined_state = combined_state + 0.1 * ho_update

        # 8) Neuromodulation (only dopamine, no random noise)
        combined_state = self.dopamine * combined_state

        # 9) LSTM readout
        lstm_in = combined_state.unsqueeze(0)  # shape = (1, 1, reservoir_size)
        lstm_out, (new_hidden, new_cell) = self.lstm(lstm_in, (lstm_hidden, lstm_cell))
        output = self.W_out(lstm_out.squeeze(0))  # shape (1, output_size)

        return output, combined_state, new_hidden, new_cell


# ---------------- Demo: Train + Predict ----------------
if __name__ == "__main__":
    # Generate data (2s at 16kHz = 32000 samples)
    t, birdsong = generate_birdsong_with_colony_noise(duration=2, fs=16000)

    # Instantiate model
    reservoir_size = 500
    model = BioInspiredESN(
        input_size=1,
        reservoir_size=reservoir_size,
        output_size=1,
        spectral_radius=0.9,  # smaller
        leak_rate=0.1,        # more conservative
        sparsity=0.3,
        lstm_hidden=256
    ).to(device)

    # Build dataset (shift birdsong by 1 sample for next-step prediction)
    train_data = torch.tensor(birdsong[:-1], dtype=torch.float32).view(-1, 1).to(device)
    train_target = torch.tensor(birdsong[1:],  dtype=torch.float32).view(-1, 1).to(device)

    # Optimizer: no weight decay, smaller LR
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0)
    criterion = nn.MSELoss()

    # Initial states
    state = torch.zeros((1, reservoir_size), dtype=torch.float32, device=device)
    lstm_hidden = torch.zeros((1, 1, 256), dtype=torch.float32, device=device)
    lstm_cell   = torch.zeros((1, 1, 256), dtype=torch.float32, device=device)

    epochs = 50
    seq_len = len(train_data)  # ~32000 for 2 seconds at 16kHz

    for epoch in range(epochs):
        # ---------------- Optional Warm-Up (commented out) ----------------
        #
        # with torch.no_grad():
        #     for i in range(1000):  # or some short warm-up length
        #         _, state, lstm_hidden, lstm_cell = model(train_data[i], state, lstm_hidden, lstm_cell)

        state = state.detach()
        lstm_hidden = lstm_hidden.detach()
        lstm_cell   = lstm_cell.detach()

        optimizer.zero_grad()

        outputs = []
        for i in range(seq_len):
            x_in = train_data[i]
            out, state, lstm_hidden, lstm_cell = model(x_in, state, lstm_hidden, lstm_cell)
            outputs.append(out)

        outputs = torch.stack(outputs).squeeze()  # shape (time, 1)
        loss = criterion(outputs, train_target.squeeze())

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.5f}")

    # ---------------- Inference ----------------
    with torch.no_grad():
        state = torch.zeros((1, reservoir_size), device=device)
        lstm_hidden = torch.zeros((1, 1, 256), device=device)
        lstm_cell = torch.zeros((1, 1, 256), device=device)

        predictions = []
        for i in range(seq_len):
            out, state, lstm_hidden, lstm_cell = model(train_data[i], state, lstm_hidden, lstm_cell)
            predictions.append(out.item())

    # ---------------- Plot ----------------
    plt.figure(figsize=(12, 4))
    plt.plot(t[:-1], birdsong[:-1], label="Generated Birdsong")
    plt.plot(t[:-1], predictions, label="Bio-Inspired ESN Prediction", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title("Bio-Inspired ESN (Reduced HO & No Noise) - Next-Step Prediction")
    plt.show()
