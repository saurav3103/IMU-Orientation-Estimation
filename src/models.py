"""
models.py

Learned sequence-to-sequence models (Transformer, LSTM) for regressing
orientation quaternions from windowed IMU data, plus window-building and
training helpers.
"""

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Models
# -------------------------------
class IMUTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, output_dim=4):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.fc_in(x)          # batch, seq, d_model
        x = self.transformer(x)    # batch, seq, d_model
        return self.fc_out(x)      # batch, seq, output_dim


class IMULSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=1, output_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


# -------------------------------
# Sliding windows
# -------------------------------
def build_windows(imu_df, qtn_df, seq_len=20):
    imu = imu_df.copy().astype(np.float64).reset_index(drop=True)
    X = imu[["ax", "ay", "az", "gx", "gy", "gz"]].values

    if {"mx", "my", "mz"}.issubset(imu.columns):
        M = imu[["mx", "my", "mz"]].values
    else:
        M = np.zeros((len(X), 3), dtype=np.float64)

    X = np.concatenate([X, M], axis=1).astype(np.float32)
    Y = qtn_df[["qw", "qx", "qy", "qz"]].values.astype(np.float32)

    n = len(X) - seq_len
    Xw = np.zeros((n, seq_len, X.shape[1]), dtype=np.float32)
    Yw = np.zeros((n, seq_len, 4), dtype=np.float32)
    for i in range(n):
        Xw[i] = X[i:i + seq_len]
        Yw[i] = Y[i:i + seq_len]
    return Xw, Yw


# -------------------------------
# Training
# -------------------------------
def train_seq_model(model, Xw, Yw, epochs=20, batch_size=32, lr=1e-3):
    model.to(DEVICE)
    dataset = torch.utils.data.TensorDataset(torch.tensor(Xw), torch.tensor(Yw))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs} - Loss: {tot/len(dataset):.6f}")

    return model


def sliding_predict_seq(model, Xw):
    """Predict the last timestep's quaternion for each window in Xw."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(Xw)):
            xb = torch.tensor(Xw[i:i + 1], dtype=torch.float32).to(DEVICE)
            out = model(xb)                          # (1, seq, 4)
            preds.append(out.cpu().numpy()[0, -1])   # last timestep
    return np.array(preds, dtype=np.float64)
