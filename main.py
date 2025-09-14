import pandas as pd
import bagpy
from bagpy import bagreader

# ---------------------------
# Load the ROS Bag
# ---------------------------
bag_file = "dataset-calib-cam7_512_16.bag"
b = bagreader(bag_file)

# ---------------------------
# List all topics
# ---------------------------
print("Available topics:", b.topic_table)

# ---------------------------
# Extract IMU data (/imu0)
# ---------------------------
imu_csv = b.message_by_topic('/imu0')
imu_df = pd.read_csv(imu_csv)
print("\nIMU Data (first 5 rows):")
print(imu_df.head())

# ---------------------------
# Extract camera data (/cam0/image_raw and /cam1/image_raw)
# ---------------------------
cam0_csv = b.message_by_topic('/cam0/image_raw')
cam1_csv = b.message_by_topic('/cam1/image_raw')

cam0_df = pd.read_csv(cam0_csv)
cam1_df = pd.read_csv(cam1_csv)

print("\nCamera 0 Data (first 5 rows):")
print(cam0_df.head())

print("\nCamera 1 Data (first 5 rows):")
print(cam1_df.head())

# ---------------------------
# Extract motion capture /vrpn_client/raw_transform (if needed)
# ---------------------------
mocap_csv = b.message_by_topic('/vrpn_client/raw_transform')
mocap_df = pd.read_csv(mocap_csv)
print("\nMoCap Data (first 5 rows):")
print(mocap_df.head())

# ---------------------------
# Save IMU data to CSV for easier use
# ---------------------------
imu_df.to_csv("imu_data.csv", index=False)
print("\n✅ IMU data saved to imu_data.csv")

#----------------------------
# full_pipeline_all_filters.py
#----------------------------

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick, Mahony
from ahrs.filters.aqua import AQUA
from scipy import linalg
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d

# -------------------------------
# Config
# -------------------------------
BAG_EXTRACTED_FOLDER = "dataset-calib-cam7_512_16"
SEQ_LEN = 20
TRANSFORMER_EPOCHS = 60
LSTM_EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Helpers: file finding + loading
# -------------------------------
def find_csv(folder, pattern):
    files = []
    for p in glob.glob(os.path.join(folder, "**", pattern), recursive=True):
        files.append(p)
    files.sort()
    return files

def load_bag_imu_csv(folder):
    candidates = find_csv(folder, "*imu*.csv")
    if len(candidates) == 0:
        candidates = find_csv(".", "imu_data.csv")
    if len(candidates) == 0:
        raise FileNotFoundError("No IMU CSV found in folder or current directory.")
    imu_csv = candidates[0]
    imu_df = pd.read_csv(imu_csv)
    # determine time column
    if "Time" in imu_df.columns:
        time_col = "Time"
    elif "time" in imu_df.columns:
        time_col = "time"
    else:
        time_col = imu_df.columns[0]
    out = pd.DataFrame()
    out["time"] = imu_df[time_col].astype(np.float64)
    # accelerometer
    acc_names = [
        ("linear_acceleration.x","linear_acceleration.y","linear_acceleration.z"),
        ("accel.x","accel.y","accel.z"),
        ("acceleration.x","acceleration.y","acceleration.z"),
    ]
    for a,b,c in acc_names:
        if a in imu_df.columns and b in imu_df.columns and c in imu_df.columns:
            out["ax"] = imu_df[a].astype(np.float64).values
            out["ay"] = imu_df[b].astype(np.float64).values
            out["az"] = imu_df[c].astype(np.float64).values
            break
    else:
        raise ValueError("Accelerometer columns not found in IMU CSV.")
    # gyroscope
    gyr_names = [
        ("angular_velocity.x","angular_velocity.y","angular_velocity.z"),
        ("gyro.x","gyro.y","gyro.z"),
        ("angular_velocity_x","angular_velocity_y","angular_velocity_z"),
    ]
    for a,b,c in gyr_names:
        if a in imu_df.columns and b in imu_df.columns and c in imu_df.columns:
            out["gx"] = imu_df[a].astype(np.float64).values
            out["gy"] = imu_df[b].astype(np.float64).values
            out["gz"] = imu_df[c].astype(np.float64).values
            break
    else:
        raise ValueError("Gyroscope columns not found in IMU CSV.")
    # magnetometer (optional)
    mag_names = [
        ("magnetic_field.x","magnetic_field.y","magnetic_field.z"),
        ("mag.x","mag.y","mag.z"),
    ]
    for a,b,c in mag_names:
        if a in imu_df.columns and b in imu_df.columns and c in imu_df.columns:
            out["mx"] = imu_df[a].astype(np.float64).values
            out["my"] = imu_df[b].astype(np.float64).values
            out["mz"] = imu_df[c].astype(np.float64).values
            break
    out = out.sort_values("time").reset_index(drop=True)
    out["time"] = out["time"] - out["time"].iloc[0]
    # ensure float dtype
    return out.astype(np.float64)

def load_bag_mocap_csv(folder):
    candidates = find_csv(folder, "*raw_transform*.csv") + find_csv(folder, "*vrpn*.csv") + find_csv(folder, "*raw_transform.csv")
    if len(candidates) == 0:
        candidates = [p for p in glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True) if "transform" in p or "vrpn" in p]
    if len(candidates) == 0:
        raise FileNotFoundError("No mocap/transform CSV found in extracted bag folder.")
    mocap_csv = candidates[0]
    mocap_df = pd.read_csv(mocap_csv)
    if "Time" in mocap_df.columns:
        tcol = "Time"
    else:
        tcol = mocap_df.columns[0]
    col_names = mocap_df.columns
    # find rotation columns, try common naming
    cand1 = ("transform.rotation.w","transform.rotation.x","transform.rotation.y","transform.rotation.z")
    cand2 = ("transform.rotation.x","transform.rotation.y","transform.rotation.z","transform.rotation.w")
    rot_cols = None
    if all(c in col_names for c in cand1):
        rot_cols = cand1
        qw_first = True
    elif all(c in col_names for c in cand2):
        rot_cols = cand2
        qw_first = False
    else:
        found = [c for c in col_names if "rotation" in c and not c.endswith("covariance")]
        if len(found) >= 4:
            mapping = {}
            for c in found:
                if c.endswith(".w"): mapping["w"] = c
                if c.endswith(".x"): mapping["x"] = c
                if c.endswith(".y"): mapping["y"] = c
                if c.endswith(".z"): mapping["z"] = c
            if set(mapping.keys()) == set(["w","x","y","z"]):
                rot_cols = (mapping["w"], mapping["x"], mapping["y"], mapping["z"])
                qw_first = True
    if rot_cols is None:
        raise ValueError(f"Could not find mocap quaternion columns. Columns present: {list(col_names)}")
    out = pd.DataFrame()
    out["time"] = mocap_df[tcol].astype(np.float64) - mocap_df[tcol].astype(np.float64).iloc[0]
    if rot_cols[0].endswith(".w"):
        out["qw"] = mocap_df[rot_cols[0]].astype(np.float64).values
        out["qx"] = mocap_df[rot_cols[1]].astype(np.float64).values
        out["qy"] = mocap_df[rot_cols[2]].astype(np.float64).values
        out["qz"] = mocap_df[rot_cols[3]].astype(np.float64).values
    else:
        out["qx"] = mocap_df[rot_cols[0]].astype(np.float64).values
        out["qy"] = mocap_df[rot_cols[1]].astype(np.float64).values
        out["qz"] = mocap_df[rot_cols[2]].astype(np.float64).values
        out["qw"] = mocap_df[rot_cols[3]].astype(np.float64).values
    return out.astype(np.float64)

def load_dataset_from_bag_folder(folder):
    imu_df = load_bag_imu_csv(folder)
    gt_df = load_bag_mocap_csv(folder)
    merged = pd.merge_asof(imu_df.sort_values("time"), gt_df.sort_values("time"), on="time", direction="nearest")
    return merged

# -------------------------------
# Quaternion → Euler helpers
# -------------------------------
def quaternion_to_euler(qw,qx,qy,qz):
    # Ensure float
    qw,qx,qy,qz = [float(x) for x in (qw,qx,qy,qz)]
    # ahrs/quaternion conventions vary; use scipy Rotation (expects x,y,z,w)
    r = R.from_quat([qx,qy,qz,qw])
    roll,pitch,yaw = r.as_euler("xyz", degrees=True)
    return roll, pitch, yaw

def quaternions_to_euler_df(df, qw_col="qw", qx_col="qx", qy_col="qy", qz_col="qz"):
    rolls,pitches,yaws = [],[],[]
    for _, row in df.iterrows():
        r,p,y = quaternion_to_euler(row[qw_col], row[qx_col], row[qy_col], row[qz_col])
        rolls.append(r); pitches.append(p); yaws.append(y)
    out = df.copy().astype(np.float64)
    out["roll_deg"] = np.array(rolls, dtype=np.float64)
    out["pitch_deg"] = np.array(pitches, dtype=np.float64)
    out["yaw_deg"] = np.array(yaws, dtype=np.float64)
    return out

# -------------------------------
# Filters (Madgwick, Mahony, Simple Kalman)
# Ensure float dtype for all arrays inside.
# -------------------------------
def run_madgwick(imu_df, freq=200.0, beta=0.1):
    imu = imu_df.copy().reset_index(drop=True).astype(np.float64)
    imu[["ax","ay","az"]] = imu[["ax","ay","az"]] / 9.81
    # gyro units detection & conversion
    gyr_median = np.median(np.abs(imu[["gx","gy","gz"]].values))
    if gyr_median > 50:
        imu[["gx","gy","gz"]] = np.deg2rad(imu[["gx","gy","gz"]])
    else:
        imu[["gx","gy","gz"]] = imu[["gx","gy","gz"]].astype(np.float64)
    has_mag = {"mx","my","mz"}.issubset(imu.columns)
    calibrated_mag = None
    if has_mag:
        mag = imu[["mx","my","mz"]].values.astype(np.float64)
        mag_norms = np.linalg.norm(mag, axis=1).reshape(-1,1)
        mag_norms[mag_norms==0] = 1.0
        calibrated_mag = mag / mag_norms
    # initial quaternion float
    if has_mag:
        aqua = AQUA()
        q0 = np.array(aqua.init_q(acc=imu.iloc[0][["ax","ay","az"]].values, mag=calibrated_mag[0]), dtype=np.float64)
        filt = Madgwick(frequency=freq, beta=beta, q0=q0)
    else:
        q0 = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
        filt = Madgwick(frequency=freq, beta=beta, q0=q0)
    quats = np.zeros((len(imu),4), dtype=np.float64)
    q = q0.copy()
    for i in range(len(imu)):
        acc = imu.loc[i, ["ax","ay","az"]].values.astype(np.float64)
        gyr = imu.loc[i, ["gx","gy","gz"]].values.astype(np.float64)
        if has_mag:
            magv = calibrated_mag[i].astype(np.float64)
            q = filt.updateMARG(q=q, gyr=gyr, acc=acc, mag=magv)
        else:
            q = filt.updateIMU(q=q, gyr=gyr, acc=acc)
        q = q.astype(np.float64)
        quats[i] = q
    est_df = pd.DataFrame(quats, columns=["qw","qx","qy","qz"]).astype(np.float64)
    est_df["time"] = imu["time"].values.astype(np.float64)
    return est_df

def run_mahony(imu_df, freq=200.0, Kp=1.0, Ki=0.0):
    imu = imu_df.copy().reset_index(drop=True).astype(np.float64)
    imu[["ax","ay","az"]] = imu[["ax","ay","az"]] / 9.81
    gyr_median = np.median(np.abs(imu[["gx","gy","gz"]].values))
    if gyr_median > 50:
        imu[["gx","gy","gz"]] = np.deg2rad(imu[["gx","gy","gz"]])
    else:
        imu[["gx","gy","gz"]] = imu[["gx","gy","gz"]].astype(np.float64)
    has_mag = {"mx","my","mz"}.issubset(imu.columns)
    mag_normed = None
    if has_mag:
        mag = imu[["mx","my","mz"]].values.astype(np.float64)
        mag_normed = mag / np.linalg.norm(mag, axis=1).reshape(-1,1)
    filt = Mahony(frequency=freq, kp=Kp, ki=Ki)
    q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    quats = np.zeros((len(imu),4), dtype=np.float64)
    for i in range(len(imu)):
        acc = imu.loc[i, ["ax","ay","az"]].values.astype(np.float64)
        gyr = imu.loc[i, ["gx","gy","gz"]].values.astype(np.float64)
        if has_mag:
            magv = mag_normed[i].astype(np.float64)
            q = filt.updateMARG(q=q, gyr=gyr, acc=acc, mag=magv)
        else:
            q = filt.updateIMU(q=q, gyr=gyr, acc=acc)
        q = q.astype(np.float64)
        quats[i] = q
    est_df = pd.DataFrame(quats, columns=["qw","qx","qy","qz"]).astype(np.float64)
    est_df["time"] = imu["time"].values.astype(np.float64)
    return est_df


# -------------------------------
# Quaternion EKF for roll, pitch, yaw
# -------------------------------
class OrientationEKF:
    def __init__(self, dt=1/200., q_var=1e-5, r_var_acc=1e-2, r_var_mag=1e-2):
        self.dt = float(dt)
        # State = quaternion (4)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 1e-3
        self.Q = np.eye(4) * q_var
        self.R_acc = np.eye(3) * r_var_acc
        self.R_mag = np.eye(3) * r_var_mag

    def normalize_quat(self, q):
        return q / np.linalg.norm(q)

    def predict(self, gyr):
        # Gyro in rad/s
        wx, wy, wz = gyr
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ], dtype=np.float64)
        F = np.eye(4) + 0.5 * self.dt * Omega
        self.q = F @ self.q
        self.q = self.normalize_quat(self.q)
        self.P = F @ self.P @ F.T + self.Q

    def update_acc(self, acc):
        # Expected gravity direction from quaternion
        q = self.q
        g_pred = np.array([
            2*(q[1]*q[3] - q[0]*q[2]),
            2*(q[0]*q[1] + q[2]*q[3]),
            q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
        ])
        g_pred = g_pred / np.linalg.norm(g_pred)
        acc = acc / (np.linalg.norm(acc) + 1e-8)
        y = acc - g_pred
        # Jacobian (approx)
        H = np.zeros((3,4))
        eps = 1e-5
        for i in range(4):
            dq = np.zeros(4); dq[i] = eps
            q2 = self.normalize_quat(q + dq)
            g2 = np.array([
                2*(q2[1]*q2[3] - q2[0]*q2[2]),
                2*(q2[0]*q2[1] + q2[2]*q2[3]),
                q2[0]**2 - q2[1]**2 - q2[2]**2 + q2[3]**2
            ])
            g2 = g2 / np.linalg.norm(g2)
            H[:,i] = (g2 - g_pred)/eps
        S = H @ self.P @ H.T + self.R_acc
        K = self.P @ H.T @ np.linalg.inv(S)
        self.q = self.q + K @ y
        self.q = self.normalize_quat(self.q)
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_mag(self, mag):
        # Expected magnetic field (in world = [1,0,0]) projected
        q = self.q
        m_pred = np.array([
            q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2,
            2*(q[1]*q[2] + q[0]*q[3]),
            2*(q[1]*q[3] - q[0]*q[2])
        ])
        m_pred = m_pred / np.linalg.norm(m_pred)
        mag = mag / (np.linalg.norm(mag)+1e-8)
        y = mag - m_pred
        H = np.zeros((3,4))
        eps = 1e-5
        for i in range(4):
            dq = np.zeros(4); dq[i] = eps
            q2 = self.normalize_quat(q + dq)
            m2 = np.array([
                q2[0]**2 + q2[1]**2 - q2[2]**2 - q2[3]**2,
                2*(q2[1]*q2[2] + q2[0]*q2[3]),
                2*(q2[1]*q2[3] - q2[0]*q2[2])
            ])
            m2 = m2 / np.linalg.norm(m2)
            H[:,i] = (m2 - m_pred)/eps
        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)
        self.q = self.q + K @ y
        self.q = self.normalize_quat(self.q)
        self.P = (np.eye(4) - K @ H) @ self.P

def run_kalman_full(imu_df, freq=200.0):
    imu = imu_df.copy().reset_index(drop=True).astype(np.float64)
    dt = 1.0/float(freq)
    ekf = OrientationEKF(dt=dt)
    quats = []
    for i in range(len(imu)):
        gyr = imu.loc[i, ["gx","gy","gz"]].values.astype(np.float64)
        # detect deg/s and convert if needed
        if np.median(np.abs(gyr)) > 50:
            gyr = np.deg2rad(gyr)
        ekf.predict(gyr)
        acc = imu.loc[i, ["ax","ay","az"]].values.astype(np.float64)
        ekf.update_acc(acc)
        if {"mx","my","mz"}.issubset(imu.columns):
            mag = imu.loc[i, ["mx","my","mz"]].values.astype(np.float64)
            ekf.update_mag(mag)
        quats.append(ekf.q.copy())
    est_df = pd.DataFrame(quats, columns=["qw","qx","qy","qz"]).astype(np.float64)
    est_df["time"] = imu["time"].values.astype(np.float64)
    return est_df


# -------------------------------
# Transformer & LSTM models (seq->seq)
# -------------------------------
class IMUTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, output_dim=4):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)
    def forward(self, x):
        x = self.fc_in(x)             # batch, seq, d_model
        x = self.transformer(x)       # batch, seq, d_model
        return self.fc_out(x)         # batch, seq, output_dim

class IMULSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=1, output_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# -------------------------------
# Build sliding windows
# -------------------------------
def build_windows(imu_df, qtn_df, seq_len=SEQ_LEN):
    imu = imu_df.copy().astype(np.float64).reset_index(drop=True)
    X = imu[["ax","ay","az","gx","gy","gz"]].values
    if {"mx","my","mz"}.issubset(imu.columns):
        M = imu[["mx","my","mz"]].values
    else:
        M = np.zeros((len(X),3), dtype=np.float64)
    X = np.concatenate([X, M], axis=1).astype(np.float32)
    Y = qtn_df[["qw","qx","qy","qz"]].values.astype(np.float32)
    n = len(X) - seq_len
    Xw = np.zeros((n, seq_len, X.shape[1]), dtype=np.float32)
    Yw = np.zeros((n, seq_len, 4), dtype=np.float32)
    for i in range(n):
        Xw[i] = X[i:i+seq_len]
        Yw[i] = Y[i:i+seq_len]
    return Xw, Yw

# -------------------------------
# Training helper
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
        if (ep+1) % max(1, epochs//5) == 0 or ep==0:
            print(f"Epoch {ep+1}/{epochs} - Loss: {tot/len(dataset):.6f}")
    return model

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("Loading dataset from folder:", BAG_EXTRACTED_FOLDER)
    data = load_dataset_from_bag_folder(BAG_EXTRACTED_FOLDER)
    print("Merged shape:", data.shape)

    # build IMU and gt
    # ensure columns exist and are floats
    imu_cols = ["time","ax","ay","az","gx","gy","gz"]
    imu_df = data.copy()
    # include magnetometer if present
    if {"mx","my","mz"}.issubset(data.columns):
        imu_df = imu_df[["time","ax","ay","az","gx","gy","gz","mx","my","mz"]].astype(np.float64)
    else:
        imu_df = imu_df[["time","ax","ay","az","gx","gy","gz"]].astype(np.float64)
    gt_qtn = data[["time","qw","qx","qy","qz"]].astype(np.float64)
    gt_euler = quaternions_to_euler_df(gt_qtn)

    # ---------- standard signal-processing filters ----------
    print("Running Madgwick...")
    mad_qtn = run_madgwick(imu_df, freq=200.0, beta=0.1)
    mad_euler = quaternions_to_euler_df(mad_qtn)

    print("Running Mahony...")
    mah_qtn = run_mahony(imu_df, freq=200.0, Kp=1.0, Ki=0.0)
    mah_euler = quaternions_to_euler_df(mah_qtn)

    print("Running Kalman (full 3D)...")
    kal_qtn = run_kalman_full(imu_df, freq=200.0)
    kal_euler = quaternions_to_euler_df(kal_qtn)


    # ---------- deep models: build windows and train ----------
    print("Building windows for seq models...")
    Xw, Yw = build_windows(imu_df, gt_qtn, seq_len=SEQ_LEN)
    print("Windows shapes:", Xw.shape, Yw.shape)

    # Transformer
    print("Training Transformer (seq->seq)...")
    transformer = IMUTransformer(input_dim=9, d_model=64, nhead=4, num_layers=2, output_dim=4).to(DEVICE)
    transformer = train_seq_model(transformer, Xw, Yw, epochs=TRANSFORMER_EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    # LSTM
    print("Training LSTM (seq->seq)...")
    lstm = IMULSTM(input_dim=9, hidden_dim=64, output_dim=4, num_layers=1).to(DEVICE)
    lstm = train_seq_model(lstm, Xw, Yw, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    # ---------- predictions (sliding windows) ----------
    def sliding_predict_seq(model, Xw):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(len(Xw)):
                xb = torch.tensor(Xw[i:i+1], dtype=torch.float32).to(DEVICE)
                out = model(xb)                                # (1, seq, 4)
                preds.append(out.cpu().numpy()[0, -1])         # last timestep
        return np.array(preds, dtype=np.float64)

    print("Predicting with Transformer...")
    trans_preds = sliding_predict_seq(transformer, Xw)
    trans_qtn = pd.DataFrame(trans_preds, columns=["qw","qx","qy","qz"]).astype(np.float64)
    trans_euler = quaternions_to_euler_df(trans_qtn)

    print("Predicting with LSTM...")
    lstm_preds = sliding_predict_seq(lstm, Xw)
    lstm_qtn = pd.DataFrame(lstm_preds, columns=["qw","qx","qy","qz"]).astype(np.float64)
    lstm_euler = quaternions_to_euler_df(lstm_qtn)

    # ---------- plotting: separate plots then combined ----------
    # For sequence models, their predictions align with gt starting at index SEQ_LEN
    t_full = gt_euler["time"].values.astype(np.float64)
    t_mad = mad_euler["time"].values.astype(np.float64)
    t_mah = mah_euler["time"].values.astype(np.float64)
    t_kal = kal_euler["time"].values.astype(np.float64)
    t_trans = t_full[SEQ_LEN:]
    t_lstm = t_full[SEQ_LEN:]

    def plot_method_single(name, est_euler, t_est):
        plt.figure(figsize=(10,6))
        for i, angle in enumerate(["roll_deg","pitch_deg","yaw_deg"], start=1):
            plt.subplot(3,1,i)
            plt.plot(t_full, gt_euler[angle], label="GT")
            plt.plot(t_est, est_euler[angle], label=name)
            plt.ylabel(angle.replace("_deg","").capitalize()+" (deg)")
            plt.legend()
        plt.suptitle(f"{name} vs GT")
        plt.tight_layout()
        plt.show()

    # show each separately
    print("Plotting Madgwick vs GT")
    plot_method_single("Madgwick", mad_euler, t_mad)
    print("Plotting Mahony vs GT")
    plot_method_single("Mahony", mah_euler, t_mah)
    print("Plotting Kalman  vs GT")
    plot_method_single("Kalman", kal_euler, t_kal)
    print("Plotting Transformer vs GT")
    plot_method_single("Transformer", trans_euler.assign(time=t_trans), t_trans)
    print("Plotting LSTM vs GT")
    plot_method_single("LSTM", lstm_euler.assign(time=t_lstm), t_lstm)

    # combined plot
    plt.figure(figsize=(12,9))
    for i, angle in enumerate(["roll_deg","pitch_deg","yaw_deg"], start=1):
        plt.subplot(3,1,i)
        plt.plot(t_full, gt_euler[angle], label="GT", linewidth=1.2)
        plt.plot(t_mad, mad_euler[angle], "--", label="Madgwick")
        plt.plot(t_mah, mah_euler[angle], "-.", label="Mahony")
        plt.plot(t_kal, kal_euler[angle], ":", label="Kalman")
        plt.plot(t_trans, trans_euler[angle], ":", label="Transformer")
        plt.plot(t_lstm, lstm_euler[angle], "--", label="LSTM")
        plt.ylabel(angle.replace("_deg","").capitalize()+" (deg)")
        plt.legend()
    plt.xlabel("Time (s)")
    plt.suptitle("All methods vs GT")
    plt.tight_layout()
    plt.show()

    # ---------- RMSE numeric comparison ----------
    # Align GT for seq models (trim first SEQ_LEN samples)
    gt_trim = gt_euler.iloc[SEQ_LEN:][["roll_deg","pitch_deg","yaw_deg"]].values
    trans_vals = trans_euler[["roll_deg","pitch_deg","yaw_deg"]].values
    lstm_vals = lstm_euler[["roll_deg","pitch_deg","yaw_deg"]].values

    # For Madgwick & Mahony & Kalman, interpolate to GT times (trim first SEQ_LEN to match)
    f_mad = interp1d(t_mad, mad_euler[["roll_deg","pitch_deg","yaw_deg"]].values.T, bounds_error=False, fill_value="extrapolate")
    mad_matched = f_mad(t_full[SEQ_LEN:]).T
    f_mah = interp1d(t_mah, mah_euler[["roll_deg","pitch_deg","yaw_deg"]].values.T, bounds_error=False, fill_value="extrapolate")
    mah_matched = f_mah(t_full[SEQ_LEN:]).T
    f_kal = interp1d(t_kal, kal_euler[["roll_deg","pitch_deg","yaw_deg"]].values.T, bounds_error=False, fill_value="extrapolate")
    kal_matched = f_kal(t_full[SEQ_LEN:]).T

    rmse_trans = np.sqrt(mean_squared_error(gt_trim, trans_vals, multioutput='raw_values'))
    rmse_lstm = np.sqrt(mean_squared_error(gt_trim, lstm_vals, multioutput='raw_values'))
    rmse_mad = np.sqrt(mean_squared_error(gt_trim, mad_matched, multioutput='raw_values'))
    rmse_mah = np.sqrt(mean_squared_error(gt_trim, mah_matched, multioutput='raw_values'))
    rmse_kal = np.sqrt(mean_squared_error(gt_trim, kal_matched, multioutput='raw_values'))

    print("RMSE (deg) [roll, pitch, yaw]:")
    print("Transformer:", np.round(rmse_trans, 4))
    print("LSTM       :", np.round(rmse_lstm, 4))
    print("Madgwick   :", np.round(rmse_mad, 4))
    print("Mahony     :", np.round(rmse_mah, 4))
    print("Kalman(r)  :", np.round(rmse_kal, 4))



