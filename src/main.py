"""
main.py

Orchestrates the full orientation-estimation pipeline:
  1. Load synchronized IMU + mocap ground truth from an extracted bag folder
  2. Run classical filters (Madgwick, Mahony, EKF)
  3. Train learned sequence models (Transformer, LSTM)
  4. Plot all estimates against ground truth
  5. Report RMSE (degrees) per method
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

from data_loader import load_dataset_from_bag_folder
from quaternion_utils import quaternions_to_euler_df
from filters import run_madgwick, run_mahony, run_kalman_full
from models import (
    IMUTransformer,
    IMULSTM,
    build_windows,
    train_seq_model,
    sliding_predict_seq,
    DEVICE,
)

# -------------------------------
# Config
# -------------------------------
BAG_EXTRACTED_FOLDER = "dataset-calib-cam7_512_16"
SEQ_LEN = 20
TRANSFORMER_EPOCHS = 60
LSTM_EPOCHS = 40
BATCH_SIZE = 32
LR = 1e-3


def plot_method_single(name, gt_euler, t_full, est_euler, t_est):
    plt.figure(figsize=(10, 6))
    for i, angle in enumerate(["roll_deg", "pitch_deg", "yaw_deg"], start=1):
        plt.subplot(3, 1, i)
        plt.plot(t_full, gt_euler[angle], label="GT")
        plt.plot(t_est, est_euler[angle], label=name)
        plt.ylabel(angle.replace("_deg", "").capitalize() + " (deg)")
        plt.legend()
    plt.suptitle(f"{name} vs GT")
    plt.tight_layout()
    plt.show()


def main():
    print("Loading dataset from folder:", BAG_EXTRACTED_FOLDER)
    data = load_dataset_from_bag_folder(BAG_EXTRACTED_FOLDER)
    print("Merged shape:", data.shape)

    # ---------- build IMU and ground truth frames ----------
    if {"mx", "my", "mz"}.issubset(data.columns):
        imu_df = data[["time", "ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]].astype(np.float64)
    else:
        imu_df = data[["time", "ax", "ay", "az", "gx", "gy", "gz"]].astype(np.float64)

    gt_qtn = data[["time", "qw", "qx", "qy", "qz"]].astype(np.float64)
    gt_euler = quaternions_to_euler_df(gt_qtn)

    # ---------- classical filters ----------
    print("Running Madgwick...")
    mad_qtn = run_madgwick(imu_df, freq=200.0, beta=0.1)
    mad_euler = quaternions_to_euler_df(mad_qtn)

    print("Running Mahony...")
    mah_qtn = run_mahony(imu_df, freq=200.0, Kp=1.0, Ki=0.0)
    mah_euler = quaternions_to_euler_df(mah_qtn)

    print("Running Kalman (full 3D)...")
    kal_qtn = run_kalman_full(imu_df, freq=200.0)
    kal_euler = quaternions_to_euler_df(kal_qtn)

    # ---------- learned models ----------
    print("Building windows for seq models...")
    Xw, Yw = build_windows(imu_df, gt_qtn, seq_len=SEQ_LEN)
    print("Windows shapes:", Xw.shape, Yw.shape)

    print("Training Transformer (seq->seq)...")
    transformer = IMUTransformer(input_dim=9, d_model=64, nhead=4, num_layers=2, output_dim=4).to(DEVICE)
    transformer = train_seq_model(transformer, Xw, Yw, epochs=TRANSFORMER_EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    print("Training LSTM (seq->seq)...")
    lstm = IMULSTM(input_dim=9, hidden_dim=64, output_dim=4, num_layers=1).to(DEVICE)
    lstm = train_seq_model(lstm, Xw, Yw, epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    # ---------- predictions ----------
    print("Predicting with Transformer...")
    trans_preds = sliding_predict_seq(transformer, Xw)
    trans_qtn = pd.DataFrame(trans_preds, columns=["qw", "qx", "qy", "qz"]).astype(np.float64)
    trans_euler = quaternions_to_euler_df(trans_qtn)

    print("Predicting with LSTM...")
    lstm_preds = sliding_predict_seq(lstm, Xw)
    lstm_qtn = pd.DataFrame(lstm_preds, columns=["qw", "qx", "qy", "qz"]).astype(np.float64)
    lstm_euler = quaternions_to_euler_df(lstm_qtn)

    # ---------- time vectors ----------
    t_full = gt_euler["time"].values.astype(np.float64)
    t_mad = mad_euler["time"].values.astype(np.float64)
    t_mah = mah_euler["time"].values.astype(np.float64)
    t_kal = kal_euler["time"].values.astype(np.float64)
    t_trans = t_full[SEQ_LEN:]
    t_lstm = t_full[SEQ_LEN:]

    # ---------- individual plots ----------
    print("Plotting Madgwick vs GT")
    plot_method_single("Madgwick", gt_euler, t_full, mad_euler, t_mad)
    print("Plotting Mahony vs GT")
    plot_method_single("Mahony", gt_euler, t_full, mah_euler, t_mah)
    print("Plotting Kalman vs GT")
    plot_method_single("Kalman", gt_euler, t_full, kal_euler, t_kal)
    print("Plotting Transformer vs GT")
    plot_method_single("Transformer", gt_euler, t_full, trans_euler.assign(time=t_trans), t_trans)
    print("Plotting LSTM vs GT")
    plot_method_single("LSTM", gt_euler, t_full, lstm_euler.assign(time=t_lstm), t_lstm)

    # ---------- combined plot ----------
    plt.figure(figsize=(12, 9))
    for i, angle in enumerate(["roll_deg", "pitch_deg", "yaw_deg"], start=1):
        plt.subplot(3, 1, i)
        plt.plot(t_full, gt_euler[angle], label="GT", linewidth=1.2)
        plt.plot(t_mad, mad_euler[angle], "--", label="Madgwick")
        plt.plot(t_mah, mah_euler[angle], "-.", label="Mahony")
        plt.plot(t_kal, kal_euler[angle], ":", label="Kalman")
        plt.plot(t_trans, trans_euler[angle], ":", label="Transformer")
        plt.plot(t_lstm, lstm_euler[angle], "--", label="LSTM")
        plt.ylabel(angle.replace("_deg", "").capitalize() + " (deg)")
        plt.legend()
    plt.xlabel("Time (s)")
    plt.suptitle("All methods vs GT")
    plt.tight_layout()
    plt.show()

    # ---------- RMSE comparison ----------
    gt_trim = gt_euler.iloc[SEQ_LEN:][["roll_deg", "pitch_deg", "yaw_deg"]].values
    trans_vals = trans_euler[["roll_deg", "pitch_deg", "yaw_deg"]].values
    lstm_vals = lstm_euler[["roll_deg", "pitch_deg", "yaw_deg"]].values

    f_mad = interp1d(t_mad, mad_euler[["roll_deg", "pitch_deg", "yaw_deg"]].values.T,
                      bounds_error=False, fill_value="extrapolate")
    mad_matched = f_mad(t_full[SEQ_LEN:]).T

    f_mah = interp1d(t_mah, mah_euler[["roll_deg", "pitch_deg", "yaw_deg"]].values.T,
                      bounds_error=False, fill_value="extrapolate")
    mah_matched = f_mah(t_full[SEQ_LEN:]).T

    f_kal = interp1d(t_kal, kal_euler[["roll_deg", "pitch_deg", "yaw_deg"]].values.T,
                      bounds_error=False, fill_value="extrapolate")
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


if __name__ == "__main__":
    main()
