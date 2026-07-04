"""
filters.py

Classical orientation estimation filters: Madgwick, Mahony, and a custom
quaternion Extended Kalman Filter (EKF) with accelerometer + magnetometer
correction.
"""

import numpy as np
import pandas as pd
from ahrs.filters import Madgwick, Mahony
from ahrs.filters.aqua import AQUA


# -------------------------------
# Madgwick filter
# -------------------------------
def run_madgwick(imu_df, freq=200.0, beta=0.1):
    imu = imu_df.copy().reset_index(drop=True).astype(np.float64)
    imu[["ax", "ay", "az"]] = imu[["ax", "ay", "az"]] / 9.81

    gyr_median = np.median(np.abs(imu[["gx", "gy", "gz"]].values))
    if gyr_median > 50:
        imu[["gx", "gy", "gz"]] = np.deg2rad(imu[["gx", "gy", "gz"]])
    else:
        imu[["gx", "gy", "gz"]] = imu[["gx", "gy", "gz"]].astype(np.float64)

    has_mag = {"mx", "my", "mz"}.issubset(imu.columns)
    calibrated_mag = None
    if has_mag:
        mag = imu[["mx", "my", "mz"]].values.astype(np.float64)
        mag_norms = np.linalg.norm(mag, axis=1).reshape(-1, 1)
        mag_norms[mag_norms == 0] = 1.0
        calibrated_mag = mag / mag_norms

    if has_mag:
        aqua = AQUA()
        q0 = np.array(
            aqua.init_q(acc=imu.iloc[0][["ax", "ay", "az"]].values, mag=calibrated_mag[0]),
            dtype=np.float64,
        )
        filt = Madgwick(frequency=freq, beta=beta, q0=q0)
    else:
        q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        filt = Madgwick(frequency=freq, beta=beta, q0=q0)

    quats = np.zeros((len(imu), 4), dtype=np.float64)
    q = q0.copy()
    for i in range(len(imu)):
        acc = imu.loc[i, ["ax", "ay", "az"]].values.astype(np.float64)
        gyr = imu.loc[i, ["gx", "gy", "gz"]].values.astype(np.float64)
        if has_mag:
            magv = calibrated_mag[i].astype(np.float64)
            q = filt.updateMARG(q=q, gyr=gyr, acc=acc, mag=magv)
        else:
            q = filt.updateIMU(q=q, gyr=gyr, acc=acc)
        q = q.astype(np.float64)
        quats[i] = q

    est_df = pd.DataFrame(quats, columns=["qw", "qx", "qy", "qz"]).astype(np.float64)
    est_df["time"] = imu["time"].values.astype(np.float64)
    return est_df


# -------------------------------
# Mahony filter
# -------------------------------
def run_mahony(imu_df, freq=200.0, Kp=1.0, Ki=0.0):
    imu = imu_df.copy().reset_index(drop=True).astype(np.float64)
    imu[["ax", "ay", "az"]] = imu[["ax", "ay", "az"]] / 9.81

    gyr_median = np.median(np.abs(imu[["gx", "gy", "gz"]].values))
    if gyr_median > 50:
        imu[["gx", "gy", "gz"]] = np.deg2rad(imu[["gx", "gy", "gz"]])
    else:
        imu[["gx", "gy", "gz"]] = imu[["gx", "gy", "gz"]].astype(np.float64)

    has_mag = {"mx", "my", "mz"}.issubset(imu.columns)
    mag_normed = None
    if has_mag:
        mag = imu[["mx", "my", "mz"]].values.astype(np.float64)
        mag_normed = mag / np.linalg.norm(mag, axis=1).reshape(-1, 1)

    filt = Mahony(frequency=freq, kp=Kp, ki=Ki)
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quats = np.zeros((len(imu), 4), dtype=np.float64)

    for i in range(len(imu)):
        acc = imu.loc[i, ["ax", "ay", "az"]].values.astype(np.float64)
        gyr = imu.loc[i, ["gx", "gy", "gz"]].values.astype(np.float64)
        if has_mag:
            magv = mag_normed[i].astype(np.float64)
            q = filt.updateMARG(q=q, gyr=gyr, acc=acc, mag=magv)
        else:
            q = filt.updateIMU(q=q, gyr=gyr, acc=acc)
        q = q.astype(np.float64)
        quats[i] = q

    est_df = pd.DataFrame(quats, columns=["qw", "qx", "qy", "qz"]).astype(np.float64)
    est_df["time"] = imu["time"].values.astype(np.float64)
    return est_df


# -------------------------------
# Quaternion EKF
# -------------------------------
class OrientationEKF:
    """Quaternion-state EKF with gyro propagation and accel/mag correction."""

    def __init__(self, dt=1 / 200.0, q_var=1e-5, r_var_acc=1e-2, r_var_mag=1e-2):
        self.dt = float(dt)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4) * 1e-3
        self.Q = np.eye(4) * q_var
        self.R_acc = np.eye(3) * r_var_acc
        self.R_mag = np.eye(3) * r_var_mag

    def normalize_quat(self, q):
        return q / np.linalg.norm(q)

    def predict(self, gyr):
        wx, wy, wz = gyr
        Omega = np.array(
            [
                [0, -wx, -wy, -wz],
                [wx, 0, wz, -wy],
                [wy, -wz, 0, wx],
                [wz, wy, -wx, 0],
            ],
            dtype=np.float64,
        )
        F = np.eye(4) + 0.5 * self.dt * Omega
        self.q = F @ self.q
        self.q = self.normalize_quat(self.q)
        self.P = F @ self.P @ F.T + self.Q

    def update_acc(self, acc):
        q = self.q
        g_pred = np.array(
            [
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[0] * q[1] + q[2] * q[3]),
                q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,
            ]
        )
        g_pred = g_pred / np.linalg.norm(g_pred)
        acc = acc / (np.linalg.norm(acc) + 1e-8)
        y = acc - g_pred

        H = np.zeros((3, 4))
        eps = 1e-5
        for i in range(4):
            dq = np.zeros(4)
            dq[i] = eps
            q2 = self.normalize_quat(q + dq)
            g2 = np.array(
                [
                    2 * (q2[1] * q2[3] - q2[0] * q2[2]),
                    2 * (q2[0] * q2[1] + q2[2] * q2[3]),
                    q2[0] ** 2 - q2[1] ** 2 - q2[2] ** 2 + q2[3] ** 2,
                ]
            )
            g2 = g2 / np.linalg.norm(g2)
            H[:, i] = (g2 - g_pred) / eps

        S = H @ self.P @ H.T + self.R_acc
        K = self.P @ H.T @ np.linalg.inv(S)
        self.q = self.q + K @ y
        self.q = self.normalize_quat(self.q)
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_mag(self, mag):
        q = self.q
        m_pred = np.array(
            [
                q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2 * (q[1] * q[2] + q[0] * q[3]),
                2 * (q[1] * q[3] - q[0] * q[2]),
            ]
        )
        m_pred = m_pred / np.linalg.norm(m_pred)
        mag = mag / (np.linalg.norm(mag) + 1e-8)
        y = mag - m_pred

        H = np.zeros((3, 4))
        eps = 1e-5
        for i in range(4):
            dq = np.zeros(4)
            dq[i] = eps
            q2 = self.normalize_quat(q + dq)
            m2 = np.array(
                [
                    q2[0] ** 2 + q2[1] ** 2 - q2[2] ** 2 - q2[3] ** 2,
                    2 * (q2[1] * q2[2] + q2[0] * q2[3]),
                    2 * (q2[1] * q2[3] - q2[0] * q2[2]),
                ]
            )
            m2 = m2 / np.linalg.norm(m2)
            H[:, i] = (m2 - m_pred) / eps

        S = H @ self.P @ H.T + self.R_mag
        K = self.P @ H.T @ np.linalg.inv(S)
        self.q = self.q + K @ y
        self.q = self.normalize_quat(self.q)
        self.P = (np.eye(4) - K @ H) @ self.P


def run_kalman_full(imu_df, freq=200.0):
    imu = imu_df.copy().reset_index(drop=True).astype(np.float64)
    dt = 1.0 / float(freq)
    ekf = OrientationEKF(dt=dt)
    quats = []

    for i in range(len(imu)):
        gyr = imu.loc[i, ["gx", "gy", "gz"]].values.astype(np.float64)
        if np.median(np.abs(gyr)) > 50:
            gyr = np.deg2rad(gyr)
        ekf.predict(gyr)

        acc = imu.loc[i, ["ax", "ay", "az"]].values.astype(np.float64)
        ekf.update_acc(acc)

        if {"mx", "my", "mz"}.issubset(imu.columns):
            mag = imu.loc[i, ["mx", "my", "mz"]].values.astype(np.float64)
            ekf.update_mag(mag)

        quats.append(ekf.q.copy())

    est_df = pd.DataFrame(quats, columns=["qw", "qx", "qy", "qz"]).astype(np.float64)
    est_df["time"] = imu["time"].values.astype(np.float64)
    return est_df
