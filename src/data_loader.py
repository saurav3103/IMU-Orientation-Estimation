"""
data_loader.py

Loads IMU and motion-capture CSVs (extracted from a ROS bag), normalizes
column names across common naming conventions, and time-synchronizes them.
"""

import os
import glob
import numpy as np
import pandas as pd


def find_csv(folder, pattern):
    files = []
    for p in glob.glob(os.path.join(folder, "**", pattern), recursive=True):
        files.append(p)
    files.sort()
    return files


def load_bag_imu_csv(folder):
    """Load IMU CSV from an extracted bag folder, normalizing column names."""
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
        ("linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"),
        ("accel.x", "accel.y", "accel.z"),
        ("acceleration.x", "acceleration.y", "acceleration.z"),
    ]
    for a, b, c in acc_names:
        if a in imu_df.columns and b in imu_df.columns and c in imu_df.columns:
            out["ax"] = imu_df[a].astype(np.float64).values
            out["ay"] = imu_df[b].astype(np.float64).values
            out["az"] = imu_df[c].astype(np.float64).values
            break
    else:
        raise ValueError("Accelerometer columns not found in IMU CSV.")

    # gyroscope
    gyr_names = [
        ("angular_velocity.x", "angular_velocity.y", "angular_velocity.z"),
        ("gyro.x", "gyro.y", "gyro.z"),
        ("angular_velocity_x", "angular_velocity_y", "angular_velocity_z"),
    ]
    for a, b, c in gyr_names:
        if a in imu_df.columns and b in imu_df.columns and c in imu_df.columns:
            out["gx"] = imu_df[a].astype(np.float64).values
            out["gy"] = imu_df[b].astype(np.float64).values
            out["gz"] = imu_df[c].astype(np.float64).values
            break
    else:
        raise ValueError("Gyroscope columns not found in IMU CSV.")

    # magnetometer (optional)
    mag_names = [
        ("magnetic_field.x", "magnetic_field.y", "magnetic_field.z"),
        ("mag.x", "mag.y", "mag.z"),
    ]
    for a, b, c in mag_names:
        if a in imu_df.columns and b in imu_df.columns and c in imu_df.columns:
            out["mx"] = imu_df[a].astype(np.float64).values
            out["my"] = imu_df[b].astype(np.float64).values
            out["mz"] = imu_df[c].astype(np.float64).values
            break

    out = out.sort_values("time").reset_index(drop=True)
    out["time"] = out["time"] - out["time"].iloc[0]
    return out.astype(np.float64)


def load_bag_mocap_csv(folder):
    """Load motion-capture (VRPN transform) CSV, normalizing quaternion columns."""
    candidates = (
        find_csv(folder, "*raw_transform*.csv")
        + find_csv(folder, "*vrpn*.csv")
        + find_csv(folder, "*raw_transform.csv")
    )
    if len(candidates) == 0:
        candidates = [
            p
            for p in glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
            if "transform" in p or "vrpn" in p
        ]
    if len(candidates) == 0:
        raise FileNotFoundError("No mocap/transform CSV found in extracted bag folder.")

    mocap_csv = candidates[0]
    mocap_df = pd.read_csv(mocap_csv)

    if "Time" in mocap_df.columns:
        tcol = "Time"
    else:
        tcol = mocap_df.columns[0]

    col_names = mocap_df.columns
    cand1 = ("transform.rotation.w", "transform.rotation.x", "transform.rotation.y", "transform.rotation.z")
    cand2 = ("transform.rotation.x", "transform.rotation.y", "transform.rotation.z", "transform.rotation.w")
    rot_cols = None

    if all(c in col_names for c in cand1):
        rot_cols = cand1
    elif all(c in col_names for c in cand2):
        rot_cols = cand2
    else:
        found = [c for c in col_names if "rotation" in c and not c.endswith("covariance")]
        if len(found) >= 4:
            mapping = {}
            for c in found:
                if c.endswith(".w"):
                    mapping["w"] = c
                if c.endswith(".x"):
                    mapping["x"] = c
                if c.endswith(".y"):
                    mapping["y"] = c
                if c.endswith(".z"):
                    mapping["z"] = c
            if set(mapping.keys()) == set(["w", "x", "y", "z"]):
                rot_cols = (mapping["w"], mapping["x"], mapping["y"], mapping["z"])

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
    """Load and time-synchronize IMU and mocap ground truth from an extracted bag folder."""
    imu_df = load_bag_imu_csv(folder)
    gt_df = load_bag_mocap_csv(folder)
    merged = pd.merge_asof(imu_df.sort_values("time"), gt_df.sort_values("time"), on="time", direction="nearest")
    return merged
