"""
quaternion_utils.py

Helpers for converting quaternions to Euler angles (roll, pitch, yaw).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler(qw, qx, qy, qz):
    """Convert a single quaternion (w,x,y,z) to Euler angles in degrees (xyz order)."""
    qw, qx, qy, qz = [float(x) for x in (qw, qx, qy, qz)]
    # scipy Rotation expects (x, y, z, w) ordering
    r = R.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler("xyz", degrees=True)
    return roll, pitch, yaw


def quaternions_to_euler_df(df, qw_col="qw", qx_col="qx", qy_col="qy", qz_col="qz"):
    """Convert a DataFrame of quaternions into roll/pitch/yaw columns (degrees)."""
    rolls, pitches, yaws = [], [], []
    for _, row in df.iterrows():
        r, p, y = quaternion_to_euler(row[qw_col], row[qx_col], row[qy_col], row[qz_col])
        rolls.append(r)
        pitches.append(p)
        yaws.append(y)

    out = df.copy().astype(np.float64)
    out["roll_deg"] = np.array(rolls, dtype=np.float64)
    out["pitch_deg"] = np.array(pitches, dtype=np.float64)
    out["yaw_deg"] = np.array(yaws, dtype=np.float64)
    return out
