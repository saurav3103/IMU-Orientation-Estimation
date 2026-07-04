"""
extract_bag.py

Extracts topics from a ROS bag file (IMU, stereo camera frames, and
motion-capture ground truth) into CSV files using bagpy.

Usage:
    python extract_bag.py
"""

import pandas as pd
from bagpy import bagreader

# ---------------------------
# Config
# ---------------------------
BAG_FILE = "dataset-calib-cam7_512_16.bag"


def extract_topics(bag_file=BAG_FILE):
    """Extract IMU, camera, and mocap topics from a ROS bag into CSVs."""
    b = bagreader(bag_file)

    print("Available topics:", b.topic_table)

    # ---------------------------
    # IMU data (/imu0)
    # ---------------------------
    imu_csv = b.message_by_topic('/imu0')
    imu_df = pd.read_csv(imu_csv)
    print("\nIMU Data (first 5 rows):")
    print(imu_df.head())

    # ---------------------------
    # Camera data (/cam0/image_raw and /cam1/image_raw)
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
    # Motion capture (/vrpn_client/raw_transform)
    # ---------------------------
    mocap_csv = b.message_by_topic('/vrpn_client/raw_transform')
    mocap_df = pd.read_csv(mocap_csv)
    print("\nMoCap Data (first 5 rows):")
    print(mocap_df.head())

    # ---------------------------
    # Save IMU data to CSV for easier downstream use
    # ---------------------------
    imu_df.to_csv("imu_data.csv", index=False)
    print("\n✅ IMU data saved to imu_data.csv")

    return imu_df, cam0_df, cam1_df, mocap_df


if __name__ == "__main__":
    extract_topics()
