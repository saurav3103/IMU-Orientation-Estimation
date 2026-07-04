# IMU Orientation Estimation

Orientation (roll/pitch/yaw) estimation from raw IMU data, benchmarking classical filters against learned sequence models on a ROS bag dataset with motion-capture ground truth.

## Overview

This project extracts synchronized IMU and motion-capture data from a ROS bag file and estimates 3D orientation using five different approaches:

- **Madgwick filter** — gradient-descent-based orientation filter (IMU/MARG)
- **Mahony filter** — complementary-filter-based orientation estimator
- **Extended Kalman Filter (EKF)** — custom quaternion-state EKF with accelerometer and magnetometer correction
- **Transformer** — sequence-to-sequence model trained to regress quaternions from windowed IMU data
- **LSTM** — recurrent sequence-to-sequence baseline for the same task

Estimates from all five methods are compared against VRPN motion-capture ground truth using RMSE (in degrees) on roll, pitch, and yaw.

## Pipeline

1. **Bag extraction** — reads `/imu0`, `/cam0/image_raw`, `/cam1/image_raw`, and `/vrpn_client/raw_transform` topics from the `.bag` file using `bagpy`, and exports them to CSV.
2. **Data loading & synchronization** — IMU and mocap CSVs are parsed, unit-normalized (gyro deg/s → rad/s where needed, accelerometer → g), and time-aligned via `merge_asof`.
3. **Classical filtering** — Madgwick, Mahony, and a custom quaternion EKF are run sample-by-sample over the IMU stream to estimate orientation quaternions.
4. **Learned models** — IMU windows (20-sample sequences of accel + gyro + mag) are used to train a Transformer and an LSTM to regress ground-truth quaternions directly.
5. **Evaluation** — All estimates are converted to Euler angles and compared against ground truth via RMSE, with per-method and combined plots.

## Dataset

Expects a ROS `.bag` file (e.g. `dataset-calib-cam7_512_16.bag`) containing:
- `/imu0` — accelerometer + gyroscope (+ optional magnetometer)
- `/vrpn_client/raw_transform` — motion-capture ground-truth pose (quaternion)
- `/cam0/image_raw`, `/cam1/image_raw` — stereo camera frames (extracted but not used in the estimation pipeline)

## Usage

```bash
pip install -r requirements.txt

# Extract bag topics to CSV
python extract_bag.py

# Run the full filtering + learning + evaluation pipeline
python full_pipeline_all_filters.py
```

Update `BAG_EXTRACTED_FOLDER` in the config section of the pipeline script to point to your extracted bag folder.

## Output

- Per-method plots of roll/pitch/yaw vs. ground truth
- Combined comparison plot across all five methods
- Console-printed RMSE table (degrees) for roll, pitch, and yaw per method

## Requirements

See `requirements.txt`. Notable dependencies: `bagpy` (ROS bag parsing without a full ROS install), `ahrs` (Madgwick/Mahony/AQUA filter implementations), `torch` (Transformer/LSTM training).

## Notes

- GPU is used automatically if available (`torch.cuda.is_available()`), otherwise falls back to CPU.
- Sequence model predictions are time-shifted by `SEQ_LEN` samples relative to ground truth; RMSE computation accounts for this offset.
- Column-name detection is fuzzy-matched across common ROS message field naming conventions to make the loader robust to different bag exports.
