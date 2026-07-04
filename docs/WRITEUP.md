# IMU-Based Orientation Estimation: Classical Filtering vs. Learned Sequence Models

## Motivation

Orientation estimation from inertial measurement units (IMUs) is a core problem in navigation, robotics, and motion tracking, and sits at the intersection of two very different design philosophies: **model-based estimation** (Kalman filtering, complementary filters) and **data-driven estimation** (learned sequence models). This project builds both from scratch on the same dataset and benchmarks them against a common ground truth, to understand where each approach earns its keep.

The motivation was twofold: (1) to get hands-on with the full sensor-fusion pipeline — from raw ROS bag extraction through quaternion kinematics to Kalman-filter derivation — rather than treating it as a black box via an off-the-shelf library, and (2) to quantify how much (if anything) a learned model gains over classical filters when both have access to the same raw accelerometer/gyroscope/magnetometer stream.

## Dataset

Data comes from the **TUM Visual-Inertial Dataset (TUM VI)**, published by the Computer Vision Group at TU Munich. Specifically, the `calib-cam7` sequence at 512×512 / 16-bit resolution (`dataset-calib-cam7_512_16.bag`) was used — a calibration sequence in which the sensor rig is moved slowly in front of an AprilGrid target, with full-trajectory ground-truth orientation supplied by a motion-capture (VRPN) system.

Relevant topics extracted from the bag:
- `/imu0` — 3-axis accelerometer + 3-axis gyroscope (200 Hz)
- `/vrpn_client/raw_transform` — ground-truth pose (quaternion) from motion capture
- `/cam0/image_raw`, `/cam1/image_raw` — stereo frames (extracted for completeness, not used in the orientation estimate itself)

The dataset is CC BY 4.0 licensed; download link and citation are included in the repo README.

## Approach

Two families of estimators were implemented and evaluated against the same ground truth:

### 1. Classical / model-based filters
- **Madgwick filter** — gradient-descent correction of a gyro-integrated quaternion using accelerometer (and magnetometer, when available) reference vectors. Implemented via `ahrs`, with unit auto-detection (deg/s vs rad/s) and magnetometer normalization handled explicitly.
- **Mahony filter** — complementary filter using proportional-integral feedback on the rotation error between measured and predicted gravity/magnetic vectors.
- **Custom quaternion Extended Kalman Filter (EKF)** — hand-derived and implemented from first principles (not from a library). The state is the orientation quaternion; the process model propagates via the discretized quaternion kinematics (`q̇ = ½Ω(ω)q`); accelerometer and magnetometer updates are incorporated via a numerically linearized measurement Jacobian (finite-difference approximation of ∂h/∂q). This was the most involved component, requiring explicit reasoning about quaternion normalization after both the predict and update steps to keep the filter well-posed.

### 2. Learned sequence models
- **Transformer encoder** — maps 20-sample windows of 9-axis IMU data (accel + gyro + mag) to quaternion sequences via a small transformer encoder (2 layers, 4 heads, 64-dim embedding).
- **LSTM** — single-layer recurrent baseline over the same windowed input, for comparison against the attention-based model.

Both are trained with direct MSE regression against ground-truth quaternions — a deliberately simple supervision signal (no explicit unit-norm constraint or geodesic loss), which is itself a design decision worth revisiting (see Limitations).

## Evaluation

All five estimators' quaternion outputs are converted to Euler angles (roll, pitch, yaw) and compared against motion-capture ground truth via **RMSE in degrees**, after time-alignment:
- Classical filters (which run sample-by-sample) are interpolated onto the ground-truth timestamps.
- Sequence models (which operate on 20-sample windows) are compared against ground truth shifted by `SEQ_LEN`, since their first valid prediction corresponds to the end of the first window.

Per-method plots (estimate vs. ground truth, one figure per method) and a combined overlay plot across all five methods provide a qualitative view alongside the RMSE table.

## Key implementation details worth noting

- **Unit robustness**: the data loader auto-detects sensor units (e.g., gyro in deg/s vs. rad/s) via a median-magnitude heuristic rather than assuming a fixed convention, since ROS bag exports are not always consistent about this.
- **Column-name robustness**: the CSV loader tries multiple common ROS message field naming conventions (`linear_acceleration.x`, `accel.x`, `acceleration.x`, etc.) so the same code works across differently-exported bags.
- **Quaternion convention care**: `ahrs` and `scipy.spatial.transform.Rotation` use different (w,x,y,z) vs (x,y,z,w) orderings; this project handles the conversion explicitly at each boundary rather than assuming consistency, which is a common and easy-to-miss source of silent bugs in orientation pipelines.

## Results

| Method | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) |
|---|---|---|---|
| Transformer | 1.15 | 0.60 | 1.67 |
| LSTM | 1.40 | 1.32 | 2.74 |
| Madgwick | 2.48 | 2.51 | 1.97 |
| Mahony | 2.27 | 1.67 | 1.82 |
| Kalman (EKF) | 3.24 | 5.29 | 28.77 |

**Observations:**
- The **Transformer** achieves the lowest error across all three axes, and by a wide margin on yaw — consistent with it having learned to exploit patterns in the windowed IMU stream (and, likely, the specific motion profile of this calibration sequence) that a hand-derived filter with fixed gains cannot adapt to.
- **Madgwick and Mahony** perform comparably to each other on roll and pitch (~2–2.5°), which tracks with both being accelerometer/magnetometer-corrected complementary-style filters of similar design.
- The **EKF's yaw error (28.77°) is a clear outlier** and the most informative negative result in the table. Roll and pitch are observable from the accelerometer's gravity reference, but yaw is only observable through the magnetometer correction — so a 10–15x larger yaw error relative to roll/pitch strongly suggests the magnetometer update is either poorly conditioned (e.g., an unmodeled hard/soft-iron disturbance, a bad world-frame reference assumption, or an issue in the finite-difference Jacobian for `update_mag`) rather than a fundamental limitation of EKF-based orientation tracking. This is the natural next debugging target before drawing conclusions about "classical filtering vs. learned models" in general.
- The **learned models outperforming the classical filters here should be read with caution**: the Transformer and LSTM are trained and evaluated on the *same* sequence (no held-out trajectory), so part of their advantage may be sequence-specific memorization rather than a generalizable improvement. The classical filters, by contrast, are gain-tuned defaults with zero exposure to this specific trajectory. A fair generalization test would evaluate the learned models on a held-out TUM VI sequence (e.g., `room1` or a different `calib-cam` run) they were not trained on.

## Limitations and possible extensions

- **Loss function**: direct MSE on raw quaternion components doesn't respect the unit-norm manifold or the double-cover ambiguity (q and −q represent the same rotation). A geodesic/angular loss, or at minimum a normalization layer, would likely improve the learned models' stability.
- **No sensor bias modeling**: the EKF's state is orientation-only; it doesn't estimate gyro/accel bias, which is typically the dominant long-term drift source in IMU-only orientation tracking. Augmenting the state vector with bias terms (a 7-state or 10-state EKF) is a natural next step.
- **Static magnetometer reference**: the EKF and classical filters assume a fixed world-frame magnetic reference vector, which is a simplification in indoor environments with magnetic disturbances from structure and electronics.
- **No comparison to a Madgwick/Mahony-tuned gain sweep**: `beta`, `Kp`, `Ki` were fixed rather than tuned per-sequence, so the classical filter results represent a reasonable default rather than a best case.

## Why this project

This project was built to strengthen the estimation/sensor-fusion side of a control-and-estimation portfolio — deriving and implementing a Kalman filter from first principles (rather than only calling a library function) and putting it head-to-head against modern learned alternatives on real (not simulated) IMU data. It complements more classical-controls-focused portfolio work (e.g., wind turbine pitch/torque control) by demonstrating the estimation half of the state-estimation-plus-control loop that appears throughout systems-and-control coursework and research.
