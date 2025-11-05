# IMU-Orientation-Estimation
This project implements a complete pipeline for orientation estimation using IMU data (accelerometer, gyroscope, magnetometer). It combines classical sensor fusion algorithms with deep learning models to estimate quaternions and Euler angles (roll, pitch, yaw).


## 📖 Overview
This project implements a complete pipeline for **orientation estimation** using IMU data (accelerometer, gyroscope, magnetometer).  
It combines **classical sensor fusion algorithms** with **deep learning models** to estimate **quaternions** and **Euler angles** (roll, pitch, yaw).  

---

## ⚙️ Implemented Methods

### 🔹 Sensor Fusion Filters
- 🌀 **Madgwick Filter** – computationally efficient quaternion estimation  
- 🌀 **Mahony Filter** – PI-corrected orientation estimation  
- 🌀 **Extended Kalman Filter (EKF)** – probabilistic sensor fusion with accelerometer & magnetometer updates  

### 🔹 Deep Learning Models
- 🔷 **Transformer Encoder** – sequence-to-sequence quaternion prediction  
- 🔷 **LSTM Network** – temporal modeling of IMU signals for attitude estimation  

### 🔹 Magnetometer Calibration
- 📐 **Ellipsoid fitting** for hard-iron and soft-iron distortion correction  

---

## ✨ Features
✅ Load IMU datasets from **ROS bag files**  
✅ Run and compare **different estimation algorithms**  
✅ Convert **quaternions → Euler angles** (roll, pitch, yaw)  
✅ Plot results for **visual comparison** of filters and learning models  
✅ Modular design for **easy extension** with new algorithms  

---

## 🚀 Applications
- 🤖 **Robotics** – navigation & localization  
- 🕹️ **VR/AR** – motion tracking  
- 🚗 **Automotive & Aerospace** – attitude estimation  
- 🏃 **Human Motion Analysis** – wearables & sports science  

---

## 📂 Dataset
This project works with datasets that contain synchronized IMU readings.  
Supported sources include:
- [📦 TUM-VI Dataset]([https://vision.in.tum.de/tumvi](https://cdn2.vision.in.tum.de/tumvi/calibrated/512_16/)) (ROS bag files with IMU + ground truth)  
- Custom IMU recordings (CSV format)  

---

### Arguments

- `--data` : Path to IMU dataset CSV file  
- `--save-plots` : Save plots in the `plots/` folder  
- `--compare` : Enable comparison between Madgwick and Kalman outputs  

---

## 📈 Example Output

- Roll, Pitch, Yaw estimates plotted over time
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/6f5de286-9043-43e3-8755-a924a0d3f4f6" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/8f4f63e8-a144-4c50-a6d3-d3f283bdf667" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/98714b0b-44ae-4b56-826e-99cda23d8dd9" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/83e9cfde-72aa-4e71-8a07-caaa64d40cd4" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/a6480ab6-54de-46e4-b9af-baae31ca88ad" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/665ea261-5e69-4be7-9a0e-88b0fb8868f8" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/ece6d241-364d-43ba-b827-9fe59b6836cf" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/1e49fe07-73b8-41a2-af35-7d0a1ddd5403" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/b9439966-dbe3-47a0-8efe-26316dcbd7f9" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/889d107d-8df0-4c13-bbf0-5310dc9526e7" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/db4e262c-86ba-444f-87e8-f9e67297f51b" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/0ce89cd9-c397-4308-8600-bd48adfe0687" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/97ce4861-6f25-4160-80f6-a439590d6d25" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/686ef996-1117-45fc-ab3b-2811992467f2" />
-<img width="989" height="593" alt="image" src="https://github.com/user-attachments/assets/c72b2c74-c017-4554-bcd8-1a7ab4f260ef" />
-<img width="1189" height="886" alt="image" src="https://github.com/user-attachments/assets/d3d48d67-c268-4cea-ba08-0aab7dceeb72" />


RMSE (deg) [roll, pitch, yaw]:
-Transformer: [1.1589 0.5996 1.6706]
-LSTM       : [1.4095 1.3241 2.7452]
-Madgwick   : [2.4871 2.518  1.972 ]
-Mahony     : [2.2793 1.6749 1.815 ]
-Kalman(r)  : [ 3.2442  5.2974 28.7669]




  

---
## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.


## ⚙️ Dependencies

- `numpy`  
- `pandas`  
- `matplotlib`  
- `ahrs` (for Madgwick filter)  
- `scipy`  

Install manually with:

```bash
pip install numpy pandas matplotlib ahrs scipy




