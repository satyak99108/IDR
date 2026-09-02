# Intelligent Dead Reckoning (IDR) — Smartphone GNSS-Denied Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-50%2F50%20passing-brightgreen.svg)]()
[![SIH MVP](https://img.shields.io/badge/SIH-MVP%20Phase%201--8%20Complete-orange.svg)]()

An end-to-end, multi-rate **Intelligent Dead Reckoning (IDR)** navigation system designed for ground vehicles operating in **GNSS-denied environments** (such as highway tunnels, underpasses, urban canyons, and multi-level parking garages).

Built for the **Smart India Hackathon (SIH)**, this system transforms noisy, consumer-grade smartphone IMU sensor streams into high-accuracy continuous navigation trajectories by coupling **Strapdown Inertial Navigation (INS)**, a **1D-CNN AI Forward Velocity Regressor**, **Non-Holonomic Constraints (NHC)**, and a **10-State Extended Kalman Filter (FilterPy EKF)**.

---

## Benchmark Performance Highlights

Evaluated on the full **51,746-sample** (~86 minutes, 38 km) **IO-VNBD** real-world driving dataset:

| Metric | Phase 3: Raw INS Baseline | Phase 7: AI Dead-Reckoning | Phase 8: GNSS+INS Fused EKF | Overall Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Full Trip RMSE (86 min)** | 1,358,306.6 m | 11,427.4 m | **38.00 m** | **>35,000× over Raw INS** |
| **Full Trip MAE** | 1,067,066.9 m | 9,534.4 m | **32.02 m** | **>33,000× over Raw INS** |
| **Tunnel Blackout Duration** | 60.0 s | 60.0 s | **60.0 s** | Full GNSS Blackout |
| **Distance in Blackout** | 838.09 m | 838.09 m | **838.09 m** | Highway Cruising |
| **Accumulated DR Error** | 23,507 m | 210.3 m | **45.33 m** | Inside Tunnel |
| **Blackout Positional Drift** | **2,804.8%** | **25.1%** | **5.41%** | **PASSED (< 10% Target)** |
| **GNSS Exit Re-acquisition** | Diverged | N/A | **< 3.0 s** | Smooth Kalman Convergence |

---

## System Architecture

```text
               ┌─────────────────────────────────────────────────────────┐
               │         Consumer Smartphone Sensors (10 Hz)            │
               │   • 3D Accelerometer (ax, ay, az)                       │
               │   • 3D Gyroscope (gx, gy, gz)                           │
               │   • GNSS Position & Course (when available)             │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │    Phase 2: Calibration & Alignment Engine               │
               │   • Gravity subtraction & static bias compensation      │
               │   • Body-to-Vehicle frame DCM rotation                  │
               └────────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │                                             │
                     ▼                                             ▼
       ┌───────────────────────────┐                 ┌───────────────────────────┐
       │ Phase 5: 1D-CNN AI Engine │                 │ Phase 6: Motion Classifier│
       │ • 50-sample IMU windows   │                 │ • 9 signal features       │
       │ • Forward speed regressor │                 │ • Dynamic trust weighting │
       │ • Pure NumPy inference    │                 │ • ZUPT state detection    │
       └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                     │ v_fwd                                       │
                     └──────────────────────┬──────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │  Phase 8: Multi-Rate GNSS + INS Fusion Engine (EKF)     │
               │                                                         │
               │  Mode: GNSS_INS (GNSS Visible)                          │
               │    • Strapdown IMU propagation (10 Hz)                  │
               │    • GNSS Position & Horizontal Velocity update (1 Hz)  │
               │    • Active sensor bias calibration (b_ax, b_ay, b_gz)  │
               │                                                         │
               │  Mode: DEAD_RECKONING (Tunnel / Blackout Outage)        │
               │    • Gyro yaw rate heading integration                  │
               │    • AI forward velocity projection: v_fwd              │
               │    • Non-Holonomic Constraints (NHC): v_lat ≈ 0         │
               │    • Direct WGS-84 ellipsoidal propagation              │
               │                                                         │
               │  Mode: RECOVERY (GNSS Re-acquisition)                   │
               │    • Innovation smoothing without position jumps        │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │             Continuous Navigation Trajectory            │
               │  [lat, lon, alt, v_north, v_east, heading, confidence]  │
               └─────────────────────────────────────────────────────────┘
```

---

## 10-State Extended Kalman Filter Formulation

The state vector tracks ground-vehicle dynamics in the local North-East-Down (NED) frame:

$$\mathbf{x} = \begin{bmatrix} p_n & p_e & p_d & v_n & v_e & v_d & \psi & b_{ax} & b_{ay} & b_{gz} \end{bmatrix}^T$$

- **Position ($p_n, p_e, p_d$)**: Local Cartesian coordinates relative to geodetic origin.
- **Velocity ($v_n, v_e, v_d$)**: 3D velocity in NED frame.
- **Heading ($\psi$)**: Vehicle azimuth angle relative to true North.
- **Biases ($b_{ax}, b_{ay}, b_{gz}$)**: Dynamic accelerometer and gyroscope sensor biases.

### Measurement Updates:
1. **GNSS PV Update**: $\mathbf{z}_{\text{gnss}} = [p_n, p_e, p_d, v_n, v_e]^T$
2. **GNSS Course Angle Update**: $\psi_{\text{meas}} = \text{atan2}(v_e, v_n)$ when moving ($v > 1.5\text{ m/s}$).
3. **AI Forward Velocity**: $h(\mathbf{x}) = v_n \cos\psi + v_e \sin\psi = v_{\text{AI}}$.
4. **Non-Holonomic Constraints (NHC)**: $v_{\text{lateral}} \approx 0$, $v_{\text{vertical}} \approx 0$.

---

## Implemented MVP Phases

- **Phase 1 — Data Ingestion & IO-VNBD Pipeline**: Automated dataset downloader and synchronization script (`scripts/download_dataset.py`).
- **Phase 2 — Sensor Calibration & Alignment**: Gravity vector estimation, static bias subtraction, and frame rotation from body to vehicle frame (`src/calibration/calibrator.py`).
- **Phase 3 — Raw Strapdown INS (Baseline)**: Quaternion-based attitude integration and double-integration demonstrating baseline sensor drift (`src/ins/strapdown.py`).
- **Phase 4 — GNSS Outage Simulation Engine**: Standardized benchmark suite simulating 10s underpasses, 30s MVP standard tunnels, 60s mountain tunnels, and stochastic dropouts (`src/simulation/`).
- **Phase 5 — AI Velocity Estimation (1D-CNN)**: Deep learning model trained on 50-sample IMU windows, featuring a zero-dependency pure-NumPy inference engine (`src/ai/`).
- **Phase 6 — Motion & Vibration Feature Classifier**: 9-feature sliding window extractor and rule-based classifier detecting `NORMAL`, `STATIONARY`, `SHOCK`, `VIBRATION`, and `ABNORMAL` motion with dynamic Kalman trust weights (`src/motion/`).
- **Phase 7 — AI-Assisted Dead Reckoning + NHC**: Integrates AI speed along gyro heading and suppresses lateral vehicle slip (`src/ins/ai_assisted.py`, `src/ins/nhc.py`).
- **Phase 8 — Multi-Rate GNSS + INS Fusion**: 10-state Extended Kalman Filter using FilterPy, ellipsoidal WGS-84 tangent-plane transformation, and autonomous mode switching (`src/fusion/`).

---

## Developer Quickstart & Setup Guide

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13)
- `git`

### 2. Clone the Repository
```bash
git clone https://github.com/satyak99108/IDR.git
cd IDR
```

### 3. Create & Activate Virtual Environment
- **On Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **On Windows (CMD):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: The core runtime dependencies (`numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `filterpy`) are lightweight and fast to install. TensorFlow is **optional** and only required if you want to retrain the AI model from scratch; out-of-the-box execution uses pre-trained weights via pure NumPy.

### 5. Download the Dataset
Download and extract the calibrated IO-VNBD dataset (`SYNC_s1_calibrated.csv`):
```bash
python scripts/download_dataset.py
```

### 6. Run Unit Tests (50 Tests)
Verify the entire mathematical and algorithmic pipeline:
```bash
python -m unittest discover tests -v
```

Expected output:
```text
Ran 50 tests in 9.5s
OK
```

---

## Running Benchmark Scripts

### Run Phase 8: Full GNSS + INS Fusion Pipeline
Evaluates the complete 10-state EKF, simulated 60-second tunnel blackout, and outputs 5 presentation plots:
```bash
python scripts/run_phase8.py --outage-start 500.0 --outage-duration 60.0
```
Outputs generated in `results/phase8/`:
- `01_fusion_trajectory_overview.png` — Full trajectory overview
- `02_tunnel_blackout_zoom.png` — Close-up view of tunnel entry, dead-reckoning, and exit recovery
- `03_velocity_and_mode_timeline.png` — Fused velocity vs ground truth with mode shading
- `04_position_error_timeline.png` — Real-time position error with tunnel shaded
- `05_sensor_biases_and_uncertainty.png` — Accelerometer/gyro bias estimates and covariance trace
- `fusion_metrics.json` — Machine-readable summary metrics
- `fused_trajectory.csv` — 51,746-point continuous trajectory

### Run Phase 7: Three-Way Dead-Reckoning Benchmark
Compares Raw INS vs AI+INS vs AI+INS+NHC:
```bash
python scripts/run_phase7.py
```

### Run Phase 6: Motion Classification & Feature Extraction
Extracts vibration features and classifies driving regimes:
```bash
python scripts/run_phase6.py
```

---

## Project Directory Structure

```text
IDR/
├── .gitignore                     # Optimized exclusion rules for lightweight repo
├── requirements.txt               # Documented production dependencies
├── README.md                      # Complete developer & architecture guide
│
├── data/                          # Dataset directory (raw/processed ignored)
│   └── download_instructions.txt
│
├── models/                        # Pre-trained AI model weights
│   ├── scaler_params.json         # Feature normalization parameters
│   └── velocity_cnn_weights.npz   # Extracted CNN weights (NumPy format, 165 KB)
│
├── src/                           # Core Algorithmic Library
│   ├── calibration/               # Sensor bias subtraction & frame alignment
│   │   └── calibrator.py
│   ├── ins/                       # Inertial Navigation & Dead Reckoning
│   │   ├── integration.py         # WGS-84 coordinate transforms & quaternion math
│   │   ├── strapdown.py           # Raw INS double-integration baseline
│   │   ├── nhc.py                 # Non-Holonomic Constraints corrector
│   │   └── ai_assisted.py         # AI forward velocity dead-reckoning engine
│   ├── ai/                        # Deep Learning Components
│   │   ├── dataset.py             # IMU sliding-window data loader
│   │   ├── model.py               # 1D-CNN Keras model architecture
│   │   ├── numpy_inference.py     # Zero-dependency NumPy CNN inference engine
│   │   └── evaluator.py           # Velocity evaluation metrics
│   ├── motion/                    # Motion Regime Classification
│   │   ├── feature_extractor.py   # 9 sliding-window signal features
│   │   └── classifier.py          # Threshold classifier with Kalman trust weights
│   ├── simulation/                # Outage Simulation & Evaluation
│   │   ├── outage_simulator.py    # Outage mask generator (tunnels, dropouts)
│   │   └── evaluator.py           # Blackout drift & RMSE evaluator
│   └── fusion/                    # State Estimation & Multi-Rate Fusion
│       ├── ekf.py                 # 10-state FilterPy Extended Kalman Filter
│       └── fusion_engine.py       # High-level GNSS/INS coordinator & mode manager
│
├── scripts/                       # Executable Phase Benchmarks & Pipelines
│   ├── download_dataset.py        # Dataset downloader & verification
│   ├── run_phase2.py              # Calibration benchmark
│   ├── run_phase3.py              # Raw INS drift benchmark
│   ├── run_phase4.py              # Outage simulation benchmark
│   ├── run_phase5.py              # AI velocity evaluation
│   ├── run_phase6.py              # Motion classification pipeline
│   ├── run_phase7.py              # Three-way dead-reckoning comparison
│   └── run_phase8.py              # Full GNSS + INS state fusion runner
│
└── tests/                         # Automated Unit Test Suite (50 Tests)
    ├── test_calibration.py
    ├── test_ins.py
    ├── test_simulation.py
    ├── test_ai.py
    ├── test_motion.py
    ├── test_ins_ai.py
    └── test_fusion.py
```

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
