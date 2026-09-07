# Intelligent Dead Reckoning (IDR) — Smartphone & Edge GNSS-Denied Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-141%2F141%20passing-brightgreen.svg)]()
[![Android JVM Tests](https://img.shields.io/badge/android%20tests-12%2F12%20passing-brightgreen.svg)]()
[![SIH MVP](https://img.shields.io/badge/SIH-MVP%20Phases%201--15%20Complete-orange.svg)]()

An end-to-end, multi-rate **Intelligent Dead Reckoning (IDR)** navigation system engineered for ground vehicles navigating in **GNSS-denied environments** (such as highway tunnels, underground passages, dense urban street canyons, and multi-level parking garages).

Built for the **Smart India Hackathon (SIH)**, this system couples **Strapdown Inertial Navigation (INS)**, a **1D-CNN AI Forward Velocity Regressor**, **Non-Holonomic Constraints (NHC)**, a **15-State Extended Kalman Filter (EKF)**, **HMM Map Matching**, a **Sensor-Agnostic Edge Streaming Engine**, and a **Native Android Application** with TFLite / LiteRT neural inference.

---

## Benchmark Performance Highlights

Evaluated on the full **51,746-sample** (~86 minutes, 38 km) **IO-VNBD** real-world vehicle driving dataset:

| Metric | Target Requirement | Phase 3: Raw INS | Phase 8: EKF Fusion | Phase 12: Full Prototype (AI + EKF + MM) | Phase 15: Edge Engine | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **60s Tunnel Blackout Drift** | **< 10.0%** | 2,804.8% | 5.41% | **3.82%** | **8.76%** | **PASSED** |
| **Blackout Positional Error** | Minimal | 23,507 m | 45.33 m | **31.28 m** | **71.77 m** | **PASSED** |
| **Streaming Throughput** | **$\ge$ 10.0 Hz** | N/A | ~45 Hz | ~120 Hz | **230 – 266 Hz** | **PASSED ($\ge 23\times$)** |
| **Processing Latency (p95)** | **< 50.0 ms** | N/A | ~15 ms | ~8 ms | **< 7.1 ms** | **PASSED** |
| **Model Size (INT8 Quantized)** | **< 1.0 MB** | N/A | N/A | 165 KB (FP32) | **44.0 KB** | **PASSED** |
| **Re-acquisition Convergence** | **< 3.0 s** | Diverged | < 2.5 s | **< 1.8 s** | **< 2.0 s** | **PASSED** |

---

## System Architecture

```text
                       +-------------------------------+
                       |      External IMU Sensor      |
                       |    (High-Rate 50 - 100 Hz)    |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |     ExternalIMUAdapter        |
                       |  - Decimation (100 -> 10 Hz)  |
                       |  - Anti-aliasing sync         |
                       +---------------+---------------+
                                       |
+--------------------------+           |
| Android App / Phone Log  |           |
| (IO-VNBD / UDP 5555)     |           |
+------------+-------------+           |
             |                         |
             v                         |
+--------------------------+           |
|    SmartphoneAdapter     |           |
| - Replay & Socket Poll   |           |
| - Calibrated Veh Frame   |           |
+------------+-------------+           |
             |                         |
             +------------+------------+
                          |
                          v  (EdgeSensorPacket contract)
        +-----------------------------------------------+
        |             EdgeNavigationEngine              |
        |  +-----------------------------------------+  |
        |  | Shared IDR Navigation Core              |  |
        |  |  - 15-State Error-State EKF             |  |
        |  |  - 1D CNN Forward Velocity Estimator    |  |
        |  |  - Motion Classifier (Stationary Lock)  |  |
        |  |  - HMM Map Matcher (Road Corridor)      |  |
        |  +-----------------------------------------+  |
        |  - Outage duration & confidence monitor    |  |
        |  - Hot-swap adapter tracker                |  |
        +-----------------------+-----------------------+
                                |
                                v  (EdgeNavigationOutput contract)
                 +-----------------------------+
                 |  Downstream Client / UI     |
                 |  (Android HUD / Telemetry)  |
                 +-----------------------------+
```

---

## Prerequisites & Installation

To run, develop, or contribute to this repository, install the following tools beforehand:

### 1. Python Environment (Core Algorithms & Benchmarks)
- **Python 3.10 to 3.13** (64-bit).
- **Git** for version control.

### 2. Android Studio Environment (For Mobile App & On-Device Testing)
- **Android Studio**: Android Studio Hedgehog (2023.1.1), Iguana, Jellyfish, Ladybug, or newer.
- **Java Development Kit (JDK)**: **JDK 17** (use the embedded OpenJDK provided with Android Studio: `Android Studio/jbr`).
- **Android SDK Components**:
  - **SDK Platform**: API 34 (Android 14.0)
  - **Build-Tools**: 34.0.0
  - **Minimum SDK**: API 26 (Android 8.0 Oreo)
- **Physical Device or Emulator**:
  - An Android device running Android 8.0+ connected via USB (with USB Debugging enabled) for real IMU sensor streaming.
  - Or an Android Virtual Device (AVD) running API 34.

---

## Getting Started (Quickstart Guide)

### Step 1: Clone the Repository
```bash
git clone https://github.com/satyak99108/IDR.git
cd IDR
```

### Step 2: Set Up Python Virtual Environment
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

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: Core execution uses pre-trained weights via pure NumPy or TFLite runtime. TensorFlow is optional and only required if retraining the 1D-CNN from scratch.

### Step 4: Download the Calibrated Dataset
Download and extract the calibrated IO-VNBD dataset (`SYNC_s1_calibrated.csv`):
```bash
python scripts/download_dataset.py
```

### Step 5: Verify Python Test Suite (141 Tests)
Run pytest across all subsystems:
```bash
python -m pytest tests/ -v
```
All 141 tests will execute in under 30 seconds and pass cleanly.

---

## Running Benchmarks & Pipelines

### 1. Phase 15: Edge Engine Streaming Benchmark (Multi-Adapter)
Tests standard contracts, real-time throughput, and 60s GNSS outage drift on both `SmartphoneAdapter` and `ExternalIMUAdapter`:
```bash
python scripts/run_phase15.py
```
Outputs in `results/phase15/`:
- `01_edge_adapter_comparison.png` — Trajectory and drift comparison
- `02_edge_streaming_latency.png` — Ingestion latency histogram and jitter waterfall
- `03_edge_throughput_benchmark.png` — Real-time throughput (Hz) vs 10 Hz MVP requirement
- `edge_benchmark.json` — Detailed JSON metrics

### 2. Phase 13: TFLite Quantization & Model Size Benchmark
Converts the Keras CNN into Float32, Float16, and INT8 TFLite models, verifying accuracy vs compression:
```bash
python scripts/run_phase13.py
```
Outputs in `results/phase13/`:
- `01_model_size_comparison.png`
- `02_latency_vs_throughput.png`
- `03_accuracy_vs_size_pareto.png`
- `04_velocity_prediction_overlay.png`

### 3. Phase 12: Production Prototype Benchmark Suite
Executes the full 4-stage progression (Raw INS $\rightarrow$ INS+NHC $\rightarrow$ AI+Fusion $\rightarrow$ Map Matching):
```bash
python scripts/run_phase12.py
```

---

## Building & Testing the Android App (`android/`)

The native Android app implements real-time hardware sensor sampling (`SensorManager`), TFLite inference via `velocity_cnn.tflite`, a HUD navigation display, and network socket streaming.

### Method A: Using Android Studio (Recommended for Development & UI Testing)
1. Open **Android Studio**.
2. Select **Open** and choose the `android/` directory inside the cloned repo (`IDR/android`).
3. Allow Gradle to synchronize dependencies.
4. Set Project JDK to **JDK 17** (`Settings` $\rightarrow$ `Build, Execution, Deployment` $\rightarrow$ `Build Tools` $\rightarrow$ `Gradle` $\rightarrow$ `Gradle JDK`).
5. Connect your Android device or start an emulator.
6. Click **Run 'app'** (`Shift + F10`) to build, install, and launch the application.

### Method B: Using Command-Line Gradle

- **On Windows (PowerShell):**
  ```powershell
  cd android
  .\gradlew assembleDebug
  ```
- **On Linux / macOS:**
  ```bash
  cd android
  chmod +x gradlew
  ./gradlew assembleDebug
  ```
The compiled APK will be generated at:
```
android/app/build/outputs/apk/debug/app-debug.apk
```

### Running Android JVM Unit Tests
Verify Android navigation engine logic, mode transitions, and sensor smoothing without an emulator:
- **Windows:**
  ```powershell
  cd android
  .\gradlew testDebugUnitTest
  ```
- **Linux / macOS:**
  ```bash
  cd android
  ./gradlew testDebugUnitTest
  ```

---

## Repository Directory Structure

```text
IDR/
├── android/                       # Native Android Application (Kotlin, TFLite, MVVM)
│   ├── app/                       # Android App module (src/main/java/com/idr/navigation)
│   ├── build.gradle.kts           # Root Gradle build script
│   └── gradlew / gradlew.bat      # Gradle wrapper scripts
│
├── edge/                          # Sensor-Agnostic Edge Engine Subsystem (Phase 15)
│   ├── contracts.py               # EdgeSensorPacket & EdgeNavigationOutput contracts
│   ├── core.py                    # EdgeNavigationEngine wrapper
│   └── adapters/                  # Stream adapters
│       ├── base.py                # BaseSensorAdapter & AdapterStats
│       ├── smartphone_adapter.py  # IO-VNBD replay & live UDP socket receiver
│       └── external_imu_adapter.py# High-rate (50-100Hz) IMU decimation adapter
│
├── models/                        # Deployed Neural Models & Scalers
│   ├── scaler_params.json         # Standardizer mean & standard deviation parameters
│   ├── velocity_cnn.tflite        # Quantized production TFLite model (44 KB)
│   └── velocity_cnn_weights.npz   # NumPy weights for zero-dependency inference
│
├── src/                           # Core Algorithmic Library
│   ├── calibration/               # Gravity subtraction & frame rotation
│   ├── ins/                       # Strapdown INS, NHC & dead reckoning
│   ├── ai/                        # 1D-CNN Keras model & NumPy inference
│   ├── motion/                    # Motion feature extractor & zero-velocity classifier
│   ├── simulation/                # Outage generator & drift evaluation
│   ├── fusion/                    # 15-state EKF & mode manager
│   ├── map_matching/              # HMM map matcher & OSM road corridor graph
│   ├── pipeline/                  # IDRNavigationEngine orchestrator
│   ├── deployment/                # TFLite quantizer & latency profiler
│   └── evaluation/                # Benchmark runner & resource profiler
│
├── scripts/                       # Executable benchmark runners (run_phase2.py to run_phase15.py)
├── tests/                         # 141 Python unit tests (pytest)
├── requirements.txt               # Documented production dependencies
├── MVP.md                         # Detailed project roadmap and specifications
└── README.md                      # Developer onboarding and architecture guide
```

---

## How to Propose Changes & Contribute

We welcome contributions from developers, researchers, and participants!

1. **Fork the Repository**: Click "Fork" on GitHub and clone your fork locally.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. **Coding Standards**:
   - Write type-annotated, PEP-8 compliant Python code.
   - For Kotlin, adhere to official Android Kotlin style conventions.
   - Preserve backward compatibility with existing standard contracts (`EdgeSensorPacket`, `EdgeNavigationOutput`).
4. **Add Tests**:
   - For new Python features, add corresponding test cases in `tests/`.
   - For Android additions, include unit tests under `android/app/src/test/`.
5. **Verify Before Committing**:
   ```bash
   python -m pytest tests/ -v
   cd android && ./gradlew testDebugUnitTest
   ```
6. **Submit a Pull Request**:
   - Push your branch to GitHub.
   - Open a PR against `main` with a clear description of changes, benchmark impacts, and verification results.

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
