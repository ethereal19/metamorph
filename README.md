# Metamorph — Synapse BCI System 🧠

Metamorph is a real-time Brain-Computer Interface (BCI) system built for **Synapse**. It processes live EEG signals from a **BioAmp EXG Pill + Arduino/ESP32** hardware setup and maps neural frequency band dynamics into 4 discrete user intents: `MOVE`, `STOP`, `SELECT`, and `OPEN`.

---

## ⚡ Key Features

- **Real-Time Band Visualization (`live_band_view.py`)**: High-performance multi-subplot live display of Delta, Theta, Alpha, Beta, and Gamma frequency bands with zero-latency serial queue handling and configurable Y-axis scale controls.
- **Welch Spectral Feature Extraction (`feature_extraction.py`)**: Computes power spectral density (`scipy.signal.welch`) over sliding windows with window-level artifact rejection (~4x baseline multiplier).
- **Multi-Model Machine Learning (`train_classifier.py`)**: Evaluates **LDA**, **SVM (Linear Kernel)**, and **Random Forest** models using 5-fold Stratified Cross-Validation and exports the winning model bundle via `joblib`.
- **Live BCI Intent Inference (`live_classifier.py`)**: Performs real-time intent prediction (`MOVE`, `STOP`, `SELECT`, `OPEN`) with confidence scores from live serial data or simulation mode.

---

## 🎯 Signal-to-Intent Mapping

| Intent | Neural Band Pattern | Functional Action |
| :--- | :--- | :--- |
| **`MOVE`** | Beta increase (13–29 Hz) | Active concentration / imagined movement |
| **`STOP`** | Alpha suppression (8–13 Hz drop) | Eyes open / attention shift |
| **`SELECT`** | Alpha suppression + Beta ratio shift | Focused selection trigger |
| **`OPEN`** | Baseline Alpha power | Eyes-closed relaxed baseline state |

*Note: Beta is capped at 29 Hz and Gamma (30–45 Hz) is treated as unreliable per BioAmp firmware bandpass constraints ([0.5, 29.5] Hz).*

---

## 📂 Repository Structure

```
Metamorph/
├── live_band_view.py        # Real-time multi-band EEG visualizer
├── feature_extraction.py    # Welch PSD band power & artifact rejection
├── train_classifier.py      # ML training & cross-validation model comparison
├── live_classifier.py       # Real-time intent classification engine
├── generate_synthetic_data.py # Synthetic training data generator
├── bci_model.joblib         # Exported ML pipeline bundle
├── move.csv                 # Labeled training dataset for MOVE intent
├── stop.csv                 # Labeled training dataset for STOP intent
├── select.csv               # Labeled training dataset for SELECT intent
├── open.csv                 # Labeled training dataset for OPEN intent
├── README.md                # Project documentation
└── .gitignore               # Environment ignore rules
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
pip install numpy scipy matplotlib pyserial scikit-learn joblib pandas
```

### 2. Live Band Visualization
Connect your BioAmp setup to `COM3` at 115200 baud and run:
```bash
python live_band_view.py
```

### 3. Model Training & Comparison
To train and compare LDA, SVM, and Random Forest models on your CSV datasets:
```bash
python train_classifier.py
```

### 4. Real-Time BCI Intent Classification
To start live inference on incoming serial stream:
```bash
python live_classifier.py
```
*(Run in simulation mode without hardware: `python live_classifier.py --sim`)*
