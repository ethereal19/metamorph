import sys
import time
import signal
import collections
import argparse
import joblib
import numpy as np
import pandas as pd
import scipy.signal as signal_processing
import serial

from feature_extraction import extract_window_features, BANDS, FS

PORT = "COM3"
BAUD = 115200
WINDOW_SEC = 1.5
WINDOW_SAMPLES = int(WINDOW_SEC * FS)  # 384 samples


def build_sos_filters(fs=FS):
    filters = {}
    nyq = 0.5 * fs
    for name, (lowcut, highcut) in BANDS.items():
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        sos = signal_processing.butter(4, [low, high], btype="bandpass", output="sos")
        filters[name] = sos
    return filters


BAND_FILTERS = build_sos_filters()


def filter_raw_window(raw_signal):
    """
    Filters raw ADC stream into frequency band signals: Delta, Theta, Alpha, Beta.
    """
    filtered_bands = {}
    for name, sos in BAND_FILTERS.items():
        padlen = 3 * (sos.shape[0] * 2)
        if len(raw_signal) > padlen:
            try:
                filtered_bands[name] = signal_processing.sosfiltfilt(sos, raw_signal)
            except Exception:
                filtered_bands[name] = np.zeros_like(raw_signal)
        else:
            filtered_bands[name] = np.zeros_like(raw_signal)
    
    filtered_bands["Gamma"] = np.zeros_like(raw_signal)
    return pd.DataFrame(filtered_bands)


def load_trained_bci_model(model_path="bci_model.joblib"):
    try:
        model_bundle = joblib.load(model_path)
        print(f" Loaded trained BCI model bundle from '{model_path}'")
        print(f" Model Type: {model_bundle['model_name']} (Training Acc: {model_bundle['accuracy']*100:.2f}%)")
        return model_bundle
    except Exception as e:
        print(f" Error loading BCI model '{model_path}': {e}")
        print(" Please run 'python train_classifier.py' first to train and export the model.")
        sys.exit(1)


def run_live_classifier(simulation_mode=False):
    model_bundle = load_trained_bci_model("bci_model.joblib")
    pipeline = model_bundle["pipeline"]
    label_encoder = model_bundle["label_encoder"]

    buffer = collections.deque(maxlen=WINDOW_SAMPLES)
    ser = None

    if not simulation_mode:
        try:
            ser = serial.Serial(PORT, BAUD, timeout=0.02)
            ser.reset_input_buffer()
            print(f" Connected to Serial Port '{PORT}' at {BAUD} baud.")
        except serial.SerialException:
            print(f"\n Could not open '{PORT}'. Switching to Simulation Mode (--sim)...")
            simulation_mode = True

    print("\n" + "=" * 65)
    print(" SYNAPSE REAL-TIME BCI INTENT CLASSIFIER INFERENCE LOOP")
    print(" Press Ctrl+C to stop.")
    print("=" * 65 + "\n")

    def signal_handler(sig, frame):
        print("\n Shutting down BCI Inference Loop...")
        if ser and ser.is_open:
            ser.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    sim_t = 0.0

    while True:
        if simulation_mode:
            sim_t += 1.0 / FS
            # Simulate shifting signal patterns
            val = np.sin(2 * np.pi * 10 * sim_t) * 20.0 + np.random.normal(0, 5.0)
            buffer.append(val)
            time.sleep(1.0 / FS)
        else:
            if ser and ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
                for line in chunk.split("\n"):
                    line_str = line.strip()
                    if line_str:
                        try:
                            buffer.append(float(line_str))
                        except ValueError:
                            pass

        if len(buffer) >= WINDOW_SAMPLES:
            raw_window = np.array(buffer)
            band_df = filter_raw_window(raw_window)

            # Feature extraction & artifact rejection
            feat_dict = extract_window_features(band_df, fs=FS, artifact_multiplier=4.0)

            if feat_dict is None:
                print(f"[{time.strftime('%H:%M:%S')}] [ARTIFACT REJECTED] Window amplitude exceeded 4x baseline.")
            else:
                feat_df = pd.DataFrame([feat_dict])
                prediction_encoded = pipeline.predict(feat_df)[0]
                intent_label = label_encoder.inverse_transform([prediction_encoded])[0]

                # Probabilities if available
                if hasattr(pipeline, "predict_proba"):
                    probs = pipeline.predict_proba(feat_df)[0]
                    conf = np.max(probs) * 100.0
                    print(f"[{time.strftime('%H:%M:%S')}]  DETECTED INTENT: >>> {intent_label.upper():<6} <<< (Confidence: {conf:.1f}%)")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}]  DETECTED INTENT: >>> {intent_label.upper():<6} <<<")

            time.sleep(0.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapse Real-Time BCI Intent Classifier")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode without hardware")
    args = parser.parse_args()

    run_live_classifier(simulation_mode=args.sim)
