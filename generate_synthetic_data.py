import os
import numpy as np
import pandas as pd

FS = 256
DURATION_PER_CLASS_SEC = 25  # 25 seconds of continuous data per intent class


def generate_eeg_class_signal(intent_label, duration_sec=DURATION_PER_CLASS_SEC, fs=FS):
    """
    Generates realistic pre-filtered band signals matching the hardware constraints
    and signal-to-intent mapping for Synapse BCI.
    """
    t = np.linspace(0, duration_sec, int(duration_sec * fs), endpoint=False)
    n_samples = len(t)

    np.random.seed(42 + hash(intent_label) % 1000)

    delta = np.random.normal(0, 4.0, n_samples)
    theta = np.random.normal(0, 3.0, n_samples)
    gamma = np.random.normal(0, 0.5, n_samples)

    if intent_label == "move":
        # MOVE: Elevated Beta band power (13-29 Hz active concentration / imagined movement)
        alpha = np.random.normal(0, 8.0, n_samples) + 5.0 * np.sin(2 * np.pi * 10.0 * t)
        beta = np.random.normal(0, 25.0, n_samples) + 22.0 * np.sin(2 * np.pi * 20.0 * t)

    elif intent_label == "stop":
        # STOP: Marked Alpha suppression (drop in Alpha amplitude)
        alpha = np.random.normal(0, 2.0, n_samples) + 1.5 * np.sin(2 * np.pi * 10.0 * t)
        beta = np.random.normal(0, 8.0, n_samples) + 6.0 * np.sin(2 * np.pi * 18.0 * t)

    elif intent_label == "select":
        # SELECT: Moderate Alpha suppression + Beta activation shift
        alpha = np.random.normal(0, 4.0, n_samples) + 3.0 * np.sin(2 * np.pi * 11.0 * t)
        beta = np.random.normal(0, 16.0, n_samples) + 14.0 * np.sin(2 * np.pi * 22.0 * t)

    elif intent_label == "open":
        # OPEN: High baseline Alpha power (relaxed eyes closed)
        alpha = np.random.normal(0, 30.0, n_samples) + 28.0 * np.sin(2 * np.pi * 10.5 * t)
        beta = np.random.normal(0, 6.0, n_samples) + 4.0 * np.sin(2 * np.pi * 16.0 * t)

    else:
        alpha = np.random.normal(0, 10.0, n_samples)
        beta = np.random.normal(0, 10.0, n_samples)

    df = pd.DataFrame({
        "timestamp": t,
        "Delta": delta,
        "Theta": theta,
        "Alpha": alpha,
        "Beta": beta,
        "Gamma": gamma,
    })

    return df


def main():
    print("Generating synthetic EEG training datasets...")
    classes = ["move", "stop", "select", "open"]
    for c in classes:
        df = generate_eeg_class_signal(c)
        filename = f"{c}.csv"
        df.to_csv(filename, index=False)
        print(f" Saved {filename} ({len(df)} samples, {DURATION_PER_CLASS_SEC}s)")

    print("\nSynthetic training data generation complete!")


if __name__ == "__main__":
    main()
