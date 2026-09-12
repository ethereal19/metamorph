import numpy as np
import pandas as pd
from scipy.signal import welch

# Frequency Band Definitions
FS = 256
BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 29.0),  # Capped at 29 Hz per firmware constraint
}

FEATURE_NAMES = [
    "alpha_power",
    "beta_power",
    "delta_power",
    "theta_power",
    "alpha_beta_ratio",
    "rel_alpha",
    "rel_beta",
    "total_power",
    "alpha_std",
    "beta_std",
    "alpha_peak",
    "beta_peak",
]


def check_artifact(window_data, baseline_median=None, multiplier=10.0):
    """
    Artifact Rejection Step:
    Rejects windows where maximum signal amplitude exceeds ~10x the median baseline of the recording
    (e.g., severe electrode pops, motion, or muscle artifacts).
    """
    if len(window_data) == 0:
        return True
    
    abs_signal = np.abs(window_data)
    max_val = np.max(abs_signal)
    
    if baseline_median is not None and baseline_median > 1e-4:
        return max_val > multiplier * baseline_median

    median_val = np.median(abs_signal)
    if median_val < 1e-4:
        return False
        
    return max_val > 15.0 * median_val


def compute_band_power(freqs, psd, low_hz, high_hz):
    """
    Integrates PSD over a specific frequency range [low_hz, high_hz].
    """
    idx = np.logical_and(freqs >= low_hz, freqs <= high_hz)
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], freqs[idx])


def extract_window_features(window_df, fs=FS, alpha_baseline=None, beta_baseline=None, artifact_multiplier=10.0):
    """
    Extracts Welch band power features from a single window DataFrame containing
    pre-filtered band signals: ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'].
    """
    alpha_raw = window_df["Alpha"].values if "Alpha" in window_df else np.zeros(len(window_df))
    beta_raw = window_df["Beta"].values if "Beta" in window_df else np.zeros(len(window_df))
    
    # Artifact Rejection Check
    if check_artifact(alpha_raw, alpha_baseline, artifact_multiplier) or check_artifact(beta_raw, beta_baseline, artifact_multiplier):
        return None  # Window rejected

    n_samples = len(window_df)
    nperseg = min(n_samples, fs)
    if nperseg < 16:
        return None

    # Welch PSD Computation
    freqs_a, psd_a = welch(alpha_raw, fs=fs, nperseg=nperseg)
    freqs_b, psd_b = welch(beta_raw, fs=fs, nperseg=nperseg)

    alpha_power = compute_band_power(freqs_a, psd_a, BANDS["Alpha"][0], BANDS["Alpha"][1])
    beta_power = compute_band_power(freqs_b, psd_b, BANDS["Beta"][0], BANDS["Beta"][1])

    if "Delta" in window_df:
        freqs_d, psd_d = welch(window_df["Delta"].values, fs=fs, nperseg=nperseg)
        delta_power = compute_band_power(freqs_d, psd_d, BANDS["Delta"][0], BANDS["Delta"][1])
    else:
        delta_power = 0.0

    if "Theta" in window_df:
        freqs_t, psd_t = welch(window_df["Theta"].values, fs=fs, nperseg=nperseg)
        theta_power = compute_band_power(freqs_t, psd_t, BANDS["Theta"][0], BANDS["Theta"][1])
    else:
        theta_power = 0.0

    # Derived Features
    total_power = alpha_power + beta_power + delta_power + theta_power + 1e-8
    alpha_beta_ratio = alpha_power / (beta_power + 1e-8)
    rel_alpha = alpha_power / total_power
    rel_beta = beta_power / total_power

    alpha_std = np.std(alpha_raw)
    beta_std = np.std(beta_raw)
    alpha_peak = np.max(np.abs(alpha_raw)) if len(alpha_raw) > 0 else 0.0
    beta_peak = np.max(np.abs(beta_raw)) if len(beta_raw) > 0 else 0.0

    return {
        "alpha_power": alpha_power,
        "beta_power": beta_power,
        "delta_power": delta_power,
        "theta_power": theta_power,
        "alpha_beta_ratio": alpha_beta_ratio,
        "rel_alpha": rel_alpha,
        "rel_beta": rel_beta,
        "total_power": total_power,
        "alpha_std": alpha_std,
        "beta_std": beta_std,
        "alpha_peak": alpha_peak,
        "beta_peak": beta_peak,
    }


def process_signal_sliding_window(df, fs=FS, window_sec=1.5, overlap_sec=0.75, artifact_multiplier=10.0):
    """
    Processes an entire CSV DataFrame using sliding windows with overlap.
    """
    window_samples = int(window_sec * fs)
    step_samples = int((window_sec - overlap_sec) * fs)

    alpha_baseline = np.median(np.abs(df["Alpha"].values)) if "Alpha" in df else None
    beta_baseline = np.median(np.abs(df["Beta"].values)) if "Beta" in df else None

    features_list = []
    total_len = len(df)

    for start_idx in range(0, total_len - window_samples + 1, step_samples):
        window_df = df.iloc[start_idx : start_idx + window_samples]
        feat = extract_window_features(
            window_df,
            fs=fs,
            alpha_baseline=alpha_baseline,
            beta_baseline=beta_baseline,
            artifact_multiplier=artifact_multiplier,
        )
        if feat is not None:
            features_list.append(feat)

    return pd.DataFrame(features_list)
