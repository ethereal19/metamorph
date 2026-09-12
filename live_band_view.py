import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import collections
import sys
import threading
import time
import signal
import numpy as np
import scipy.signal as signal_processing
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial

# ==============================================================================
# Configuration
# ==============================================================================
PORT = "COM3"
BAUD = 115200
FS = 256
WINDOW_SECONDS = 4

# Y-Axis Display Settings
# Set AUTO_SCALE = False if you want a strictly fixed Y-axis limit (FIXED_Y_LIMIT).
# This prevents open-circuit floating noise from expanding the plot when the headband is taken off.
AUTO_SCALE = True
FIXED_Y_LIMIT = 200.0   # Range used when AUTO_SCALE is False (-200 to +200)
MIN_Y_RANGE = 150.0     # Minimum floor when AUTO_SCALE is True (-150 to +150)

# Calculated rolling buffer size (1024 samples)
BUFFER_LEN = FS * WINDOW_SECONDS

# EEG Frequency Band Definitions: Name -> (low_hz, high_hz, note)
BANDS = {
    "Delta": (0.5, 4.0, ""),
    "Theta": (4.0, 8.0, ""),
    "Alpha": (8.0, 13.0, ""),
    "Beta": (13.0, 29.0, "Capped at 29 Hz per firmware"),
    "Gamma": (30.0, 45.0, "Mostly filtered out by firmware"),
}


# Pre-compute SOS (Second-Order Sections) Butterworth filters for speed & stability
def build_filters(fs, order=4):
    filters = {}
    nyq = 0.5 * fs
    for name, (lowcut, highcut, _) in BANDS.items():
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        if low < high:
            sos = signal_processing.butter(order, [low, high], btype="bandpass", output="sos")
            filters[name] = sos
        else:
            filters[name] = None
    return filters


BAND_FILTERS = build_filters(FS, order=4)


def apply_band_filter(sos, data):
    if sos is None:
        return np.zeros_like(data)
    padlen = 3 * (sos.shape[0] * 2)
    if len(data) <= padlen:
        return np.zeros_like(data)
    try:
        return signal_processing.sosfiltfilt(sos, data)
    except Exception:
        return np.zeros_like(data)


class SerialReader:
    """
    Zero-latency serial reader with automatic queue backlog draining.
    Guarantees the displayed signal is always strictly real-time.
    """
    def __init__(self, port, baud, maxlen):
        self.port = port
        self.baud = baud
        self.buffer = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.running = False
        self.ser = None
        self.thread = None

    def start(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.02)
            # Flush any stale buffer data left over from before launch
            self.ser.reset_input_buffer()
        except serial.SerialException:
            print("\n" + "=" * 65)
            print(f"ERROR: Could not open serial port '{self.port}'.")
            print("-" * 65)
            print("Common Causes & Solutions:")
            print(" 1. Arduino IDE Serial Monitor or Serial Plotter is currently open.")
            print("    --> Please close any Serial Monitor/Plotter and try again.")
            print(" 2. Arduino is unplugged or assigned to a different COM port.")
            print(" 3. Another process or script is using the port.")
            print("=" * 65 + "\n")
            sys.exit(1)

        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        raw_accumulator = ""
        while self.running:
            try:
                if self.ser and self.ser.is_open:
                    waiting = self.ser.in_waiting
                    # Flush backlog (> 2048 bytes) to stay zero-lag
                    if waiting > 2048:
                        self.ser.reset_input_buffer()
                        raw_accumulator = ""
                        time.sleep(0.01)
                        continue

                    if waiting > 0:
                        chunk = self.ser.read(waiting).decode("utf-8", errors="ignore")
                        raw_accumulator += chunk
                        if "\n" in raw_accumulator:
                            lines = raw_accumulator.split("\n")
                            raw_accumulator = lines[-1]
                            parsed_values = []
                            for line in lines[:-1]:
                                line_str = line.strip()
                                if line_str:
                                    try:
                                        parsed_values.append(float(line_str))
                                    except ValueError:
                                        pass
                            if parsed_values:
                                with self.lock:
                                    self.buffer.extend(parsed_values)
                    else:
                        time.sleep(0.003)
            except Exception:
                if not self.running:
                    break
                time.sleep(0.01)

    def get_data(self):
        with self.lock:
            return np.array(self.buffer, dtype=float)

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass


def main():
    reader = SerialReader(PORT, BAUD, maxlen=BUFFER_LEN)

    def cleanup():
        print("\nClosing serial connection and shutting down...")
        reader.stop()
        plt.close("all")

    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    reader.start()

    fig, axes = plt.subplots(len(BANDS), 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title("SYNAPSE EEG - Live Band View")
    fig.patch.set_facecolor("#121212")

    band_colors = {
        "Delta": "#00d2ff",
        "Theta": "#00f2fe",
        "Alpha": "#38ef7d",
        "Beta": "#ff4e50",
        "Gamma": "#f9d423",
    }

    lines = {}
    band_items = list(BANDS.items())

    # Template time axis
    t_axis_full = np.linspace(-WINDOW_SECONDS, 0, BUFFER_LEN)
    initial_limit = FIXED_Y_LIMIT if not AUTO_SCALE else MIN_Y_RANGE
    current_ylim = {name: initial_limit for name in BANDS}

    for i, (band_name, (low, high, note)) in enumerate(band_items):
        ax = axes[i]
        ax.set_facecolor("#1e1e1e")
        color = band_colors.get(band_name, "#ffffff")

        (line,) = ax.plot([], [], color=color, linewidth=1.2)
        lines[band_name] = line

        title_text = f"{band_name} ({low}–{high} Hz)"
        if note:
            title_text += f" — [{note}]"

        ax.set_title(title_text, color="#e0e0e0", fontsize=9, loc="left", pad=4, weight="bold")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.grid(True, color="#2c2c2c", linestyle="--", linewidth=0.5)
        ax.set_xlim(-WINDOW_SECONDS, 0)
        ax.set_ylim(-initial_limit, initial_limit)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    axes[-1].set_xlabel("Time (seconds relative to now)", color="#e0e0e0", fontsize=9)

    def update(frame):
        raw_data = reader.get_data()
        n = len(raw_data)
        if n < 32:
            return lines.values()

        # Decimate 2x for rendering if buffer is full to keep plot fast
        step = 2 if n >= 512 else 1
        sub_raw = raw_data[::step]
        t_axis = t_axis_full[-n::step]

        for i, (band_name, _) in enumerate(band_items):
            sos = BAND_FILTERS[band_name]
            filtered = apply_band_filter(sos, sub_raw)
            lines[band_name].set_data(t_axis, filtered)

            ax = axes[i]
            if AUTO_SCALE:
                max_amp = np.max(np.abs(filtered))
                target_limit = max(max_amp * 1.25, MIN_Y_RANGE)

                curr = current_ylim[band_name]
                if abs(target_limit - curr) / curr > 0.30:
                    current_ylim[band_name] = target_limit
                    ax.set_ylim(-target_limit, target_limit)
            else:
                # Fixed scale mode
                if current_ylim[band_name] != FIXED_Y_LIMIT:
                    current_ylim[band_name] = FIXED_Y_LIMIT
                    ax.set_ylim(-FIXED_Y_LIMIT, FIXED_Y_LIMIT)

        return lines.values()

    fig.tight_layout()
    fig.canvas.mpl_connect("close_event", lambda event: reader.stop())

    # 80ms interval (~12.5 FPS)
    ani = animation.FuncAnimation(fig, update, interval=80, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
