import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(ecg: np.ndarray, fs: int = 360, low: float = 0.5, high: float = 45.0) -> np.ndarray:
    """Bandpass filter for ECG signal."""
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, ecg)