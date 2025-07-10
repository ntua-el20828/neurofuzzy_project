import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def detect_r_peaks(ecg: np.ndarray, fs: int = 360) -> np.ndarray:
    """
    Pan-Tompkins R-peak detection algorithm.
    
    Parameters:
    -----------
    ecg : np.ndarray
        The input ECG signal
    fs : int
        Sampling frequency in Hz
        
    Returns:
    --------
    r_peaks : np.ndarray
        Array of R-peak indices
    """
    # 1. Bandpass filter (5-15 Hz)
    nyq = 0.5 * fs
    low, high = 5 / nyq, 15 / nyq
    b, a = butter(3, [low, high], btype='band')
    filtered = filtfilt(b, a, ecg)
    
    # 2. Derivative
    derivative = np.diff(filtered)
    derivative = np.append(derivative, derivative[-1])
    
    # 3. Squaring
    squared = derivative ** 2
    
    # 4. Moving window integration
    window_size = int(0.15 * fs)  # 150 ms window
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    # 5. Find peaks with adaptive thresholding
    r_peaks, _ = find_peaks(integrated, distance=int(0.2*fs))
    
    # TODO: Implement adaptive thresholding and backtracking to exact R-peak
    
    return r_peaks