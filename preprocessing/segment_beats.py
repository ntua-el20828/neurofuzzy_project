import numpy as np

def segment_beats(ecg: np.ndarray, r_peaks: np.ndarray, fs: int = 360) -> np.ndarray:
    """
    Segment beats around each R-peak.
    
    Parameters:
    -----------
    ecg : np.ndarray
        The input ECG signal
    r_peaks : np.ndarray
        Array of R-peak indices
    fs : int
        Sampling frequency in Hz
        
    Returns:
    --------
    beats : np.ndarray
        Array of segmented beats with shape (n_beats, window_len)
    """
    pre = int(0.1 * fs)   # 100 ms before
    post = int(0.2 * fs)  # 200 ms after
    window_len = pre + post
    
    beats = []
    for r in r_peaks:
        start = max(r - pre, 0)
        end = min(r + post, len(ecg))
        beat = ecg[start:end]
        
        # Pad if needed (for beats at the beginning or end of the signal)
        if len(beat) < window_len:
            pad_width = window_len - len(beat)
            if start == 0:  # Beginning of signal
                beat = np.pad(beat, (0, pad_width), mode='edge')
            else:  # End of signal
                beat = np.pad(beat, (pad_width, 0), mode='edge')
                
        beats.append(beat)
        
    return np.stack(beats) if beats else np.empty((0, window_len))