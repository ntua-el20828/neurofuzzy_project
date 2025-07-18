import numpy as np
from scipy.signal import find_peaks

def safe_mean(arr):
    return np.mean(arr) if arr.size > 0 else 0.0

def extract_features(beats: np.ndarray, r_peaks: np.ndarray, fs: int = 360) -> np.ndarray:
    """
    Extract six features per beat:
    QRS_dur, RR_prev, RR_post, P_absent, T_inv, PR_int
    
    Parameters:
    -----------
    beats : np.ndarray
        Array of segmented beats with shape (n_beats, window_len)
    r_peaks : np.ndarray
        Array of R-peak indices
    fs : int
        Sampling frequency in Hz
        
    Returns:
    --------
    features : np.ndarray
        Array of features with shape (n_beats, 6)
    """
    n_beats = beats.shape[0]
    features = np.zeros((n_beats, 6))
    
    # Pre-compute RR intervals
    rr_intervals = np.diff(r_peaks)
    rr_intervals = np.insert(rr_intervals, 0, rr_intervals[0])  # Duplicate first for first beat
    rr_post = np.append(rr_intervals[1:], rr_intervals[-1])     # Duplicate last for last beat
    
    pre_window = int(0.1 * fs)  # 100ms before R-peak
    
    for i, beat in enumerate(beats):
        if beat.size == 0:
            continue  # or append np.zeros(shape) as a placeholder
        # 1. QRS duration
        # Simple method: zero-crossing points around R-peak
        mid_point = pre_window  # R-peak position in the beat
        
        # Find QRS onset (before R-peak)
        qrs_onset = mid_point
        for j in range(mid_point, max(0, mid_point-50), -1):
            if beat[j] * beat[j-1] <= 0:  # Zero crossing
                qrs_onset = j
                break
                
        # Find QRS offset (after R-peak)
        qrs_offset = mid_point
        for j in range(mid_point, min(len(beat)-1, mid_point+70)):
            if beat[j] * beat[j+1] <= 0:  # Zero crossing
                qrs_offset = j
                break
                
        qrs_duration = (qrs_offset - qrs_onset) * 1000 / fs  # in ms
        features[i, 0] = qrs_duration
        
        # 2. RR_prev
        features[i, 1] = rr_intervals[i] * 1000 / fs  # in ms
        
        # 3. RR_post
        features[i, 2] = rr_post[i] * 1000 / fs  # in ms
        
        # 4. P_absent (P wave detection in segment before QRS)
        p_segment = beat[max(0, qrs_onset-int(0.2*fs)):qrs_onset]
        p_peaks, _ = find_peaks(np.abs(p_segment), height=0.1*np.max(np.abs(beat)))
        features[i, 3] = 1.0 if len(p_peaks) == 0 else 0.0  # 1 if P is absent
        
        # 5. T-wave polarity (compared to QRS)
        t_segment = beat[qrs_offset:min(len(beat), qrs_offset+int(0.3*fs))]
        if len(t_segment) > 0:
            qrs_sign = np.sign(safe_mean(beat[qrs_onset:qrs_offset]))
            t_sign = np.sign(safe_mean(t_segment))
            features[i, 4] = 1.0 if qrs_sign * t_sign < 0 else 0.0  # 1 if opposite polarity
        else:
            features[i, 4] = 0.0
            
        # 6. PR interval
        if features[i, 3] == 0.0 and len(p_peaks) > 0:  # P-wave exists
            p_peak = p_peaks[-1]  # Last peak in the P segment
            pr_interval = (qrs_onset - p_peak) * 1000 / fs  # in ms
            features[i, 5] = pr_interval
        else:
            features[i, 5] = 0.0  # No P-wave, no PR interval
    
    return features