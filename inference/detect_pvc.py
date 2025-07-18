import wfdb
import numpy as np
import torch
import os
from preprocessing.filter_ecg import bandpass_filter
from preprocessing.pan_tompkins import detect_r_peaks
from preprocessing.segment_beats import segment_beats
from preprocessing.feature_extraction import extract_features
from model.neurofuzzy_net import NeuroFuzzyNet
from utils.helpers import load_model_config
    

def load_annotations(record_name, fs):
    """
    Load beat annotations from a record.
    
    Parameters:
    -----------
    record_name : str
        Record name without extension
    fs : int
        Sampling frequency
        
    Returns:
    --------
    tuple
        R-peak indices and labels (0: normal, 1: PVC)
    """
    try:
        ann = wfdb.rdann(record_name, 'atr')
        # Convert annotation indices to array
        r_peaks = ann.sample
        # Extract labels (N: normal, V: PVC)
        labels = np.array([1 if symbol == 'V' else 0 for symbol in ann.symbol])
        return r_peaks, labels
    except:
        return None, None

def run_pvc_detection(ecg_path, ann_path=None, model_path=None):
    """
    Run the full pipeline on a single ECG record.
    
    Parameters:
    -----------
    ecg_path : str
        Path to ECG record
    ann_path : str, optional
        Path to annotation file
    model_path : str, optional
        Path to trained model
        
    Returns:
    --------
    tuple
        Beat predictions and ground truth
    """
    # Remove file extension if present
    record_name = ecg_path.replace('.dat', '')
    
    # Load ECG record
    try:
        record = wfdb.rdrecord(record_name, channels=[0])
        ecg = record.p_signal.flatten()
        fs = record.fs
        print(f"Loaded ECG record: {record_name}, length: {len(ecg)}, fs: {fs}")
    except Exception as e:
        print(f"Error loading ECG record: {e}")
        return [], []
    
    # Pre-filter ECG
    ecg_filt = bandpass_filter(ecg, fs)
    
    # Detect R-peaks
    if ann_path:
        # Use annotated R-peaks if available
        r_peaks, ground_truth = load_annotations(record_name, fs)
        print(f"Using annotated R-peaks: {len(r_peaks)}")
    else:
        # Detect R-peaks using Pan-Tompkins
        r_peaks = detect_r_peaks(ecg_filt, fs)
        ground_truth = np.zeros(len(r_peaks))
        print(f"Detected R-peaks: {len(r_peaks)}")
    
    if len(r_peaks) == 0:
        print("No R-peaks detected or loaded.")
        return [], []
    
    # Segment beats
    beats = segment_beats(ecg_filt, r_peaks, fs)
    print(f"Segmented beats: {beats.shape}")
    
    # Extract features
    features = extract_features(beats, r_peaks, fs)
    print(f"Extracted features: {features.shape}")
    
    # Load model if path provided, else return ground truth only
    if model_path:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Load config if available
            config_path = model_path.replace(".pth", "_config.json")
            seq_len = beats.shape[1]
            if os.path.exists(config_path):
                config = load_model_config(config_path)
                seq_len = config.get("seq_len", seq_len)
            model = NeuroFuzzyNet(seq_len=seq_len).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Convert to torch tensors and move to device
            X = torch.FloatTensor(beats).to(device)
            f = torch.FloatTensor(features).to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(X, f)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int).tolist()
                
            print(f"Made predictions for {len(preds)} beats")
            return preds, ground_truth.tolist()
        except Exception as e:
            print(f"Error during prediction: {e}")
            return [], ground_truth.tolist()
    else:
        # No model provided, just return features and ground truth for training
        print("No model provided, returning features for training")
        return beats.tolist(), features.tolist(), ground_truth.tolist()