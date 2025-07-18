import os
import wfdb
import numpy as np
from tqdm import tqdm
import time

from preprocessing.filter_ecg import bandpass_filter
from preprocessing.pan_tompkins import detect_r_peaks
from preprocessing.segment_beats import segment_beats
from preprocessing.feature_extraction import extract_features
from model.train import train_model

DATA_DIR = "training data"

def get_record_names(data_dir):
    """Return list of record names (without extension) in the data directory."""
    records = []
    for fname in os.listdir(data_dir):
        if fname.endswith('.dat'):
            records.append(os.path.splitext(fname)[0])
    return sorted(list(set(records)))

def load_record(record_name, data_dir):
    """Load ECG and annotation for a record."""
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path, channels=[0])
    ecg = record.p_signal.flatten()
    fs = record.fs
    ann = wfdb.rdann(record_path, 'atr')
    r_peaks = ann.sample
    labels = np.array([1 if s == 'V' else 0 for s in ann.symbol])
    return ecg, fs, r_peaks, labels

def main():
    all_beats = []
    all_features = []
    all_labels = []

    record_names = get_record_names(DATA_DIR)
    print(f"Found {len(record_names)} records: {record_names}")

    for record_name in tqdm(record_names, desc="Extracting beats/features"):
        try:
            ecg, fs, r_peaks, labels = load_record(record_name, DATA_DIR)
            ecg_filt = bandpass_filter(ecg, fs)
            beats = segment_beats(ecg_filt, r_peaks, fs)
            features = extract_features(beats, r_peaks, fs)
            # Align labels to beats (should be same length)
            min_len = min(len(beats), len(labels))
            all_beats.append(beats[:min_len])
            all_features.append(features[:min_len])
            all_labels.append(labels[:min_len])
        except Exception as e:
            print(f"Error processing {record_name}: {e}")

    # Concatenate all data
    all_beats = np.vstack(all_beats)
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    print(f"Total beats: {all_beats.shape[0]}")

    # Train the model with progress bar and time estimation
    print("Starting training...")
    start_time = time.time()
    model, metrics = train_model(
        all_beats, all_features, all_labels,
        epochs=100, batch_size=32, lr=0.0001, patience=7, weight_decay=1e-4
    )
    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.1f} minutes.")
    print("Final metrics:", metrics)

if __name__ == "__main__":
    main()