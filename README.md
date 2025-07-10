# Fuzzy Neural PVC Detector

A hybrid neuro-fuzzy approach for detecting Premature Ventricular Contractions (PVCs) in ECG signals.

## Overview

This project implements a PVC detection system that combines:
- Signal processing techniques for ECG preprocessing
- Feature extraction based on cardiological characteristics
- Fuzzy logic for domain knowledge incorporation
- Neural networks for waveform pattern recognition

## Pipeline

1. Load ECG data (MIT-BIH Arrhythmia Database)
2. Preprocess ECG signals (bandpass filter 0.5-45 Hz)
3. Detect R-peaks using Pan-Tompkins algorithm
4. Segment individual heartbeats
5. Extract six critical features:
   - QRS duration
   - Previous RR interval
   - Post RR interval
   - P-wave absence
   - T-wave inversion
   - PR interval
6. Apply fuzzy membership functions
7. Process through fuzzy rule base
8. Combine with neural network for final classification

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running Detection

```bash
python main.py --ecg data/100.dat
```

### Training

```python
from preprocessing.filter_ecg import bandpass_filter
from preprocessing.pan_tompkins import detect_r_peaks
from preprocessing.segment_beats import segment_beats
from preprocessing.feature_extraction import extract_features
from model.train import train_model

# Load and preprocess data
# ... (load ECG signals and annotations)

# Train model
model, metrics = train_model(beats, features, labels, epochs=100)
```

## Project Structure

- `preprocessing/`: Signal processing modules
- `fuzzification/`: Fuzzy logic components
- `model/`: Neural network architecture
- `inference/`: Detection pipeline
- `utils/`: Helper functions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- wfdb
- scikit-fuzzy
- NumPy, SciPy
- Matplotlib
- scikit-learn

## Performance

The model is evaluated on MIT-BIH Arrhythmia Database using:
- Accuracy
- Sensitivity (Recall)
- Specificity
- F1 Score