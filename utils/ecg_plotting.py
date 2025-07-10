import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import wfdb

def plot_ecg_with_annotations(ecg, fs, r_peaks=None, pvc_indices=None, window_sec=10):
    """
    Plot ECG signal with annotations.
    
    Parameters:
    -----------
    ecg : np.ndarray
        ECG signal
    fs : int
        Sampling frequency
    r_peaks : np.ndarray, optional
        R-peak indices
    pvc_indices : list, optional
        Indices of beats classified as PVCs
    window_sec : int
        Window size in seconds
    """
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    
    # Set up a grid with 1 row and 1 column
    gs = GridSpec(1, 1)
    
    # Create axes
    ax = fig.add_subplot(gs[0, 0])
    
    # Calculate time vector
    time = np.arange(len(ecg)) / fs
    
    # Plot ECG signal
    ax.plot(time, ecg, 'b', linewidth=1, label='ECG')
    
    # Plot R-peaks if provided
    if r_peaks is not None:
        r_times = r_peaks / fs
        ax.plot(r_times, ecg[r_peaks], 'ro', markersize=6, label='R-peaks')
    
    # Mark PVCs if provided
    if pvc_indices is not None and r_peaks is not None:
        pvc_locs = r_peaks[pvc_indices]
        pvc_times = pvc_locs / fs
        ax.plot(pvc_times, ecg[pvc_locs], 'go', markersize=8, label='PVCs')
    
    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title('ECG Signal with Annotations')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set initial view to first window_sec seconds
    ax.set_xlim(0, window_sec)
    
    # Add grid
    ax.grid(True)
    
    # Add navigation toolbar for interactive zooming
    plt.tight_layout()
    
    return fig, ax

def plot_beat_examples(beats, labels, title="ECG Beat Examples"):
    """
    Plot examples of normal and PVC beats.
    
    Parameters:
    -----------
    beats : np.ndarray
        Array of segmented beats
    labels : np.ndarray
        Beat labels (0: normal, 1: PVC)
    title : str
        Plot title
    """
    # Find indices of normal and PVC beats
    normal_idx = np.where(labels == 0)[0]
    pvc_idx = np.where(labels == 1)[0]
    
    # Take at most 5 examples of each
    normal_idx = normal_idx[:min(5, len(normal_idx))]
    pvc_idx = pvc_idx[:min(5, len(pvc_idx))]
    
    fig, axs = plt.subplots(2, max(len(normal_idx), len(pvc_idx)), figsize=(15, 6))
    
    # Plot normal beats
    for i, idx in enumerate(normal_idx):
        axs[0, i].plot(beats[idx])
        axs[0, i].set_title(f"Normal #{i+1}")
        axs[0, i].axvline(x=beats.shape[1]//3, color='r', linestyle='--')  # Mark R-peak
    
    # Plot PVC beats
    for i, idx in enumerate(pvc_idx):
        axs[1, i].plot(beats[idx])
        axs[1, i].set_title(f"PVC #{i+1}")
        axs[1, i].axvline(x=beats.shape[1]//3, color='r', linestyle='--')  # Mark R-peak
    
    # Hide unused subplots
    for i in range(len(normal_idx), axs.shape[1]):
        axs[0, i].axis('off')
    for i in range(len(pvc_idx), axs.shape[1]):
        axs[1, i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig