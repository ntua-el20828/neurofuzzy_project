import argparse
import json
import os
import logging
from inference.detect_pvc import run_pvc_detection
from utils.metrics import compute_metrics
from utils.helpers import set_seed
from utils.ecg_plotting import plot_ecg_with_annotations

def main():
    """
    Command-line interface for the Neuro-fuzzy PVC Detector.
    """
    parser = argparse.ArgumentParser(description="Neuro-fuzzy PVC Detector CLI")
    parser.add_argument("--ecg", type=str, required=True, help="Path to ECG record (e.g., 100.dat)")
    parser.add_argument("--ann", type=str, default=None, help="Path to annotation file (optional)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (e.g., best_model.pth)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON (optional)")
    parser.add_argument("--plot", action="store_true", help="Plot ECG with detected PVCs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Set random seed
    set_seed(args.seed)

    # Validate file paths
    if not os.path.isfile(args.ecg):
        raise FileNotFoundError(f"ECG file not found: {args.ecg}")
    if args.ann and not os.path.isfile(args.ann):
        raise FileNotFoundError(f"Annotation file not found: {args.ann}")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    try:
        # Run full pipeline
        beat_preds, y_true = run_pvc_detection(args.ecg, args.ann, args.model)
        metrics = compute_metrics(beat_preds, y_true)

        results = {
            "per_beat_predictions": beat_preds,
            "metrics": metrics
        }
        print(json.dumps(results, indent=2))
        logging.info(f"Accuracy: {metrics['accuracy']:.3f} | Sensitivity: {metrics['sensitivity']:.3f} | "
                     f"Specificity: {metrics['specificity']:.3f} | F1: {metrics['f1']:.3f}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {args.output}")

        if args.plot:
            # Optional: plot ECG with detected PVCs
            import wfdb
            record_name = args.ecg.replace('.dat', '')
            record = wfdb.rdrecord(record_name, channels=[0])
            ecg = record.p_signal.flatten()
            fs = record.fs
            from preprocessing.pan_tompkins import detect_r_peaks
            r_peaks = detect_r_peaks(ecg, fs)
            pvc_indices = [i for i, p in enumerate(beat_preds) if p == 1]
            fig, ax = plot_ecg_with_annotations(ecg, fs, r_peaks, pvc_indices)
            fig.savefig("ecg_pvc_detection.png")
            logging.info("ECG plot with PVCs saved as ecg_pvc_detection.png")

    except Exception as e:
        logging.error(f"Error during inference: {e}")

if __name__ == "__main__":
    main()