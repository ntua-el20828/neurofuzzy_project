import argparse
import json
from inference.detect_pvc import run_pvc_detection
from utils.metrics import compute_metrics

def main():
    parser = argparse.ArgumentParser(description="Neuro-fuzzy PVC Detector CLI")
    parser.add_argument("--ecg", type=str, required=True, help="Path to ECG record (e.g., 100.dat)")
    parser.add_argument("--ann", type=str, default=None, help="Path to annotation file (optional)")
    args = parser.parse_args()

    # Run full pipeline
    beat_preds, y_true = run_pvc_detection(args.ecg, args.ann)
    metrics = compute_metrics(beat_preds, y_true)

    print(json.dumps({
        "per_beat_predictions": beat_preds,
        "metrics": metrics
    }, indent=2))

if __name__ == "__main__":
    main()