import os
import torch
import numpy as np
import random
import wfdb
import json

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_model(model, path):
    """Save model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, device=None):
    """Load model from disk."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def save_model_config(config, path):
    """Save model config (e.g., seq_len) as JSON."""
    with open(path, "w") as f:
        json.dump(config, f)

def load_model_config(path):
    """Load model config from JSON."""
    with open(path, "r") as f:
        return json.load(f)

def get_available_records():
    """Get list of available MIT-BIH records."""
    try:
        # This assumes data is in the standard WFDB path
        records = wfdb.get_record_list('mitdb')
        return records
    except:
        # Fallback to common record numbers
        return [f"{i:03d}" for i in range(100, 235)]