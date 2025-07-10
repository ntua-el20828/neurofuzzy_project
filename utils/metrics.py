import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_pred, y_true):
    """
    Calculate classification metrics.
    
    Parameters:
    -----------
    y_pred : np.ndarray
        Predicted labels (0: normal, 1: PVC)
    y_true : np.ndarray
        Ground truth labels
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    # Ensure binary classification
    y_pred = np.asarray(y_pred).astype(int)
    y_true = np.asarray(y_true).astype(int)
    
    # Handle empty arrays
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0
        }
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # If there are no positive examples, handle gracefully
    if np.sum(y_true) == 0:
        sensitivity = 1.0 if np.sum(y_pred) == 0 else 0.0
        precision = 1.0 if np.sum(y_pred) == 0 else 0.0
    else:
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
    
    # If there are no negative examples, handle gracefully
    if np.sum(1 - y_true) == 0:
        specificity = 1.0 if np.sum(1 - y_pred) == 0 else 0.0
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # F1 score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1)
    }

def compute_metrics(preds, targets):
    """
    Wrapper for calculate_metrics (compatibility with CLI).
    """
    return calculate_metrics(preds, targets)