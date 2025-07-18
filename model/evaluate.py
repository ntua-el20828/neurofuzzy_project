import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils.metrics import calculate_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def evaluate_model(model, beats, features, labels, batch_size=32):
    """
    Evaluate the model on test data.
    
    Parameters:
    -----------
    model : NeuroFuzzyNet
        Trained model
    beats : np.ndarray
        Beat waveforms
    features : np.ndarray
        Extracted features
    labels : np.ndarray
        Ground truth labels
    batch_size : int
        Batch size
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Convert to torch tensors
    X = torch.FloatTensor(beats)
    f = torch.FloatTensor(features)
    y = torch.FloatTensor(labels).view(-1, 1)
    
    # Create dataset and loader
    dataset = TensorDataset(X, f, y)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    # Predictions
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, f_batch, y_batch in loader:
            X_batch, f_batch = X_batch.to(device), f_batch.to(device)
            
            outputs = model(X_batch, f_batch)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    # Concatenate batches
    if len(all_probs) == 0 or len(all_preds) == 0 or len(all_targets) == 0:
        return {
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "confusion_matrix": np.zeros((2,2)),
            "roc_auc": 0.0,
            "fpr": [0.0, 1.0],
            "tpr": [0.0, 1.0]
        }
    all_probs = np.vstack(all_probs).flatten()
    all_preds = np.vstack(all_preds).flatten()
    all_targets = np.vstack(all_targets).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    
    metrics.update({
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr
    })
    
    return metrics

def plot_evaluation(metrics):
    """
    Plot evaluation results.
    
    Parameters:
    -----------
    metrics : dict
        Evaluation metrics from evaluate_model
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot confusion matrix
    cm = metrics["confusion_matrix"]
    ax1.imshow(cm, cmap="Blues")
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    # Plot ROC curve
    ax2.plot(metrics["fpr"], metrics["tpr"], label=f"ROC curve (AUC = {metrics['roc_auc']:.3f})")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")
    
    # Add text with metrics
    plt.figtext(0.5, 0.01, 
                f"Accuracy: {metrics['accuracy']:.3f}, Sensitivity: {metrics['sensitivity']:.3f}, "
                f"Specificity: {metrics['specificity']:.3f}, F1: {metrics['f1']:.3f}",
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})
    
    plt.tight_layout()
    plt.savefig("evaluation_results.png")
    plt.show()