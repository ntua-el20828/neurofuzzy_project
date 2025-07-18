import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from .neurofuzzy_net import NeuroFuzzyNet
from utils.metrics import calculate_metrics
from utils.helpers import save_model, save_model_config


def prepare_data(beats, features, labels, test_size=0.2):
    """
    Prepare data for training.
    
    Parameters:
    -----------
    beats : np.ndarray
        Beat waveforms
    features : np.ndarray
        Extracted features
    labels : np.ndarray
        Labels (0: normal, 1: PVC)
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        Train and test data
    """
    X_train, X_test, f_train, f_test, y_train, y_test = train_test_split(
        beats, features, labels, test_size=test_size, stratify=labels
    )
    
    # Convert to torch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    f_train = torch.FloatTensor(f_train)
    f_test = torch.FloatTensor(f_test)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    y_test = torch.FloatTensor(y_test).view(-1, 1)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, f_train, y_train)
    test_dataset = TensorDataset(X_test, f_test, y_test)
    
    return train_dataset, test_dataset

def train_model(beats, features, labels, epochs=100, batch_size=32, lr=0.001):
    """
    Train the NeuroFuzzyNet model.
    
    Parameters:
    -----------
    beats : np.ndarray
        Beat waveforms
    features : np.ndarray
        Extracted features
    labels : np.ndarray
        Labels (0: normal, 1: PVC)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate
        
    Returns:
    --------
    tuple
        Trained model and metrics
    """
    # Prepare data
    train_dataset, test_dataset = prepare_data(beats, features, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Compute class weights for imbalanced data
    pos_weight = torch.tensor([(len(labels) - labels.sum()) / max(labels.sum(), 1)])
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuroFuzzyNet(seq_len=beats.shape[1], class_weights=pos_weight).to(device)
    
    # Loss and optimizer
    loss_fn = model.get_loss_fn()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    best_config = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, f_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, f_batch, y_batch = X_batch.to(device), f_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch, f_batch)
            loss = loss_fn(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, f_batch, y_batch in test_loader:
                X_batch, f_batch, y_batch = X_batch.to(device), f_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch, f_batch)
                loss = loss_fn(outputs, y_batch)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(test_loader)
        scheduler.step(val_loss)
        
        # Compute metrics
        all_preds = np.vstack(all_preds).flatten()
        all_targets = np.vstack(all_targets).flatten()
        metrics = calculate_metrics(all_preds, all_targets)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}, Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            best_config = {"seq_len": beats.shape[1]}
            save_model(model, "best_model.pth")
            save_model_config(best_config, "best_model_config.json")

    
    # Load best model
    model.load_state_dict(best_model)
    
    return model, metrics