import torch
import torch.nn as nn
from fuzzification.fuzzy_layer import NeuroFuzzyLayer
from .cnn_backbone import BeatCNN

class NeuroFuzzyNet(nn.Module):
    def __init__(self, seq_len: int = 108, class_weights: torch.Tensor = None):
        """
        Combined neural-fuzzy network for PVC detection.
        
        Parameters:
        -----------
        seq_len : int
            Length of input sequence
        class_weights : torch.Tensor, optional
            Class weights for handling imbalance
        """
        super().__init__()
        self.cnn = BeatCNN(seq_len=seq_len)
        self.fuzzy = NeuroFuzzyLayer()
        
        # Concatenated features from CNN and fuzzy layer
        concat_features = self.cnn.out_features + 1
        
        # Dense layer for final prediction
        self.classifier = nn.Sequential(
            nn.Linear(concat_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
        self.class_weights = class_weights

    def forward(self, beat_waveform, features):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        beat_waveform : torch.Tensor
            Beat waveform, shape (batch_size, seq_len)
        features : torch.Tensor
            Extracted features, shape (batch_size, 6)
            
        Returns:
        --------
        torch.Tensor
            PVC detection logits, shape (batch_size, 1)
        """
        # Extract CNN features from waveform
        cnn_emb = self.cnn(beat_waveform)
        
        # Get fuzzy confidence score
        pvc_score = self.fuzzy(features)
        
        # Concatenate features
        combined = torch.cat([cnn_emb, pvc_score], dim=1)
        
        # Final classification
        return self.classifier(combined)
    
    def get_loss_fn(self):
        """Return appropriate loss function."""
        if self.class_weights is not None:
            return nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        else:
            return nn.BCEWithLogitsLoss()