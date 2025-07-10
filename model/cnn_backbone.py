import torch
import torch.nn as nn

class BeatCNN(nn.Module):
    def __init__(self, in_channels: int = 1, seq_len: int = 108):
        """
        CNN backbone for ECG beat waveform feature extraction.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels (1 for single-lead ECG)
        seq_len : int
            Length of input sequence (default: 108 samples = 300ms at 360Hz)
        """
        super().__init__()
        self.conv = nn.Sequential(
            # First convolutional block
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second convolutional block
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third convolutional block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        self.flatten = nn.Flatten()
        self.out_features = 64

    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input beat waveform, shape (batch_size, 1, seq_len)
            
        Returns:
        --------
        torch.Tensor
            Beat embedding, shape (batch_size, out_features)
        """
        # Ensure x is properly shaped (batch, channels, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.conv(x)
        x = self.flatten(x)
        return x