import torch
import torch.nn as nn
from .membership_functions import FuzzyMembershipLayer
from .fuzzy_rules import FuzzyRuleBase

class NeuroFuzzyLayer(nn.Module):
    def __init__(self):
        """Initialize the neuro-fuzzy layer with membership functions and rules."""
        super().__init__()
        self.membership_layer = FuzzyMembershipLayer()
        self.rule_base = FuzzyRuleBase()

    def forward(self, x):
        """
        Forward pass through the neuro-fuzzy layer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features, shape (batch_size, 6)
            
        Returns:
        --------
        torch.Tensor
            PVC confidence score, shape (batch_size, 1)
        """
        # Fuzzify inputs
        mf_outputs = self.membership_layer(x)
        
        # Apply fuzzy rules
        pvc_score = self.rule_base(mf_outputs)
        
        return pvc_score