import torch
import torch.nn as nn

class FuzzyRuleBase(nn.Module):
    def __init__(self):
        """Initialize the fuzzy rule base with learnable weights."""
        super().__init__()
        # Initialize rule weights with suggested values
        self.rule_weights = nn.Parameter(
            torch.tensor([1.0, 1.0, 1.0, 0.6, 0.6, 0.2], dtype=torch.float32)
        )
        
    def forward(self, mf_outputs):
        """
        Apply fuzzy rules using the given membership function outputs.
        
        Parameters:
        -----------
        mf_outputs : dict
            Dictionary of membership values from FuzzyMembershipLayer
            
        Returns:
        --------
        torch.Tensor
            Fuzzy output for PVC confidence (batch_size, 1)
        """
        batch_size = next(iter(mf_outputs.values()))["normal"].shape[0]
        rule_activations = torch.zeros((batch_size, 6), device=self.rule_weights.device)
        
        # Rule 1: IF QRS is wide AND RR_prev is short THEN PVC_conf = High
        rule_activations[:, 0] = torch.min(
            mf_outputs["qrs"]["wide"],
            mf_outputs["rr_prev"]["short"]
        )
        
        # Rule 2: IF QRS is wide AND P_absent is true THEN PVC_conf = High
        rule_activations[:, 1] = torch.min(
            mf_outputs["qrs"]["wide"],
            mf_outputs["p_absent"]["true"]
        )
        
        # Rule 3: IF QRS is wide AND T_inv is true THEN PVC_conf = High
        rule_activations[:, 2] = torch.min(
            mf_outputs["qrs"]["wide"],
            mf_outputs["t_inv"]["true"]
        )
        
        # Rule 4: IF QRS is wide AND RR_post is long THEN PVC_conf = Medium
        rule_activations[:, 3] = torch.min(
            mf_outputs["qrs"]["wide"],
            mf_outputs["rr_post"]["long"]
        )
        
        # Rule 5: IF QRS is normal AND RR_prev is short AND P_absent is true THEN PVC_conf = Medium
        rule_activations[:, 4] = torch.min(
            torch.min(
                mf_outputs["qrs"]["normal"],
                mf_outputs["rr_prev"]["short"]
            ),
            mf_outputs["p_absent"]["true"]
        )
        
        # Rule 6: ELSE PVC_conf = Low (default rule)
        # Compute negation of all other rule activations
        rule_activations[:, 5] = (1 - torch.max(rule_activations[:, :5], dim=1)[0])
        
        # Weight the rule activations
        weighted_activations = rule_activations * self.rule_weights
        
        # Defuzzify using weighted average (simplification of center of gravity)
        # Map rules 0-2 to high (1.0), 3-4 to medium (0.5), and 5 to low (0.0)
        rule_outputs = torch.zeros_like(weighted_activations)
        rule_outputs[:, :3] = 1.0  # High confidence rules
        rule_outputs[:, 3:5] = 0.5  # Medium confidence rules
        rule_outputs[:, 5] = 0.0    # Low confidence rule
        
        # Final defuzzified output (weighted average)
        numerator = torch.sum(weighted_activations * rule_outputs, dim=1, keepdim=True)
        denominator = torch.sum(weighted_activations, dim=1, keepdim=True) + 1e-10
        pvc_score = numerator / denominator
        
        return pvc_score