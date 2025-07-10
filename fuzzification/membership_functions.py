import torch
import torch.nn as nn

class GaussianMF(nn.Module):
    def __init__(self, mu: float, sigma: float, learnable: bool = True):
        """
        Gaussian Membership Function.
        
        Parameters:
        -----------
        mu : float
            Center of the Gaussian
        sigma : float
            Width of the Gaussian
        learnable : bool
            Whether parameters should be learnable
        """
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32), requires_grad=learnable)
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float32), requires_grad=learnable)

    def forward(self, x):
        """
        Compute membership values.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input values, shape (batch_size,)
            
        Returns:
        --------
        torch.Tensor
            Membership values, shape (batch_size,)
        """
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)

class FuzzyMembershipLayer(nn.Module):
    def __init__(self):
        """Initialize membership functions for all features."""
        super().__init__()
        
        # QRS duration membership functions
        self.qrs_narrow = GaussianMF(mu=80.0, sigma=15.0)
        self.qrs_normal = GaussianMF(mu=100.0, sigma=15.0)
        self.qrs_wide = GaussianMF(mu=130.0, sigma=20.0)
        
        # RR interval previous membership functions
        self.rr_prev_short = GaussianMF(mu=600.0, sigma=100.0)
        self.rr_prev_normal = GaussianMF(mu=800.0, sigma=100.0)
        self.rr_prev_long = GaussianMF(mu=1000.0, sigma=150.0)
        
        # RR interval post membership functions
        self.rr_post_short = GaussianMF(mu=600.0, sigma=100.0)
        self.rr_post_normal = GaussianMF(mu=800.0, sigma=100.0)
        self.rr_post_long = GaussianMF(mu=1000.0, sigma=150.0)
        
        # P absent membership function (already binary, but use MF for consistency)
        self.p_absent_false = GaussianMF(mu=0.0, sigma=0.1)
        self.p_absent_true = GaussianMF(mu=1.0, sigma=0.1)
        
        # T wave inversion membership function
        self.t_inv_false = GaussianMF(mu=0.0, sigma=0.1)
        self.t_inv_true = GaussianMF(mu=1.0, sigma=0.1)
        
        # PR interval membership functions
        self.pr_short = GaussianMF(mu=120.0, sigma=20.0)
        self.pr_normal = GaussianMF(mu=160.0, sigma=20.0)
        self.pr_long = GaussianMF(mu=200.0, sigma=30.0)
        
    def forward(self, x):
        """
        Compute membership values for all features.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input features, shape (batch_size, 6)
            
        Returns:
        --------
        dict
            Dictionary of membership values
        """
        qrs_dur, rr_prev, rr_post, p_absent, t_inv, pr_int = (
            x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        )
        
        return {
            "qrs": {
                "narrow": self.qrs_narrow(qrs_dur),
                "normal": self.qrs_normal(qrs_dur),
                "wide": self.qrs_wide(qrs_dur)
            },
            "rr_prev": {
                "short": self.rr_prev_short(rr_prev),
                "normal": self.rr_prev_normal(rr_prev),
                "long": self.rr_prev_long(rr_prev)
            },
            "rr_post": {
                "short": self.rr_post_short(rr_post),
                "normal": self.rr_post_normal(rr_post),
                "long": self.rr_post_long(rr_post)
            },
            "p_absent": {
                "false": self.p_absent_false(p_absent),
                "true": self.p_absent_true(p_absent)
            },
            "t_inv": {
                "false": self.t_inv_false(t_inv),
                "true": self.t_inv_true(t_inv)
            },
            "pr_int": {
                "short": self.pr_short(pr_int),
                "normal": self.pr_normal(pr_int),
                "long": self.pr_long(pr_int)
            }
        }