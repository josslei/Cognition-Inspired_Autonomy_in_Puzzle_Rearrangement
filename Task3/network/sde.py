import torch
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def marginal_prob_std(t, sigma: float, device=DEVICE):
    t = torch.as_tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2 / np.log(sigma))

def diffusion_coeff(t, sigma: float, device=DEVICE):
    return torch.as_tensor(sigma ** t, device=device)