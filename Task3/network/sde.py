import torch
import numpy as np


def marginal_prob_std(t: torch.Tensor, sigma: float):
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2 / np.log(sigma))

def diffusion_coeff(t: torch.Tensor, sigma: float):
    return sigma ** t