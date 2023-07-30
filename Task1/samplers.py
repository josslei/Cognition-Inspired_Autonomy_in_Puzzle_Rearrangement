import copy
from typing import List
from functools import partial

import numpy as np
import tqdm

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPEED_UP_COEF: float = 10.0
RAND_STEP_COEF: float = 1.0 / 1000.0


def euler_maruyama_sampler(model: torch.nn.Module,
                           diffusion_coeff_fn: partial[torch.Tensor],
                           omega: torch.Tensor,
                           omega_sequences: List[List[np.ndarray]],
                           t_final: float = 0.05,
                           num_steps: int = 500,
                           eps: float = 1e-5,
                           device=DEVICE) -> List[np.ndarray]:
    """The Euler Maruyama sampler

    Args:
        model (torch.nn.Module): _description_
        omega (torch.Tensor): _description_
        omega_sequences (List[List[np.ndarray]]): _description_
        t_final (float, optional): _description_. Defaults to 0.05.
        num_steps (int, optional): _description_. Defaults to 500.
        eps (float, optional): _description_. Defaults to 1e-5.
        device (_type_, optional): _description_. Defaults to DEVICE.

    Returns:
        List[np.ndarray]: Result of sampling; omega of every rearranged tangram in a batch.
    """
    # Define steps
    time_steps = torch.linspace(t_final, eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    # Rearrange tangram pieces
    print('Rearranging...')
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step: torch.Tensor = torch.tensor([1.]).view(-1, 1).to(device)
            batch_time_step *= time_step
            g = diffusion_coeff_fn(batch_time_step)
            # Iterate one step
            omega += SPEED_UP_COEF * (g**2) * __inference(model, omega, batch_time_step) * step_size
            omega += SPEED_UP_COEF * torch.sqrt(step_size) * g * (RAND_STEP_COEF * torch.randn_like(omega))
            # Record process
            append_omega_batch(omega_sequences, omega)
    return [o.cpu().numpy() for o in omega]


def append_omega_batch(omega_sequences: List[List[np.ndarray]], batch: torch.Tensor) -> None:
    batch_size: int = batch.shape[0]
    for i, omega in enumerate(batch):
        if len(omega_sequences) <= i + 1:
            omega_sequences += [[omega.view(7, 3).cpu().numpy()]]
        else:
            omega_sequences[i] += [omega.view(7, 3).cpu().numpy()]
    pass


def __inference(model, omega: torch.Tensor, t: torch.Tensor, device=DEVICE) -> torch.Tensor:
    omega = copy.deepcopy(omega.to(device))
    # -> (batch_size, num_objs * 3)

    score = model(omega, t)
    # -> (batch_size, num_objs * 3)
    return score