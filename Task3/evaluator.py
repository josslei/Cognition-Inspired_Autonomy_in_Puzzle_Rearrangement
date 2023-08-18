from typing import List

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(omega: torch.Tensor,
             gt: torch.Tensor,
             device=DEVICE) -> List[float]:
    """ Evaluates the result of tangram rearrangement

    Args:
        omega (torch.Tensor): States of arranged tangrams. Shape: (batch_size, 7 * 3)
        gt (List[torch.Tensor]): Original & correct arrangements of tangrams.
            Shape: (batch_size, 7 * 3)

    Returns:
        List[float]: Value of errors.
    """
    batch_size: int = omega.shape[0]
    all_errors: List[float] = []
    for i in range(batch_size):
        inference: torch.Tensor = omega[i].view(7, 3).to(device)
        error = tangram_error(gt, inference)
        all_errors += [error]

    return all_errors

def tangram_error(tangram_1: torch.Tensor, tangram_2: torch.Tensor) -> float:
    """ Calculate distance as error

    Args:
        tangram_1 (torch.Tensor): Tangram 1. Shape: (7, 3)
        tangram_2 (torch.Tensor): Tangram 2. Shape: (7, 3)

    Returns:
        float: Error
    """
    def __rho(x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([x[0], x[1], torch.cos(torch.pi * x[2])], device=x.device)
    error: float = 0.0
    for i, _ in enumerate(tangram_1):
        error += torch.norm(__rho(tangram_1[i]) - __rho(tangram_2[i]), p=2).item() # type: ignore
    return error