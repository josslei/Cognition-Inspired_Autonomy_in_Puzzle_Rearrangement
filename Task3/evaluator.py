from typing import List, Dict, Tuple

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(omega: torch.Tensor,
             ground_truth: torch.Tensor,
             device=DEVICE) -> List[Dict[str, float]]:
    """ Evaluates the result of tangram rearrangement

    Args:
        omega (torch.Tensor): States of arranged tangrams. Shape: (batch_size, 7 * 3)
        gt (List[torch.Tensor]): Original & correct arrangements of tangrams.
            Shape: (batch_size, 7 * 3)

    Returns:
        List[float]: Value of errors.
    """
    batch_size: int = omega.shape[0]
    all_errors: List[Dict[str, float]] = []
    for i in range(batch_size):
        inference: torch.Tensor = omega[i].view(7, 3).to(device)
        gt: torch.Tensor = ground_truth[i].view(7, 3).to(device)
        _err_both: float; _err_disp: float; _err_rot: float
        _err_both, _err_disp, _err_rot = tangram_error(gt, inference)
        _error = { 'error_both': _err_both, 'error_displacement': _err_disp, 'error_rotation': _err_rot }
        all_errors += [_error]

    return all_errors

def tangram_error(tangram_1: torch.Tensor, tangram_2: torch.Tensor) -> Tuple[float, float, float]:
    """ Calculate distance as error

    Args:
        tangram_1 (torch.Tensor): Tangram 1. Shape: (7, 3)
        tangram_2 (torch.Tensor): Tangram 2. Shape: (7, 3)

    Returns:
        float: Error
    """
    def __rho(x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([x[0], x[1], torch.cos(torch.pi * x[2])], device=x.device)
    def __get_displacement(x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([x[0], x[1]], device=x.device)
    def __get_rotation(x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([torch.cos(torch.pi * x[2])], device=x.device)
    error_both: float = 0.0
    error_displacement: float = 0.0
    error_rotation: float = 0.0
    for i, _ in enumerate(tangram_1):
        delta: torch.Tensor
        delta = __rho(tangram_1[i]) - __rho(tangram_2[i])
        error_both += torch.norm(delta, p=2).item() # type: ignore
        delta = __get_displacement(tangram_1[i]) - __get_displacement(tangram_2[i])
        error_displacement += torch.norm(delta, p=2).item() # type: ignore
        delta = __get_rotation(tangram_1[i]) - __get_rotation(tangram_2[i])
        error_rotation += torch.norm(delta, p=2).item() # type: ignore
    return error_both, error_displacement, error_rotation