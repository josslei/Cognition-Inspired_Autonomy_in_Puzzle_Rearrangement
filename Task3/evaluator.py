from typing import List, Dict, Tuple

import torch
import numpy as np

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
        _es: float; _ed: float; _er: float
        _es, _ed, _er = tangram_error(gt, inference)
        _error = { 'error_sum': _es, 'error_displacement': _ed, 'error_rotation': _er }
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
    def l2_square(x: torch.Tensor, y: torch.Tensor) -> float:
        return (x.item() - y.item())**2
    error_sum: float = 0.0
    error_displacement: float = 0.0
    error_rotation: float = 0.0
    for i, _ in enumerate(tangram_1):
        ed_2: float = l2_square(tangram_1[i][0], tangram_2[i][0]) + l2_square(tangram_1[i][1], tangram_2[i][1])
        er_2: float = l2_square(torch.cos(tangram_1[i][2] * np.pi), torch.cos(tangram_2[i][2] * np.pi))
        er_2 += l2_square(torch.sin(tangram_1[i][2] * np.pi), torch.sin(tangram_2[i][2] * np.pi))
        error_sum += np.sqrt(ed_2 + er_2)
        error_displacement += np.sqrt(ed_2)
        error_rotation += np.sqrt(er_2)
    return error_sum, error_displacement, error_rotation
