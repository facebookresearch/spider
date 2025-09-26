"""
Define functions to interpolate the control signal.

Author: Chaoyi Pan

Date: 2025-08-10
"""

from __future__ import annotations
import sys
import torch

def interp(src: torch.Tensor, n: int) -> torch.Tensor:
    """
    Interpolate the source tensor to the destination time step with zero-order hold
    Example:
        src = torch.tensor([[[1,2], [3,4]], [[5,6], [7,8]]])
        n = 2
        dst = interp(src, n)
        print(dst)
        # tensor([[[1, 2], [1, 2], [3, 4], [3, 4]], [[5, 6], [5, 6], [7, 8], [7, 8]]])
    Args:
        src: Source tensor, shape (N, H, D)
        n: Number of time steps to repeat each time step
    Returns:
        Interpolated tensor, shape (N, H * n, D)
    """
    N, H, D = src.shape
    # Repeat each time step n times
    # First, reshape to (N, H, 1, D) to prepare for expansion
    src_expanded = src.unsqueeze(2)  # Shape: (N, H, 1, D)
    # Repeat along the time dimension
    src_repeated = src_expanded.repeat(1, 1, n, 1)  # Shape: (N, H, n, D)
    # Reshape to final output shape
    dst = src_repeated.reshape(N, H * n, D)  # Shape: (N, H * n, D)
    return dst


def get_slice(
    src: tuple[torch.Tensor, ...], start: int, end: int
) -> tuple[torch.Tensor, ...]:
    """
    Get a slice of the source tensor
    """
    return tuple(s[start:end] for s in src)
