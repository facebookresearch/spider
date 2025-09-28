"""Define functions to interpolate the control signal.

Author: Chaoyi Pan

Date: 2025-08-10
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# def interp(src: torch.Tensor, n: int) -> torch.Tensor:
#     """
#     Interpolate the source tensor to the destination time step with zero-order hold
#     Example:
#         src = torch.tensor([[[1,2], [3,4]], [[5,6], [7,8]]])
#         n = 2
#         dst = interp(src, n)
#         print(dst)
#         # tensor([[[1, 2], [1, 2], [3, 4], [3, 4]], [[5, 6], [5, 6], [7, 8], [7, 8]]])
#     Args:
#         src: Source tensor, shape (N, H, D)
#         n: Number of time steps to repeat each time step
#     Returns:
#         Interpolated tensor, shape (N, H * n, D)
#     """
#     N, H, D = src.shape
#     # Repeat each time step n times
#     # First, reshape to (N, H, 1, D) to prepare for expansion
#     src_expanded = src.unsqueeze(2)  # Shape: (N, H, 1, D)
#     # Repeat along the time dimension
#     src_repeated = src_expanded.repeat(1, 1, n, 1)  # Shape: (N, H, n, D)
#     # Reshape to final output shape
#     dst = src_repeated.reshape(N, H * n, D)  # Shape: (N, H * n, D)
#     return dst


def interp(src: torch.Tensor, n: int, order: int = 1) -> torch.Tensor:
    """Interpolate the source tensor using zeroth, first, or second-order hold.

    This function uses torch.nn.functional.interpolate for an efficient implementation
    of all interpolation methods.

    - order=0: Zeroth-order hold (Nearest Neighbor). Steps between values.
    - order=1: First-order hold (Linear Interpolation). Smooth lines between values.
    - order=2: Second-order hold (Quadratic Interpolation). Smooth curves between values.

    Args:
        src: Source tensor, shape (N, H, D).
        n: The integer upsampling factor.
        order: The order of interpolation. Must be 0, 1, or 2.

    Returns:
        Interpolated tensor, shape (N, H * n, D).
    """
    if order not in [0, 1, 2]:
        raise ValueError("Order must be an integer: 0, 1, or 2.")

    N, H, D = src.shape

    # If there's only one time step, interpolation is not meaningful.
    # The only possible behavior is to repeat the value (zero-order hold).
    if H <= 1:
        return src.repeat(1, n, 1)

    # Determine the interpolation mode string for the backend function.
    # Also handle cases where the input is too short for the chosen order.
    if order == 0:
        mode = "nearest"
    elif order == 1:
        mode = "linear"
    elif order == 2:
        # Quadratic interpolation requires at least 3 points to define a curve.
        if H < 3:
            # Gracefully fall back to linear if we can't do quadratic.
            print(
                f"Warning: Source tensor has H={H} < 3 time steps. "
                "Falling back to linear interpolation for order=2."
            )
            mode = "linear"
        else:
            mode = "quadratic"

    # Ensure the input tensor is a floating-point type for interpolation,
    # as linear and quadratic modes require it.
    if not src.is_floating_point():
        src = src.to(torch.float32)

    # `F.interpolate` expects the dimension to be interpolated as the last one.
    # The input shape should be (N, Channels, Length).
    # We treat our D dimension as "channels" and H as "length".
    # So, we permute the tensor from (N, H, D) to (N, D, H).
    src_permuted = src.permute(0, 2, 1)

    # Calculate the desired output length.
    # We subtract 1 from H, multiply by n, then add 1 to ensure that the
    # total number of points is correct after upsampling.
    # However, for simplicity and direct control, setting size=H*n works well.
    dst_len = H * n

    # align_corners=True is important for signal-like data. It ensures that the
    # endpoint values of the input and output sequences match perfectly.
    # It does not apply to 'nearest' mode.
    align = mode != "nearest"

    # Perform the 1D interpolation
    dst_permuted = F.interpolate(
        src_permuted, size=dst_len, mode=mode, align_corners=align
    )

    # Permute the dimensions back to the desired output shape: (N, D, H*n) -> (N, H*n, D)
    dst = dst_permuted.permute(0, 2, 1)

    return dst


# --- Example Usage ---
if __name__ == "__main__":
    # Create a source tensor with 4 time steps to clearly see the curve for quadratic.
    # Shape: (1, 4, 2) -> N=1, H=4, D=2
    src_tensor = torch.tensor(
        [[[1.0, 10.0], [3.0, 12.0], [2.0, 8.0], [4.0, 9.0]]], dtype=torch.float32
    )

    # Upsampling factor
    n_factor = 3

    print("--- Original Tensor (H=4) ---")
    print(src_tensor)
    print("-" * 30)

    # --- Zeroth-Order Hold (ZOH) ---
    dst_zoh = interp(src_tensor, n_factor, order=0)
    print(f"--- Zeroth-Order Hold (order=0, H={dst_zoh.shape[1]}) ---")
    # Values are simply repeated
    print(dst_zoh.squeeze(0).numpy())
    print("-" * 30)
    #

    # --- First-Order Hold (FOH) ---
    dst_foh = interp(src_tensor, n_factor, order=1)
    print(f"--- First-Order Hold (order=1, H={dst_foh.shape[1]}) ---")
    # Linear interpolation between points
    print(dst_foh.squeeze(0).numpy())
    print("-" * 30)
    #

    # --- Second-Order Hold ---
    dst_soh = interp(src_tensor, n_factor, order=2)
    print(f"--- Second-Order Hold (order=2, H={dst_soh.shape[1]}) ---")
    # Quadratic interpolation (curved) between points
    print(dst_soh.squeeze(0).numpy())
    print("-" * 30)
    #


def get_slice(
    src: tuple[torch.Tensor, ...], start: int, end: int
) -> tuple[torch.Tensor, ...]:
    """Get a slice of the source tensor"""
    return tuple(s[start:end] for s in src)
