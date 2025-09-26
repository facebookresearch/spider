"""
Math utilities

Author: Chaoyi Pan
Date: 2025-08-10
"""

import torch


def quat_xyzw2wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion from xyzw to wxyz

    Args:
        quat_xyzw: Quaternion in xyzw format, shape (..., 4)
    Returns:
        quat_wxyz: Quaternion in wxyz format, shape (..., 4)
    """
    return torch.cat([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], dim=-1)


def quat_wxyz2xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion from wxyz to xyzw

    Args:
        quat_wxyz: Quaternion in wxyz format, shape (..., 4)
    Returns:
        quat_xyzw: Quaternion in xyzw format, shape (..., 4)
    """
    return torch.cat([quat_wxyz[..., 0:3], quat_wxyz[..., 3:4]], dim=-1)


def quat_to_vel(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to angular velocity
    Args:
    quat: (..., 4) quaternion in (w, x, y, z) format
    Returns:
    vel: (..., 3) angular velocity
    """
    axis = quat[..., 1:4]  # Extract x, y, z components
    sin_a_2 = torch.norm(axis, dim=-1, keepdim=True)

    # Handle zero case
    zero_mask = sin_a_2[..., 0] == 0.0
    result = torch.zeros_like(axis)

    # Non-zero case
    non_zero_mask = ~zero_mask
    if torch.any(non_zero_mask):
        speed = 2.0 * torch.atan2(sin_a_2[non_zero_mask, 0], quat[non_zero_mask, 0])
        # when axis-angle is larger than pi, rotation is in the opposite direction
        speed = torch.where(speed > torch.pi, speed - 2.0 * torch.pi, speed)

        result[non_zero_mask] = (
            axis[non_zero_mask] * speed[..., None] / sin_a_2[non_zero_mask]
        )

    return result


def mul_quat(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions
    Args:
    u: (..., 4) quaternion in (w, x, y, z) format
    v: (..., 4) quaternion in (w, x, y, z) format
    Returns:
    result: (..., 4) quaternion product
    """
    result = torch.zeros_like(u)
    result[..., 0] = (
        u[..., 0] * v[..., 0]
        - u[..., 1] * v[..., 1]
        - u[..., 2] * v[..., 2]
        - u[..., 3] * v[..., 3]
    )
    result[..., 1] = (
        u[..., 0] * v[..., 1]
        + u[..., 1] * v[..., 0]
        + u[..., 2] * v[..., 3]
        - u[..., 3] * v[..., 2]
    )
    result[..., 2] = (
        u[..., 0] * v[..., 2]
        - u[..., 1] * v[..., 3]
        + u[..., 2] * v[..., 0]
        + u[..., 3] * v[..., 1]
    )
    result[..., 3] = (
        u[..., 0] * v[..., 3]
        + u[..., 1] * v[..., 2]
        - u[..., 2] * v[..., 1]
        + u[..., 3] * v[..., 0]
    )
    return result


def quat_sub(qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
    """
    Subtract quaternion qa from qb, in (w, x, y, z) format
    Args:
        qa: (..., 4) quaternion array
        qb: (..., 4) quaternion array
    Returns:
        quat_diff: (..., 3) angular difference
    """
    qneg = qb.clone()
    qneg[..., 1:] = -qneg[..., 1:]  # Negate x, y, z components
    qdif = mul_quat(qneg, qa)

    # convert to 3D velocity
    return quat_to_vel(qdif)
