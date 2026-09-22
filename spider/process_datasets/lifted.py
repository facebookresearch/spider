# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Process unified-format "lifted" datasets for SPIDER retargeting.

Adapted from the archived fh/lifted-bench processor. This exposes the benchmark
baseline only: no experimental hand normalization or fingertip tightening.

Handles any dataset following the unified lifted layout, e.g.:
  - /efs/data/dexycb_lifted/right/<sequence>/3d/seed_default_lifted_centered/
  - /efs/data/hot3d_3d_lifted/right/<sequence>/3d/seed_default_lifted_centered/
  - /efs/data/oakink_3d_hold_lifted/<sequence>/3d/seed_default_centered/

Each variant directory contains data.npz (MANO keypoints + object transforms at
30fps) and object_1_mesh.glb (+ convex/ decomposition). The trajectory is
resampled duration-preserving to ref_dt, wrist pose is derived from MANO
keypoints, and everything is shifted so the object's lowest vertex at frame 0
sits on the ground plane (z=0).

Outputs to: {dataset_dir}/processed/{dataset_name}/mano/{embodiment}/{task}/{data_id}/
"""

import json
import os
import shutil
from pathlib import Path

import loguru
import numpy as np
import trimesh
import tyro
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

import spider
from spider.io import get_mesh_dir, get_processed_data_dir

# variant preference order: "centered" variants are gravity-aligned + ground-referenced
VARIANT_CANDIDATES = [
    "seed_default_lifted_centered",
    "seed_default_centered",
]


def interpolate_transforms(transforms: np.ndarray, num_frames_out: int) -> np.ndarray:
    """Interpolate 4x4 transforms: linear for translation, SLERP for rotation."""
    num_frames = len(transforms)
    t_in = np.linspace(0, 1, num_frames)
    t_out = np.linspace(0, 1, num_frames_out)

    positions = transforms[:, :3, 3]
    pos_interp = interp1d(t_in, positions, axis=0, kind="linear")(t_out)

    rotations = Rotation.from_matrix(transforms[:, :3, :3])
    slerp = Slerp(t_in, rotations)
    rot_interp = slerp(t_out).as_matrix()

    out = np.zeros((num_frames_out, 4, 4))
    out[:, :3, :3] = rot_interp
    out[:, :3, 3] = pos_interp
    out[:, 3, 3] = 1.0
    return out


def interpolate_keypoints(keypoints: np.ndarray, num_frames_out: int) -> np.ndarray:
    """Linearly interpolate (T, 21, 3) keypoints to new frame count."""
    num_frames = len(keypoints)
    t_in = np.linspace(0, 1, num_frames)
    t_out = np.linspace(0, 1, num_frames_out)
    flat = keypoints.reshape(num_frames, -1)
    interped = interp1d(t_in, flat, axis=0, kind="linear")(t_out)
    return interped.reshape(num_frames_out, keypoints.shape[1], keypoints.shape[2])


def compute_wrist_orientation(keypoints: np.ndarray, side: str) -> np.ndarray:
    """Compute wrist orientation quaternion (wxyz) from 21 MANO keypoints."""
    num_frames = len(keypoints)
    wxyz = np.zeros((num_frames, 4))

    for i in range(num_frames):
        z_axis = keypoints[i, 9] - keypoints[i, 0]
        z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis_aux = keypoints[i, 5] - keypoints[i, 13]
        if side == "left":
            y_axis_aux = -y_axis_aux
        y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)

        x_axis = np.cross(y_axis_aux, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        rot_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)
        r = Rotation.from_matrix(rot_matrix)
        xyzw = r.as_quat()
        wxyz[i] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    return wxyz


def keypoints_to_qpos(kpts: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    """MANO keypoints (num_frames,21,3) -> wrist qpos (num_frames,7) and fingertip qpos (num_frames,5,7)."""
    num_frames = len(kpts)
    wrist = np.zeros((num_frames, 7))
    wrist[:, :3] = kpts[:, 0, :]
    wrist[:, 3:] = compute_wrist_orientation(kpts, side)

    # thumb(4), index(8), middle(12), ring(16), pinky(20)
    fingertip_indices = [4, 8, 12, 16, 20]
    finger = np.zeros((num_frames, 5, 7))
    finger[:, :, 3] = 1.0
    for j, idx in enumerate(fingertip_indices):
        finger[:, j, :3] = kpts[:, idx, :]
    return wrist, finger


def append_lift_to_threshold(
    hand_kpts: np.ndarray,
    obj_transforms: np.ndarray,
    target_lift: float,
    lift_speed: float,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Append a vertical lift so the object ends `target_lift` above frame 0.

    Mirrors PickAnything's ``add_lift_to_package`` (hand + object translate
    rigidly along +Z at constant velocity from their last pose), but the lift
    distance is ``target_lift - (z_end - z_0)`` so the reference reaches the
    same final height regardless of how much the human demo already lifted.
    """
    z = obj_transforms[:, 2, 3]
    already = float(z[-1] - z[0])
    remaining = target_lift - already
    if remaining <= 1e-3:
        return hand_kpts, obj_transforms
    n_lift = max(int(round(remaining / lift_speed * fps)), 1)
    dz = np.linspace(remaining / n_lift, remaining, n_lift).astype(hand_kpts.dtype)

    hand_tail = np.repeat(hand_kpts[-1:], n_lift, axis=0)
    hand_tail[:, :, 2] += dz[:, None]
    obj_tail = np.repeat(obj_transforms[-1:], n_lift, axis=0)
    obj_tail[:, 2, 3] += dz
    loguru.logger.info(
        f"Appended lift: demo lifted {already:+.3f}m, adding {remaining:.3f}m over "
        f"{n_lift} frames ({lift_speed} m/s) to reach {target_lift:.2f}m"
    )
    return (
        np.concatenate([hand_kpts, hand_tail], axis=0),
        np.concatenate([obj_transforms, obj_tail], axis=0),
    )


def main(
    source_dir: str,
    sequence: str,
    dataset_name: str,
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    embodiment_type: str = "right",
    variant: str = "auto",
    source_fps: float = 30.0,
    ref_dt: float = 0.08,
    min_frames: int = 12,
    max_frames: int = 64,
    time_scale: float = 1.0,
    ensure_lift: float = 0.0,
    lift_speed: float = 0.3,
    fixed_num_frames: int = 0,
):
    """Process one unified-format sequence into SPIDER's keypoint format.

    The benchmark baseline uses ensure_lift=0.30, ref_dt=0.08, and
    fixed_num_frames=24 only for HRDexDB. Source meshes retain their GLB
    scene transforms; synthetic lift translates the hand and object together.
    """
    if embodiment_type not in {"right", "left"}:
        raise ValueError("Lifted source clips require right or left embodiment")
    dataset_dir = os.path.abspath(dataset_dir)
    seq_dir = Path(source_dir) / sequence / "3d"

    if variant == "auto":
        for cand in VARIANT_CANDIDATES:
            if (seq_dir / cand / "data.npz").exists():
                variant = cand
                break
        else:
            raise FileNotFoundError(
                f"No centered variant with data.npz under {seq_dir}"
            )
    source_path = seq_dir / variant
    data_path = source_path / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    task = sequence
    data_id = 0

    loguru.logger.info(
        f"Processing {sequence} ({variant}) -> {dataset_name}/{embodiment_type}"
    )

    hand_key = f"hand_{embodiment_type}_keypoints"
    with np.load(data_path, allow_pickle=False) as data:
        hand_kpts = data[hand_key].copy()
        obj_transforms = data["object_1_transforms"].copy()
    if hand_kpts.shape[1:] != (21, 3) or obj_transforms.shape != (len(hand_kpts), 4, 4):
        raise ValueError("Expected aligned (T,21,3) keypoints and (T,4,4) transforms")
    if (
        len(hand_kpts) < 2
        or not np.isfinite(obj_transforms).all()
        or np.isinf(hand_kpts).any()
    ):
        raise ValueError(
            "Invalid source length or nonfinite source transforms/keypoints"
        )

    # repair sparse NaN frames (hand-tracking dropouts); refuse heavy corruption
    nan_frames = np.isnan(hand_kpts.reshape(len(hand_kpts), -1)).any(axis=1)
    if nan_frames.any():
        frac = nan_frames.mean()
        if frac > 0.2:
            raise ValueError(
                f"{nan_frames.sum()}/{len(nan_frames)} frames have NaN keypoints"
            )
        good = np.where(~nan_frames)[0]
        flat = hand_kpts.reshape(len(hand_kpts), -1)
        for c in range(flat.shape[1]):
            flat[:, c] = np.interp(np.arange(len(flat)), good, flat[good, c])
        hand_kpts = flat.reshape(hand_kpts.shape)
        loguru.logger.warning(
            f"Interpolated over {nan_frames.sum()} NaN keypoint frames ({frac:.0%})"
        )

    if ensure_lift > 0:
        hand_kpts, obj_transforms = append_lift_to_threshold(
            hand_kpts, obj_transforms, ensure_lift, lift_speed, source_fps
        )

    source_frames = len(hand_kpts)
    if fixed_num_frames > 0:
        # legacy recipe (spider/process_datasets/hrdexdb.py): squeeze the whole
        # demo into a fixed number of frames regardless of its duration
        num_frames = fixed_num_frames
    else:
        # duration-preserving resample: source_frames @ source_fps -> num_frames @ 1/ref_dt
        num_frames = int(
            np.clip(
                round(source_frames / source_fps / ref_dt * time_scale),
                min_frames,
                max_frames,
            )
        )
    loguru.logger.info(
        f"Resampling: {source_frames} frames @ {source_fps}fps -> "
        f"{num_frames} frames @ ref_dt={ref_dt}s"
    )

    hand_kpts = interpolate_keypoints(hand_kpts, num_frames)
    obj_transforms = interpolate_transforms(obj_transforms, num_frames)
    num_frames = num_frames

    # Load mesh (also used below for the ground z-offset)
    glb_path = source_path / "object_1_mesh.glb"
    scene = trimesh.load(str(glb_path))
    # Preserve node transforms: HOT3D geometry is in millimetres under a scale node.
    mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"unexpected mesh type {type(mesh)} in {glb_path}")

    qpos_wrist, qpos_finger = keypoints_to_qpos(hand_kpts, embodiment_type)

    # Object pose from 4x4 transforms
    qpos_obj = np.zeros((num_frames, 7))
    qpos_obj[:, :3] = obj_transforms[:, :3, 3]
    xyzw = Rotation.from_matrix(obj_transforms[:, :3, :3]).as_quat()
    qpos_obj[:, 3] = xyzw[:, 3]  # w
    qpos_obj[:, 4:] = xyzw[:, :3]  # xyz

    # Z-offset: shift everything so object's lowest vertex at frame 0 sits at z=0
    rot0 = Rotation.from_matrix(obj_transforms[0, :3, :3])
    verts_world_frame0 = rot0.apply(mesh.vertices) + qpos_obj[0, :3]
    z_offset = verts_world_frame0[:, 2].min()
    if abs(z_offset) > 1e-6:
        qpos_wrist[:, 2] -= z_offset
        qpos_finger[:, :, 2] -= z_offset
        qpos_obj[:, 2] -= z_offset
        loguru.logger.info(
            f"Applied z-offset: {z_offset:+.4f}m (object lowest z now at 0)"
        )

    # Inactive hand: zeros with identity quaternion
    zeros_wrist = np.zeros((num_frames, 7))
    zeros_wrist[:, 3] = 1.0
    zeros_finger = np.zeros((num_frames, 5, 7))
    zeros_finger[:, :, 3] = 1.0
    zeros_obj = np.zeros((num_frames, 7))
    zeros_obj[:, 3] = 1.0

    if embodiment_type == "right":
        qpos_wrist_right, qpos_finger_right, qpos_obj_right = (
            qpos_wrist,
            qpos_finger,
            qpos_obj,
        )
        qpos_wrist_left, qpos_finger_left, qpos_obj_left = (
            zeros_wrist,
            zeros_finger,
            zeros_obj,
        )
    else:
        qpos_wrist_left, qpos_finger_left, qpos_obj_left = (
            qpos_wrist,
            qpos_finger,
            qpos_obj,
        )
        qpos_wrist_right, qpos_finger_right, qpos_obj_right = (
            zeros_wrist,
            zeros_finger,
            zeros_obj,
        )

    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(output_dir, exist_ok=True)

    npz_path = f"{output_dir}/trajectory_keypoints.npz"
    np.savez(
        npz_path,
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
    )
    loguru.logger.info(
        f"Saved trajectory_keypoints.npz ({num_frames} frames) to {npz_path}"
    )

    # Save mesh as OBJ + copy convex decomposition
    obj_name = task
    mesh_dir = get_mesh_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        object_name=obj_name,
    )
    os.makedirs(mesh_dir, exist_ok=True)

    obj_path = f"{mesh_dir}/visual.obj"
    mesh.export(obj_path)
    loguru.logger.info(f"Saved mesh to {obj_path}")

    convex_src = source_path / "convex"
    convex_dst = Path(mesh_dir) / "convex"
    if convex_src.exists():
        os.makedirs(convex_dst, exist_ok=True)
        for f in sorted(convex_src.glob("*.obj")):
            shutil.copy2(f, convex_dst / f.name)
        loguru.logger.info(
            f"Copied {len(list(convex_dst.glob('*.obj')))} convex hulls to {convex_dst}"
        )

    mesh_dir_relative = str(Path(mesh_dir).relative_to(Path(dataset_dir)))
    convex_dir_relative = f"{mesh_dir_relative}/convex"

    right_side = embodiment_type == "right"
    task_info = {
        "task": task,
        "dataset_name": dataset_name,
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": mesh_dir_relative if right_side else None,
        "left_object_mesh_dir": None if right_side else mesh_dir_relative,
        "right_object_convex_dir": convex_dir_relative if right_side else None,
        "left_object_convex_dir": None if right_side else convex_dir_relative,
        "ref_dt": ref_dt,
    }

    task_info_path = f"{output_dir}/../task_info.json"
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info.json to {task_info_path}")


if __name__ == "__main__":
    tyro.cli(main)
