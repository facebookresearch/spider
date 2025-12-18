# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Process HOT3D sequences and save hand/object trajectories.

This script should be run insider the HOT3D dataset root folder, assume the following structure:
  hot3d/
    hot3d/
        process_hot3d.py
        dataset/
            P0001_4bf4e21a/
            mano_hand_pose_trajectory.jsonl

Usage:
    python process_hot3d.py --sequence-name P0003_c701bd11

Author: Changhao Wang
Date: 2025-11-09
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import loguru
import numpy as np
import pymeshlab
import torch
import tyro
try:
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
except ImportError:
    raise ImportError("projectaria_tools is not installed. Please install it from https://github.com/facebookresearch/projectaria_tools.")
from scipy.spatial.transform import Rotation

from dataset_api import Hot3dDataProvider  # type: ignore  # noqa: E402
from data_loaders.loader_hand_poses import (  # type: ignore # noqa: E402
    Handedness,
    load_mano_shape_params,
)
from data_loaders.loader_object_library import load_object_library  # type: ignore # noqa: E402
from data_loaders.mano_layer import MANOHandModel  # type: ignore # noqa: E402
from data_loaders.pytorch3d_rotation.rotation_conversions import (  # type: ignore # noqa: E402
    matrix_to_axis_angle,
)


def _to_numpy(array) -> np.ndarray:
    """Convert torch tensors to numpy arrays without gradients."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def convert_glb_to_obj(glb_path: str, obj_path: str, assets_path: str) -> bool:
    """Convert GLB meshes provided by HOT3D into OBJ files for downstream use."""
    try:
        full_glb_path = os.path.join(assets_path, glb_path)
        if not os.path.exists(full_glb_path):
            loguru.logger.error(f"GLB file not found: {full_glb_path}")
            return False

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(full_glb_path, load_in_a_single_layer=True)
        ms.set_texture_per_mesh(use_dummy_texture=True)
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        ms.save_current_mesh(
            obj_path,
            save_face_color=False,
            save_textures=False,
            save_vertex_color=False,
        )
        loguru.logger.info(f"Converted {full_glb_path} to {obj_path}")
        return True
    except Exception as exc:  # pragma: no cover
        loguru.logger.error(f"Failed to convert {glb_path} to OBJ: {exc}")
        return False


def extract_hand_data_from_hot3d(
    hand_data,
    hand_id: str,
    mano_model: MANOHandModel,
    use_root_pose: bool = False,
    T_global=np.eye(4),
):
    """Extract MANO wrist/root poses and fingertip trajectories from HOT3D dicts."""
    joint_angles = hand_data["joint_angles"].float()
    shape_params = hand_data["mano_shape_params"].float()
    wrist_pose_matrix = hand_data["wrist_pose"].float()

    if isinstance(T_global, np.ndarray):
        T_global = torch.from_numpy(T_global).float()
    pose_matrix = T_global @ wrist_pose_matrix

    pose_position = pose_matrix[:3, 3]
    pose_rotation_matrix = pose_matrix[:3, :3]
    axis_angle = matrix_to_axis_angle(pose_rotation_matrix.unsqueeze(0)).squeeze(0)
    global_xform = torch.cat([axis_angle, pose_position])
    is_right = torch.tensor([hand_id == "right"], dtype=torch.bool)

    _, hand_landmarks = mano_model.forward_kinematics(
        shape_params,
        joint_angles,
        global_xform,
        is_right,
    )

    fingertip_indices = [4, 8, 12, 16, 20]
    hand_landmarks_np = _to_numpy(hand_landmarks)
    fingertip_positions = hand_landmarks_np[fingertip_indices]
    wrist_position = hand_landmarks_np[0]
    root_position = hand_landmarks_np[9]

    z_axis = hand_landmarks_np[9] - hand_landmarks_np[0]
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis_aux = hand_landmarks_np[5] - hand_landmarks_np[13]
    y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)
    x_axis = np.cross(y_axis_aux, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    if hand_id == "left":
        x_axis = -x_axis
        y_axis = -y_axis
    pose_rotation = Rotation.from_matrix(np.stack([x_axis, y_axis, z_axis], axis=1))
    quaternion = pose_rotation.as_quat()
    quaternion = quaternion[[3, 0, 1, 2]]

    pose_translation = root_position if use_root_pose else wrist_position
    hand_pose = np.concatenate([pose_translation, quaternion])
    finger_poses = np.zeros((5, 7), dtype=np.float32)
    finger_poses[:, :3] = fingertip_positions
    finger_poses[:, 3:] = [1, 0, 0, 0]
    return hand_pose, finger_poses


def extract_object_data_from_hot3d(object_data):
    """Convert HOT3D homogeneous object transforms into (pos, quat)."""
    T_object = object_data["T_object"]
    position = _to_numpy(T_object[:3, 3])
    rotation_matrix = _to_numpy(T_object[:3, :3])
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    quaternion = quaternion[[3, 0, 1, 2]]
    return np.concatenate([position, quaternion])


def _hand_pose_to_dict(
    hand_pose_data, mano_shape_params: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Convert Hot3D HandPose to the dict format used downstream."""
    if (
        hand_pose_data is None
        or hand_pose_data.wrist_pose is None
        or hand_pose_data.joint_angles is None
    ):
        return {}

    wrist_pose = torch.from_numpy(hand_pose_data.wrist_pose.to_matrix()).float()
    joint_angles = torch.tensor(hand_pose_data.joint_angles, dtype=torch.float32)
    return {
        "wrist_pose": wrist_pose,
        "joint_angles": joint_angles,
        "mano_shape_params": mano_shape_params.clone(),
    }


def build_trajectory_from_hot3d_provider(
    hot3d_provider: Hot3dDataProvider,
    object_uid: str,
    embodiment_type: str,
    mano_shape_params: torch.Tensor,
) -> List[Dict]:
    """Leverage the official HOT3D data provider to align hand/object poses."""
    object_provider = hot3d_provider.object_pose_data_provider
    hand_provider = hot3d_provider.mano_hand_data_provider
    if hand_provider is None:
        raise RuntimeError(
            "MANO hand data provider unavailable — ensure MANO model files are installed."
        )

    object_timestamps = set(object_provider.timestamp_ns_list)
    hand_timestamps = set(hand_provider.timestamp_ns_list)
    candidate_timestamps = sorted(object_timestamps & hand_timestamps)
    if not candidate_timestamps:
        candidate_timestamps = object_provider.timestamp_ns_list

    frames: List[Dict] = []
    object_uid_str = str(object_uid)
    for timestamp in candidate_timestamps:
        object_with_dt = object_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if object_with_dt is None:
            continue

        object_pose = object_with_dt.pose3d_collection.poses.get(object_uid_str)
        if object_pose is None or object_pose.T_world_object is None:
            continue

        hand_with_dt = hand_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        if hand_with_dt is None:
            continue

        hands = {"left": {}, "right": {}}
        for handedness, pose in hand_with_dt.pose3d_collection.poses.items():
            hand_dict = _hand_pose_to_dict(pose, mano_shape_params)
            if not hand_dict:
                continue
            if handedness == Handedness.Left:
                hands["left"] = hand_dict
            elif handedness == Handedness.Right:
                hands["right"] = hand_dict

        has_left = bool(hands["left"])
        has_right = bool(hands["right"])
        if embodiment_type == "left" and not has_left:
            continue
        if embodiment_type == "right" and not has_right:
            continue
        if embodiment_type == "bimanual" and not (has_left and has_right):
            continue

        T_world_object = torch.from_numpy(
            object_pose.T_world_object.to_matrix()
        ).float()

        frames.append(
            {
                "timestamp_ns": timestamp,
                "object_data": {"T_object": T_world_object},
                "left": hands["left"],
                "right": hands["right"],
            }
        )

    if not frames:
        raise ValueError(
            "No aligned frames found via the HOT3D data provider — check object UID and embodiment filters."
        )
    return frames


def compute_ref_dt(timestamps: List[int]) -> float:
    if len(timestamps) < 2:
        return 1.0 / 120.0
    timestamps = np.array(timestamps, dtype=np.int64)
    diffs = np.diff(timestamps) * 1e-9
    return float(np.median(diffs))


def prepare_task_info(
    dataset_dir: str,
    dataset_name: str,
    embodiment_type: str,
    task: str,
    data_id: int,
    object_uid: str,
    sequence_name: str,
    ref_dt: float,
    object_mesh_dir: str | None,
) -> Dict:
    return {
        "task": task,
        "dataset_name": dataset_name,
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": object_mesh_dir,
        "left_object_mesh_dir": None,
        "object_uid": object_uid,
        "sequence_name": sequence_name,
        "ref_dt": ref_dt,
        "dataset_dir": dataset_dir,
    }


def get_processed_data_dir(
    dataset_dir: str,
    dataset_name: str,
    robot_type: str,
    embodiment_type: str,
    task: str,
    data_id: int,
) -> str:
    return os.path.join(
        dataset_dir,
        "processed",
        dataset_name,
        robot_type,
        embodiment_type,
        task,
        str(data_id),
    )


def get_mesh_dir(dataset_dir: str, dataset_name: str, object_name: str) -> str:
    return os.path.join(
        dataset_dir, "processed", dataset_name, "assets", "objects", object_name
    )


def list_sequence_names(
    hot3d_dataset_root: Path, prefix: Optional[str] = "P00"
) -> List[str]:
    if not hot3d_dataset_root.exists():
        return []
    sequence_names = [
        entry.name for entry in hot3d_dataset_root.iterdir() if entry.is_dir()
    ]
    if prefix:
        sequence_names = [name for name in sequence_names if name.startswith(prefix)]
    return sorted(sequence_names)


def resolve_object_uids(
    object_provider: Any, requested_object_uid: Optional[str], sequence_name: str
) -> List[str]:
    if requested_object_uid is not None:
        requested_str = str(requested_object_uid)
        if (
            object_provider is not None
            and requested_str not in object_provider.object_uids_with_poses
        ):
            raise ValueError(
                f"Object UID {requested_str} not found in sequence {sequence_name}."
            )
        return [requested_str]
    if object_provider is None:
        raise RuntimeError(
            f"Object pose data provider unavailable for sequence {sequence_name}."
        )
    object_uids = sorted(object_provider.object_uids_with_poses)
    if not object_uids:
        raise ValueError(f"No object poses found for sequence {sequence_name}.")
    return object_uids


def process_object_sequence(
    *,
    sequence_name: str,
    object_uid: str,
    dataset_dir: Path,
    assets_dir: Path,
    hot3d_provider: Hot3dDataProvider,
    mano_model: MANOHandModel,
    mano_shape_tensor: torch.Tensor,
    data_id: int,
    embodiment_type: str,
    first_frame: int,
    last_frame: Optional[int],
    use_root_pose: bool,
    normalize_object_frame: bool,
    object_frame_offset_deg: Tuple[float, float, float],
):
    loguru.logger.info(
        f"[{sequence_name} | obj {object_uid}] Aligning timestamps via Hot3dDataProvider..."
    )
    trajectory_data = build_trajectory_from_hot3d_provider(
        hot3d_provider, str(object_uid), embodiment_type, mano_shape_tensor
    )
    total_frames = len(trajectory_data)
    if total_frames == 0:
        raise ValueError(
            f"No frames available after alignment for sequence {sequence_name}, object {object_uid}."
        )

    first_idx = max(0, first_frame)
    if first_idx >= total_frames:
        raise ValueError(
            f"first_frame={first_frame} is outside the available range (0-{total_frames-1})."
        )

    if last_frame is None:
        last_idx = total_frames - 1
    else:
        if last_frame < first_idx:
            raise ValueError(
                f"last_frame ({last_frame}) must be >= first_frame ({first_idx})."
            )
        last_idx = min(last_frame, total_frames - 1)

    if first_idx != 0 or last_idx != total_frames - 1:
        loguru.logger.info(
            f"[{sequence_name} | obj {object_uid}] Cropping frames from {total_frames} total to range [{first_idx}, {last_idx}]."
        )
    trajectory_data = trajectory_data[first_idx : last_idx + 1]

    timestamps = [frame["timestamp_ns"] for frame in trajectory_data]
    ref_dt = compute_ref_dt(timestamps)
    loguru.logger.info(
        f"[{sequence_name} | obj {object_uid}] Aligned {len(trajectory_data)} frames (ref_dt ≈ {ref_dt:.4f}s)."
    )

    dataset_dir_str = str(dataset_dir)
    task = f"{sequence_name}-obj{object_uid}"
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir_str,
        dataset_name="hot3d",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(output_dir, exist_ok=True)

    object_mesh_dir = get_mesh_dir(
        dataset_dir=dataset_dir_str,
        dataset_name="hot3d",
        object_name=str(object_uid),
    )
    os.makedirs(object_mesh_dir, exist_ok=True)
    glb_converted = convert_glb_to_obj(
        glb_path=f"{object_uid}.glb",
        obj_path=f"{object_mesh_dir}/visual.obj",
        assets_path=str(assets_dir),
    )
    if not glb_converted:
        loguru.logger.warning(
            f"[{sequence_name} | obj {object_uid}] Failed to convert {object_uid}.glb; downstream visualization may need a fallback mesh."
        )

    task_info = prepare_task_info(
        dataset_dir=dataset_dir_str,
        dataset_name="hot3d",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
        object_uid=object_uid,
        sequence_name=sequence_name,
        ref_dt=ref_dt,
        object_mesh_dir=object_mesh_dir if glb_converted else None,
    )

    task_info_path = Path(output_dir).parent / "task_info.json"
    os.makedirs(task_info_path.parent, exist_ok=True)
    with open(task_info_path, "w", encoding="utf-8") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(
        f"[{sequence_name} | obj {object_uid}] Saved task metadata to {task_info_path}"
    )

    N = len(trajectory_data)
    qpos_wrist_right = np.zeros((N, 7))
    qpos_finger_right = np.zeros((N, 5, 7))
    qpos_obj_right = np.zeros((N, 7))
    qpos_wrist_left = np.zeros((N, 7))
    qpos_finger_left = np.zeros((N, 5, 7))
    qpos_obj_left = np.zeros((N, 7))

    first_object_tensor = trajectory_data[0]["object_data"]["T_object"]
    T_global_np = np.eye(4, dtype=np.float32)
    if normalize_object_frame:
        T_object_init = first_object_tensor.detach().clone().numpy()
        R_object_init = Rotation.from_matrix(T_object_init[:3, :3])
        R_offset = Rotation.from_euler("xyz", object_frame_offset_deg, degrees=True)
        R_object = R_object_init * R_offset
        T_object = np.eye(4)
        T_object[:3, :3] = R_object.as_matrix()
        T_object[:3, 3] = T_object_init[:3, 3]
        T_global_np = np.linalg.inv(T_object)
    T_global = torch.from_numpy(T_global_np).float()

    loguru.logger.info(
        f"[{sequence_name} | obj {object_uid}] Extracting qpos for hands and objects..."
    )
    for i, frame_data in enumerate(trajectory_data):
        object_data = frame_data["object_data"]
        object_data["T_object"] = T_global @ object_data["T_object"]
        object_pose = extract_object_data_from_hot3d(object_data)

        left_hand_data = frame_data["left"]
        right_hand_data = frame_data["right"]

        if left_hand_data:
            hand_pose_left, finger_poses_left = extract_hand_data_from_hot3d(
                left_hand_data, "left", mano_model, use_root_pose, T_global
            )
            qpos_wrist_left[i] = hand_pose_left
            qpos_finger_left[i] = finger_poses_left

        if right_hand_data:
            hand_pose_right, finger_poses_right = extract_hand_data_from_hot3d(
                right_hand_data, "right", mano_model, use_root_pose, T_global
            )
            qpos_wrist_right[i] = hand_pose_right
            qpos_finger_right[i] = finger_poses_right

        qpos_obj_right[i] = object_pose
        qpos_obj_left[i] = np.array([0, 0, 0, 1, 0, 0, 0])

    output_npz = f"{output_dir}/trajectory_keypoints.npz"
    np.savez(
        output_npz,
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
        timestamps=np.array(timestamps, dtype=np.int64),
    )
    loguru.logger.info(
        f"[{sequence_name} | obj {object_uid}] Saved trajectory data to {output_npz}"
    )


def main(
    dataset_dir: str = "../example_datasets",
    hot3d_dataset_root: str = "./dataset",
    sequence_name: str = "P0001_4bf4e21a",
    object_uid: Optional[str] = None,
    embodiment_type: str = "bimanual",
    data_id: int = 0,
    first_frame: int = 0,
    last_frame: Optional[int] = None,
    mano_model_path: str = "./dataset/mano",
    assets_dir: str = "./dataset/assets",
    use_root_pose: bool = True,
    normalize_object_frame: bool = False,
    object_frame_offset_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    process_all_sequences: bool = False,
):
    dataset_dir = Path(dataset_dir).expanduser()
    hot3d_dataset_root = Path(hot3d_dataset_root).expanduser()
    assets_dir = Path(assets_dir).expanduser()
    mano_model_path = Path(mano_model_path).expanduser()

    if not assets_dir.exists():
        raise FileNotFoundError(f"Object library folder {assets_dir} not found.")
    if not mano_model_path.exists():
        raise FileNotFoundError(f"MANO model folder {mano_model_path} not found.")

    if process_all_sequences:
        sequence_names = list_sequence_names(hot3d_dataset_root, prefix=None)
        if not sequence_names:
            raise ValueError(
                f"No sequences found under {hot3d_dataset_root}."
            )
    else:
        sequence_names = [sequence_name]

    object_library = load_object_library(object_library_folderpath=str(assets_dir))

    for seq_name in sequence_names:
        sequence_dir = hot3d_dataset_root / seq_name
        if not sequence_dir.exists():
            loguru.logger.warning(f"Sequence folder {sequence_dir} not found; skipping.")
            continue

        loguru.logger.info(f"Processing sequence {seq_name}...")
        mano_model = MANOHandModel(str(mano_model_path))
        hot3d_provider = Hot3dDataProvider(
            sequence_folder=str(sequence_dir),
            object_library=object_library,
            mano_hand_model=mano_model,
        )

        mano_shape_params = load_mano_shape_params(
            str(sequence_dir / "mano_hand_pose_trajectory.jsonl")
        )
        if mano_shape_params is None:
            loguru.logger.warning(
                f"[{seq_name}] MANO shape parameters missing; defaulting to zeros."
            )
            mano_shape_params = [0.0] * 10
        mano_shape_tensor = torch.tensor(mano_shape_params, dtype=torch.float32)

        object_provider = hot3d_provider.object_pose_data_provider
        try:
            object_uids = resolve_object_uids(object_provider, object_uid, seq_name)
        except Exception as exc:
            loguru.logger.warning(f"[{seq_name}] {exc}")
            continue

        for uid in object_uids:
            try:
                process_object_sequence(
                    sequence_name=seq_name,
                    object_uid=uid,
                    dataset_dir=dataset_dir,
                    assets_dir=assets_dir,
                    hot3d_provider=hot3d_provider,
                    mano_model=mano_model,
                    mano_shape_tensor=mano_shape_tensor,
                    data_id=data_id,
                    embodiment_type=embodiment_type,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    use_root_pose=use_root_pose,
                    normalize_object_frame=normalize_object_frame,
                    object_frame_offset_deg=object_frame_offset_deg,
                )
            except Exception as exc:
                loguru.logger.error(
                    f"[{seq_name} | obj {uid}] Failed to process object: {exc}"
                )


if __name__ == "__main__":
    tyro.cli(main)
