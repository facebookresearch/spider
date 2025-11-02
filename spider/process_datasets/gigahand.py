"""This script process gigahand dataset to our target format.

Process:
1. Load object and hand pose data
2. Convert original mano point to finger tip and wrist pose
3. Extract position and pose data for hands and objects

Input: gigahand dataset folder
Output: npz file containing:
    qpos_wrist_left (3pos+4quat(wxyz)), qpos_finger_left, qpos_obj_left, qpos_wrist_right, qpos_finger_right, qpos_obj_right

Author: Chaoyi Pan
Date: 2025-07-28
"""

import glob
import json
import os
from contextlib import contextmanager
from os.path import join
from pathlib import Path

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import pymeshlab
import retarget
import tyro
from loop_rate_limiters import RateLimiter
from retarget.io import get_mesh_dir, get_processed_data_dir
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def select_nf(params_all, nf):
    output = {}
    for key in ["poses", "Rh", "Th"]:
        output[key] = params_all[key][nf : nf + 1, :]
    if "expression" in params_all.keys():
        output["expression"] = params_all["expression"][nf : nf + 1, :]
    if params_all["shapes"].shape[0] == 1:
        output["shapes"] = params_all["shapes"]
    else:
        output["shapes"] = params_all["shapes"][nf : nf + 1, :]
    return output


def load_model(
    gender="neutral",
    use_cuda=True,
    model_type="smpl",
    skel_type="body25",
    device=None,
    model_path="data/smplx",
    **kwargs,
):
    # prepare SMPL model
    # print('[Load model {}/{}]'.format(model_type, gender))
    import torch

    if device is None:
        if use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    from easymocap.smplmodel.body_model import SMPLlayer

    if model_type == "smpl":
        if skel_type == "body25":
            reg_path = join(model_path, "J_regressor_body25.npy")
        elif skel_type == "h36m":
            reg_path = join(model_path, "J_regressor_h36m.npy")
        else:
            raise NotImplementedError
        body_model = SMPLlayer(
            join(model_path, "smpl"),
            gender=gender,
            device=device,
            regressor_path=reg_path,
            **kwargs,
        )
    elif model_type == "smplh":
        body_model = SMPLlayer(
            join(model_path, "smplh/SMPLH_MALE.pkl"),
            model_type="smplh",
            gender=gender,
            device=device,
            regressor_path=join(model_path, "J_regressor_body25_smplh.txt"),
            **kwargs,
        )
    elif model_type == "smplx":
        body_model = SMPLlayer(
            join(model_path, f"smplx/SMPLX_{gender.upper()}.pkl"),
            model_type="smplx",
            gender=gender,
            device=device,
            regressor_path=join(model_path, "J_regressor_body25_smplx.txt"),
            **kwargs,
        )
    elif model_type == "manol" or model_type == "manor":
        lr = {"manol": "LEFT", "manor": "RIGHT"}
        body_model = SMPLlayer(
            join(model_path, f"smplh/MANO_{lr[model_type]}.pkl"),
            model_type="mano",
            gender=gender,
            device=device,
            regressor_path=join(model_path, f"J_regressor_mano_{lr[model_type]}.txt"),
            **kwargs,
        )
    else:
        body_model = None
    body_model.to(device)
    return body_model


def hand_pose_loader(keypoints3d_path):
    """Find frame indices for which both left and right hand pose data are present. Code from GigaHand."""
    chosen_path_left = Path(keypoints3d_path) / "chosen_frames_left.json"
    chosen_path_right = Path(keypoints3d_path) / "chosen_frames_right.json"
    with open(chosen_path_right) as f:
        chosen_frames_right = set(json.load(f))
    with open(chosen_path_left) as f:
        chosen_frames_left = set(json.load(f))
    chosen_hand_union_frames = list(chosen_frames_right | chosen_frames_left)
    chosen_hand_intersect_frames = list(chosen_frames_right & chosen_frames_left)
    return chosen_hand_union_frames, chosen_hand_intersect_frames


def object_pose_loader(object_pose_path, use_filter=True, use_smoother=True):
    """Load and optionally filter/smooth/interpolate object pose sequence from JSON."""
    with open(object_pose_path) as f:
        object_poses = json.load(f)
    tracked_object_frames = sorted([int(frame_id) for frame_id in object_poses.keys()])
    chosen_object_poses, chosen_object_frames = interpolate_object_poses(
        object_poses,
        tracked_object_frames,
        use_filter=use_filter,
        use_smoother=use_smoother,
    )
    return chosen_object_poses, chosen_object_frames


def interpolate_object_poses(
    object_poses, tracked_frames, use_filter=True, use_smoother=True
):
    """Interpolate and smooth translation and rotation in the dictionary."""
    trans, rots, idxs = [], [], []
    for cid in tracked_frames:
        frame = str(cid).zfill(6)
        if frame in object_poses:
            trans.append(np.asarray(object_poses[frame]["mesh_translation"]).squeeze())
            rots.append(np.asarray(object_poses[frame]["mesh_rotation"]))
            idxs.append(cid)
    trans, rots, idxs = np.array(trans), np.array(rots), np.array(idxs)
    # Interpolate translations and rotations
    full_idx = np.arange(idxs[0], idxs[-1] + 1)
    interp_t = interp1d(idxs, trans, axis=0, kind="linear", fill_value="extrapolate")(
        full_idx
    )
    if rots.shape[1] == 3:
        r = Rotation.from_rotvec(rots)
        out_fmt = "rotvec"
    else:
        r = Rotation.from_quat(rots)
        out_fmt = "quat"
    interp_r = Slerp(idxs, r)(full_idx)
    interp_r = interp_r.as_rotvec() if out_fmt == "rotvec" else interp_r.as_quat()
    # Smoothing using moving average
    if use_smoother:
        interp_t = moving_average_filter(interp_t, window_size=9)
        interp_r = moving_average_filter(interp_r, window_size=9)
    inter_poses = {
        str(fid).zfill(6): {
            "mesh_translation": interp_t[i].tolist(),
            "mesh_rotation": interp_r[i].tolist(),
        }
        for i, fid in enumerate(full_idx)
    }
    frame_idx = [fid for i, fid in enumerate(full_idx)]
    return inter_poses, frame_idx


def moving_average_filter(signal, window_size=5):
    """Apply moving average filter with edge-padding to smooth the signal."""
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    pad_len = window_size // 2
    padded = np.pad(signal, ((pad_len, pad_len), (0, 0)), mode="edge")
    kernel = np.ones(window_size) / window_size
    smoothed_signal = np.array(
        [
            np.convolve(padded[:, i], kernel, mode="valid")
            for i in range(signal.shape[1])
        ]
    ).T
    return smoothed_signal.squeeze()


def hand_mano_loader(path, idx_in_hand_iou_indices):
    """Load and select MANO hand model parameters for frames of interest."""
    with open(path) as f:
        manos_params = json.load(f)
    params_left_list = manos_params["left"]
    params_right_list = manos_params["right"]
    params_left = {k: np.asarray(v) for k, v in params_left_list.items()}
    params_right = {k: np.asarray(v) for k, v in params_right_list.items()}
    choosen_frame = np.asarray(list(range(len(params_left_list["poses"]))))
    param_right_all, param_left_all = [], []
    for nf in choosen_frame[idx_in_hand_iou_indices]:
        param_right = select_nf(params_right, nf)
        param_left = select_nf(params_left, nf)
        param_right_all.append(param_right)
        param_left_all.append(param_left)
    return param_right_all, param_left_all


def pytorch3d_quat_to_rotmat(quat_wxyz):
    """Convert a quaternion from PyTorch3D ([w, x, y, z]) to a rotation matrix using scipy, handling coordinate conventions."""
    # Convert to [x, y, z, w]
    quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    # .T is required for our dataset's handedness (PyTorch3D to scipy)
    return Rotation.from_quat(quat_xyzw).as_matrix().T


def object_transform_loader(valid_frames, chosen_object_poses):
    """For each valid frame, construct a 4x4 transformation matrix from quaternion/rotvec and translation.
    The quaternion case expects [w, x, y, z] as from PyTorch3D and converts to scipy format.
    """
    all_transforms = []
    for cid in valid_frames:
        pose = chosen_object_poses[str(cid).zfill(6)]
        mesh_translation = np.asarray(pose["mesh_translation"]).reshape(
            3,
        )
        mesh_rotation = np.asarray(pose["mesh_rotation"])
        if mesh_rotation.shape[0] == 3:
            # Rotation vector (axis-angle)
            R = Rotation.from_rotvec(mesh_rotation).as_matrix()
        else:
            # Quaternion [w, x, y, z] from PyTorch3D
            R = pytorch3d_quat_to_rotmat(mesh_rotation)
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = mesh_translation
        all_transforms.append(transform)
    return all_transforms


def load_body_model(model_path="body_models"):
    """Load left and right MANO hand models."""
    body_model_right = load_model(
        gender="neutral",
        model_type="manor",
        model_path=model_path,
        num_pca_comps=6,
        use_pose_blending=True,
        use_shape_blending=True,
        use_pca=False,
        use_flat_mean=False,
    )
    body_model_left = load_model(
        gender="neutral",
        model_type="manol",
        model_path=model_path,
        num_pca_comps=6,
        use_pose_blending=True,
        use_shape_blending=True,
        use_pca=False,
        use_flat_mean=False,
    )
    return body_model_right, body_model_left


def extract_hand_data(body_model, mano_params, hand_id):
    """Extract hand keypoints and base pose from MANO model."""
    # Get hand keypoints (21 joints including fingertips)
    keypoints = body_model(return_verts=False, return_tensor=False, **mano_params)[0]

    # Fingertip indices in MANO (thumb, index, middle, ring, pinky)
    FINGERTIP_INDICES = [4, 8, 12, 16, 20]
    fingertip_positions = keypoints[FINGERTIP_INDICES]  # Shape: (5, 3)

    # Extract wrist/base pose from MANO parameters
    # Base position from translation parameter
    base_position = np.array(mano_params["Th"]).flatten()[:3]

    # method 1: read wrist rotation from base rotation (less)
    # # Base orientation from global rotation parameter
    # if "Rh" in mano_params:
    #     global_rotation = np.array(mano_params["Rh"]).flatten()[:3]
    #     if np.linalg.norm(global_rotation) > 1e-6:
    #         base_rotation = Rotation.from_rotvec(global_rotation)
    #     else:
    #         base_rotation = Rotation.identity()
    # else:
    #     base_rotation = Rotation.identity()
    # method 2: get wrist rotation from hand landmark
    # Convert rotation matrix to quaternion for pose
    z_axis = keypoints[9] - keypoints[0]
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis_aux = keypoints[5] - keypoints[13]
    y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)
    x_axis = np.cross(y_axis_aux, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    if hand_id == "left":
        x_axis = -x_axis
        y_axis = -y_axis
    pose_rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)  # Column vectors
    base_rotation = Rotation.from_matrix(pose_rotation_matrix)

    # Convert to pose format (position + quaternion)
    xyzw = base_rotation.as_quat()
    wxyz = xyzw[[3, 0, 1, 2]]
    wrist_pose = np.concatenate([base_position, wxyz])  # (7,)

    # For fingertips, assume identity orientation (only position matters for now)
    finger_poses = np.zeros((5, 7))  # 5 fingertips, 7-element pose each
    finger_poses[:, :3] = fingertip_positions  # Position
    finger_poses[:, 3:] = [1, 0, 0, 0]  # Identity quaternion [w, x, y, z]

    return wrist_pose, finger_poses


def compute_mesh_centroid(mesh_path):
    """Compute the centroid of mesh vertices to find the offset needed for centering."""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    # Get vertices
    vertices = ms.current_mesh().vertex_matrix()

    # Compute centroid
    centroid = np.mean(vertices, axis=0)

    return centroid


def extract_object_data(transform):
    """Extract object pose from 4x4 transformation matrix."""
    # Extract position
    position = transform[:3, 3]

    # Extract rotation and convert to quaternion
    rotation_matrix = transform[:3, :3]
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # [x, y, z, w]
    quaternion = quaternion[[3, 0, 1, 2]]  # [w, x, y, z]

    # Combine position and quaternion
    object_pose = np.concatenate([position, quaternion])  # (7,)

    return object_pose


def main(
    dataset_dir: str = "../../example_datasets",
    participant: str = "p36",
    scene_name: str = "17_instruments",
    object_name: str = "ukelele_scan",
    scene: str = "tea",
    sequence_id: str = "0010",
    embodiment_type: str = "bimanual",
    show_viewer: bool = True,
    save_video: bool = False,
    use_example_dataset: bool = True,  # load example dataset from gigahand, which has different file structure
):
    """Process gigahand dataset to our target format."""
    # Resolve dataset paths
    dataset_dir = os.path.abspath(dataset_dir)
    gigahand_path = Path(dataset_dir) / "raw" / "gigahand"
    # Compose identifiers
    task_id = f"{participant}-{scene}-{sequence_id}"
    try:
        data_id = int(sequence_id)
    except Exception:
        data_id = 0
    if use_example_dataset:
        object_pose_path = (
            gigahand_path / "object_poses" / task_id / "pose" / "optimized_pose.json"
        )
    else:
        object_pose_path = (
            gigahand_path
            / "objectposes"
            / scene_name
            / object_name
            / f"{participant}-{scene}_{sequence_id}"
            / "pose"
            / "optimized_pose.json"
        )
    # task_name: before the second "-"
    if use_example_dataset:
        hand_pose_path = gigahand_path / "hand_poses" / task_id
    else:
        hand_pose_path = gigahand_path / "handposes" / f"{participant}-{scene}"
    keypoints3d_path = hand_pose_path / "keypoints_3d" / sequence_id[1:]
    mano_main_path = hand_pose_path / "params" / f"{sequence_id[1:]}.json"

    # Load object and hand poses
    print("Loading object poses...")
    chosen_object_poses, chosen_object_frames = object_pose_loader(
        object_pose_path, use_filter=True, use_smoother=True
    )

    print("Loading hand pose frames...")
    chosen_hand_union_frames, chosen_hand_intersect_frames = hand_pose_loader(
        keypoints3d_path
    )

    # Find joint-valid frame numbers
    chosen_video_frames, _, _ = np.intersect1d(
        chosen_object_frames, chosen_hand_intersect_frames, return_indices=True
    )
    valid_length = len(chosen_video_frames)

    if valid_length == 0:
        print("No frames with both object and hand pose detected.")
        return

    print(f"Found {valid_length} valid frames")

    # Load MANO parameters of joint-valid frames
    video_indices_in_hand_iou_indices = np.asarray(
        [chosen_hand_union_frames.index(f) for f in chosen_video_frames]
    )
    mano_params_right, mano_params_left = hand_mano_loader(
        mano_main_path, video_indices_in_hand_iou_indices
    )

    # Load object transforms for valid frames
    object_transforms = object_transform_loader(
        chosen_video_frames, chosen_object_poses
    )

    # Load MANO body models
    print("Loading MANO models...")
    # TODO: integrate to our pipeline
    body_model_right, body_model_left = load_body_model(
        model_path=f"{retarget.ROOT}/../../GigaHands/body_models"
    )

    # Initialize output arrays
    N = valid_length
    qpos_wrist_right = np.zeros((N, 7))
    qpos_finger_right = np.zeros((N, 5, 7))
    qpos_obj_right = np.zeros((N, 7))  # Same object for both hands
    qpos_wrist_left = np.zeros((N, 7))
    qpos_finger_left = np.zeros((N, 5, 7))
    qpos_obj_left = np.zeros((N, 7))  # Same object for both hands

    # Compute mesh centroid offset before processing frames
    # This will be computed when mesh is processed below, but we need it earlier
    # for trajectory compensation, so we compute it here first
    if use_example_dataset:
        mesh_path_for_centroid = (
            gigahand_path / "object_poses" / task_id / "pose" / "transform_mesh.obj"
        )
        mesh_path_for_centroid = str(mesh_path_for_centroid)
    else:
        mesh_pattern_for_centroid = f"{gigahand_path}/object_meshes/publish/{scene_name}/{object_name}/{object_name.split('_')[0]}.obj"
        mesh_matches_for_centroid = glob.glob(mesh_pattern_for_centroid)
        if not mesh_matches_for_centroid:
            raise FileNotFoundError(
                f"No files matching pattern {mesh_pattern_for_centroid}"
            )
        mesh_path_for_centroid = mesh_matches_for_centroid[0]

    mesh_centroid_offset = compute_mesh_centroid(mesh_path_for_centroid)
    loguru.logger.info(
        f"Mesh centroid offset for trajectory compensation: {mesh_centroid_offset}"
    )

    # Process each frame
    print("Processing frames...")
    # in example dataset, object should be transformed into initial frame
    if use_example_dataset:
        init_transform = object_transforms[0].copy()
        object_transforms = [
            transform @ np.linalg.inv(init_transform) for transform in object_transforms
        ]
    # Transform object for the mesh offset
    mesh_transform = np.eye(4)
    mesh_transform[:3, 3] = mesh_centroid_offset
    object_transforms = [transform @ mesh_transform for transform in object_transforms]

    T_global = np.eye(4)
    T_global[:3, 3] = [-0.5, 0.32, -0.32]
    # TODO: this is calibrate by hand. need to make it more automatic
    r_global = Rotation.from_euler("xyz", [np.pi / 2, 0, 0.0], degrees=False)
    # pos_offset = np.array([0.4, -1.0, 0.0])
    pos_offset = np.array([0.0, 0.0, 0.0])
    T_global[:3, :3] = r_global.as_matrix()
    for i in range(N):
        # Get MANO parameters for this frame
        param_right = mano_params_right[i]
        param_left = mano_params_left[i]

        # Extract hand data
        wrist_pose_right, finger_poses_right = extract_hand_data(
            body_model_right, param_right, "right"
        )
        wrist_pos_right = wrist_pose_right[:3]
        wrist_pos_right_expand = np.append(wrist_pos_right, 1)
        wrist_pos_right_global = T_global @ wrist_pos_right_expand
        wrist_pose_right[:3] = wrist_pos_right_global[:3] + pos_offset
        wrist_quat_right = wrist_pose_right[3:]
        xyzw = wrist_quat_right[[1, 2, 3, 0]]
        r_wrist_right = Rotation.from_quat(xyzw)
        r_wrist_right = r_global * r_wrist_right  # * r_wrist_right_global
        xyzw_global = r_wrist_right.as_quat()
        wxyz_global = xyzw_global[[3, 0, 1, 2]]
        wrist_pose_right[3:] = wxyz_global
        wrist_pose_left, finger_poses_left = extract_hand_data(
            body_model_left, param_left, "left"
        )
        for finger_idx in range(5):
            finger_pos_right = finger_poses_right[finger_idx, :3]
            finger_pos_right_expand = np.append(finger_pos_right, 1)
            finger_pos_right_global = T_global @ finger_pos_right_expand
            finger_poses_right[finger_idx, :3] = (
                finger_pos_right_global[:3] + pos_offset
            )
        wrist_pos_left = wrist_pose_left[:3]
        wrist_pos_left_expand = np.append(wrist_pos_left, 1)
        wrist_pos_left_global = T_global @ wrist_pos_left_expand
        wrist_pose_left[:3] = wrist_pos_left_global[:3] + pos_offset
        wrist_quat_left = wrist_pose_left[3:]
        xyzw = wrist_quat_left[[1, 2, 3, 0]]
        r_wrist_left = Rotation.from_quat(xyzw)
        r_wrist_left = r_global * r_wrist_left  # * r_wrist_left_global
        xyzw_global = r_wrist_left.as_quat()
        wxyz_global = xyzw_global[[3, 0, 1, 2]]
        wrist_pose_left[3:] = wxyz_global
        for finger_idx in range(5):
            finger_pos_left = finger_poses_left[finger_idx, :3]
            finger_pos_left_expand = np.append(finger_pos_left, 1)
            finger_pos_left_global = T_global @ finger_pos_left_expand
            finger_poses_left[finger_idx, :3] = finger_pos_left_global[:3] + pos_offset

        # Extract object data
        # Get object transform (relative to initial pose)
        transform = T_global @ object_transforms[i]
        object_pose = extract_object_data(transform)

        # Store data
        qpos_wrist_right[i] = wrist_pose_right
        qpos_finger_right[i] = finger_poses_right
        qpos_obj_right[i] = object_pose
        qpos_obj_right[i, :3] = qpos_obj_right[i, :3] + pos_offset

        qpos_wrist_left[i] = wrist_pose_left
        qpos_finger_left[i] = finger_poses_left
        # ignore left object pose since it doesn't exsit

    # Copy mesh is handled below via standardized mesh_dir/visual.obj

    # Prepare task info and copy meshes into standardized mesh dirs
    if use_example_dataset:
        task = f"{participant}-{scene}"
    else:
        task = f"{participant}@{scene}@{scene_name}@{object_name}@{sequence_id}"
    task_info = {
        "task": task,
        "dataset_name": "gigahand",
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "right_object_mesh_dir": None,
        "left_object_mesh_dir": None,
        "ref_dt": 0.02,
    }

    # Read right object mesh and convert to standardized location
    if use_example_dataset:
        right_obj_mesh_path = (
            gigahand_path / "object_poses" / task_id / "pose" / "transform_mesh.obj"
        )
        right_obj_mesh_path = str(right_obj_mesh_path)
    else:
        right_obj_mesh_pattern = f"{gigahand_path}/object_meshes/publish/{scene_name}/{object_name}/{object_name.split('_')[0]}.obj"
        right_obj_mesh_matches = glob.glob(right_obj_mesh_pattern)
        if not right_obj_mesh_matches:
            raise FileNotFoundError(
                f"No files matching pattern {right_obj_mesh_pattern}"
            )
        right_obj_mesh_path = right_obj_mesh_matches[0]

    # Convert/copy mesh to mesh_dir/visual.obj
    if use_example_dataset:
        right_object_name = task
    else:
        # right_object_name = os.path.splitext(os.path.basename(right_obj_mesh_path))[0]
        right_object_name = object_name
    mesh_dir = get_mesh_dir(
        dataset_dir=dataset_dir,
        dataset_name="gigahand",
        object_name=right_object_name,
    )

    # Load and center the mesh using the previously computed centroid offset
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(right_obj_mesh_path)

    # Center the mesh by translating vertices to have centroid at origin
    # Use the correct function for geometric transformation
    ms.apply_filter(
        "compute_coord_by_function",
        x=f"x-({mesh_centroid_offset[0]})",
        y=f"y-({mesh_centroid_offset[1]})",
        z=f"z-({mesh_centroid_offset[2]})",
    )

    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    ms.save_current_mesh(f"{mesh_dir}/visual.obj")
    loguru.logger.info(
        f"Saved centered mesh (offset: {mesh_centroid_offset}) from {right_obj_mesh_path} to {mesh_dir}/visual.obj"
    )
    task_info["right_object_mesh_dir"] = mesh_dir
    # single object
    task_info["left_object_mesh_dir"] = None
    # persist
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="gigahand",
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(output_dir, exist_ok=True)
    task_info_path = f"{output_dir}/../task_info.json"
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info to {task_info_path}")

    # Visualize the data
    qpos_list = np.concatenate(
        [
            qpos_wrist_right[:, None],
            qpos_finger_right,
            qpos_wrist_left[:, None],
            qpos_finger_left,
            qpos_obj_right[:, None],
            qpos_obj_left[:, None],
        ],
        axis=1,
    )

    # visualize the data
    mj_spec = mujoco.MjSpec.from_file(f"{retarget.ROOT}/assets/mano/empty_scene.xml")

    # add right object to body "right_object"
    object_right_handle = mj_spec.worldbody.add_body(
        name="right_object",
        mocap=True,
    )
    object_right_handle.add_site(
        name="right_object",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[1, 0, 0, 1],
        group=0,
    )

    if (
        embodiment_type in ["right", "bimanual"]
        and task_info["right_object_mesh_dir"] is not None
    ):
        mj_spec.add_mesh(
            name="right_object",
            file=f"{task_info['right_object_mesh_dir']}/visual.obj",
        )
        object_right_handle.add_geom(
            name="right_object",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="right_object",
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            group=0,
            condim=1,
        )
        loguru.logger.info("Added right object to body 'right_object'")

    # add left object to body "left_object"
    object_left_handle = mj_spec.worldbody.add_body(
        name="left_object",
        mocap=True,
    )
    object_left_handle.add_site(
        name="left_object",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.01, 0.02, 0.03],
        rgba=[0, 1, 0, 1],
        group=0,
    )
    bimanual_single_object = (
        embodiment_type == "bimanual" and task_info["left_object_mesh_dir"] is None
    )
    if (
        embodiment_type in ["left", "bimanual"]
        and not bimanual_single_object
        and task_info["left_object_mesh_dir"] is not None
    ):
        # add left object to body "left_object"
        mj_spec.add_mesh(
            name="left_object",
            file=f"{task_info['left_object_mesh_dir']}/visual.obj",
        )
        object_left_handle.add_geom(
            name="left_object",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="left_object",
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            group=0,
            condim=1,
            contype=0,
            conaffinity=0,
        )

    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)
    rate_limiter = RateLimiter(120.0)
    if show_viewer:
        run_viewer = lambda: mujoco.viewer.launch_passive(mj_model, mj_data)
    else:
        cam = mujoco.MjvCamera()
        cam.type = 2
        cam.fixedcamid = 0

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {
                    "is_running": lambda: True,
                    "sync": lambda: None,
                    "cam": cam,
                },
            )

    if save_video:
        import imageio

        mj_model.vis.global_.offwidth = 720
        mj_model.vis.global_.offheight = 480
        renderer = mujoco.Renderer(mj_model, height=480, width=720)
        images = []
    with run_viewer() as gui:
        cnt = 0
        contact_seq = np.zeros((N, 10))
        while gui.is_running():
            mj_data.mocap_pos[:] = qpos_list[cnt, :, :3]
            mj_data.mocap_quat[:] = qpos_list[cnt, :, 3:]
            mujoco.mj_step(mj_model, mj_data)
            cnt = (cnt + 1) % N
            if save_video:
                renderer.update_scene(mj_data, gui.cam)
                img = renderer.render()
                images.append(img)
            if cnt == (N - 1):
                # save to standardized output dir
                np.savez(
                    f"{output_dir}/trajectory_kinematic.npz",
                    qpos_wrist_right=qpos_wrist_right,
                    qpos_finger_right=qpos_finger_right,
                    qpos_obj_right=qpos_obj_right,
                    qpos_wrist_left=qpos_wrist_left,
                    qpos_finger_left=qpos_finger_left,
                    qpos_obj_left=qpos_obj_left,
                    contact=np.zeros((N, 10)),
                )
                loguru.logger.info(
                    f"Saved data to {output_dir}/trajectory_kinematic.npz"
                )
                if save_video:
                    imageio.mimsave(f"{output_dir}/visualization.mp4", images, fps=120)
                    loguru.logger.info(f"Saved video to {output_dir}/visualization.mp4")
                if not show_viewer:
                    break
            if show_viewer:
                gui.sync()
                rate_limiter.sleep()


if __name__ == "__main__":
    tyro.cli(main)
