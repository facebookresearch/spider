"""This script process hot3d dataset to our target format.

Process:
1. Load object and hand pose data
2. Convert original mano point to finger tip and wrist/root pose
3. Extract position and pose data for hands and objects

Input: hot3d dataset folder
Output: npz file containing:
    qpos_wrist_left (3pos+4quat(wxyz)), qpos_finger_left, qpos_obj_left, qpos_wrist_right, qpos_finger_right, qpos_obj_right

Note:
- By default, the script now uses root pose (centered at middle finger root) instead of wrist pose
- This matches the reference implementation in the hot3d dataset
- Use --use_root_pose=False to revert to wrist pose if needed

Author: Chaoyi Pan
Date: 2025-08-04
Updated: 2025-01-27 (added wrist to root pose conversion)
"""

import os
import pickle
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import pymeshlab
import torch
import tyro
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation

# Add the imitation_learning_3d path to import HOT3D utilities
sys.path.append("../../../imitation_learning_3d")
from robo3d.common.hot3d.mano_layer import loadManoHandModel
from robo3d.common.rotation_utils import from_homogeneous


def convert_numpy_to_torch(input_dict):
    """Credit: hot3d dataset
    Convert all numpy arrays in the input dictionary to PyTorch tensors with float32 data type.

    Args:
        input_dict (dict): The input dictionary containing numpy arrays.

    Returns:
        dict: A new dictionary with numpy arrays converted to PyTorch tensors.
    """
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            # Recursively convert nested dictionaries
            output_dict[key] = convert_numpy_to_torch(value)
        elif isinstance(value, np.ndarray):
            # Convert numpy array to torch tensor
            output_dict[key] = torch.from_numpy(value).float()
        else:
            # Keep the value as is if it's not a numpy array
            output_dict[key] = value
    return output_dict


def load_data(
    hot3d_dir: str, object_id: str, sequence_id: int = 4, trajectory_id: int = 0
):
    """Load data from hot3d dataset for a specific trajectory ID within a sequence.
    Loads pregrasp folders first, then postgrasp folders for the specified trajectory.
    """
    # Use the specified sequence (default trajectories4)
    base_path = Path(hot3d_dir) / f"trajectories{sequence_id}" / "data" / object_id

    if not base_path.exists():
        loguru.logger.error(f"Object path not found: {base_path}")
        return None

    # Load data from specific trajectory folder
    trajectory_path = base_path / f"trajectory_{trajectory_id}"

    if not trajectory_path.exists():
        loguru.logger.error(f"Trajectory path not found: {trajectory_path}")
        return None

    loguru.logger.info(f"Loading data from trajectory_{trajectory_id}")

    all_pregrasp_files = []
    all_postgrasp_files = []

    # Process the specific trajectory folder
    pregrasp_folder = trajectory_path / "pregrasp"
    postgrasp_folder = trajectory_path / "postgrasp"

    # Check if folders exist and have data
    if pregrasp_folder.exists() and len(os.listdir(pregrasp_folder)) > 0:
        pregrasp_files = [
            pregrasp_folder / item for item in os.listdir(pregrasp_folder)
        ]
        pregrasp_files = sorted(pregrasp_files, key=lambda x: int(x.stem))
        # Filter out empty folders - check if required files exist
        valid_pregrasp_files = []
        for folder in pregrasp_files:
            required_files = ["object_data.pkl", "right.pkl", "left.pkl"]
            if all((folder / file).exists() for file in required_files):
                valid_pregrasp_files.append(folder)
            else:
                loguru.logger.warning(f"Skipping empty/incomplete folder: {folder}")
        all_pregrasp_files.extend(valid_pregrasp_files)
        loguru.logger.info(
            f"Added {len(valid_pregrasp_files)} valid pregrasp files from {trajectory_path}"
        )

    if postgrasp_folder.exists() and len(os.listdir(postgrasp_folder)) > 0:
        postgrasp_files = [
            postgrasp_folder / item for item in os.listdir(postgrasp_folder)
        ]
        postgrasp_files = sorted(postgrasp_files, key=lambda x: int(x.stem))
        # Filter out empty folders - check if required files exist
        valid_postgrasp_files = []
        for folder in postgrasp_files:
            required_files = ["object_data.pkl", "right.pkl", "left.pkl"]
            if all((folder / file).exists() for file in required_files):
                valid_postgrasp_files.append(folder)
            else:
                loguru.logger.warning(f"Skipping empty/incomplete folder: {folder}")
        all_postgrasp_files.extend(valid_postgrasp_files)
        loguru.logger.info(
            f"Added {len(valid_postgrasp_files)} valid postgrasp files from {trajectory_path}"
        )

    # Concatenate pregrasp files first, then postgrasp files for this trajectory
    files = all_pregrasp_files + all_postgrasp_files

    if not files:
        loguru.logger.error(
            f"No data files found for object {object_id} in trajectory_{trajectory_id}"
        )
        return None

    loguru.logger.info(
        f"Total files to process from trajectory_{trajectory_id}: {len(all_pregrasp_files)} pregrasp + {len(all_postgrasp_files)} postgrasp = {len(files)}"
    )

    # load data
    trajectory_data = []
    hand_list = ["right", "left"]
    for datapoint_folder in files:
        _datapoint = {}
        filename_list = ["object_data"] + hand_list
        for filename in filename_list:
            with open(Path(datapoint_folder / (filename + ".pkl")), "rb") as file:
                # datapoint has keys: ['object_data', 'right', 'left']
                _datapoint[filename] = convert_numpy_to_torch(pickle.load(file))
        trajectory_data.append(_datapoint)

    loguru.logger.info(f"Loaded {len(trajectory_data)} total frames")
    return trajectory_data


def wrist_to_root_pose(
    wrist_pose_matrix, joint_angles, shape_params, hand_id, mano_model
):
    """Convert wrist pose to root pose (centered at root of middle finger).

    Args:
        wrist_pose_matrix: 4x4 transformation matrix for wrist pose
        joint_angles: MANO joint angles (15,)
        shape_params: MANO shape parameters (10,)
        hand_id: "left" or "right"
        mano_model: MANO hand model instance

    Returns:
        root_pose_matrix: 4x4 transformation matrix for root pose
    """
    # Extract translation and rotation from wrist pose matrix
    trans, rot = from_homogeneous(wrist_pose_matrix)

    # Use MANO model's wrist2root conversion
    root_pose_matrix = mano_model.wrist2root(
        trans, rot, joint_angles, shape_params, hand_id
    )

    return root_pose_matrix


def extract_hand_data_from_hot3d(
    hand_data, hand_id, mano_model, use_root_pose=False, T_global=np.eye(4)
):
    """Extract hand pose data from HOT3D format using MANO forward kinematics.
    Returns root/wrist pose and fingertip poses.

    Args:
        hand_data: HOT3D hand data containing wrist_pose, joint_angles, mano_shape_params
        hand_id: "left" or "right"
        mano_model: MANO hand model instance
        use_root_pose: If True, convert wrist pose to root pose (centered at middle finger root)

    Returns:
        hand_pose: 7-element pose (3 position + 4 quaternion) for wrist or root
        finger_poses: (5, 7) array of fingertip poses
    """
    # Extract MANO parameters from HOT3D data
    joint_angles = hand_data["joint_angles"]  # Shape: (15,) pose parameters
    shape_params = hand_data["mano_shape_params"]  # Shape: (10,) shape parameters

    # Get wrist pose from wrist_pose matrix (4x4)
    wrist_pose_matrix = hand_data["wrist_pose"]  # Shape: (4, 4)

    # Convert wrist to root pose if requested
    if use_root_pose:
        # Convert wrist pose to root pose (centered at root of middle finger)
        pose_matrix = wrist_to_root_pose(
            wrist_pose_matrix, joint_angles, shape_params, hand_id, mano_model
        )
        pose_name = "root"
    else:
        pose_matrix = wrist_pose_matrix
        pose_name = "wrist"

    pose_matrix = T_global @ pose_matrix

    # Extract position and rotation from pose matrix
    pose_position = pose_matrix[:3, 3]
    pose_rotation_matrix = pose_matrix[:3, :3]

    # Use MANO forward kinematics with the original pose rotation matrix
    mesh_vertices, hand_landmarks = mano_model.forward_kinematics(
        pose_position,
        pose_rotation_matrix.detach().clone(),
        joint_angles,
        shape_params,
        hand_id,
    )

    # Apply joint mapping to get correct landmark order
    # MANO fingertip indices after joint mapping (thumb, index, middle, ring, pinky)
    FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # These are the final indices after mapping
    fingertip_positions = hand_landmarks[FINGERTIP_INDICES]  # Shape: (5, 3)

    # Calculate hand orientation from landmarks (example: using middle finger direction)
    # Get wrist position (joint index 0) and middle finger tip (joint index 12)
    wrist_pos = hand_landmarks[0]  # Wrist joint
    middle_finger_tip = hand_landmarks[12]  # Middle finger tip

    # Convert rotation matrix to quaternion for pose
    z_axis = hand_landmarks[9] - hand_landmarks[0]
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis_aux = hand_landmarks[5] - hand_landmarks[13]
    y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)
    x_axis = np.cross(y_axis_aux, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    if hand_id == "left":
        x_axis = -x_axis
        y_axis = -y_axis
    pose_rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)  # Column vectors
    rotation = Rotation.from_matrix(pose_rotation_matrix)
    quaternion = rotation.as_quat()  # [x, y, z, w]
    quaternion = quaternion[[3, 0, 1, 2]]  # Convert to [w, x, y, z]

    # Combine position and quaternion for hand pose
    hand_pose = np.concatenate([pose_position.numpy(), quaternion])  # (7,)

    # For fingertips, assume identity orientation (only position matters for now)
    finger_poses = np.zeros((5, 7))  # 5 fingertips, 7-element pose each
    finger_poses[:, :3] = fingertip_positions.numpy()  # Position
    finger_poses[:, 3:] = [1, 0, 0, 0]  # Identity quaternion [w, x, y, z]

    return hand_pose, finger_poses


def extract_object_data_from_hot3d(object_data):
    """Extract object pose from HOT3D object data."""
    # Get object transformation matrix
    T_object = object_data["T_object"]  # Shape: (4, 4)

    # Extract position and rotation
    position = T_object[:3, 3].numpy()
    rotation_matrix = T_object[:3, :3].numpy()

    # Convert rotation matrix to quaternion
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # [x, y, z, w]
    quaternion = quaternion[[3, 0, 1, 2]]  # Convert to [w, x, y, z]

    # Combine position and quaternion
    object_pose = np.concatenate([position, quaternion])  # (7,)

    return object_pose


def convert_glb_to_obj(glb_path: str, obj_path: str, assets_path: str) -> bool:
    """Convert GLB file to OBJ format using pymeshlab.

    Args:
        glb_path: Path to the GLB file (relative to assets_path)
        obj_path: Output path for the OBJ file
        assets_path: Base path for HOT3D assets

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Construct full path to GLB file
        full_glb_path = os.path.join(assets_path, glb_path)

        if not os.path.exists(full_glb_path):
            loguru.logger.error(f"GLB file not found: {full_glb_path}")
            return False

        # Create MeshSet and load GLB file
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(full_glb_path, load_in_a_single_layer=True)
        ms.set_texture_per_mesh(use_dummy_texture=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)

        # Save as OBJ file
        ms.save_current_mesh(
            obj_path,
            save_face_color=False,
            save_textures=False,
            save_vertex_color=False,
        )

        loguru.logger.info(f"Successfully converted {full_glb_path} to {obj_path}")
        return True

    except Exception as e:
        loguru.logger.error(f"Failed to convert {glb_path} to OBJ: {str(e)}")
        return False


# def apply_transform_to_qpos(qpos, T_apply):
#     """
#     Apply transformation to qpos.
#     """
#     pos = qpos[:3]
#     wxyz = qpos[3:]
#     xyzw = wxyz[[3, 0, 1, 2]]
#     T_in = np.eye(4)
#     T_in[:3, 3] = pos
#     T_in[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
#     T_out = T_apply @ T_in
#     pos_out = T_out[:3, 3]
#     mat_out = T_out[:3, :3]
#     xyzw_out = Rotation.from_matrix(mat_out).as_quat()
#     wxyz_out = xyzw_out[[3, 0, 1, 2]]
#     qpos_out = np.concatenate([pos_out, wxyz_out])
#     return qpos_out


def main(
    hot3d_dir: str = "/checkpoint/gum/francoishogan/hot3d/processed",
    object_id: str = "106434519822892",
    sequence_id: int = 4,
    trajectory_id: int = 2,
    hand_type: str = "bimanual",
    mano_model_path: str = "/checkpoint/gum/francoishogan/hot3d/mano_v1_2/models",
    assets_path: str = "../../datasets/hot3d_assets",
    show_viewer: bool = True,
    save_video: bool = False,
    use_root_pose: bool = True,
):
    """Process hot3d dataset to our target format.

    Args:
        hot3d_dir: Path to HOT3D processed dataset
        object_id: Object ID to process
        sequence_id: Sequence ID to process (default: 4 for trajectories4)
        trajectory_id: Specific trajectory ID to process (default: 0 for trajectory_0)
        hand_type: Type of hand processing ("left", "right", "bimanual")
        mano_model_path: Path to MANO model files (defaults to HOT3D's MANO path)
        assets_path: Path to HOT3D assets containing GLB files
        show_viewer: Whether to show MuJoCo viewer
        save_video: Whether to save video output
        use_root_pose: Whether to convert wrist pose to root pose (centered at middle finger root)
    """
    task = f"hot3d-seq{sequence_id}-obj{object_id}-traj{trajectory_id}"

    # Load trajectory data
    trajectory_data = load_data(hot3d_dir, object_id, sequence_id, trajectory_id)

    if trajectory_data is None or len(trajectory_data) == 0:
        loguru.logger.error("No trajectory data loaded")
        return

    N = len(trajectory_data)
    loguru.logger.info(f"Processing {N} frames")

    # Load MANO hand model using HOT3D's MANO model path
    print("Loading MANO models...")
    mano_model = loadManoHandModel(mano_model_files_dir=mano_model_path)
    if mano_model is None:
        loguru.logger.error(f"Failed to load MANO hand model from {mano_model_path}")
        return

    # Initialize output arrays
    qpos_wrist_right = np.zeros((N, 7))
    qpos_finger_right = np.zeros((N, 5, 7))
    qpos_obj_right = np.zeros((N, 7))
    qpos_wrist_left = np.zeros((N, 7))
    qpos_finger_left = np.zeros((N, 5, 7))
    qpos_obj_left = np.zeros((N, 7))

    # Get object initial transformation
    T_object_init = trajectory_data[0]["object_data"]["T_object"]  # .cpu().numpy()
    R_object_init = Rotation.from_matrix(T_object_init[:3, :3])
    R_object_offset = Rotation.from_euler("xyz", [-np.pi / 2, 0, 0])
    R_object = R_object_init * R_object_offset
    T_object = np.eye(4)
    T_object[:3, :3] = R_object.as_matrix()
    T_object[:3, 3] = T_object_init[:3, 3]
    T_global = np.linalg.inv(T_object)
    T_global = torch.from_numpy(T_global).float()

    # Process each frame
    print("Processing frames...")
    for i, frame_data in enumerate(trajectory_data):
        object_data = frame_data["object_data"]
        left_hand_data = frame_data["left"]
        right_hand_data = frame_data["right"]

        # Get the transformation matrix for normalizing to base frame
        # T_object = object_data["T_object"]
        # T_inv = torch.inverse(T_object)

        # # Transform point cloud to base frame
        # ones = torch.ones((object_data['pc'].shape[0], 1), device=object_data['pc'].device, dtype=object_data['pc'].dtype)
        # points = torch.cat((object_data["pc"], ones), dim=1)
        # pc_base = (points @ T_inv.T)[:, :3]

        # # Create transformed object data with normalized object position (identity transform)
        # object_data_base = object_data.copy()
        # object_data_base["pc"] = pc_base
        # object_data_base["T_object"] = torch.eye(4, device=object_data["T_object"].device, dtype=object_data["T_object"].dtype)

        # Extract object pose (should be identity/origin now)
        object_data["T_object"] = T_global @ object_data["T_object"]
        object_pose = extract_object_data_from_hot3d(object_data)

        # Process left hand if data is available
        if (
            "wrist_pose" in left_hand_data
            and "joint_angles" in left_hand_data
            and "mano_shape_params" in left_hand_data
        ):
            # Transform wrist pose to base frame
            # wrist_pose_left_base = T_inv @ left_hand_data["wrist_pose"]
            wrist_pose_left_base = left_hand_data["wrist_pose"]

            # Create transformed hand data
            left_hand_data_base = left_hand_data.copy()
            left_hand_data_base["wrist_pose"] = wrist_pose_left_base

            # Extract hand poses using MANO forward kinematics
            hand_pose_left, finger_poses_left = extract_hand_data_from_hot3d(
                left_hand_data_base, "left", mano_model, use_root_pose, T_global
            )
            qpos_wrist_left[i] = hand_pose_left
            qpos_finger_left[i] = finger_poses_left

        # Process right hand if data is available
        if (
            "wrist_pose" in right_hand_data
            and "joint_angles" in right_hand_data
            and "mano_shape_params" in right_hand_data
        ):
            # Transform wrist pose to base frame
            # wrist_pose_right_base = T_inv @ right_hand_data["wrist_pose"]
            wrist_pose_right_base = right_hand_data["wrist_pose"]

            # Create transformed hand data
            right_hand_data_base = right_hand_data.copy()
            right_hand_data_base["wrist_pose"] = wrist_pose_right_base

            # Extract hand poses using MANO forward kinematics
            hand_pose_right, finger_poses_right = extract_hand_data_from_hot3d(
                right_hand_data_base, "right", mano_model, use_root_pose, T_global
            )
            qpos_wrist_right[i] = hand_pose_right
            qpos_finger_right[i] = finger_poses_right

        # Store object pose for both hands (same object, normalized to origin)
        qpos_obj_right[i] = object_pose
        qpos_obj_left[i] = np.array([0, 0, 0, 1, 0, 0, 0])

    # apply global transform to qpos
    # for i in range(N):
    #     qpos_wrist_right[i] = apply_transform_to_qpos(qpos_wrist_right[i], T_global)
    #     for j in range(5):
    #         qpos_finger_right[i, j] = apply_transform_to_qpos(qpos_finger_right[i, j], T_global)
    #     qpos_obj_right[i] = apply_transform_to_qpos(qpos_obj_right[i], T_global)
    #     qpos_wrist_left[i] = apply_transform_to_qpos(qpos_wrist_left[i], T_global)
    #     for j in range(5):
    #         qpos_finger_left[i, j] = apply_transform_to_qpos(qpos_finger_left[i, j], T_global)
    # qpos_obj_left is already at origin, so no need to transform

    # Create object mesh files using GLB to OBJ conversion
    if len(trajectory_data) > 0:
        # Construct GLB file path based on object_id
        glb_filename = f"{object_id}.glb"

        # Create mesh directories and convert GLB to OBJ
        for side in ["right"]:
            file_name = f"{side}_{task}"
            export_dir = f"../assets/objects/{file_name}"
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            # Convert GLB to OBJ using pymeshlab
            obj_path = f"{export_dir}/{file_name}.obj"
            success = convert_glb_to_obj(glb_filename, obj_path, assets_path)
            loguru.logger.info(f"Successfully converted GLB to OBJ for {file_name}")

            if not success:
                loguru.logger.warning(
                    f"Failed to convert GLB to OBJ for {file_name}, mesh file may be missing"
                )

        for side in ["left"]:
            # create a dummy left object
            file_name = f"{side}_{task}"
            export_dir = f"../assets/objects/{file_name}"
            # delete the export_dir if it exists
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)
            # create empty left folder
            os.makedirs(export_dir)
            loguru.logger.info(f"Created empty left object folder: {export_dir}")

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

    # Set up MuJoCo visualization similar to gigahand.py
    mj_spec = mujoco.MjSpec.from_file("../assets/mano/empty_scene.xml")

    # Add right object
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

    if hand_type in ["right", "bimanual"]:
        mj_spec.add_mesh(
            name="right_object",
            file=f"../objects/right_{task}/right_{task}.obj",
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

    # Add left object
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

    if hand_type in ["left"]:
        mj_spec.add_mesh(
            name="left_object",
            file=f"../objects/left_{task}/left_{task}.obj",
        )
        object_left_handle.add_geom(
            name="left_object",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="left_object",
            pos=[0, 0, 0],
            quat=[1, 0, 0, 0],
            group=0,
            condim=1,
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
                save_dir = "../../datasets/raw/mano"
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save data to NPZ
                np.savez(
                    f"{save_dir}/{hand_type}_{task}.npz",
                    qpos_wrist_right=qpos_wrist_right,
                    qpos_finger_right=qpos_finger_right,
                    qpos_obj_right=qpos_obj_right,
                    qpos_wrist_left=qpos_wrist_left,
                    qpos_finger_left=qpos_finger_left,
                    qpos_obj_left=qpos_obj_left,
                    contact=np.zeros((N, 10)),
                )
                loguru.logger.info(f"Saved data to {hand_type}_{task}.npz")

                if save_video:
                    imageio.mimsave(
                        f"{save_dir}/{hand_type}_{task}.mp4",
                        images,
                        fps=10,
                    )
                    loguru.logger.info(
                        f"Saved video to {save_dir}/{hand_type}_{task}.mp4"
                    )

                if not show_viewer:
                    break

            if show_viewer:
                gui.sync()
                rate_limiter.sleep()


if __name__ == "__main__":
    tyro.cli(main)
