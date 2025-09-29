"""Process real world data from FAIR Montreal Campus.

1. Convert mesh to obj
2. Convert original data to finger tip, object and wrist position and rotation using hand landmarks
3. Visualize the data in mujoco viewer

File path example: {dataset_dir}/raw/fair_mon/{task}_{hand_type}/{data_id}.pkl

Input: pickle file from FAIR MON dataset.
Output: npz file containing:
    qpos_wrist_left, qpos_finger_left, qpos_obj_left, qpos_wrist_right, qpos_finger_right, qpos_obj_right

Author: Chaoyi Pan
Date: 2025-01-28
"""

import json
import os
import pickle
import time

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import open3d as o3d
import rerun as rr
import torch
import trimesh
import tyro
from loop_rate_limiters import RateLimiter
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from spider.io import get_mesh_dir, get_processed_data_dir


def recover_original_array(array_20d, fill_value=0.0):
    """Recovers the original 21-dimensional array from a 20-dimensional array (from Francois).

    Args:
        array_20d: The 20-dimensional array (as a list or NumPy array).
        fill_value: The value to use for the missing element at index 13.
                    Defaults to 0.0.

    Returns:
        The recovered 21-dimensional array (as a NumPy array if the input
        was a NumPy array, otherwise as a list).
    """
    mano_joint_mapping = [
        16,
        17,
        18,
        19,
        20,
        0,
        14,
        15,
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        11,
        12,
        7,
        8,
        9,
    ]

    # Check if the input is a NumPy array to determine the output type
    is_numpy = isinstance(array_20d, np.ndarray)

    if is_numpy:
        # Create a new array with 21 elements, filled with the fill_value
        # This handles multidimensional arrays as well, e.g., (20, 3) -> (21, 3)
        new_shape = list(array_20d.shape)
        new_shape[0] = 21
        array_21d = np.full(new_shape, fill_value, dtype=array_20d.dtype)
    else:
        # If the input is a list, create a new list of size 21
        array_21d = [fill_value] * 21

    # Place the elements from the 20d array into their original positions
    for i, original_index in enumerate(mano_joint_mapping):
        array_21d[original_index] = array_20d[i]

    return array_21d


def index_original_array(array_21d):
    """Index the original 21-dimensional array to a 21-dimensional array (from Changhao)."""
    mano_joint_mapping = [
        0,
        13,  # thumb
        14,
        15,
        16,
        1,  # index
        2,
        3,
        17,
        4,  # middle
        5,
        6,
        18,
        10,  # ring
        11,
        12,
        19,
        7,  # pinky
        8,
        9,
        20,
    ]

    return array_21d[mano_joint_mapping]


def downsample_point_cloud(points, target_num_points=512):
    """Downsample point cloud to target number of points using random sampling.

    Args:
        points (np.ndarray): Input point cloud of shape (N, 3)
        target_num_points (int): Target number of points after downsampling

    Returns:
        np.ndarray: Downsampled point cloud of shape (target_num_points, 3)
    """
    num_points = points.shape[0]

    if num_points <= target_num_points:
        # If we already have fewer points than target, return as is
        return points

    # Random sampling without replacement
    indices = np.random.choice(num_points, target_num_points, replace=False)
    return points[indices]


def add_ground(size=3.0, z_height=0.2):
    """Add ground visualization for rerun"""
    # Create a flat plane (thin box)
    half = size / 2.0

    vertices = np.array(
        [
            [-half, -half, z_height],  # bottom-left
            [half, -half, z_height],  # bottom-right
            [half, half, z_height],  # top-right
            [-half, half, z_height],  # top-left
        ],
        dtype=np.float32,
    )

    # Two triangles forming the quad
    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.uint32,
    )

    # Simple uniform color (light gray)
    colors = np.array([[50, 50, 50, 0] * 4], dtype=np.uint8)

    rr.log(
        "ground",
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=indices,
            vertex_colors=colors,
        ),
    )


def interpolate_pointclouds(pointclouds, M):
    """Interpolates along the first dimension of a (N, P, 3) or (N, 3) array to get M frames.

    Args:
        pointclouds (np.ndarray): shape (N, P, 3) or (N, 3)
        M (int): number of frames after interpolation

    Returns:
        np.ndarray: shape (M, P, 3) or (M, 3)
    """
    if len(pointclouds.shape) == 2:
        # Handle (N, 3) case - single point trajectory
        N, C = pointclouds.shape
        assert C == 3, "Last dimension must be 3 (x, y, z)"
    elif len(pointclouds.shape) == 3:
        # Handle (N, P, 3) case - multiple points trajectory
        N, P, C = pointclouds.shape
        assert C == 3, "Last dimension must be 3 (x, y, z)"
    else:
        raise ValueError(f"Expected 2D or 3D array, got {len(pointclouds.shape)}D")

    # Original and target "time" coordinates
    x_old = np.linspace(0, 1, N)
    x_new = np.linspace(0, 1, M)

    # Create interpolation function along axis 0
    f = interp1d(x_old, pointclouds, axis=0, kind="linear")

    return f(x_new)


def main(
    dataset_dir: str = "../../example_datasets",
    hand_type: str = "right",
    data_id: int = 0,
    task: str = "coke",
    right_object_name: str = "coke",
    left_object_name: str | None = None,
    show_viewer: bool = True,
    show_rerun: bool = False,
    max_steps: int = -1,
    save_video: bool = False,
    filter_sudden_changes: bool = True,
    num_interpolation_frames: int = 20,
    mesh_downsample_points: int = 512,
    downsample_mesh: bool = False,
    z_offset: float = 0.0,
    center_initial_frame: bool = True,
):
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name="fair_mon",
        robot_type="mano",
        hand_type=hand_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    file_path = f"{dataset_dir}/raw/fair_mon/{task}_{hand_type}/{data_id}.pkl"
    with open(file_path, "rb") as f:
        data_raw = pickle.load(f)

    loguru.logger.info(f"Loaded data from {file_path}")

    # Extract mesh data and save as obj
    prediction = data_raw["hand_predictions"][0]
    object_pc = data_raw["object_pc"]

    # Process mesh if available and convert to obj
    task_info = {
        "task": task,
        "dataset_name": "fair_mon",
        "robot_type": "mano",
        "hand_type": hand_type,
        "data_id": data_id,
        "right_object_mesh_dir": None,
        "left_object_mesh_dir": None,
    }

    # Convert object point cloud to mesh
    mesh_dir = get_mesh_dir(
        dataset_dir=dataset_dir,
        dataset_name="fair_mon",
        object_name=right_object_name,
    )
    os.makedirs(mesh_dir, exist_ok=True)

    # Check if mesh already exists
    mesh_file_path = f"{mesh_dir}/visual.obj"
    if os.path.exists(mesh_file_path):
        loguru.logger.info(
            f"Mesh already exists at {mesh_file_path}, skipping mesh conversion"
        )
    else:
        # Create mesh from object point cloud using Open3D alpha shape
        # Get the initial object transformation to apply inverse transform to the mesh
        trajectory_data_temp = trajectory_from_grasp(data_raw, num_interpolation_frames)
        initial_transform = trajectory_data_temp["object"][0]  # (4, 4) matrix

        # Optionally downsample point cloud for faster mesh generation
        if downsample_mesh:
            loguru.logger.info(
                f"Original point cloud has {len(object_pc)} points, downsampling to {mesh_downsample_points}"
            )
            object_pc_processed = downsample_point_cloud(
                object_pc, target_num_points=mesh_downsample_points
            )
        else:
            loguru.logger.info(
                f"Using full point cloud with {len(object_pc)} points for mesh generation"
            )
            object_pc_processed = object_pc

        # Apply inverse transformation to center the point cloud properly
        # Convert point cloud to homogeneous coordinates
        object_pc_homo = np.hstack(
            [object_pc_processed, np.ones((object_pc_processed.shape[0], 1))]
        )
        # Apply inverse transformation
        object_pc_transformed = (np.linalg.inv(initial_transform) @ object_pc_homo.T).T[
            :, :3
        ]

        # Create mesh using Open3D alpha shape method
        loguru.logger.info(
            f"Converting point cloud with {len(object_pc_processed)} points to mesh using alpha shape"
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_pc_transformed)
        pcd.estimate_normals()
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=0.02
        )
        mesh.compute_vertex_normals()

        object_mesh = trimesh.Trimesh(
            vertices=np.array(mesh.vertices), faces=np.array(mesh.triangles)
        )

        # Save the mesh as OBJ
        object_mesh.export(mesh_file_path)

        loguru.logger.info(
            f"Saved transformed object mesh from point cloud to {mesh_file_path}"
        )

    task_info["right_object_mesh_dir"] = mesh_dir

    # Extract hand landmarks for trajectory computation
    hand_landmarks = prediction[
        "hand_landmarks"
    ]  # Shape should be (21, 3) for 21 landmarks

    # Create trajectory by interpolation
    # FAIR MON provides single grasp, we need to create a trajectory
    trajectory_data = trajectory_from_grasp(data_raw, num_interpolation_frames)

    # Extract trajectories
    hand_landmark_trajectory = trajectory_data["hand_landmarks"]
    object_trajectory = trajectory_data["object"]

    # Convert landmark order from Francois to Changhao
    landmark_list = []
    for i in range(hand_landmark_trajectory.shape[0]):
        recovered_landmarks = recover_original_array(hand_landmark_trajectory[i])
        new_landmarks = index_original_array(recovered_landmarks)
        landmark_list.append(new_landmarks)
    hand_landmark_trajectory = np.array(landmark_list)

    N = hand_landmark_trajectory.shape[0]

    # Compute wrist pose from hand landmarks
    # Use landmarks to estimate wrist position and orientation
    # Landmark 0 is typically the wrist
    wrist_positions = hand_landmark_trajectory[:, 0, :]  # Wrist landmark

    # Estimate wrist orientation from hand landmarks
    # Use the vector from wrist to middle finger MCP joint for orientation
    wrist_orientations = []
    for i in range(N):
        # Vector from wrist (0) to middle finger MCP (9)
        forward_vec = (
            hand_landmark_trajectory[i, 9, :] - hand_landmark_trajectory[i, 0, :]
        )
        forward_vec = forward_vec / np.linalg.norm(forward_vec)

        # Vector from wrist to index finger MCP (5)
        right_vec = (
            hand_landmark_trajectory[i, 5, :] - hand_landmark_trajectory[i, 0, :]
        )
        right_vec = right_vec / np.linalg.norm(right_vec)

        # Cross product for up vector
        up_vec = np.cross(forward_vec, right_vec)
        up_vec = up_vec / np.linalg.norm(up_vec)

        # Recompute right vector to ensure orthogonality
        right_vec = np.cross(up_vec, forward_vec)

        # Create rotation matrix
        rot_matrix = np.column_stack([right_vec, up_vec, forward_vec])
        wrist_orientations.append(R.from_matrix(rot_matrix).as_rotvec())

    wrist_orientations = np.array(wrist_orientations)

    # Extract fingertip positions (landmarks 4, 8, 12, 16, 20 are fingertips)
    fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    fingertip_positions = hand_landmark_trajectory[:, fingertip_indices, :]

    # Handle left/right hand data
    if hand_type in ["right", "bimanual"]:
        right_wrist_pos = wrist_positions
        right_wrist_rot = wrist_orientations
        right_mano_joints = fingertip_positions
        right_obj_trajectory = object_trajectory
    else:
        right_wrist_pos = np.zeros((N, 3))
        right_wrist_rot = np.zeros((N, 3))
        right_mano_joints = np.zeros((N, 5, 3))
        right_obj_trajectory = np.tile(np.eye(4), (N, 1, 1))

    if hand_type in ["left", "bimanual"]:
        left_wrist_pos = wrist_positions
        left_wrist_rot = wrist_orientations
        left_mano_joints = fingertip_positions
        left_obj_trajectory = object_trajectory
    else:
        left_wrist_pos = np.zeros((N, 3))
        left_wrist_rot = np.zeros((N, 3))
        left_mano_joints = np.zeros((N, 5, 3))
        left_obj_trajectory = np.tile(np.eye(4), (N, 1, 1))

    # Apply filtering if requested
    if filter_sudden_changes:
        for i in range(1, N):
            if np.linalg.norm(right_wrist_pos[i] - right_wrist_pos[i - 1]) > 0.1:
                right_wrist_pos[i] = right_wrist_pos[i - 1]
                right_wrist_rot[i] = right_wrist_rot[i - 1]
                right_mano_joints[i] = right_mano_joints[i - 1]
            if np.linalg.norm(left_wrist_pos[i] - left_wrist_pos[i - 1]) > 0.1:
                left_wrist_pos[i] = left_wrist_pos[i - 1]
                left_wrist_rot[i] = left_wrist_rot[i - 1]
                left_mano_joints[i] = left_mano_joints[i - 1]

    # Convert to qpos format (position + quaternion)
    if max_steps > 0:
        N = min(N, max_steps)
        right_wrist_pos = right_wrist_pos[:N]
        right_wrist_rot = right_wrist_rot[:N]
        right_mano_joints = right_mano_joints[:N]
        left_wrist_pos = left_wrist_pos[:N]
        left_wrist_rot = left_wrist_rot[:N]
        left_mano_joints = left_mano_joints[:N]
        right_obj_trajectory = right_obj_trajectory[:N]
        left_obj_trajectory = left_obj_trajectory[:N]

    # Convert to qpos format
    unit_quat = np.array([1, 0, 0, 0])
    qpos_wrist_right = np.zeros((N, 7))
    qpos_finger_right = np.zeros((N, 5, 7))
    qpos_obj_right = np.zeros((N, 7))
    qpos_wrist_left = np.zeros((N, 7))
    qpos_finger_left = np.zeros((N, 5, 7))
    qpos_obj_left = np.zeros((N, 7))

    # Global rotation (identity for now)
    if center_initial_frame:
        r_global = R.from_matrix(right_obj_trajectory[0][:3, :3]).inv()
        pos_offset = -right_obj_trajectory[0][:3, 3]
    else:
        r_global = R.from_euler("xyz", [0, 0, 0])
        pos_offset = np.zeros(3)

    for i in range(N):
        # Right hand
        qpos_wrist_right[i, :3] = r_global.apply(right_wrist_pos[i] + pos_offset)
        r = r_global * R.from_rotvec(right_wrist_rot[i])
        xyzw = r.as_quat()
        wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
        qpos_wrist_right[i, 3:] = wxyz

        # Left hand
        qpos_wrist_left[i, :3] = r_global.apply(left_wrist_pos[i] + pos_offset)
        r = r_global * R.from_rotvec(left_wrist_rot[i])
        xyzw = r.as_quat()
        wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
        qpos_wrist_left[i, 3:] = wxyz

        # Fingertips
        for j in range(5):
            qpos_finger_right[i, j, :3] = r_global.apply(
                right_mano_joints[i, j] + pos_offset
            )
            qpos_finger_right[i, j, 3:] = unit_quat
            qpos_finger_left[i, j, :3] = r_global.apply(
                left_mano_joints[i, j] + pos_offset
            )
            qpos_finger_left[i, j, 3:] = unit_quat

        # Objects
        if len(right_obj_trajectory.shape) == 3:  # (N, 4, 4)
            T_obj = right_obj_trajectory[i]
            qpos_obj_right[i, :3] = r_global.apply(T_obj[:3, 3] + pos_offset)
            r = r_global * R.from_matrix(T_obj[:3, :3])
            xyzw = r.as_quat()
            wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
            qpos_obj_right[i, 3:] = wxyz
        else:  # (N, 3) position only
            qpos_obj_right[i, :3] = r_global.apply(right_obj_trajectory[i] + pos_offset)
            qpos_obj_right[i, 3:] = unit_quat

        if len(left_obj_trajectory.shape) == 3:  # (N, 4, 4)
            T_obj = left_obj_trajectory[i]
            qpos_obj_left[i, :3] = r_global.apply(T_obj[:3, 3] + pos_offset)
            r = r_global * R.from_matrix(T_obj[:3, :3])
            xyzw = r.as_quat()
            wxyz = np.concatenate([xyzw[3:], xyzw[:3]])
            qpos_obj_left[i, 3:] = wxyz
        else:  # (N, 3) position only
            qpos_obj_left[i, :3] = r_global.apply(left_obj_trajectory[i] + pos_offset)
            qpos_obj_left[i, 3:] = unit_quat

    # add z_offset
    qpos_wrist_right[:, 2] += z_offset
    qpos_wrist_left[:, 2] += z_offset
    qpos_finger_right[:, :, 2] += z_offset
    qpos_finger_left[:, :, 2] += z_offset
    qpos_obj_right[:, 2] += z_offset
    qpos_obj_left[:, 2] += z_offset

    # Save task info
    task_info_path = f"{output_dir}/../task_info.json"
    if os.path.exists(task_info_path):
        # Load existing task info and update it
        with open(task_info_path) as f:
            existing_task_info = json.load(f)
        existing_task_info.update(task_info)
        with open(task_info_path, "w") as f:
            json.dump(existing_task_info, f, indent=2)
    else:
        # Create new task info file
        with open(task_info_path, "w") as f:
            json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Saved task_info to {task_info_path}")

    # Save processed data
    np.savez(
        f"{output_dir}/trajectory_keypoints.npz",
        qpos_wrist_right=qpos_wrist_right,
        qpos_finger_right=qpos_finger_right,
        qpos_obj_right=qpos_obj_right,
        qpos_wrist_left=qpos_wrist_left,
        qpos_finger_left=qpos_finger_left,
        qpos_obj_left=qpos_obj_left,
    )
    loguru.logger.info(f"Saved qpos to {output_dir}/trajectory_keypoints.npz")

    # Rerun visualization
    if show_rerun:
        visualize_rerun(trajectory_data)

    # Visualization (simplified version)
    if show_viewer:
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

        # Load MuJoCo model for visualization
        mj_spec = mujoco.MjSpec.from_file("../assets/mano/empty_scene.xml")

        # Add right object to body "right_object"
        object_right_handle = mj_spec.worldbody.add_body(
            name="right_object",
            mocap=True,
        )
        collision_right_handle = mj_spec.worldbody.add_body(
            name="right_object_collision",
            mocap=False,
        )
        collision_right_handle.add_joint(
            name="right_object_collision_joint",
            type=mujoco.mjtJoint.mjJNT_FREE,
            armature=0.0001,
            frictionloss=0.0001,
        )
        object_right_handle.add_site(
            name="right_object",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.02, 0.03],
            rgba=[1, 0, 0, 1],
            group=0,
        )
        if (
            hand_type in ["right", "bimanual"]
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
            collision_right_handle.add_geom(
                name="right_collision",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="right_object",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                group=0,
                condim=1,
                contype=0,
                conaffinity=0,
            )

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
        collision_left_handle = mj_spec.worldbody.add_body(
            name="left_object_collision",
            mocap=False,
        )
        collision_left_handle.add_joint(
            name="left_object_collision_joint",
            type=mujoco.mjtJoint.mjJNT_FREE,
            armature=0.0001,
            frictionloss=0.0001,
        )
        bimanual_single_object = (
            hand_type == "bimanual" and task_info["left_object_mesh_dir"] is None
        )
        if (
            hand_type in ["left", "bimanual"]
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
            collision_left_handle.add_geom(
                name="left_collision",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="left_object",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                group=0,
                condim=1,
                contype=1,
                conaffinity=1,
            )
        else:
            # add placeholder to avoid simulation error
            collision_left_handle.add_geom(
                name="left_collision",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.01, 0.02, 0.03],
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                group=0,
            )

        # manually specify contact pairs
        fingertips_right = [
            "right_thumb_tip",
            "right_index_tip",
            "right_middle_tip",
            "right_ring_tip",
            "right_pinky_tip",
        ]
        fingertips_left = [
            "left_thumb_tip",
            "left_index_tip",
            "left_middle_tip",
            "left_ring_tip",
            "left_pinky_tip",
        ]
        if hand_type == "right":
            fingertips = fingertips_right
            objects = ["right_collision"]
        elif hand_type == "left":
            fingertips = fingertips_left
            objects = ["left_collision"]
        elif hand_type == "bimanual":
            fingertips = fingertips_right + fingertips_left
            objects = ["right_collision", "left_collision"]
        else:
            raise ValueError(f"Invalid hand type: {hand_type}")

        for fingertip in fingertips:
            for object_name in objects:
                mj_spec.add_pair(
                    name=f"{fingertip}_{object_name}",
                    geomname1=fingertip,
                    geomname2=object_name,
                    condim=1,
                )

        mj_model = mj_spec.compile()
        mj_data = mujoco.MjData(mj_model)

        # Simple visualization loop
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            rate_limiter = RateLimiter(30.0)
            cnt = 0
            while viewer.is_running():
                mj_data.qpos[:7] = qpos_obj_right[cnt]
                mj_data.qpos[7:14] = qpos_obj_left[cnt]
                mj_data.mocap_pos[:] = qpos_list[cnt, :, :3]
                mj_data.mocap_quat[:] = qpos_list[cnt, :, 3:]
                mujoco.mj_step(mj_model, mj_data)
                cnt = (cnt + 1) % N
                viewer.sync()
                rate_limiter.sleep()


def trajectory_from_grasp(data, num_frames=20):
    """Create a trajectory from a single grasp by interpolating between start, grasp, and end poses.
    Uses hand landmarks instead of vertices for position and orientation computation.
    """
    prediction = data["hand_predictions"][0]
    object_pc = data["object_pc"]

    # Use hand landmarks for trajectory generation
    hand_landmarks = prediction["hand_landmarks"]  # Should be (21, 3)

    # Convert to numpy if it's a tensor
    if isinstance(hand_landmarks, torch.Tensor):
        hand_landmarks = hand_landmarks.cpu().numpy()

    # Create three poses: begin (lifted), grasp (at object), end (lifted)
    # Lift amount in z direction
    lift_offset = np.array([0, 0, 0.4])

    hand_landmarks_beg = hand_landmarks + lift_offset
    hand_landmarks_grasp = hand_landmarks
    hand_landmarks_end = hand_landmarks + lift_offset

    # Object trajectory - center position from object point cloud
    object_center = np.mean(object_pc, axis=0)  # Compute center of object

    # Object should stay stationary during pre-grasp and grasp, only move during post-grasp
    object_beg = object_center
    object_grasp = object_center
    object_end = object_center + lift_offset

    # Create lists for interpolation
    hand_landmarks_list = [hand_landmarks_beg, hand_landmarks_grasp, hand_landmarks_end]
    object_list = [object_beg, object_grasp, object_end]

    # Interpolate trajectories
    new_hand_landmarks = interpolate_pointclouds(
        np.array(hand_landmarks_list), num_frames
    )
    new_object_list = interpolate_pointclouds(np.array(object_list), num_frames)

    # Create transformation matrices for object trajectory
    object_trajectory_matrices = []
    for i in range(num_frames):
        T = np.eye(4)
        T[:3, 3] = new_object_list[i]  # Set translation
        object_trajectory_matrices.append(T)

    trajectory_data = {
        "hand_landmarks": new_hand_landmarks,  # (num_frames, 21, 3)
        "object": np.array(object_trajectory_matrices),  # (num_frames, 4, 4)
        "normals": prediction["hand_normals"],
        "faces": prediction["hand_faces"],
    }
    return trajectory_data


def visualize_rerun(trajectory_data):
    """Visualize trajectory data using rerun"""
    rr.init("fair_mon_visualization", spawn=True)
    add_ground()

    hand_landmarks_trajectory = trajectory_data["hand_landmarks"]
    object_trajectory = trajectory_data["object"]

    # Extract object positions from transformation matrices
    if len(object_trajectory.shape) == 3:  # (N, 4, 4)
        object_positions = object_trajectory[:, :3, 3]
    else:  # (N, 3)
        object_positions = object_trajectory

    # Visualize frame by frame
    for i, (landmarks, obj_pos) in enumerate(
        zip(hand_landmarks_trajectory, object_positions, strict=False)
    ):
        rr.log(
            "object", rr.Points3D(obj_pos.reshape(1, 3), colors=[0, 255, 0], radii=0.02)
        )
        rr.log("landmarks", rr.Points3D(landmarks, colors=[255, 0, 0], radii=0.008))
        time.sleep(0.1)


if __name__ == "__main__":
    tyro.cli(main)
