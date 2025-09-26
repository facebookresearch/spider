import pickle
import rerun as rr
import torch
import numpy as np
from scipy.interpolate import interp1d
import time


def add_ground(size=3.0, z_height=0.2):
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
    """
    Interpolates along the first dimension of a (N, P, 3) array to get M frames.

    Args:
        pointclouds (np.ndarray): shape (N, P, 3)
        M (int): number of frames after interpolation

    Returns:
        np.ndarray: shape (M, P, 3)
    """
    N, P, C = pointclouds.shape
    assert C == 3, "Last dimension must be 3 (x, y, z)"

    # Original and target "time" coordinates
    x_old = np.linspace(0, 1, N)
    x_new = np.linspace(0, 1, M)

    # Create interpolation function along axis 0
    f = interp1d(x_old, pointclouds, axis=0, kind="linear")

    return f(x_new)


def load_data(filename="coke.pkl"):
    # Replace 'file.pkl' with your file path
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def trajectory_from_grasp(data):
    prediction = data["hand_predictions"][0]
    object = data["object_pc"]

    hand_vertices_beg = prediction["hand_vertices"] + torch.tensor([0, 0, 0.4]).to(
        prediction["hand_vertices"].dtype
    )
    hand_vertices_grasp = prediction["hand_vertices"]
    hand_vertices_end = prediction["hand_vertices"] + torch.tensor([0, 0, 0.4]).to(
        prediction["hand_vertices"].dtype
    )

    object_beg = object
    object_grasp = object
    object_end = object + np.array([0, 0, 0.4])
    hand_list = [hand_vertices_beg, hand_vertices_grasp, hand_vertices_end]
    object_list = [object_beg, object_grasp, object_end]
    for counter, _ in enumerate(hand_list):
        hand_list[counter] = hand_list[counter].cpu().numpy()

    new_hand_list = interpolate_pointclouds(np.array(hand_list), 20)
    new_object_list = interpolate_pointclouds(np.array(object_list), 20)
    trajectory_data = {
        "hand": new_hand_list,
        "object": new_object_list,
        "normals": prediction["hand_normals"],
        "faces": prediction["hand_faces"],
    }
    return trajectory_data


def visualize(trajectory_data):
    rr.init("rerun_example_points3d", spawn=True)
    add_ground()

    prediction = data["hand_predictions"][0]
    object = data["object_pc"]

    # hand mesh
    hand_vertices_beg = prediction["hand_vertices"] + torch.tensor([0, 0, 0.4]).to(
        prediction["hand_vertices"].dtype
    )
    hand_vertices_grasp = prediction["hand_vertices"]
    hand_vertices_end = prediction["hand_vertices"] + torch.tensor([0, 0, 0.4]).to(
        prediction["hand_vertices"].dtype
    )

    # hand landmarks
    hand_landmarks_beg = prediction["hand_landmarks"] + torch.tensor([0, 0, 0.4]).to(
        prediction["hand_landmarks"].dtype
    )
    hand_landmarks_grasp = prediction["hand_landmarks"]
    hand_landmarks_end = prediction["hand_landmarks"] + torch.tensor([0, 0, 0.4]).to(
        prediction["hand_landmarks"].dtype
    )

    # object pointcloud
    object_beg = object
    object_grasp = object
    object_end = object + np.array([0, 0, 0.4])

    hand_list = [hand_vertices_beg, hand_vertices_grasp, hand_vertices_end]
    landmarks_list = [hand_landmarks_beg, hand_landmarks_grasp, hand_landmarks_end]
    object_list = [object_beg, object_grasp, object_end]
    for counter, _ in enumerate(hand_list):
        hand_list[counter] = hand_list[counter].cpu().numpy()
        landmarks_list[counter] = landmarks_list[counter].cpu().numpy()

    new_hand_list = interpolate_pointclouds(np.array(hand_list), 20)
    new_landmarks_list = interpolate_pointclouds(np.array(landmarks_list), 20)
    new_object_list = interpolate_pointclouds(np.array(object_list), 20)

    # for prediction in data["hand_predictions"]:
    for object, hand, landmarks in zip(
        new_object_list, new_hand_list, new_landmarks_list
    ):
        rr.log("points", rr.Points3D(object))
        rr.log(
            "triangle",
            rr.Mesh3D(
                vertex_positions=hand,
                vertex_normals=prediction["hand_normals"],
                triangle_indices=prediction["hand_faces"],
            ),
        )
        rr.log("landmarks", rr.Points3D(landmarks, colors=[255, 0, 0], radii=0.008))
        time.sleep(0.2)


if __name__ == "__main__":
    data = load_data(
        filename="/home/pcy/Research/code/retarget/example_datasets/raw/fair_mon/cat_right/0.pkl"
    )
    trajectory_data = trajectory_from_grasp(data)
    visualize(trajectory_data)
