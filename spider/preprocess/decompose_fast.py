try:
    import trimesh
except ModuleNotFoundError:
    print("trimesh is required. Please install with `pip install trimesh`")
    raise SystemExit(1)

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import tyro
from loguru import logger


MeshPart = Tuple[np.ndarray, np.ndarray]


def fast_voxel_convex_decomp_from_pointcloud(
    points: np.ndarray, pitch: float = 0.1, min_points: int = 20
) -> List[MeshPart]:
    """Approximate convex decomposition via voxel clusters and convex hulls."""

    coords = np.floor(points / pitch).astype(int)
    unique_voxels, inverse = np.unique(coords, axis=0, return_inverse=True)

    hulls: List[MeshPart] = []
    for idx, _ in enumerate(unique_voxels):
        cluster_points = points[inverse == idx]
        if len(cluster_points) < min_points:
            continue

        cluster_mesh = trimesh.Trimesh(vertices=cluster_points, faces=[])
        hull = cluster_mesh.convex_hull
        vertices = np.asarray(hull.vertices)
        faces = np.asarray(hull.faces, dtype=int)
        hulls.append((vertices, faces))

    return hulls


def flatten_base(hulls: Iterable[MeshPart], thickness: float = 0.01) -> List[MeshPart]:
    """Append a thin plate that flattens the bottom of the decomposition."""

    hull_list = list(hulls)
    if not hull_list:
        return hull_list

    all_vertices = np.vstack([vertices for vertices, _ in hull_list])
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
    min_z = np.min(all_vertices[:, 2])

    z0 = min_z
    z1 = min_z + thickness
    plate_vertices = np.array(
        [
            [min_x, min_y, z0],
            [max_x, min_y, z0],
            [max_x, max_y, z0],
            [min_x, max_y, z0],
            [min_x, min_y, z1],
            [max_x, min_y, z1],
            [max_x, max_y, z1],
            [min_x, max_y, z1],
        ]
    )
    plate_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=int,
    )

    hull_list.append((plate_vertices, plate_faces))
    return hull_list


def main(
    dataset_dir: str = "../../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    hand_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    data_id: int = 0,
) -> None:
    dataset_path = Path(dataset_dir).expanduser().resolve()

    if hand_type == "right":
        hands = ["right"]
    elif hand_type == "left":
        hands = ["left"]
    elif hand_type == "bimanual":
        hands = ["right", "left"]
    else:
        raise ValueError(f"Invalid hand type: {hand_type}")

    processed_dir = (
        dataset_path
        / "processed"
        / dataset_name
        / "mano"
        / hand_type
        / task
        / str(data_id)
    )
    task_info_path = processed_dir.parent / "task_info.json"

    if not task_info_path.exists():
        logger.error(
            "Missing task_info at {}. Run dataset preprocessing first.",
            task_info_path,
        )
        return

    with task_info_path.open("r", encoding="utf-8") as file:
        task_info = json.load(file)

    for hand in hands:
        mesh_dir_key = (
            "right_object_mesh_dir" if hand == "right" else "left_object_mesh_dir"
        )
        mesh_dir = task_info.get(mesh_dir_key)
        if not mesh_dir:
            logger.warning("No mesh_dir for {} hand; skipping.", hand)
            continue

        mesh_path = Path(mesh_dir)
        input_file = mesh_path / "visual.obj"
        output_dir = mesh_path / "convex"

        if not input_file.exists():
            logger.warning("Input mesh {} does not exist. Skipping {} hand.", input_file, hand)
            continue

        mesh = trimesh.load(
            str(input_file), force="mesh", process=False, skip_materials=True
        )

        hulls = fast_voxel_convex_decomp_from_pointcloud(np.asarray(mesh.vertices))
        if not hulls:
            logger.warning("No convex parts generated for {}; skipping export.", hand)
            continue

        hulls = flatten_base(hulls)
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (vertices, faces) in enumerate(hulls):
            mesh_part = trimesh.Trimesh(vertices, faces)
            part_path = output_dir / f"{idx}.obj"
            mesh_part.export(part_path)
            logger.info("Exported mesh part {} to {}", idx, part_path)

        convex_key = (
            "right_object_convex_dir" if hand == "right" else "left_object_convex_dir"
        )
        task_info[convex_key] = str(output_dir)

    with task_info_path.open("w", encoding="utf-8") as file:
        json.dump(task_info, file, indent=2)

    logger.info("Updated task_info with convex dirs at {}", task_info_path)


if __name__ == "__main__":
    tyro.cli(main)
