try:
    import trimesh
except ModuleNotFoundError:
    print("trimesh is required. Please install with `pip install trimesh`")
    exit(1)

import os
import json
import coacd
import tyro
import loguru

import spider
from spider.io import get_mesh_dir


def main(
    dataset_dir: str = f"{spider.ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    hand_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    data_id: int = 0,
):
    dataset_dir = os.path.abspath(dataset_dir)
    if hand_type == "right":
        hands = ["right"]
    elif hand_type == "left":
        hands = ["left"]
    elif hand_type == "bimanual":
        hands = ["right", "left"]
    else:
        raise ValueError(f"Invalid hand type: {hand_type}")

    # load task info produced during dataset preprocessing
    processed_dir = (
        f"{dataset_dir}/processed/{dataset_name}/mano/{hand_type}/{task}/{data_id}"
    )
    task_info_path = f"{processed_dir}/../task_info.json"
    if not os.path.exists(task_info_path):
        loguru.logger.error(
            f"Missing task_info at {task_info_path}. Run dataset preprocessing first."
        )
        return
    with open(task_info_path, "r") as f:
        task_info = json.load(f)

    for hand in hands:
        if hand == "right":
            mesh_dir = task_info.get("right_object_mesh_dir")
        else:
            mesh_dir = task_info.get("left_object_mesh_dir")
        if mesh_dir is None:
            loguru.logger.warning(f"No mesh_dir for {hand} hand; skipping.")
            continue
        input_file = f"{mesh_dir}/visual.obj"
        output_dir = f"{mesh_dir}/convex"
        if not os.path.exists(input_file):
            loguru.logger.warning(
                f"Input mesh {input_file} does not exist. Skipping {hand} hand."
            )
            continue

        mesh = trimesh.load(
            input_file, force="mesh", process=False, skip_materials=True
        )
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        result = coacd.run_coacd(
            mesh,
            threshold=0.07,
            max_convex_hull=8,
            preprocess_mode="auto",
            preprocess_resolution=50,
            resolution=2000,
            mcts_nodes=50,
            mcts_iterations=200,
            mcts_max_depth=5,
            pca=False,
            merge=True,
            decimate=True,
            max_ch_vertex=32,
            extrude=True,
            extrude_margin=0.1,
            apx_mode="ch",
            seed=1,
        )
        # ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, (vs, fs) in enumerate(result):
            mesh_part = trimesh.Trimesh(vs, fs)
            part_filename = f"{output_dir}/{i}.obj"
            mesh_part.export(part_filename)
            loguru.logger.info(f"Exported mesh part {i} to {part_filename}")

        # persist decomposed path back to task_info for future reference
        key = "right_object_convex_dir" if hand == "right" else "left_object_convex_dir"
        task_info[key] = output_dir

    # save updated task_info
    with open(task_info_path, "w") as f:
        json.dump(task_info, f, indent=2)
    loguru.logger.info(f"Updated task_info with convex dirs at {task_info_path}")


if __name__ == "__main__":
    tyro.cli(main)
