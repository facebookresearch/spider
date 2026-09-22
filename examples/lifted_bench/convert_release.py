# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Convert downloaded lifted runs into the current SPIDER example-dataset layout.

Inputs: raw/<dataset>/<task>/3d/<variant>/ and
runs/<dataset>/<robot>/<task>/{scene.xml,config.yaml,RECEIPT.json,*.npz}.
Robot assets must come from the benchmark code archive (including Sharpa).
This converts source keypoints and reconstructs assets; it does not rerun MPC.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import fields
from pathlib import Path

import mujoco
import numpy as np
import tyro
import yaml

from spider.config import Config
from spider.preprocess.decompose_fast import main as decompose
from spider.process_datasets.lifted import main as process_lifted


def sha256(path: Path) -> str:
    """Hash a file without loading the whole file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write readable release metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def copy_scene_assets(scene: Path, asset_dir: Path, robot_assets: Path) -> None:
    """Materialize referenced robot files and reject missing object dependencies."""
    root = ET.parse(scene).getroot()
    for element in root.iter():
        filename = element.get("file")
        if filename is None:
            continue
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Nonportable scene dependency: {filename}")
        destination = asset_dir / relative
        if relative.parts[0] == "robots":
            source = robot_assets.joinpath(*relative.parts[1:])
            if not source.is_file():
                raise FileNotFoundError(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and sha256(destination) != sha256(source):
                raise ValueError(f"Conflicting shared asset: {destination}")
            shutil.copy2(source, destination)
        if not destination.is_file():
            raise FileNotFoundError(f"Missing scene dependency: {destination}")


def recover_sparse_convex(source_task: Path, object_dir: Path) -> dict[str, str]:
    """Recover source convex meshes only when voxel decomposition produced none.

    The archived visual OBJ must match the converted GLB vertices, establishing
    the coordinate system before using its accompanying convex parts. This does
    not establish equivalence to the original optimizer's collision geometry.
    """
    import trimesh

    if any((object_dir / "convex").glob("*.obj")):
        return {}
    for variant in ("seed_default_lifted_centered", "seed_default_centered"):
        source = source_task / "3d" / variant
        if (source / "data.npz").exists():
            break
    parts = sorted((source / "convex").glob("*.obj"))
    if not parts or not (source / "visual.obj").is_file():
        raise FileNotFoundError(
            f"Sparse mesh needs archived visual.obj and convex/*.obj: {source}"
        )
    vertices = []
    for path in (source / "visual.obj", object_dir / "visual.obj"):
        mesh = trimesh.load(path, force="mesh", process=False, skip_materials=True)
        vertices.append(np.unique(np.round(np.asarray(mesh.vertices), 7), axis=0))
    if vertices[0].shape != vertices[1].shape or not np.allclose(
        vertices[0], vertices[1], atol=1e-7, rtol=0
    ):
        raise ValueError(
            f"Archived convex mesh coordinate system is not verified: {source}"
        )
    (object_dir / "convex").mkdir(exist_ok=True)
    hashes = {"visual.obj": sha256(source / "visual.obj")}
    for path in parts:
        mesh = trimesh.load(path, force="mesh", process=False, skip_materials=True)
        if not np.isfinite(mesh.vertices).all() or len(mesh.faces) < 4:
            raise ValueError(f"Invalid archived convex part: {path}")
        shutil.copy2(path, object_dir / "convex" / path.name)
        hashes[f"convex/{path.name}"] = sha256(path)
    return hashes


def main(
    source_dir: Path,
    robot_assets: Path,
    output_dir: Path,
    manifest: Path = Path("examples/lifted_bench/smoke_manifest.json"),
) -> None:
    """Convert a manifest, preserving the standard processed directory layout.

    Use a new output directory: rerunning into an existing release is rejected to
    avoid silently changing previously validated assets or mixing manifests.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must be empty; use a new release directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(manifest.read_text())
    prepared = set()
    recovered_assets = {}
    entries = []
    config_fields = {field.name for field in fields(Config)}
    for row in rows:
        dataset, task, robot = (row[k] for k in ("dataset", "task", "robot_type"))
        side, data_id = row["embodiment_type"], row["data_id"]
        if side != "right" or data_id != 0 or row["seed"] not in (0, 1):
            raise ValueError("Supported archive runs are right-hand seeds 0 and 1")
        for part in (dataset, task, robot):
            if not part or Path(part).name != part or part in {".", ".."}:
                raise ValueError(f"Invalid path component: {part}")
        source = source_dir / "runs" / dataset / robot / task
        receipt = json.loads((source / "RECEIPT.json").read_text())
        if (receipt["dataset"], receipt["seq"], receipt["hand"]) != (
            row.get("source_dataset", dataset),
            task,
            robot,
        ):
            raise ValueError(f"Receipt identity mismatch: {source}")
        if not (
            receipt.get("stage") == "done"
            and receipt.get("success") is True
            and receipt.get("bimanual") is False
            and receipt.get("data_weak") is False
        ):
            print(f"Excluded unsuccessful or unscored source run: {source}", flush=True)
            continue
        dataset_root = output_dir / "processed" / dataset
        task_dir = dataset_root / robot / side / task
        trial_dir = task_dir / str(data_id)
        trial_dir.mkdir(parents=True)

        if (dataset, task) not in prepared:
            process_lifted(
                source_dir=str(source_dir / "raw" / dataset),
                sequence=task,
                dataset_name=dataset,
                dataset_dir=str(output_dir),
                embodiment_type=side,
                ensure_lift=0.30,
                fixed_num_frames=24 if dataset == "hrdexdb_24f" else 0,
            )
            # The source processor may copy archived convex parts. Remove them
            # before regeneration so an empty decomposition takes the explicit,
            # coordinate-checked recovery path and records its provenance.
            convex_dir = dataset_root / "assets" / "objects" / task / "convex"
            if convex_dir.exists():
                shutil.rmtree(convex_dir)
            decompose(
                dataset_dir=str(output_dir),
                dataset_name=dataset,
                embodiment_type=side,
                task=task,
            )
            recovered_assets[(dataset, task)] = recover_sparse_convex(
                source_dir / "raw" / dataset / task,
                dataset_root / "assets" / "objects" / task,
            )
            prepared.add((dataset, task))

        # Same task-level scene placement as retarget/retarget_example.
        scene = task_dir / "scene.xml"
        shutil.copy2(source / "scene.xml", scene)
        copy_scene_assets(scene, dataset_root / "assets", robot_assets)
        model = mujoco.MjModel.from_xml_path(str(scene))
        for name in ("trajectory_mjwp.npz", "trajectory_kinematic.npz"):
            shutil.copy2(source / name, trial_dir / name)
            with np.load(trial_dir / name, allow_pickle=False) as data:
                if data["qpos"].shape[-1] != model.nq:
                    raise ValueError(f"Scene/trajectory dimensions differ: {trial_dir}")

        info = json.loads(
            (dataset_root / "mano" / side / task / "task_info.json").read_text()
        )
        info.update(robot_type=robot, hand_type=side, data_id=data_id, sim_dt=0.01)
        write_json(task_dir / "task_info.json", info)
        config = yaml.safe_load((source / "config.yaml").read_text())
        config = {k: v for k, v in config.items() if k in config_fields}
        config.update(
            dataset_name=dataset,
            dataset_dir=".",
            model_path=scene.relative_to(output_dir).as_posix(),
            data_path=(trial_dir / "trajectory_kinematic.npz")
            .relative_to(output_dir)
            .as_posix(),
            output_dir=trial_dir.relative_to(output_dir).as_posix(),
            load_config_path="",
        )
        (trial_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        public_receipt = {
            k: v for k, v in receipt.items() if k not in {"host", "shard", "error"}
        }
        write_json(trial_dir / "metrics.json", public_receipt)
        entry = dict(row)
        entry.update(
            path=trial_dir.relative_to(output_dir).as_posix(),
            scene=scene.relative_to(output_dir).as_posix(),
            source_sha256={
                name: sha256(source / name)
                for name in (
                    "trajectory_mjwp.npz",
                    "trajectory_kinematic.npz",
                    "scene.xml",
                    "config.yaml",
                    "RECEIPT.json",
                )
            },
            asset_recovery="Object meshes regenerated with the benchmark baseline processor and current decompose_fast; robot meshes recovered from benchmark archive.",
            validation_scope="state replay; original collision geometry equivalence is not established",
            sim_dt=0.01,
            ref_dt=0.08,
        )
        if recovered_assets[(dataset, task)]:
            entry["source_asset_sha256"] = recovered_assets[(dataset, task)]
            entry["asset_recovery"] = (
                "Sparse object mesh: archived source convex parts recovered after matching source OBJ vertices to converted GLB; robot meshes recovered from benchmark archive."
            )
        entries.append(entry)
        print(f"Converted {dataset}/{robot}/{task}", flush=True)
    if not entries:
        raise ValueError("No successful, scored trajectories in the source manifest")
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "layout": "retarget_example",
            "subset": "smoke",
            "successful_only": True,
            "selection": "one sequence successful for all four robots per source; not representative",
            "runs": entries,
        },
    )
    for dataset in sorted({r["dataset"] for r in entries}):
        write_json(
            output_dir / "processed" / dataset / "dataset_summary.json",
            {
                "dataset_name": dataset,
                "runs": [r for r in entries if r["dataset"] == dataset],
            },
        )
    write_json(
        output_dir / "checksums.json",
        {
            p.relative_to(output_dir).as_posix(): sha256(p)
            for p in sorted(output_dir.rglob("*"))
            if p.is_file()
        },
    )


if __name__ == "__main__":
    tyro.cli(main)
