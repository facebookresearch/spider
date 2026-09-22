# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Verify a downloaded release and render every trial with SPIDER's default renderer."""

from __future__ import annotations

import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro

from examples.lifted_bench.convert_release import sha256, write_json
from spider.trajectory import load_saved_trajectory
from spider.viewers import render_image, setup_renderer


def verify_paths(scene: Path, root: Path) -> None:
    """Ensure every external MJCF dependency resolves inside the downloaded root."""
    xml = ET.parse(scene).getroot()
    compiler = xml.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    for node in xml.iter():
        filename = node.get("file")
        if filename is None:
            continue
        directory = (
            meshdir
            if node.tag == "mesh"
            else texturedir
            if node.tag == "texture"
            else ""
        )
        path = (scene.parent / directory / filename).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)


def main(
    dataset_dir: Path,
    output_dir: Path,
    render: bool = True,
    selection: Path | None = None,
    verify_checksums: bool = True,
) -> None:
    """Check file hashes, current-loader compatibility, metrics, and fresh rendering."""
    dataset_dir = dataset_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    checksums = (
        json.loads((dataset_dir / "checksums.json").read_text())
        if verify_checksums
        else {}
    )
    for relative, expected in checksums.items():
        path = (dataset_dir / relative).resolve()
        path.relative_to(dataset_dir)
        if sha256(path) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    results = []
    cards = []
    rows = (
        json.loads(selection.read_text()) if selection is not None else manifest["runs"]
    )
    for row in rows:
        verify_paths(dataset_dir / row["scene"], dataset_dir)
        run = load_saved_trajectory(
            dataset_dir,
            dataset_name=row["dataset"],
            robot_type=row["robot_type"],
            embodiment_type=row["embodiment_type"],
            task=row["task"],
            data_id=row["data_id"],
        )
        # Reproduce the lifted-benchmark metric, distinct from the renderer's
        # optimizer reference interpolation in spider.io.load_data.
        reference_xyz = run.reference_source[:, -7:-4]
        sim_xyz = run.qpos[:, -7:-4]
        ref_at_sim = np.stack(
            [
                np.interp(
                    np.arange(len(sim_xyz)) * run.config.sim_dt,
                    np.arange(len(reference_xyz)) * run.config.ref_dt,
                    reference_xyz[:, i],
                )
                for i in range(3)
            ],
            axis=1,
        )
        sim_lift = float(sim_xyz[-1, 2] - sim_xyz[0, 2])
        final_err = float(np.linalg.norm(sim_xyz[-1] - ref_at_sim[-1]))
        ref_lift = float(reference_xyz[-1, 2] - reference_xyz[0, 2])
        success = sim_lift >= 0.10 and final_err <= 0.10
        for name, value in (
            ("sim_lift", sim_lift),
            ("final_err", final_err),
            ("ref_lift", ref_lift),
        ):
            if not np.isclose(value, run.metrics[name], atol=0.000051, rtol=0):
                raise ValueError(f"{row['path']}: {name} differs from source receipt")
        if (
            success != run.metrics["success"]
            or (ref_lift < 0.15) != run.metrics["data_weak"]
        ):
            raise ValueError(f"{row['path']}: metric classification differs")
        slug = f"{row['dataset']}__{row['robot_type']}__{row['task']}__{row['data_id']}"
        video_path = output_dir / f"{slug}.mp4"
        if render:
            data, ref = mujoco.MjData(run.model), mujoco.MjData(run.model)
            renderer = setup_renderer(run.config, run.model)
            stride = max(1, round(run.config.render_dt / run.config.sim_dt))
            try:
                with imageio.get_writer(
                    video_path,
                    fps=1 / (stride * run.config.sim_dt),
                    macro_block_size=16,
                ) as writer:
                    for index in range(0, len(run.qpos), stride):
                        data.qpos[:] = run.qpos[index]
                        data.qvel[:] = run.qvel[index]
                        data.ctrl[:] = run.ctrl[index]
                        ref.qpos[:] = run.reference[index]
                        frame = render_image(run.config, renderer, run.model, data, ref)
                        writer.append_data(frame)
                        if index == ((len(run.qpos) - 1) // stride) * stride:
                            imageio.imwrite(output_dir / f"{slug}.jpg", frame)
            finally:
                renderer.close()
        result = {
            "dataset": row["dataset"],
            "robot": row["robot_type"],
            "task": row["task"],
            "nq": run.model.nq,
            "nv": run.model.nv,
            "nu": run.model.nu,
            "frames": len(run.qpos),
            "sim_lift": sim_lift,
            "final_err": final_err,
            "success": success,
            "hashes_verified": verify_checksums,
            "loader_passed": True,
            "rendered": render,
            "video": video_path.name if render else None,
        }
        results.append(result)
        label = html.escape(f"{row['dataset']} / {row['robot_type']}")
        cards.append(
            f'<article><h2>{label}</h2><video controls loop muted preload="metadata" poster="{slug}.jpg" src="{slug}.mp4"></video><p>{len(run.qpos)} frames · lift {sim_lift:.3f} m · error {final_err:.4f} m</p></article>'
        )
        print(
            f"PASS {slug}: {len(run.qpos)} frames, lift={sim_lift:.4f}, error={final_err:.4f}",
            flush=True,
        )
        write_json(
            output_dir / "verification.json",
            {"files_verified": len(checksums), "runs": results},
        )
    (output_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>SPIDER dataset smoke test</title>'
        "<style>body{font:16px system-ui;margin:32px;background:#f4f6f8;color:#18202b}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}"
        "article{background:white;padding:18px;border-radius:12px}video{width:100%}</style>"
        "<h1>SPIDER · converted dataset smoke test</h1>"
        "<p>Fresh local renders of saved trajectories. Reference left; simulation right. "
        "Each file was checked with the current SPIDER loader. State replay does not establish dynamics reproducibility.</p>"
        "<main>" + "".join(cards) + "</main>"
    )


if __name__ == "__main__":
    tyro.cli(main)
