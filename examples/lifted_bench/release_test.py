# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Regression tests for the successful-only upload boundary."""

import json
import tempfile
import unittest
from pathlib import Path

import trimesh

from examples.lifted_bench.convert_release import recover_sparse_convex
from examples.lifted_bench.full_release import select_runs
from examples.lifted_bench.roundtrip_release import require_successful_subset


class SuccessfulUploadTest(unittest.TestCase):
    """Reject excluded outcomes and trajectories hidden outside the manifest."""

    def test_sparse_mesh_coordinate_check(self):
        """Accept aligned archived convex parts and reject shifted coordinates."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_task = root / "source"
            source = source_task / "3d/seed_default_centered"
            (source / "convex").mkdir(parents=True)
            (source / "data.npz").touch()
            target = root / "target"
            target.mkdir()
            mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.03])
            mesh.export(source / "visual.obj")
            mesh.export(source / "convex/0.obj")
            mesh.export(target / "visual.obj")
            hashes = recover_sparse_convex(source_task, target)
            self.assertIn("convex/0.obj", hashes)
            self.assertTrue((target / "convex/0.obj").is_file())
            (target / "convex/0.obj").unlink()
            mesh.apply_translation([0.2, 0, 0])
            mesh.export(source / "visual.obj")
            with self.assertRaises(ValueError):
                recover_sparse_convex(source_task, target)

    def test_seed_precedence_and_strict_eligibility(self):
        """Prefer seed zero and exclude unknown, weak, or failed outcomes."""
        receipt = {
            "dataset": "hot3d_v2",
            "seq": "episode",
            "hand": "allegro",
            "stage": "done",
            "success": True,
            "bimanual": False,
            "data_weak": False,
        }
        fallback = {**receipt, "dataset": "hot3d_v2_s1"}
        rows, counts = select_runs([("s1full", fallback), ("hv3", receipt)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 0)
        self.assertEqual(counts["duplicate_successes"], 1)
        rows, _ = select_runs(
            [("hv3", {**receipt, "success": False}), ("s1full", fallback)]
        )
        self.assertEqual(rows[0]["seed"], 1)
        self.assertEqual(rows[0]["dataset"], "hot3d_v2")
        self.assertEqual(rows[0]["source_dataset"], "hot3d_v2_s1")
        for key, value in (
            ("stage", "prep"),
            ("success", False),
            ("bimanual", True),
            ("data_weak", True),
            ("success", None),
        ):
            rows, _ = select_runs([("hv3", {**receipt, key: value})])
            self.assertEqual(rows, [])

    def test_upload_selection(self):
        """Only completed, scored successes and their references may be uploaded."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial = root / "processed/test/allegro/right/task/0"
            trial.mkdir(parents=True)
            for name in ("trajectory_mjwp.npz", "trajectory_kinematic.npz"):
                (trial / name).touch()
            human = root / "processed/test/mano/right/task/0"
            human.mkdir(parents=True)
            (human / "trajectory_keypoints.npz").touch()
            row = {
                "dataset": "test",
                "embodiment_type": "right",
                "task": "task",
                "data_id": 0,
                "path": trial.relative_to(root).as_posix(),
            }
            (root / "manifest.json").write_text(json.dumps({"runs": [row]}))
            metrics = {
                "stage": "done",
                "success": True,
                "bimanual": False,
                "data_weak": False,
            }
            metrics_path = trial / "metrics.json"
            metrics_path.write_text(json.dumps(metrics))
            self.assertEqual(require_successful_subset(root), 1)
            for key, value in (
                ("success", False),
                ("stage", "prep"),
                ("bimanual", True),
                ("data_weak", True),
            ):
                with self.subTest(key=key):
                    metrics_path.write_text(json.dumps({**metrics, key: value}))
                    with self.assertRaises(ValueError):
                        require_successful_subset(root)
            metrics_path.write_text(json.dumps(metrics))
            (root / "trajectory_unlisted.npz").touch()
            with self.assertRaises(ValueError):
                require_successful_subset(root)


if __name__ == "__main__":
    unittest.main()
