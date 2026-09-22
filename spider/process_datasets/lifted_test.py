# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Regression checks for lifted source units and synthetic-lift conversion."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from spider.process_datasets.lifted import append_lift_to_threshold, main


class LiftedConversionTest(unittest.TestCase):
    """Check physical invariants that can silently corrupt converted demonstrations."""

    def test_synthetic_lift_preserves_hand_object_offsets(self):
        """Appending a lift must translate the hand and object together."""
        hand = np.zeros((3, 21, 3))
        hand[:, :, 0] = 0.1
        objects = np.repeat(np.eye(4)[None], 3, axis=0)
        lifted_hand, lifted_objects = append_lift_to_threshold(
            hand, objects, 0.3, 0.3, 30.0
        )
        np.testing.assert_allclose(lifted_hand[:3], hand)
        np.testing.assert_allclose(lifted_objects[:3], objects)
        np.testing.assert_allclose(
            lifted_hand[:, :, :3] - lifted_objects[:, None, :3, 3],
            np.broadcast_to([0.1, 0, 0], lifted_hand.shape),
        )
        self.assertGreaterEqual(lifted_objects[-1, 2, 3], 0.3 - 1e-12)

    def test_glb_node_scale_and_ground_alignment(self):
        """A millimetre GLB under a scale node must produce a metre-scale scene."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant = root / "source" / "sample" / "3d" / "seed_default_centered"
            variant.mkdir(parents=True)
            scene = trimesh.Scene()
            transform = np.diag([0.001, 0.001, 0.001, 1.0])
            scene.add_geometry(
                trimesh.creation.box(extents=[100, 100, 100]), transform=transform
            )
            (variant / "object_1_mesh.glb").write_bytes(scene.export(file_type="glb"))
            hand = np.zeros((10, 21, 3))
            hand[:, 9, 2] = 0.1
            hand[:, 5, 1] = 0.1
            hand[:, 13, 1] = -0.1
            hand[4] = np.nan  # Sparse tracking dropout is interpolated.
            np.savez(
                variant / "data.npz",
                hand_right_keypoints=hand,
                object_1_transforms=np.repeat(np.eye(4)[None], 10, axis=0),
            )
            main(str(root / "source"), "sample", "test", str(root / "output"))
            output = root / "output" / "processed" / "test"
            mesh = trimesh.load(
                output / "assets" / "objects" / "sample" / "visual.obj", force="mesh"
            )
            np.testing.assert_allclose(mesh.extents, [0.1, 0.1, 0.1], atol=1e-7)
            with np.load(
                output / "mano/right/sample/0/trajectory_keypoints.npz"
            ) as data:
                self.assertTrue(np.isfinite(data["qpos_wrist_right"]).all())
                np.testing.assert_allclose(
                    data["qpos_obj_right"][:, 2], 0.05, atol=1e-7
                )
                np.testing.assert_allclose(
                    np.linalg.norm(data["qpos_wrist_right"][:, 3:], axis=1), 1
                )


if __name__ == "__main__":
    unittest.main()
