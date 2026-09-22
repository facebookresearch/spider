# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Upload a converted smoke subset, download its pinned commit, and verify locally."""

from __future__ import annotations

import json
from pathlib import Path

import tyro
from huggingface_hub import HfApi, snapshot_download

from examples.lifted_bench.convert_release import write_json
from examples.lifted_bench.verify_release import main as verify


def require_successful_subset(stage_dir: Path) -> int:
    """Refuse failed/unscored runs and any trajectories absent from the manifest."""
    manifest = json.loads((stage_dir / "manifest.json").read_text())
    expected = set()
    if not manifest["runs"]:
        raise ValueError("No trajectories selected for upload")
    for row in manifest["runs"]:
        trial = stage_dir / row["path"]
        trial.resolve().relative_to(stage_dir.resolve())
        metrics = json.loads((trial / "metrics.json").read_text())
        if not (
            metrics.get("stage") == "done"
            and metrics.get("success") is True
            and metrics.get("bimanual") is False
            and metrics.get("data_weak") is False
        ):
            raise ValueError(f"Upload rejected: unsuccessful or unscored {row['path']}")
        expected.add(trial / "trajectory_mjwp.npz")
        expected.add(trial / "trajectory_kinematic.npz")
        expected.add(
            stage_dir
            / "processed"
            / row["dataset"]
            / "mano"
            / row["embodiment_type"]
            / row["task"]
            / str(row["data_id"])
            / "trajectory_keypoints.npz"
        )
    actual = set(stage_dir.rglob("trajectory_*.npz"))
    if actual != expected:
        raise ValueError("Unindexed or missing trajectories in upload directory")
    return len(manifest["runs"])


def main(
    stage_dir: Path,
    download_dir: Path,
    report_dir: Path,
    repo_id: str = "retarget/retarget_full",
    revision: str = "main",
) -> None:
    """Perform the real remote round trip; require an authenticated writable repo."""
    if download_dir.exists() and any(download_dir.iterdir()):
        raise ValueError(
            "download_dir must be empty for an independent round-trip test"
        )
    count = require_successful_subset(stage_dir)
    verify(stage_dir, report_dir / "before_upload", render=False)
    results = json.loads(
        (report_dir / "before_upload" / "verification.json").read_text()
    )
    if not all(row["success"] for row in results["runs"]):
        raise ValueError("Recomputed metrics include unsuccessful trajectories")
    checksums = json.loads((stage_dir / "checksums.json").read_text())
    api = HfApi()
    api.whoami()  # Fail before mutation if the user has not logged in.
    api.repo_info(repo_id, repo_type="dataset", revision=revision)
    # Assets/trajectories first; completion metadata last. No remote deletion.
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        folder_path=str(stage_dir),
        allow_patterns=list(checksums),
        ignore_patterns=["manifest.json", "checksums.json"],
        commit_message=f"Add {count} successful SPIDER examples in the example dataset layout",
    )
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        folder_path=str(stage_dir),
        allow_patterns=["manifest.json", "checksums.json"],
        commit_message="Record manifest and checksums for the converted smoke subset",
    )
    # Local directory starts empty and transfer uses an independent cache.
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=commit.oid,
        local_dir=str(download_dir),
        cache_dir=str(report_dir / "download_cache"),
        allow_patterns=[*checksums, "checksums.json"],
        force_download=True,
    )
    verify(download_dir, report_dir / "after_download", render=True)
    write_json(
        report_dir / "roundtrip.json",
        {
            "repo_id": repo_id,
            "revision": commit.oid,
            "runs": len(
                json.loads((download_dir / "manifest.json").read_text())["runs"]
            ),
            "download_verified": True,
            "rendered": True,
        },
    )
    print(f"Verified https://huggingface.co/datasets/{repo_id}/tree/{commit.oid}")


if __name__ == "__main__":
    tyro.cli(main)
