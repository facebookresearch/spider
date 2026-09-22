# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Download the small conversion manifest and recover archived robot assets."""

from __future__ import annotations

import json
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tyro

SOURCE_DATASETS = {
    "dexycb_lifted": "dexycb_lifted/right",
    "oakink_lifted": "oakink_3d_hold_lifted",
    "hrdexdb_24f": "HRDexDB_lifted",
    "hot3d_v2": "hot3d_3d_v2/right",
}


def main(
    output_dir: Path,
    manifest: Path = Path("examples/lifted_bench/smoke_manifest.json"),
    profile: str = "",
    source_uri: str = "s3://far-research-internal/fhogan/spider/arxiv/data",
    code_uri: str = "s3://far-research-internal/fhogan/spider_full_bench/code/spider_src.tar.gz",
) -> None:
    """Fetch raw source packages and archived runs using the authenticated AWS CLI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(manifest.read_text())
    aws = ["aws", *(["--profile", profile] if profile else []), "s3"]
    jobs = []
    raw_jobs = set()
    for row in rows:
        dataset, robot, task = (row[k] for k in ("dataset", "robot_type", "task"))
        if row["seed"] != 0 or row["data_id"] != 0 or row["embodiment_type"] != "right":
            raise ValueError(
                "Fetcher currently supports the seed-0/right-hand smoke manifest"
            )
        remote = f"{source_uri}/retarget/{row['campaign']}/processed/{dataset}/{robot}/right/{task}/0/"
        local = output_dir / "runs" / dataset / robot / task
        jobs.append(
            [
                *aws,
                "sync",
                "--quiet",
                remote,
                str(local),
                "--exclude",
                "*",
                *[
                    arg
                    for pattern in ("scene.xml", "config.yaml", "RECEIPT.json", "*.npz")
                    for arg in ("--include", pattern)
                ],
            ]
        )
        raw_jobs.add((dataset, task))
    for dataset, task in sorted(raw_jobs):
        remote = f"{source_uri}/3d/{SOURCE_DATASETS[dataset]}/{task}/3d/"
        local = output_dir / "raw" / dataset / task / "3d"
        jobs.append(
            [
                *aws,
                "sync",
                "--quiet",
                remote,
                str(local),
                "--exclude",
                "*",
                "--include",
                "seed_default_lifted_centered/*",
                "--include",
                "seed_default_centered/*",
            ]
        )
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda command: subprocess.run(command, check=True), jobs))
    archive = output_dir / "spider_src.tar.gz"
    subprocess.run([*aws, "cp", "--quiet", code_uri, str(archive)], check=True)
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            prefix = "spider-src/spider/assets/robots/"
            if member.isfile() and member.name.startswith(prefix):
                relative = Path(member.name).relative_to("spider-src")
                if ".." in relative.parts:
                    raise ValueError(f"Invalid archive member: {member.name}")
                target = output_dir / "code" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("wb") as dest:
                    import shutil

                    shutil.copyfileobj(source, dest)
    print(
        f"Downloaded {len(rows)} runs and {len(raw_jobs)} source clips to {output_dir}"
    )


if __name__ == "__main__":
    tyro.cli(main)
