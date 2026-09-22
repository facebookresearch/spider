# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the repository root.

"""Download the public full release in bulk and verify its ordinary file layout."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tyro
from huggingface_hub import HfApi

from examples.lifted_bench.full_release import dump
from examples.lifted_bench.full_roundtrip import REPO, download


def main(
    output_dir: Path = Path("example_datasets/retarget_full"),
    revision: str = "main",
) -> None:
    """Fetch a pinned archive, check its files against the repo, and prepare Viser.

    Use an empty output directory initially; interrupted transfers can resume.
    Transfer checkpoints and the archive are retained in a sibling hidden folder.
    """
    output_dir = output_dir.resolve()
    report = output_dir.parent / f".{output_dir.name}-download"
    report.mkdir(parents=True, exist_ok=True)
    state = report / "candidate.json"
    resolved = HfApi().repo_info(REPO, repo_type="dataset", revision=revision).sha
    if state.exists() and json.loads(state.read_text())["data_revision"] != resolved:
        raise ValueError(
            "This directory belongs to another revision; use its pinned revision or a new output directory"
        )
    dump(state, {"repo_id": REPO, "data_revision": resolved})
    download(output_dir, report)
    hashes = json.loads((output_dir / "checksums.json").read_text())

    def check(item):
        relative, expected = item
        path = (output_dir / relative).resolve()
        path.relative_to(output_dir)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Extracted file checksum mismatch: {relative}")

    with ThreadPoolExecutor(16) as executor:
        list(executor.map(check, hashes.items()))
    print(f"Downloaded and checked {REPO}@{resolved} into {output_dir}")
    print(
        f"Inspect with: uv run examples/inspect_dataset.py --dataset-dir {output_dir}"
    )


if __name__ == "__main__":
    tyro.cli(main)
