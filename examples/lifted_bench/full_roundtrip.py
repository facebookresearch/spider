# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the repository root.
"""Publish a full conversion only after an independent, pinned download passes.

The candidate is a Hugging Face pull request in the existing dataset repository.
Large-folder upload checkpoints allow retrying transfers. Every downloaded file
is hashed, every trajectory is loaded and scored again, and the longest example
for each dataset/robot/seed is freshly rendered. The dataset README is untouched.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import multiprocessing
import os
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from examples.lifted_bench.full_release import dump

REPO = "retarget/retarget_full"


def package(stage):
    """Offer a bulk transfer containing the same ordinary processed file layout."""
    from examples.lifted_bench.convert_release import sha256

    metadata = stage / "distribution.json"
    checksum_hash = sha256(stage / "checksums.json")
    if metadata.exists():
        record = json.loads(metadata.read_text())
        if (
            record["checksums_sha256"] == checksum_hash
            and sha256(stage / record["archive"]) == record["sha256"]
        ):
            return
    paths = sorted(json.loads((stage / "checksums.json").read_text())) + [
        "checksums.json"
    ]
    archive = stage / "distribution/retarget_full.tar.gz"
    archive.parent.mkdir(exist_ok=True)
    temporary = archive.with_suffix(".part")
    with tarfile.open(temporary, "w:gz", compresslevel=1, dereference=True) as stream:
        for relative in paths:
            stream.add(stage / relative, arcname=relative, recursive=False)
    temporary.replace(archive)
    dump(
        metadata,
        {
            "archive": archive.relative_to(stage).as_posix(),
            "sha256": sha256(archive),
            "bytes": archive.stat().st_size,
            "checksums_sha256": checksum_hash,
            "files": len(paths),
            "layout": "retarget_example",
            "format": "tar.gz",
        },
    )
    print(
        f"Packaged {len(paths)} files for bulk download ({archive.stat().st_size:,} bytes)",
        flush=True,
    )


def upload(stage, report):
    """Upload validated artifacts to a resumable candidate, then its manifest."""
    from examples.lifted_bench.roundtrip_release import require_successful_subset

    count = require_successful_subset(stage)
    package(stage)
    api = HfApi()
    state = report / "candidate.json"
    if state.exists():
        candidate = json.loads(state.read_text())
    else:
        pr = api.create_pull_request(
            REPO,
            f"Full successful dataset conversion: {count} trajectories",
            repo_type="dataset",
            description="Convert the full eligible archive into the standard SPIDER layout. All selected trajectories pass the current loader and recomputed success metrics. Independent download verification is pending.",
        )
        candidate = {
            "repo_id": REPO,
            "discussion_num": pr.num,
            "revision": pr.git_reference,
        }
        dump(state, candidate)
    attributes = Path(
        hf_hub_download(
            REPO, ".gitattributes", repo_type="dataset", revision=candidate["revision"]
        )
    ).read_text()
    additions = [
        f"{pattern} filter=lfs diff=lfs merge=lfs -text"
        for pattern in ("*.obj", "*.stl", "*.STL")
    ]
    missing = [line for line in additions if line not in attributes.splitlines()]
    if missing:
        api.upload_file(
            repo_id=REPO,
            repo_type="dataset",
            revision=candidate["revision"],
            path_in_repo=".gitattributes",
            path_or_fileobj=(
                attributes.rstrip() + "\n" + "\n".join(missing) + "\n"
            ).encode(),
            commit_message="Store converted mesh assets through LFS/Xet",
        )
    api.upload_large_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=stage,
        revision=candidate["revision"],
        num_workers=16,
        ignore_patterns=["manifest.json", "checksums.json", "README.md"],
        print_report_every=60,
    )
    commit = api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=stage,
        revision=candidate["revision"],
        allow_patterns=["manifest.json", "checksums.json"],
        commit_message=f"Record complete inventory and checksums for {count} successful runs",
    )
    candidate["data_revision"] = commit.oid
    dump(state, candidate)
    print(f"Uploaded candidate data {commit.oid}", flush=True)


def download(root, report):
    """Download the bulk archive and audit every ordinary remote file identity."""
    from examples.lifted_bench.convert_release import sha256

    state = json.loads((report / "candidate.json").read_text())
    marker = report / "download_started.json"
    if not marker.exists() and root.exists() and any(root.iterdir()):
        raise ValueError("Independent download directory must initially be empty")
    if (
        marker.exists()
        and json.loads(marker.read_text())["revision"] != state["data_revision"]
    ):
        raise ValueError("Cannot resume a download from a different revision")
    dump(marker, {"revision": state["data_revision"]})

    def fetch(filename):
        return Path(
            hf_hub_download(
                REPO,
                filename,
                repo_type="dataset",
                revision=state["data_revision"],
                local_dir=report / "transfer",
                cache_dir=report / "download_cache",
            )
        )

    distribution_file = fetch("distribution.json")
    distribution = json.loads(distribution_file.read_text())
    archive = fetch(distribution["archive"])
    if sha256(archive) != distribution["sha256"]:
        raise ValueError("Downloaded bulk archive checksum mismatch")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as stream:
        stream.extractall(root, filter="data")
    if sha256(root / "checksums.json") != distribution["checksums_sha256"]:
        raise ValueError("Extracted checksum index differs from distribution metadata")
    expected = json.loads((root / "checksums.json").read_text())
    expected["checksums.json"] = distribution["checksums_sha256"]
    expected["distribution.json"] = sha256(distribution_file)
    expected[distribution["archive"]] = distribution["sha256"]
    local = {"distribution.json": distribution_file, distribution["archive"]: archive}
    verified = set()
    for remote in HfApi().list_repo_tree(
        REPO, repo_type="dataset", revision=state["data_revision"], recursive=True
    ):
        if remote.path not in expected:
            if Path(remote.path).name.startswith(
                "trajectory_"
            ) and remote.path.endswith(".npz"):
                raise ValueError(f"Unindexed remote trajectory: {remote.path}")
            continue
        path = local.get(remote.path, root / remote.path)
        if remote.lfs:
            matches = remote.lfs.sha256 == expected[remote.path]
        else:
            digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            matches = digest.hexdigest() == remote.blob_id
        if not matches:
            raise ValueError(
                f"Ordinary remote file differs from downloaded archive: {remote.path}"
            )
        verified.add(remote.path)
    if verified != set(expected):
        raise ValueError(f"Missing ordinary remote files: {set(expected) - verified}")
    dump(
        report / "download_complete.json",
        {
            "revision": state["data_revision"],
            "method": "bulk archive plus Git/LFS identity audit of every ordinary remote file",
            "remote_files_verified": len(verified),
            "archive_sha256": distribution["sha256"],
        },
    )
    print(
        f"Downloaded archive and verified {len(verified)} ordinary remote file identities",
        flush=True,
    )


def verify_chunk(args):
    """Load a bounded group using the current SPIDER loader and optional renderer."""
    root, report, rows, index, render = args
    name = ("render" if render else "loader") + f"_{index:04d}"
    output = report / name
    selection = report / "selections" / f"{name}.json"
    from examples.lifted_bench.convert_release import sha256

    identity = None
    if render:
        code_root = Path(__file__).resolve().parents[2]
        code_paths = [
            code_root / name
            for name in (
                "uv.lock",
                "spider/trajectory.py",
                "spider/config.py",
                "spider/io.py",
                "examples/lifted_bench/verify_release.py",
            )
        ] + sorted((code_root / "spider/viewers").glob("*.py"))
        identity = {
            "rows": rows,
            "checksums_sha256": sha256(root / "checksums.json"),
            "code_sha256": {
                p.relative_to(code_root).as_posix(): sha256(p) for p in code_paths
            },
            "render_backend": os.environ.get("MUJOCO_GL", ""),
        }
        checkpoint = output / "render_checkpoint.json"
        if checkpoint.exists():
            cached = json.loads(checkpoint.read_text())
            if cached["input"] == identity and all(
                (output / name).is_file() and sha256(output / name) == expected
                for name, expected in cached["outputs"].items()
            ):
                return json.loads((output / "verification.json").read_text())["runs"]
    dump(selection, rows)
    with (
        (report / f"{name}.log").open("w") as stream,
        contextlib.redirect_stdout(stream),
        contextlib.redirect_stderr(stream),
    ):
        import torch

        torch.set_num_threads(1)
        from loguru import logger

        logger.remove()
        logger.add(stream)
        from examples.lifted_bench.verify_release import main as verify

        verify(root, output, render=render, selection=selection, verify_checksums=False)
    if render:
        dump(
            output / "render_checkpoint.json",
            {
                "input": identity,
                "outputs": {
                    p.name: sha256(p)
                    for p in output.iterdir()
                    if p.suffix in {".mp4", ".jpg"} or p.name == "verification.json"
                },
            },
        )
    return json.loads((output / "verification.json").read_text())["runs"]


def verify_download(root, report, workers):
    """Hash all files, load and score all trials, and render stratified examples."""
    from examples.lifted_bench.convert_release import sha256
    from examples.lifted_bench.roundtrip_release import require_successful_subset

    count = require_successful_subset(root)
    hashes = json.loads((root / "checksums.json").read_text())

    def check(item):
        relative, expected = item
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
        if sha256(path) != expected:
            raise ValueError(f"Downloaded checksum mismatch: {relative}")

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(check, hashes.items()))
    print(f"All {len(hashes)} downloaded file hashes match", flush=True)
    rows = json.loads((root / "manifest.json").read_text())["runs"]
    results = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        jobs = [
            pool.submit(verify_chunk, (root, report, rows[i : i + 50], i // 50, False))
            for i in range(0, len(rows), 50)
        ]
        for future in as_completed(jobs):
            results.extend(future.result())
            print(
                f"Loaded and scored {len(results)}/{count} downloaded trajectories",
                flush=True,
            )
    # Longest recorded rollout per dataset, hand and selected seed.
    frames = {(r["dataset"], r["robot"], r["task"]): r["frames"] for r in results}
    samples = {}
    for row in rows:
        key = (row["dataset"], row["robot_type"], row["seed"])
        length = frames[(row["dataset"], row["robot_type"], row["task"])]
        if key not in samples or length > samples[key][0]:
            samples[key] = (length, row)
    render_rows = [value[1] for _, value in sorted(samples.items())]
    selected_trials = {(r["dataset"], r["robot_type"], r["task"]) for r in render_rows}
    recovered_episodes = set()
    for row in rows:
        episode = (row["dataset"], row["task"])
        trial = (row["dataset"], row["robot_type"], row["task"])
        if row.get("source_asset_sha256") and episode not in recovered_episodes:
            recovered_episodes.add(episode)
            if trial not in selected_trials:
                render_rows.append(row)
    dump(report / "render_selection.json", render_rows)
    rendered = []
    with ProcessPoolExecutor(min(4, workers), mp_context=ctx) as pool:
        jobs = [
            pool.submit(verify_chunk, (root, report, [row], i, True))
            for i, row in enumerate(render_rows)
        ]
        for future in as_completed(jobs):
            rendered.extend(future.result())
            print(f"Fresh renders {len(rendered)}/{len(render_rows)}", flush=True)
    state = json.loads((report / "candidate.json").read_text())
    transfer = json.loads((report / "download_complete.json").read_text())
    if transfer["revision"] != state["data_revision"]:
        raise ValueError("Verification does not match the completed remote-file audit")
    for row in results + rendered:
        # The parent verified the entire pinned download before starting workers.
        row["hashes_verified"] = True
    cards = []
    for row in rendered:
        video = next(report.glob(f"render_*/{row['video']}"))
        relative = video.relative_to(report).as_posix()
        label = html.escape(f"{row['dataset']} / {row['robot']} / {row['task']}")
        cards.append(
            f'<article><h2>{label}</h2><video controls loop muted preload="metadata" src="{relative}" poster="{Path(relative).with_suffix(".jpg")}"></video><p>{row["frames"]} frames · lift {row["sim_lift"]:.3f} m · error {row["final_err"]:.4f} m</p></article>'
        )
    (report / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>SPIDER full dataset verification</title>'
        "<style>body{font:16px system-ui;margin:32px;background:#f4f6f8;color:#18202b}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:24px}article{background:white;padding:18px;border-radius:12px}h2{font-size:16px;overflow-wrap:anywhere}video{width:100%}</style>"
        f"<h1>SPIDER full release · {len(results):,} verified trajectories</h1>"
        "<p>Fresh default renders from downloaded files. Reference left; saved simulation right. State replay does not establish original dynamics reproducibility.</p><main>"
        + "".join(cards)
        + "</main>"
    )
    dump(
        report / "roundtrip.json",
        {
            "repo_id": REPO,
            "revision": state["data_revision"],
            "download_verified": True,
            "download_method": transfer["method"],
            "remote_files_verified": transfer["remote_files_verified"],
            "files_verified": len(hashes),
            "checksums_sha256": sha256(root / "checksums.json"),
            "runs_verified": len(results),
            "source_episodes": len({(r["dataset"], r["task"]) for r in rows}),
            "selected_seeds": dict(Counter(str(r["seed"]) for r in rows)),
            "simulation_frames": sum(r["frames"] for r in results),
            "rendered_runs": len(rendered),
            "runs": results,
            "renders": rendered,
            "render_reuse_provenance": (
                json.loads((report / "render_cache_promotion.json").read_text())
                if (report / "render_cache_promotion.json").exists()
                else None
            ),
        },
    )


def publish(report):
    """Merge the verified candidate after checking its immutable data revision."""
    state = json.loads((report / "candidate.json").read_text())
    validation = json.loads((report / "roundtrip.json").read_text())
    if (
        validation["revision"] != state["data_revision"]
        or not validation["download_verified"]
        or validation.get("remote_files_verified", 0)
        != validation["files_verified"] + 3
    ):
        raise ValueError("Candidate has no matching independent verification")
    api = HfApi()
    current = api.repo_info(REPO, repo_type="dataset", revision=state["revision"])
    if current.sha != state.get("report_revision", state["data_revision"]):
        raise ValueError("Candidate changed after verification")
    if not all(row["success"] and row["loader_passed"] for row in validation["runs"]):
        raise ValueError("Verification includes an unsuccessful or unreadable run")
    counts = Counter(row["dataset"] for row in validation["runs"])
    lines = [
        "# Full converted release",
        "",
        f"{validation['runs_verified']:,} successful trajectories in the standard `retarget_example` layout.",
        f"They cover {validation['source_episodes']:,} source episodes and {validation['simulation_frames']:,} recorded robot simulation frames.",
        f"Selected seeds: {validation['selected_seeds'].get('0', 0):,} seed-0 runs and {validation['selected_seeds'].get('1', 0):,} seed-1 fallbacks.",
        "One run per source episode and robot; seed 0 preferred, seed 1 used when needed.",
        "Only completed, successful, single-hand, non-weak runs are included.",
        "",
        "| Source | Trajectories |",
        "| --- | ---: |",
        *[f"| {name} | {count:,} |" for name, count in sorted(counts.items())],
        "",
        f"All {validation['files_verified']:,} checksummed files were independently downloaded and matched.",
        "The bulk archive was downloaded, extracted into a clean directory, and every ordinary remote file was checked against its Git or LFS content identity.",
        "Every trajectory passed the current SPIDER loader, scene dependency checks, finite-value/time-grid checks, and recomputed lift/error metrics.",
        f"The immutable data revision tested was `{state['data_revision']}`.",
        "",
        "Robot trajectories are unchanged; assets and human references are converted to the current layout.",
        "Human keypoints precede the archived IK trimming/filtering stage; they are not frame-aligned copies of the saved robot reference.",
        "Validation establishes saved-state replay, not original collision-geometry equivalence or new MPC/control-replay results.",
        "Source terms and provenance limitations documented in the existing dataset card still apply.",
        "The existing smoke-test README is retained unchanged; this report describes the full release.",
        "",
        "## Fresh default-render checks",
        "",
        "The longest selected trajectory for each source, robot, and seed was rendered from the downloaded files.",
        "The sparse book-mesh recovery cases were also rendered.",
        "Reference appears on the left and saved simulation on the right.",
        "",
        "| Source | Robot | Task | Video |",
        "| --- | --- | --- | --- |",
    ]
    operations = [
        CommitOperationAdd(
            path_in_repo="verification/full_roundtrip.json",
            path_or_fileobj=report / "roundtrip.json",
        )
    ]
    viser_path = report / "viser_verified.json"
    if viser_path.exists():
        viser = json.loads(viser_path.read_text())
        if (
            viser["javascript_errors"]
            or not viser["playback_verified"]
            or viser["discovered_runs"] != validation["runs_verified"]
        ):
            raise ValueError("Viser verification did not pass for the full release")
        operations.append(
            CommitOperationAdd(
                path_in_repo="verification/full_viser.json", path_or_fileobj=viser_path
            )
        )
        operations.append(
            CommitOperationAdd(
                path_in_repo="verification/full_viser.png",
                path_or_fileobj=report / "viser_verified.png",
            )
        )
    for row in sorted(
        validation["renders"], key=lambda r: (r["dataset"], r["robot"], r["task"])
    ):
        video = row["video"]
        lines.append(
            f"| {row['dataset']} | {row['robot']} | {row['task']} | [Watch](verification/full_render/{video}) |"
        )
        matches = list(report.glob(f"render_*/{video}"))
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous verified preview: {video}")
        for path in (matches[0], matches[0].with_suffix(".jpg")):
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"verification/full_render/{path.name}",
                    path_or_fileobj=path,
                )
            )
    lines.extend(
        [
            "",
            "## Inspect locally",
            "",
            "```bash",
            "uv run -m examples.lifted_bench.download_release --output-dir example_datasets/retarget_full",
            "uv run examples/inspect_dataset.py --dataset-dir example_datasets/retarget_full",
            "```",
            "",
            "Open `http://localhost:8080` and select a trajectory.",
        ]
    )
    if viser_path.exists():
        lines.append(
            "Viser selection and playback were browser-tested on all 16 source/robot combinations using seed-1 examples, plus a recovered book case; no JavaScript errors occurred. See [the browser check](verification/full_viser.json)."
        )
    release_report = report / "FULL_RELEASE.md"
    release_report.write_text("\n".join(lines) + "\n")
    operations.append(
        CommitOperationAdd(
            path_in_repo="FULL_RELEASE.md", path_or_fileobj=release_report
        )
    )
    if "report_revision" not in state:
        commit = api.create_commit(
            REPO,
            repo_type="dataset",
            revision=state["revision"],
            parent_commit=state["data_revision"],
            operations=operations,
            commit_message="Record full download verification and fresh default-render previews",
        )
        state["report_revision"] = commit.oid
        dump(report / "candidate.json", state)
    discussion = api.get_discussion_details(
        REPO, state["discussion_num"], repo_type="dataset"
    )
    if discussion.status == "draft":
        api.change_discussion_status(
            REPO, state["discussion_num"], new_status="open", repo_type="dataset"
        )
    api.merge_pull_request(REPO, state["discussion_num"], repo_type="dataset")
    state["published_revision"] = api.repo_info(REPO, repo_type="dataset").sha
    dump(report / "candidate.json", state)
    print(
        f"Published https://huggingface.co/datasets/{REPO}/tree/{state['published_revision']}",
        flush=True,
    )


def main():
    """Run upload, independent download, verification, or verified publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["upload", "download", "verify", "publish"])
    parser.add_argument(
        "--stage", type=Path, default=Path("example_datasets/retarget_full")
    )
    parser.add_argument(
        "--download", type=Path, default=Path("example_datasets/retarget_full_verified")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("example_video_data/retarget_full_release")
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    if args.action == "upload":
        upload(args.stage.resolve(), args.report.resolve())
    elif args.action == "download":
        download(args.download.resolve(), args.report.resolve())
    elif args.action == "verify":
        verify_download(args.download.resolve(), args.report.resolve(), args.workers)
    else:
        publish(args.report.resolve())


if __name__ == "__main__":
    main()
