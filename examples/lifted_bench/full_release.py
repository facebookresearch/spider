# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the repository root.
"""Resumable full archive conversion; only scored successful single-hand runs.

Install with uv sync --extra release. Inventory is rebuilt from S3 receipts,
not the incomplete benchmark summary JSON. Source files and private logs live
outside the public stage. Conversion checkpoints are written only after the
current loader and reconstructed success metrics pass for every selected run.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import shutil
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "fhogan/spider/arxiv/data"
BUCKET = "far-research-internal"
DATASETS = {
    "dexycb_lifted": "dexycb_lifted/right",
    "oakink_lifted": "oakink_3d_hold_lifted",
    "hrdexdb_24f": "HRDexDB_lifted",
    "hot3d_v2": "hot3d_3d_v2/right",
}
REQUIRED = (
    "scene.xml",
    "config.yaml",
    "RECEIPT.json",
    "trajectory_mjwp.npz",
    "trajectory_kinematic.npz",
)


def dump(path, value):
    """Atomically write a resumable JSON checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def select_runs(receipts):
    """Deduplicate canonical episodes, preferring seed zero over seed one."""
    selected = {}
    counts = Counter()
    for campaign, receipt in receipts:
        counts["receipts"] += 1
        if not (
            receipt.get("stage") == "done"
            and receipt.get("success") is True
            and receipt.get("bimanual") is False
            and receipt.get("data_weak") is False
        ):
            counts["not_eligible"] += 1
            continue
        source_dataset = receipt["dataset"]
        dataset = source_dataset.removesuffix("_s1")
        if dataset not in DATASETS or receipt["hand"] not in (
            "allegro",
            "xhand",
            "inspire",
            "sharpa",
        ):
            raise ValueError(f"Unknown eligible source: {receipt}")
        seed = 1 if campaign == "s1full" else 0
        if seed != int(source_dataset.endswith("_s1")):
            raise ValueError("Seed/dataset alias mismatch")
        key = (dataset, receipt["seq"], receipt["hand"])
        row = {
            "dataset": dataset,
            "source_dataset": source_dataset,
            "task": receipt["seq"],
            "robot_type": receipt["hand"],
            "campaign": campaign,
            "seed": seed,
            "embodiment_type": "right",
            "data_id": 0,
        }
        counts["eligible_receipts"] += 1
        if key not in selected or seed < selected[key]["seed"]:
            selected[key] = row
    rows = [selected[k] for k in sorted(selected)]
    counts["selected_runs"] = len(rows)
    counts["duplicate_successes"] = counts["eligible_receipts"] - len(rows)
    return rows, dict(counts)


def client(profile):
    """Create an authenticated S3 client with bounded concurrent connections."""
    import boto3
    from botocore.config import Config

    return boto3.Session(profile_name=profile or None).client(
        "s3",
        config=Config(
            max_pool_connections=64, retries={"max_attempts": 8, "mode": "adaptive"}
        ),
    )


def fetch_file(s3, key, path, size=None):
    """Download atomically and check length before marking a file complete."""
    path = Path(path)
    if (
        path.is_file()
        and path.stat().st_size > 0
        and (size is None or path.stat().st_size == size)
    ):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    response = s3.get_object(Bucket=BUCKET, Key=key)
    with response["Body"] as source, temporary.open("wb") as dest:
        shutil.copyfileobj(source, dest)
    if temporary.stat().st_size != response["ContentLength"]:
        raise ValueError(f"Incomplete download: {key}")
    temporary.replace(path)


def inventory(work, s3):
    """Freeze an authoritative selection from all archived receipts."""
    jobs = []
    for campaign in ("full1", "hv3", "s1full"):
        prefix = f"{BASE}/retarget/{campaign}/receipts/"
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".json"):
                    local = work / "receipts" / campaign / Path(obj["Key"]).name
                    jobs.append((obj["Key"], local, obj["Size"]))
    with ThreadPoolExecutor(48) as pool:
        list(pool.map(lambda job: fetch_file(s3, *job), jobs))
    rows, counts = select_runs(
        (path.parent.name, json.loads(path.read_text())) for _, path, _ in jobs
    )
    counts["by_dataset"] = dict(Counter(row["dataset"] for row in rows))
    counts["by_seed"] = dict(Counter(str(row["seed"]) for row in rows))
    counts["by_dataset_hand"] = dict(
        Counter(f"{row['dataset']}/{row['robot_type']}" for row in rows)
    )
    counts["source_episodes"] = len({(r["dataset"], r["task"]) for r in rows})
    dump(work / "full_manifest.json", rows)
    dump(work / "inventory.json", counts)
    print(json.dumps(counts, indent=2), flush=True)
    return rows


def fetch_sources(work, s3, rows):
    """Fetch only selected robot artifacts and their raw source dependencies."""
    jobs = []
    episodes = {(r["dataset"], r["task"]) for r in rows}
    for dataset, task in sorted(episodes):
        prefix = f"{BASE}/3d/{DATASETS[dataset]}/{task}/3d/"
        objects = {}
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=BUCKET, Prefix=prefix
        ):
            objects.update(
                {o["Key"][len(prefix) :]: o for o in page.get("Contents", [])}
            )
        for variant in ("seed_default_lifted_centered", "seed_default_centered"):
            if (
                f"{variant}/data.npz" in objects
                and f"{variant}/object_1_mesh.glb" in objects
            ):
                # Keep original convex parts available for sparse meshes that
                # yield no voxel hulls; their visual OBJ verifies coordinates.
                names = ["data.npz", "object_1_mesh.glb"]
                names.extend(
                    relative.removeprefix(f"{variant}/")
                    for relative in objects
                    if relative == f"{variant}/visual.obj"
                    or (
                        relative.startswith(f"{variant}/convex/")
                        and relative.endswith(".obj")
                    )
                )
                for name in names:
                    obj = objects[f"{variant}/{name}"]
                    jobs.append(
                        (
                            obj["Key"],
                            work / "raw" / dataset / task / "3d" / variant / name,
                            obj["Size"],
                        )
                    )
                break
        else:
            raise FileNotFoundError(f"No complete source package: {prefix}")
    for row in rows:
        prefix = f"{BASE}/retarget/{row['campaign']}/processed/{row['source_dataset']}/{row['robot_type']}/right/{row['task']}/0/"
        local = work / "runs" / row["dataset"] / row["robot_type"] / row["task"]
        # Bind resumed downloads to the selected campaign, not merely a path.
        identity = local / "selection.json"
        if identity.exists() and json.loads(identity.read_text()) != row:
            raise ValueError(f"Source selection changed: {local}")
        dump(identity, row)
        jobs.extend((prefix + name, local / name, None) for name in REQUIRED)
    print(
        f"Downloading {len(jobs)} files for {len(episodes)} episodes and {len(rows)} runs",
        flush=True,
    )
    with ThreadPoolExecutor(48) as pool:
        pending = [pool.submit(fetch_file, s3, *job) for job in jobs]
        for index, future in enumerate(as_completed(pending), 1):
            future.result()
            if index % 1000 == 0:
                print(f"Downloaded {index}/{len(jobs)}", flush=True)
    dump(work / "download_complete.json", {"runs": len(rows), "files": len(jobs)})


def group_id(rows):
    """Identify an immutable selection of hands for one source episode."""
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:20]


def convert_group(args):
    """Convert and validate one episode, recording failures outside the release."""
    work, robots, rows = args
    name = group_id(rows)
    root = work / "groups" / name
    log = work / "logs" / f"{name}.log"
    checkpoint = work / "completed" / f"{name}.json"
    if checkpoint.exists():
        return json.loads(checkpoint.read_text())
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        if root.exists():
            shutil.rmtree(root)
        manifest = work / "group_manifests" / f"{name}.json"
        dump(manifest, rows)
        with (
            log.open("w") as stream,
            contextlib.redirect_stdout(stream),
            contextlib.redirect_stderr(stream),
        ):
            import torch

            torch.set_num_threads(1)
            from loguru import logger

            logger.remove()
            logger.add(stream)
            from examples.lifted_bench.convert_release import main as convert
            from examples.lifted_bench.verify_release import main as verify

            convert(work, robots, root, manifest)
            verify(root, work / "verification" / name, render=False)
        result = {"group": name, "rows": rows, "ok": True}
        dump(checkpoint, result)
        return result
    except Exception:
        error = traceback.format_exc()
        with log.open("a") as stream:
            stream.write(error)
        return {"group": name, "rows": rows, "ok": False, "error": error}


def convert_all(work, robots, rows, workers):
    """Convert independent episodes with bounded CPU parallelism."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["task"])].append(row)
    results = []
    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(convert_group, (work, robots, group))
            for group in groups.values()
        ]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if not result["ok"]:
                print(
                    f"FAILED {result['rows'][0]['dataset']}/{result['rows'][0]['task']}: {result['error'].splitlines()[-1]}",
                    flush=True,
                )
            if index % 25 == 0:
                print(
                    f"Converted and verified {index}/{len(groups)} episodes; failures={sum(not r['ok'] for r in results)}",
                    flush=True,
                )
    dump(work / "conversion_results.json", results)
    return results


def assemble(work, output, rows):
    """Merge only verified groups and reject conflicting shared assets."""
    from examples.lifted_bench.convert_release import sha256

    entries = []
    materialized_hashes = {}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["task"])].append(row)
    output.mkdir(parents=True, exist_ok=True)
    for group in groups.values():
        name = group_id(group)
        if not (work / "completed" / f"{name}.json").is_file():
            raise ValueError(
                f"Unverified conversion group: {name}; do not publish incomplete inventory"
            )
        root = work / "groups" / name
        manifest = json.loads((root / "manifest.json").read_text())
        if len(manifest["runs"]) != len(group):
            raise ValueError(f"Incomplete converted group: {name}")
        expected = json.loads((root / "checksums.json").read_text())
        entries.extend(manifest["runs"])
        for path in (root / "processed").rglob("*"):
            if not path.is_file() or path.name == "dataset_summary.json":
                continue
            dest = output / path.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            expected_hash = expected[path.relative_to(root).as_posix()]
            if dest.exists():
                if dest not in materialized_hashes:
                    materialized_hashes[dest] = sha256(dest)
                if materialized_hashes[dest] != expected_hash:
                    raise ValueError(f"Conflicting release asset: {dest}")
            else:
                if sha256(path) != expected_hash:
                    raise ValueError(f"Converted file changed after validation: {path}")
                os.link(path, dest)
                materialized_hashes[dest] = expected_hash
    dump(
        output / "manifest.json",
        {
            "schema_version": 1,
            "layout": "retarget_example",
            "subset": "full",
            "successful_only": True,
            "selection": "One successful scored single-hand run per source episode and robot; seed 0 preferred, seed 1 fallback.",
            "runs": entries,
        },
    )
    for dataset in DATASETS:
        dump(
            output / "processed" / dataset / "dataset_summary.json",
            {
                "dataset_name": dataset,
                "runs": [r for r in entries if r["dataset"] == dataset],
            },
        )
    dump(output / "inventory.json", json.loads((work / "inventory.json").read_text()))
    source_paths = sorted(
        path
        for dataset, task in groups
        for path in (work / "raw" / dataset / task).rglob("*")
        if path.is_file() and path.name in {"data.npz", "object_1_mesh.glb"}
    )
    with ThreadPoolExecutor(16) as pool:
        source_hashes = dict(
            pool.map(
                lambda path: (path.relative_to(work / "raw").as_posix(), sha256(path)),
                source_paths,
            )
        )
    dump(
        output / "source_packages.json",
        {
            "source_prefix": f"s3://{BUCKET}/{BASE}/3d/",
            "dataset_prefixes": DATASETS,
            "sha256": source_hashes,
        },
    )
    paths = [
        p
        for p in sorted(output.rglob("*"))
        if p.is_file()
        and p.name not in {"checksums.json", "distribution.json", "README.md"}
        and ".cache" not in p.relative_to(output).parts
        and p.relative_to(output).parts[0] != "distribution"
    ]
    with ThreadPoolExecutor(16) as pool:
        hashes = dict(
            pool.map(lambda p: (p.relative_to(output).as_posix(), sha256(p)), paths)
        )
    dump(output / "checksums.json", hashes)
    print(
        f"Assembled {len(entries)} successful runs, {len(paths)} files, {sum(p.stat().st_size for p in paths)} bytes",
        flush=True,
    )


def main():
    """Run one resumable stage of the full release pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inventory", "fetch", "convert", "assemble"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("example_datasets/retarget_full")
    )
    parser.add_argument("--robot-assets", type=Path, required=True)
    parser.add_argument("--profile", default="far-compute")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    work = args.work.resolve()
    if args.stage == "inventory":
        inventory(work, client(args.profile))
        return
    rows = json.loads((work / "full_manifest.json").read_text())
    if args.stage == "fetch":
        fetch_sources(work, client(args.profile), rows)
    elif args.stage == "convert":
        results = convert_all(work, args.robot_assets.resolve(), rows, args.workers)
        if any(not r["ok"] for r in results):
            raise SystemExit(
                "Some groups failed validation; see conversion_results.json and logs."
            )
    else:
        assemble(work, args.output.resolve(), rows)


if __name__ == "__main__":
    main()
