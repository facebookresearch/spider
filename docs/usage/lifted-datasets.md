# Lifted datasets: conversion and inspection

The destination is [`retarget/retarget_full`](https://huggingface.co/datasets/retarget/retarget_full).
Converted data follows the **same directory convention as `retarget/retarget_example`**.
The full release contains **7,876 successful trajectories from 2,885 source episodes**
across DexYCB, OakInk, HRDexDB, and HOT3D v2, using Allegro, Xhand, Inspire, and Sharpa.
It selects one successful right-hand run per source episode and robot, preferring
seed 0 and using seed 1 when needed. The original 16-run smoke subset remains
available at revision `081bb1bd5825c07e7af7645e6663a9824ed6df28`.

Verified published revision: `b6050b02146fc412d3db8cd539634dc71a117474`.
Every trajectory passed loader and success-metric checks after download; 35 default
renders and Viser browser checks cover the source/robot combinations and recovered
sparse meshes. See the [public full-release report](https://huggingface.co/datasets/retarget/retarget_full/blob/b6050b02146fc412d3db8cd539634dc71a117474/FULL_RELEASE.md).

For the full download, use the [bulk downloader](#full-archive-conversion).
The [smoke-test report](../development/dataset-smoke-report.md) retains the original
per-example results. Conversion, download, and inspection tools are included in SPIDER.

## Current dataset convention

```text
processed/<dataset>/assets/objects/<task>/...
processed/<dataset>/assets/robots/<robot>/...
processed/<dataset>/mano/right/<task>/task_info.json
processed/<dataset>/mano/right/<task>/0/trajectory_keypoints.npz
processed/<dataset>/<robot>/right/<task>/scene.xml
processed/<dataset>/<robot>/right/<task>/task_info.json
processed/<dataset>/<robot>/right/<task>/0/trajectory_kinematic.npz
processed/<dataset>/<robot>/right/<task>/0/trajectory_mjwp.npz
processed/<dataset>/<robot>/right/<task>/0/config.yaml
processed/<dataset>/<robot>/right/<task>/0/metrics.json
```

The converter creates SPIDER wrist, fingertip, and object-pose arrays from human
source packages, reconstructs object meshes, recovers archived robot meshes,
restores task-level scene placement, and writes portable configuration/metadata.
Recorded robot trajectories remain byte-for-byte unchanged. `manifest.json`
records source hashes and asset recovery; `checksums.json` covers converted files.
Replay has no dependency on `/efs`, `/nfs`, or external robot-mesh symlinks.

## Inspect with Viser

```bash
uv sync --frozen
uv run -m examples.lifted_bench.download_release --output-dir example_datasets/retarget_full
uv run examples/inspect_dataset.py --dataset-dir example_datasets/retarget_full
```

Open `http://localhost:8080`. Select a trial, scrub Frame, or enable Play. The
blue overlay is the IK reference; toggle visual/collision/reference geometry as
needed. The inspector uses the existing SPIDER Viser builder and current
`process_config()` / `load_data()`. It needs no GPU or optimization run.

Optional filters: `--dataset-name hot3d_v2 --robot-type sharpa --port 8081`.
For remote machines, forward the port: `ssh -L 8080:localhost:8080 <host>`.
Before uploading, inspect `example_datasets/retarget_full_stage` instead.

## Fetch and convert the smoke subset

The source download requires an AWS profile with access to the internal archive.
The selection is tracked in `examples/lifted_bench/smoke_manifest.json`.

```bash
uv run -m examples.lifted_bench.fetch_sources \
  --output-dir /tmp/spider-release-source --profile far-compute

uv run -m examples.lifted_bench.convert_release \
  --source-dir /tmp/spider-release-source \
  --robot-assets /tmp/spider-release-source/code/spider/assets/robots \
  --output-dir example_datasets/retarget_full_stage
```

Use an empty conversion output directory. Human data is ground-aligned and
resampled to 12.5 Hz, with a synthetic lift appended when required. HRDexDB uses
the benchmark's fixed 24-frame recipe. No experimental hand normalization or
fingertip tightening is applied. HOT3D GLB node transforms preserve metre scale.
Sharpa assets are recovered from the benchmark archive and included in the dataset.

## Validate and render

```bash
MUJOCO_GL=egl uv run -m examples.lifted_bench.verify_release \
  --dataset-dir example_datasets/retarget_full_stage \
  --output-dir example_video_data/retarget_full_local
```

This checks hashes, portable mesh paths, current-loader compatibility, dimensions,
time grids, finite values, and benchmark outcomes. It renders every example using
the repository's default `setup_renderer()` / `render_image()` functions. Open the
output `index.html` for fresh videos: reference left, simulation right.
`verification.json` contains per-example results. Use `--no-render` for data checks.
EGL also works with a suitable software OpenGL driver.

This establishes **saved-state replay compatibility**, not control-replay dynamics
equivalence. Object collision meshes were reconstructed using the benchmark's
preprocessing code, which matches the current decomposition code, but original
processed mesh hashes are unavailable for comparison. The manifest records this.

## Upload, download, and verify

Authenticate with an account that can write to the destination:

```bash
uv run hf auth login
MUJOCO_GL=egl uv run -m examples.lifted_bench.roundtrip_release \
  --stage-dir example_datasets/retarget_full_stage \
  --download-dir example_datasets/retarget_full_download \
  --report-dir example_video_data/retarget_full_roundtrip \
  --repo-id retarget/retarget_full
```

This command writes to the specified dataset. It verifies staging, uploads ordinary
files followed by completion metadata, pins the resulting commit, downloads into
an empty directory with an independent cache, checks every hash, and renders all
downloaded examples. It does not delete remote files or change repo visibility.
`roundtrip.json` records the commit only after all checks pass. `--revision` can
target an existing candidate branch instead of `main`.

To reproduce the original 16-run smoke subset without conversion:

```bash
uv run hf download retarget/retarget_full --repo-type dataset \
  --revision 081bb1bd5825c07e7af7645e6663a9824ed6df28 --local-dir example_datasets/retarget_full_download
```

## Read arrays

```python
from pathlib import Path
from spider.trajectory import discover_trajectories, load_saved_trajectory

root = Path("example_datasets/retarget_full_download")
trial = discover_trajectories(root)[0]
run = load_saved_trajectory(root, **trial)
print(run.qpos.shape, run.qvel.shape, run.ctrl.shape)
print(run.metrics)
```

The reader flattens control-step batches along time. Simulation is 100 Hz, the IK
reference is 12.5 Hz. State/velocity/action widths can differ. For these single-object
scenes, the last seven `qpos` values are object XYZ in metres and quaternion `wxyz`;
use the model for robot joint and actuator order.

Human keypoints and robot IK references are separate pipeline stages. In this
archive the IK recipe trims the final human frame, applies a three-frame moving
average, and drops one more frame when computing velocities, giving four fewer
IK samples. Its object poses are optimized tracking states. Do not zip the human
and IK files frame-by-frame; saved-trajectory replay uses the IK file and `ref_dt`.

Scored runs require completion, a single-hand demo, and reference rise ≥ 0.15 m.
Success means simulated final rise ≥ 0.10 m and final position error ≤ 0.10 m.
Synthetic lifts and robot/seed variants are not independent captured demonstrations.
See the [release plan](../development/dataset-release-plan.md) for source provenance
and the remaining larger-release requirements.

## Full archive conversion

The full pipeline reads the actual receipts from `full1`, `hv3`, and `s1full`;
archived summary JSON files are incomplete. It selects one completed, scored,
successful single-hand run per canonical source episode and robot, preferring
seed 0 and using seed 1 when needed. Seed-1 dataset aliases are converted back to
the same canonical dataset names, with the selected seed retained in the manifest.

Install the release extra, recover the archived robot assets using the source
fetcher above, and use a work directory with room for source data and intermediate
conversion groups. Keep this work directory separate from the public stage.

```bash
uv sync --frozen --extra release
for stage in inventory fetch convert assemble; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
    uv run --extra release -m examples.lifted_bench.full_release "$stage" \
    --work example_datasets/full_release_work \
    --robot-assets /tmp/spider-release-source/code/spider/assets/robots \
    --output example_datasets/retarget_full \
    --profile far-compute --workers 12 || break
done
```

Each episode is converted once for its selected robots. Completion checkpoints
are written after scenes compile, the current loader reads each trajectory, and
recomputed lift/error metrics agree with its receipt. Failed conversions stay in
the private work directory and prevent assembly; inspect `conversion_results.json`
and `logs/`, fix the cause, then rerun `convert`. Do not change the selected source
campaign in an existing work directory.

The full round trip uses a candidate pull request in `retarget/retarget_full`.
It leaves the dataset README unchanged. Upload and download can resume. Every
downloaded file is checked against its SHA-256; every trajectory is loaded and
scored again. Fresh default-render videos cover the longest selected trajectory
for each source, robot, and seed. Run `publish` only after `verify` succeeds:

```bash
for action in upload download verify publish; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
    uv run --extra release -m examples.lifted_bench.full_roundtrip "$action" \
    --stage example_datasets/retarget_full \
    --download example_datasets/retarget_full_verified \
    --report example_video_data/retarget_full_release --workers 12 || break
done
```

`candidate.json` records the immutable uploaded revision and final published
revision. `roundtrip.json` records all loader/metric results and the rendered
sample. The current inspector can browse the entire downloaded release:

```bash
uv run examples/inspect_dataset.py --dataset-dir example_datasets/retarget_full_verified
```

The full release also contains `distribution/retarget_full.tar.gz`, a bulk download
of the same converted `processed/` files and checksum index. `distribution.json`
records its SHA-256. This avoids one transfer request per small file; it does not
change the dataset layout. The round-trip script downloads this archive into a
clean directory, checks every extracted file, and audits every ordinary remote
file against Hugging Face's Git/LFS content identities.

For a bulk download, fetch only the distribution files, verify the archive hash
against `distribution.json`, and extract into the directory passed to the inspector.
The public downloader performs these steps, checks the extracted file hashes, and
audits the ordinary repository files automatically:

```bash
uv run -m examples.lifted_bench.download_release --output-dir example_datasets/retarget_full
uv run examples/inspect_dataset.py --dataset-dir example_datasets/retarget_full
```

Use `--revision <commit>` to pin a release. An interrupted download resumes from
the same revision; use a new directory when switching revisions.

For an ordinary folder download without the duplicate archive, use:

```bash
uv run hf download retarget/retarget_full --repo-type dataset \
  --exclude 'distribution/*' --local-dir example_datasets/retarget_full
```

Three sparse HRDexDB book meshes produce no voxel hulls. The converter recovers
the source package's archived convex parts only after checking that its visual OBJ
vertices match the converted GLB coordinates. Their manifest entries record the
recovered asset hashes and distinguish this path from regenerated voxel hulls.
