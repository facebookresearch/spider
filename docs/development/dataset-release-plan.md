# Larger dataset release plan

Status: full conversion published: **7,876 successful trajectories**, independently downloaded and verified, with 35 default-render checks and Viser validation. Published revision: `b6050b02146fc412d3db8cd539634dc71a117474`. Based on the 2026-09-21 `LIFTED_BENCH_HANDOFF.md` at
`s3://far-research-internal/fhogan/spider/arxiv/`. Inspected against SPIDER
`71238456bf97a7eeb3d0471aa31974e2d404d4ae`. The smoke subset is published in `retarget/retarget_full`.

The [smoke-test report](dataset-smoke-report.md) records local conversion, rendering,
and Viser results for all 16 requested source/robot combinations.

## Recommendation and scope

Release the existing lifted benchmark first, then generate additional demonstrations.
Keep `retarget/retarget_example` as the small quickstart download. Propose a separate
Hugging Face dataset, [`retarget/retarget_full`](https://huggingface.co/datasets/retarget/retarget_full),
with versioned releases and selective downloads by source dataset, robot, and seed.
Use the same `processed/` layout as `retarget_example` for compatibility with current code.
The destination already exists; the smoke manifest covers four sources × four robot
hands, one example per combination (16 runs).

The handoff reports approximately 77 GB of source packages and 16 GB of retargeting
outputs across about 230,000 objects. These are planning estimates, not a fresh
bucket inventory. Exact release size depends on selected runs and recovered assets.

| Source | Main single-hand sequences | Successful seed-0 runs | Approximate successful sequence/robot pairs across two seeds |
| --- | ---: | ---: | ---: |
| DexYCB | 499 | 870 | 1,088 |
| OakInk-Image | 169 | 241 | 312 |
| HRDexDB | 441 | 870 | 970 |
| HOT3D v2 | 2,718 | 4,542 | 5,455 |
| Total | 3,827 | 6,523 | 7,825 |

Counts are from the handoff, include all four benchmark robots, and are not final
public-release counts. Two-seed figures are approximate paired-analysis yields,
not counts of independent human demonstrations. HRDexDB provenance and Sharpa
asset availability remain unresolved. Recalculate exact counts from receipts.

## Full conversion inventory

The 2026-09-21 receipt scan found 31,738 runs across `full1`, `hv3`, and `s1full`.
12,949 were completed, successful, single-hand runs with non-weak references.
Deduplicating source episode/robot pairs with seed-0 preference selects **7,876
trajectories from 2,885 source episodes**: 6,519 seed-0 runs and 1,357 seed-1 fallbacks.
The other 5,073 successful receipts are duplicate seed variants. The archived
seed-1 summary is incomplete; these exact counts come from the receipts.

| Source | Allegro | Xhand | Inspire | Sharpa | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| DexYCB | 306 | 272 | 274 | 259 | 1,111 |
| HOT3D v2 | 1,638 | 1,462 | 1,363 | 1,016 | 5,479 |
| HRDexDB | 252 | 241 | 234 | 246 | 973 |
| OakInk | 74 | 73 | 75 | 91 | 313 |
| Total | 2,270 | 2,048 | 1,946 | 1,612 | 7,876 |

All selected runs passed conversion and local loader/metric validation. Eleven
runs from three sparse HRDexDB book episodes use recovered source convex meshes;
the source visual OBJ vertices were checked against the converted GLB coordinates.
No selected run was excluded. See the [full conversion commands](../usage/lifted-datasets.md#full-archive-conversion)
for checkpointed conversion and independent remote verification.

## 1. Freeze the source inventory

- Inventory the canonical `data/retarget/`, `data/3d/`, and
  `data/spider_full_bench_meta/` prefixes. Record object keys, sizes, source versions
  where available, and downloaded SHA-256 hashes. Use the original receipts as
  provenance; an S3 ETag is not a general-purpose SHA-256 checksum.
- Use `full1` and `hv3` for seed 0 and `s1full` for seed 1. Preserve the A/B campaigns
  as separately labelled benchmark material. Exclude superseded HOT3D `hv1`/`abh1`
  from the main release.
- Normalize aliases such as `hot3d_v2_s1` to a source dataset plus an explicit seed;
  retain the original alias and campaign. Key runs by source sequence, robot,
  embodiment, recipe, and seed. Do not flatten seed 1 onto seed 0's paths.
- Reconcile every expected manifest entry against receipts and actual files. The
  handoff reports six missing `full1` tasks. Its worker uses unchecked upload
  commands, so a final receipt alone does not prove artifact completeness.
- Inspect the archived branch separately before incorporating code. The handoff
  says three patches, but the S3 listing contains four top-level patches. Pin actual
  branch commits and the per-campaign code archives. Do not merge the history branch.

Deliverable: an inventory with complete, missing, failed, and excluded entries,
plus exact bytes and counts for each proposed release subset.

## 2. Define what users download

Publish successful trajectories only. Keep full benchmark accounting internally:

- **Training subset:** completed, single-hand, non-weak-reference successful runs.
  Choose one successful seed per sequence/robot using a documented deterministic
  rule (seed 0 first, otherwise seed 1). Offer both seeds as optional augmentation.
- **Internal benchmark accounting:** retain all main-campaign receipts, failures,
  weak references, and exclusion reasons for reproducible denominators. Failed,
  incomplete, bimanual, and weak-reference trajectories are excluded from the
  Hugging Face upload. The converter and uploader enforce this selection.

Match the handoff's metric: final simulated object rise at least 0.10 m and final
position error at most 0.10 m. Scored runs require `stage=done`, a single-hand demo,
and reference lift at least 0.15 m. Do not substitute the generic tracking-error
evaluator in `spider/postprocess/get_success_rate.py` for this benchmark metric.
Check thresholds using full-precision trajectories when recomputing; receipt
metrics are rounded. Synthetic lift must be labelled in the dataset card.

Split by original participant/recording or source sequence as appropriate before
expanding into robot/seed variants. Keep overlapping clips and all variants of a
demonstration in the same split. Define a separate object-held-out split if desired.

Before public distribution, record source-specific redistribution terms for
derived trajectories and meshes. HRDexDB needs its source, citation, and terms
filled in; omit it from the first public version if unresolved. Do not package MANO
model files. The repository's license alone does not establish terms for every
source dataset or robot asset.

## 3. Build portable packages

Each selected run needs its optimized trajectory, IK reference, portable scene,
resolved configuration, public metadata, and every referenced mesh. Preserve the
original receipt internally; export a documented metadata view without worker host
names or internal paths. Keep original logs and full MPC diagnostics in the internal
archive; offer diagnostics and videos separately from the core training download.

Two verified compatibility issues must be resolved:

1. The worker copies `scene.xml` from the task directory into the run's `0/`
   directory. The inspected XML still uses `meshdir="../../../assets/"`.
   `process_config()` expects the scene at the task level, one directory above
   `0/`. Restore that layout and validate relative paths when materializing data.
2. The worker uploads no shared mesh directory. Recover the exact processed object
   and robot assets from the originating workspace, or regenerate them from pinned
   inputs and verify equivalence. Sharpa is ignored by Git and absent in this
   checkout; resolve access/distribution or mark those runs as unavailable for
   standalone replay and exclude them from the replay-ready release.

Deduplicate assets by content hash. Preserve the SPIDER directory structure after
extraction; provide a materializer that downloads selected shards and installs the
shared asset dependencies. Keep provenance mapping from every released file to
its source object and recipe. Do not silently regenerate an old run with changed
collision geometry.

Remote layout follows the current example dataset directly (no mandatory archive
extraction or custom materializer):

```text
README.md
manifest.json
checksums.json
processed/<dataset>/dataset_summary.json
processed/<dataset>/assets/{objects,robots}/...
processed/<dataset>/mano/right/<task>/task_info.json
processed/<dataset>/mano/right/<task>/0/trajectory_keypoints.npz
processed/<dataset>/<robot>/right/<task>/scene.xml
processed/<dataset>/<robot>/right/<task>/task_info.json
processed/<dataset>/<robot>/right/<task>/0/{trajectory_kinematic,trajectory_mjwp}.npz
processed/<dataset>/<robot>/right/<task>/0/{config.yaml,metrics.json}
```

Use Hugging Face revisions/tags for releases. Keep different recipe/seed aliases
separate when their task-level scene or metadata differs. The current smoke subset
contains seed 0 only. Optional shards can be added later, while retaining this
ordinary directory interface. Preserve original NPZ arrays and record source hashes
in the manifest; conversion adds portable metadata, human keypoints, and assets.

Implementation: `examples/lifted_bench/{fetch_sources,convert_release,verify_release,roundtrip_release}.py`.
See [the loading and smoke-test guide](../usage/lifted-datasets.md) for commands.

Optionally offer compact source packages. The handoff estimates roughly 30× savings
for HOT3D by removing per-frame MANO vertex arrays. Measure that on a pilot and
verify the processor still reads the compact packages before promising savings.
Do not remove those arrays from the canonical archive.

## 4. Validate and upload incrementally

1. Start with the requested 16-run compatibility smoke test: one successful example
   for each source/robot combination. This selected subset is not a benchmark of
   success rates. Expand local validation to both seeds, failures, and incomplete cases afterward;
   only successful scored runs may be uploaded. Validate NPZ shapes,
   finite state/control values, time grids, scene dimensions, and mesh resolution.
2. Load every selected scene on a clean machine with no `/efs` or `/nfs` mounts.
   Verify every core run's required artifacts and hashes. Replay a stratified sample
   using the standard MuJoCo rendering path, including long clips and large objects.
   State replay verifies visualization; use a separate control replay if claiming
   dynamics reproducibility.
3. Compare recomputed lift/error outcomes and aggregate tables with the frozen
   source receipts. Account explicitly for missing runs and exclusions.
4. Stage an immutable release directory internally. Upload the pilot to a private
   candidate dataset, download it into an empty directory, materialize it, and run
   the documented loading example and replay checks.
5. Upload the remaining files with the current Hugging Face `upload_folder()` /
   `hf upload` flow and Xet enabled. Pin the uploader version and resume interrupted
   transfers. Current guidance supports resumable, batched uploads; see the
   [official upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload).
6. Verify remote counts and downloaded hashes. Publish the final manifest and
   completion marker only after all dependencies exist. Keep the candidate private
   until its card, terms, validation report, and final counts are reviewed; then
   publish a version tag and update the repository's dataset links.

Release acceptance: no missing dependencies in the replay-ready subset, no split
leakage across variants, reproducible reported denominators, successful clean
download/read/replay, and documented source terms. No full GPU regeneration is
needed for this first release; prioritize packaging and dependency recovery.

## 5. Expand after the first release

Prioritize deterministic mesh/preprocessing failures and the six missing tasks,
then new source coverage. Repair under a new recipe/campaign ID so published
benchmark results stay immutable. Pilot left-hand support before scaling: the
lifted benchmark processor/runner currently use right-hand targets, so existing
left-hand packages are not a drop-in extension.

The handoff lists 498 DexYCB and 1,679 HOT3D left-hand clips: up to 2,177 additional
source episodes, or 8,708 runs for four robots and one seed before filtering. At
the historical aggregate rate of about 2,140 GPU-hours / 32,594 receipts, that is
roughly 570 GPU-hours per seed. This is a coarse capacity estimate including past
overheads/failures; measure a representative pilot before scheduling a fleet.
Two seeds would roughly double that compute. Do not count left-hand runs until
embodiment mappings, IK, scene generation, and evaluation have been validated.

Treat the 141 bimanual HOT3D clips as a separate future task; the existing right-hand
runs do not establish bimanual retargeting support. For HOT3D quality improvements,
evaluate collision-aware IK and grasp-target fidelity separately. The handoff's
fingertip-tightening and shorter-approach experiments did not improve the baseline.

Suggested work order: inventory and provenance → dependency recovery and reader →
pilot upload/replay → full existing-data release → left-hand pilot → expansion.
