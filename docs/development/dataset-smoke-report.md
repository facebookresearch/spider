# Lifted dataset smoke-test report

Date: 2026-09-21. Branch: `chaoyi/new_dataset`.

**Success-only upload/download/render: 16/16 passed.**

Dataset: https://huggingface.co/datasets/retarget/retarget_full

Verified data revision: `081bb1bd5825c07e7af7645e6663a9824ed6df28`. All 408 checksummed files matched after a
clean download; all 16 fresh videos are byte-identical to their published previews.
The public [verification report](https://huggingface.co/datasets/retarget/retarget_full/blob/main/SMOKE_TEST.md)
records rendering and Viser inspection of the downloaded data.

## Verification evidence

- Converted four source clips into the current `retarget_example` directory layout,
  with one saved rollout for each of four robot hands (16 combinations).
- 408 hashed files, 109,581,007 bytes before the checksum index (about 104.5 MiB).
- Every scene resolves its mesh dependencies within the converted dataset.
- All runs passed current `process_config()` and `load_data()`, shape/time-grid/finite
  checks, and comparison of recomputed metrics to the source receipts.
- All 16 fresh videos used local `setup_renderer()` / `render_image()` with EGL.
- All 16 trials were loaded and scrubbed in a real Chromium browser using Viser;
  no browser JavaScript errors. Playback, trial switching, and visibility toggles passed.
- Three regression tests passed: GLB node scale/ground alignment, preservation of
  hand/object offsets during synthetic lift, and success-only upload filtering. New Python files pass Ruff; the existing
  Viser module retains its 37 pre-existing lint findings with no new findings.

## Per-example results

| Dataset | Robot | Frames | Lift (m) | Final error (m) | Load / render / Viser |
| --- | --- | ---: | ---: | ---: | --- |
| dexycb_lifted | allegro | 175 | 0.3362 | 0.0072 | Pass |
| dexycb_lifted | xhand | 175 | 0.3333 | 0.0038 | Pass |
| dexycb_lifted | inspire | 175 | 0.3321 | 0.0026 | Pass |
| dexycb_lifted | sharpa | 175 | 0.3332 | 0.0072 | Pass |
| hrdexdb_24f | allegro | 175 | 0.2811 | 0.0167 | Pass |
| hrdexdb_24f | xhand | 175 | 0.2835 | 0.0199 | Pass |
| hrdexdb_24f | inspire | 175 | 0.2819 | 0.0178 | Pass |
| hrdexdb_24f | sharpa | 175 | 0.2709 | 0.0081 | Pass |
| oakink_lifted | allegro | 150 | 0.2506 | 0.0058 | Pass |
| oakink_lifted | xhand | 150 | 0.2542 | 0.0091 | Pass |
| oakink_lifted | inspire | 150 | 0.2491 | 0.0044 | Pass |
| oakink_lifted | sharpa | 150 | 0.2359 | 0.0034 | Pass |
| hot3d_v2 | allegro | 225 | 0.2437 | 0.0101 | Pass |
| hot3d_v2 | xhand | 225 | 0.2553 | 0.0169 | Pass |
| hot3d_v2 | inspire | 225 | 0.2580 | 0.0185 | Pass |
| hot3d_v2 | sharpa | 225 | 0.2560 | 0.0206 | Pass |

## Inspect locally

- Gallery: `http://localhost:8082` while the local gallery server is running.
- Viser: `http://localhost:8080` while the inspector is running.
- Gallery files: `example_video_data/retarget_full_roundtrip/after_download/index.html`.
- Verification data: `example_video_data/retarget_full_roundtrip/after_download/verification.json`.
- Browser snapshots: `example_video_data/retarget_full_roundtrip/viser/`.
- Downloaded data: `example_datasets/retarget_full_download/`.

These artifacts are local, Git-ignored outputs; use the guide to reproduce them.

## Limits

The samples were chosen for successful outcomes, not representative success rates.
The conversion preserves recorded robot NPZ files and reconstructs missing object
collision meshes. This validates saved-state playback and current-loader compatibility,
not original collision-mesh equivalence or control-replay dynamics.
Only successful scored trajectories were uploaded; failed and weak-reference runs are excluded.
