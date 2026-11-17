# Workflow: CPU (MuJoCo batch)

SPIDER's sampling MPC on MuJoCo's CPU batch API (`mujoco.batch`), the default simulator (`simulator=cpu`). It
retargets a demonstration of two hands and an object onto two Sharpa hands. It needs no GPU and no external
library: MuJoCo, NumPy and Torch.

It has the interface of the [MJWP workflow](workflow-mjwp.md): `spider/simulators/cpu.py` provides the same
functions with the same arguments (`setup_mj_model`, `load_data`, `setup_env`, `step_env`, `get_reward`,
`get_terminal_reward`, `get_terminate`, `get_trace`, `save_state` / `load_state`, `sync_env`, `copy_sample_state`,
`get_qpos` / `set_qpos` / `get_qvel`, `save_env_params` / `load_env_params`), and `examples/run_cpu.py` follows the
protocol of `run_mjwp.py`: load the reference, set up the worlds, optimise a receding-horizon plan with SPIDER's
sampler, execute it, broadcast the executed world. As in the HDMI and DexMachina simulators, the worlds keep the
reference clock: the reward compares every world with the reference frame it is at.

## Prerequisites

- An environment with MuJoCo 3.17 or later, which has `mujoco.batch`. SPIDER's own environment pins MuJoCo 3.7
  for MuJoCo Warp, so this workflow gets a small one of its own, with SPIDER installed into it without its
  dependencies:

  ```bash
  uv venv --python 3.12 .venv-cpu
  uv pip install --python .venv-cpu/bin/python "mujoco>=3.17" torch numpy scipy loguru hydra-core tyro \
      opencv-python-headless "imageio[ffmpeg]" "rerun-sdk==0.26.2" viser trimesh
  uv pip install --python .venv-cpu/bin/python --no-deps -e .
  ```

- A scene: a directory with `scene.xml` (the MuJoCo model) and `scene.npz` (the reference motion and the tables of
  the control law). Scenes live where SPIDER keeps processed data:

  ```
  example_datasets/processed/chord/sharpa/bimanual/<task>/<data_id>/scene.xml
  example_datasets/processed/chord/sharpa/bimanual/<task>/<data_id>/scene.npz
  ```

  The [example dataset](https://huggingface.co/datasets/retarget/retarget_example) has the scenes of 277
  [ARCTIC](https://arctic.is.tue.mpg.de) sequences, each with the motion, the metrics and the video of a run with
  the defaults (seed 0; `processed/chord/README.md` there says how):

  ```bash
  git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
  ```

  Their task is `<subject>-<object>-<use|grab>` and their `data_id` ARCTIC's take number: `s01/box_use_01` is
  `task=s01-box-use data_id=1`, the default.
  Scenes come from [FlashCHORD](workflow-chord.md#export-a-scene-for-the-cpu) (exporting one needs it; running
  one does not).

## Run the demo

```bash
# open the viewer at http://localhost:8080, pick a task and the settings there, press Start
.venv-cpu/bin/python examples/run_cpu.py

# start a clip at once (defaults: task=s01-box-use data_id=1), still with the viewer
.venv-cpu/bin/python examples/run_cpu.py task=s01-laptop-use data_id=2 seed=1 autostart=true

# a scene anywhere on disk, results somewhere else
.venv-cpu/bin/python examples/run_cpu.py model_path=SCENE_DIR/scene.xml output_dir=OUT autostart=true

# no viewer, on 16 threads, with half the planning time
.venv-cpu/bin/python examples/run_cpu.py show_viewer=false num_threads=16 max_num_iterations=2
```

A run prints the object error frame by frame, and at the end one line with the result (the default clip, on 32
threads):

```
object 0.72 cm 1.3 deg hinge 0.4 deg openings 5/5 success 1 frac_ok 1.00 | wrist 0.66 cm 1.5 deg fingers 2.9 deg | shake x0.92 | wall 149 s
```

It writes, into the scene's directory unless `output_dir` is set:

- `trajectory_cpu.npz`: the executed motion (`qpos` per control frame, in the model's coordinates), the reference
  (`qpos_ref`), the controls and the planner's statistics;
- `metrics.json`: the numbers of the line above and more (see `spider/postprocess/evaluate_cpu.py`; `python -m
  spider.postprocess.evaluate_cpu RUN_DIR` scores a saved run again);
- `visualization_cpu.mp4`: the motion with the reference drawn over it in green. Rendering needs an OpenGL
  context; on a machine without a display set `MUJOCO_GL=egl` (or `osmesa`), or pass `save_video=false`.

## Watching a run

The default viewer is [Viser](https://viser.studio): open `http://localhost:8080`. Runs are driven from its panel:

- **Retargeting**: the task (every scene of the dataset directory), the key settings (seed, samples, iterations per
  frame, temperature, the sampling noise of the wrists and the fingers, the horizon, how many frames to run),
  **Start** and **Stop**, a progress bar and the result of the last run. Picking a task shows its scene at the first
  frame of the reference. Stop ends a run where it is; what was executed is scored and saved.
- **Recovery: plan again, go back**: what a run does when the object slips (see below): how many times a frame is
  planned again and with how much more noise; when the object counts as slipping, how far before the slip the run
  goes back, how many times, and with how much more noise; when it counts as lost. The status line counts the
  frames planned again and the returns.
- **Metrics**: while a run is going, the distance of the object and the wrists from the reference, the angle of the
  object and its hinge from the reference, and the planning time per frame.
- The scene follows the run: the simulated hands and object, and the reference as a translucent ghost of everything
  that moves.
- The **traces** of every replanning tick: where the sampled rollouts take the object (red), and the wrists and
  the ten fingertips (blue), over the planning horizon, from white in the first optimizer iteration to full colour
  in the last, with the reference's paths in green (object) and yellow (wrists and fingertips).
- **Timeline**: move the frame slider back to look at an earlier moment (the scene stops following the run until
  the slider is back at the end); after a run it plays the motion back. When a run goes back to an earlier frame,
  the timeline does too: it keeps the motion that is kept.

`autostart=true` starts the configured task without waiting for Start. The server stays up, ready for the next run,
until Ctrl+C; with `wait_on_finish=false` the process ends with the first run. Every run writes into its task's
scene directory; with `output_dir` set, all runs write there and each replaces the files of the one before.

The server listens on this machine only. To watch a run on another machine, forward the port
(`ssh -L 8080:127.0.0.1:8080 HOST`), or set `SPIDER_VISER_HOST=0.0.0.0` on a network you trust.
`viewer=mujoco` opens MuJoCo's native viewer instead, `viewer=rerun` Rerun, `show_viewer=false` none; these run the
configured task at once.

## What the planner does

The control is a residual on the reference: per hand 3 wrist position residuals, 3 wrist rotation residuals and the
finger joint residuals. The scene's control law filters it and tracks `reference + residual` with stiff drives, so a
zero control replays the demonstration. At every control frame (20 Hz) the planner draws 128 residual plans over
the next 0.4 s around the plan of the frame before, rolls them out in parallel worlds, and keeps the weighted mean
of the best tenth; four such iterations per frame. The reward is SPIDER's tracking cost: object pose and hinge
angle, wrists and fingers against the reference.

Three things keep a run from failing for good when the object slips (`recovery` in `examples/config/cpu.yaml`):

- when even the best sample expects the object off its path, the frame is planned again with more noise;
- when the executed object slips (more than 5 cm off, or turned more than 0.5 rad, for two frames in a row)
  while it is still in hand, the run goes back to 1 s before it was last within 2 cm and plans that stretch
  again with new samples (then 2 s, then 3 s before it), looking 0.8 s ahead instead of 0.4 s so that the
  planner sees the slip coming;
- when the object has been lost (more than 10 cm off for 2 s), turned, or a lid opening of the reference was
  missed, the run goes back the same way. These returns have tries of their own; after three at one place, a run
  that has lost the object stops and saves the motion up to there.

`examples/config/cpu.yaml` holds the settings; [the GPU page](workflow-chord.md#what-the-defaults-are-and-why) says
why they are what they are (the searches ran on this simulator).

## How fast

Seconds of planning per second of motion, on an otherwise idle 64-core machine:

| `num_threads` | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| defaults (128 samples, 4 iterations per frame) | 27 | 15 | 9.2 | 6.7 | 6.0 |
| `max_num_iterations=2` | 14 | 7.7 | 4.7 | 3.3 | |

A 30 s clip takes about 5 minutes on 16 threads. More than 16 to 32 threads gain little: 128 small rollouts do not
fill more cores. Results do not depend on the thread count: a scene, a seed and the settings always give the same
run, on any machine.

## The scene format

`scene.xml` is an ordinary MJCF with the meshes inline. `scene.npz` holds, in that model's coordinates:

| Array | Shape | |
|---|---|---|
| `qpos_ref` | (T, nq) | the reference motion, one row per control frame |
| `frame_dt`, `substeps` | | control frame [s] and physics steps per frame |
| `targets0` | (T, nv) | joint targets of a zero residual |
| `position_actuator`, `position_actuator_dof` | (P,) | actuators that track a target, and the dof each tracks |
| `wrist_pos_dof`, `wrist_ori_dof`, `finger_dof`, `wrist_qpos` | (H, ...) | the residual's layout per hand |
| `wrist_kp`, `wrist_tau_max` | (H,) | the wrist rotation's proportional torque and its limit |
| `action_clip`, `action_ema` | (nu,), () | the residual's clip and filter |
| `scalar_q`, `quat_q`, `vel_q`, `vel_dof`, `scalar_group`, `quat_group`, `articulation_q` | | what the tracking cost compares |
| `object_body`, `wrist_body`, `object_pose_ref`, `object_pose_reset`, `body_names`, `sides` | | bodies and poses for scoring |

Gains are the model's actuator gains; the timestep is the model's.

## Tests

```bash
pytest spider/simulators/cpu_test.py spider/postprocess/evaluate_cpu_test.py spider/viewers/viser_viewer_test.py
pytest spider/simulators/chord_export_test.py   # the export of FlashCHORD scenes, on synthetic ones (no FlashCHORD)
```
