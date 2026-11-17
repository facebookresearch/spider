# Workflow: FlashCHORD (GPU)

This workflow runs SPIDER's sampling MPC directly on the environment of
[FlashCHORD](https://nvidia-isaac.github.io/video_to_data/) (part of NVIDIA's `video_to_data` toolkit) instead of
SPIDER's own MuJoCo Warp scene. Nothing is ported: the physics (Newton `SolverMuJoCo` on MuJoCo Warp), the control
law (residual hand-pose actions on PD-controlled wrists and fingers) and the reference are FlashCHORD's, and
SPIDER's sampler drives them through the usual simulator interface.

The same scene also runs in the [CPU workflow](workflow-cpu.md), SPIDER's default, without a GPU and without
FlashCHORD. That page is the place to start: it has the demo, the viewer and what the planner does. This page
covers the GPU environment, the export of scenes for the CPU, and the reasons behind the settings both use.
`examples/run_chord.py` is the GPU twin of `examples/run_cpu.py` (the same loop, recovery and viewer on
`spider/simulators/chord.py`), and `examples/config/chord.yaml` starts from `cpu.yaml`.

## Prerequisites

- FlashCHORD (`flash_chord`, with Newton and MuJoCo Warp) installed as its own README describes, and SPIDER
  installed into the same environment without its dependencies: `pip install --no-deps -e /path/to/spider`, plus
  `hydra-core`, `loguru`, `tyro`, `scipy`, `opencv-python`, `imageio[ffmpeg]`, `rerun-sdk==0.26.2`, `viser` and
  `trimesh` if they are missing.
- A FlashCHORD reference: a `robot_name=...` directory holding the retargeted parquet, with the object assets and the
  support-surface stage beside it as FlashCHORD's loaders write them. FlashCHORD ships `tissue_box_simple`; see
  [ARCTIC references](#arctic-references) for building more.

## Run

```bash
python examples/run_chord.py "data_path='REFERENCE_DIR'" output_dir=OUT

# reference replay without planning, as a baseline
python examples/run_chord.py "data_path='REFERENCE_DIR'" output_dir=OUT max_num_iterations=0

# half the planning time (2 iterations per control frame instead of 4)
python examples/run_chord.py "data_path='REFERENCE_DIR'" output_dir=OUT max_num_iterations=2
```

Keep the inner quotes around the reference path: FlashCHORD's reference directories contain `=`
(`sequence_id=.../robot_name=...`), which Hydra otherwise reads as part of the override.

With the default Viser viewer a run waits for **Start** in the browser (`http://localhost:8080`), where the settings
and the recovery can be changed first; `autostart=true` starts it at once, `show_viewer=false` runs it without a
viewer.

The viewer, the outputs (`trajectory_chord.npz`, `metrics.json`, `visualization_chord.mp4`) and the metrics are
those of the [CPU workflow](workflow-cpu.md#watching-a-run). The gains, the action filter and the solver sizes of
the recipe are constants at the top of `spider/simulators/chord.py`; the planner's settings are in
`examples/config/chord.yaml`.

Two things differ from the CPU simulator. The reward adds CHORD's contact terms to SPIDER's tracking cost
(`CONTACT_SCALE`), which only FlashCHORD's environment computes; on the GPU they help (55 of 60 runs pass with
them, 24 of 29 with the tracking cost alone). And GPU physics is not repeatable: the same clip, settings and seed
can pass on one run and fail on the next, so count runs, not seeds.

## Export a scene for the CPU

```bash
python -m spider.simulators.chord_export --parquet REFERENCE_DIR \
    --out example_datasets/processed/chord/sharpa/bimanual/TASK/DATA_ID
```

The export builds the scene once with the settings of `spider/simulators/chord.py` and writes `scene.xml` and
`scene.npz` ([format](workflow-cpu.md#the-scene-format)): the MJCF that FlashCHORD compiles, with the support
table fixed to the world, the object's hinges passive and held by friction, and the reference and the control
tables in that model's coordinates. Gains and the action filter are fixed at export.

The export does not need a GPU: with `--device cpu` it builds the scene on Warp's CPU device in about half a minute
(FlashCHORD's environment with the CPU build of torch is enough), and writes the same tables.

FlashCHORD decomposes the object meshes into convex pieces when it builds a scene, and the result differs from
machine to machine (the ketchup bottle came out as 90, 91 or 92 pieces on thirteen machines). The GPU environment
plans on whatever that machine built; an exported scene is one decomposition, which is what makes CPU runs
comparable.

## What the defaults are, and why

The settings were chosen by three searches on twenty ARCTIC clips: for tracking, for speed, and for a steady hand
base; the last two on the CPU simulator, where a scene, a seed and the settings always give the same run. Pass
counts below are of 60 runs (twenty clips, three seeds) on the CPU unless they say otherwise.

| Setting | Default | Why |
|---|---|---|
| Hand gains (`chord.py`) | wrists 1600 N/m with 85 N s/m, fingers 40 N m/rad | The largest effect of all. FlashCHORD's hands are tuned soft (25 N/m, 1.7 N m/rad) for reinforcement learning that starts with a virtual wrench carrying the object. The residual can move a wrist target 20 cm at most, which at 25 N/m is 5 N per hand: not enough to carry a 1 kg object. Stiff hands also go where the plan sends them, so every sample tells the planner more: with wrists at 200 N/m and fingers at 10 N m/rad 50 runs pass instead of 60. Wrists from 1600 to 6400 N/m and fingers from 20 to 80 N m/rad track the same. |
| Action filter (`chord.py`) | 0.5 | With the damping above, what keeps the hand base steady; see [A steady hand base](#a-steady-hand-base). |
| `ctrl_dt` / `max_num_iterations` | 0.05 s / 4 | Replan at every control frame with a few iterations. The plan of the last frame is the starting point of the next, so iterations are not lost, and planning often with few iterations beats planning seldom with many for the same rollout physics: 4 per frame pass 60 runs, 8 every 0.1 s 56, 16 every 0.2 s 53; at half the physics 57, 50 and 43. The planning time is proportional to the iterations: 2 halve it, 8 double it without passing more runs. |
| `num_samples` | 128 | With stiff gains 128 samples track as well as 256; 64 lose more runs than halving the iterations does. |
| `temperature` / `final_noise_scale` | 0.3 / 1.0 | The best tenth of the samples is averaged with these weights; 0.1 follows the single best sample too closely (53 runs pass, and the hand shakes half as much again). With a few iterations per tick the noise is not shrunk from one iteration to the next: leaving 0.7 of it in the last iteration passes 58 runs, 0.5 55, 0.3 53, 0.1 54. |
| `knot_dt` / `first_ctrl_noise_scale` | 0.2 s / 0.5 | Two knots over the horizon, the first with half the noise of the last. Four knots shake more; a quieter first knot steadies little and loses runs (0.25: 53). |
| `pos_noise_scale` / `rot_noise_scale` / `joint_noise_scale` | 0.01 m / 0.01 rad / 0.10 rad | Residual noise per knot. Once an object is in hand, 2 cm of wrist noise alone loses it in about half of the samples within the horizon; 0.5 cm is too little to correct the hands (55 runs pass). |
| `horizon` | 0.4 s | With a good plan to start from, a longer look ahead buys nothing: 0.4 s tracks as well as 0.6 s and 0.8 s; 0.3 s and shorter lose accuracy. On a stretch planned again after going back it does help (see `recovery`). |
| Reward | SPIDER's tracking cost; `base_pos_rew_scale` / `base_rot_rew_scale` / `joint_rew_scale` 1.0 / 0.3 / 0.01 | SPIDER's cost does the tracking; CHORD's objective as the planner reward tracked worse in every comparison (4 cm instead of 1 cm). The wrist terms keep the hands near the demonstration: without them they end 3.4 cm away and fewer runs pass. |
| No object guidance | | A virtual wrench pulling the object to its reference inside planning rollouts helped the soft hands of an earlier recipe. With stiff hands it hurts: 49 runs pass with it. It is no longer an option. |
| `recovery` | plan a frame again up to twice; go back when the object slips (5 cm or 0.5 rad off for two frames) to 1 s before it was last within 2 cm, and when it is lost; 2 s and 3 s on the next tries at the place; plan the stretch again looking 0.8 s ahead | Searching harder in place is what gets lids opened; going back is what recovers a lost object, which planning on rarely does because the hands follow the demonstration and the sampling noise no longer reaches the object. Without going back 47 runs pass, without the second planning pass 51, without either 36. Going back only once the object was lost (10 cm off for 2 s) returned to a grasp that was already failing; catching the slip while the object is still in hand passes 263 of the 277 ARCTIC sequences instead of 254. Looking 0.8 s ahead on the stretch planned again lets the planner see the failure coming: 267 pass (+8, -4; the ketchup bottle 27 of 28 instead of 24), at 2.1 returns a run; with seed 1, 267 against 257 (+13, -3). Going back less far loses runs: to 0.05, 0.25 or 0.5 s before the slip 212, 220 and 236 pass; with the longer look ahead, 0.5 s passes 245 and 1 s on every try 251. A fixed 2 s from the slip's first frame passes 266, steps of 0.5 s 264, a fourth try 266. A check at 2 to 4 cm fires on ordinary wobbles and spends the tries before the real slip. |
| Hinge friction (`HINGE_HOLD`) | 1.5 | Dry friction of object hinges, as a multiple of what holds the hinged part against gravity. FlashCHORD gives every hinge 0.1 N m, less than a laptop lid weighs, so the lid swings as soon as the object is tilted. |
| Kinematic armature (`KINEMATIC_ARMATURE`) | 1e4 | Newton compiles FlashCHORD's kinematic support table with an armature of 1e10, which makes MuJoCo Warp's constraint solver stop after one iteration. This lowers it and recomputes the solver statistic. (The CPU scene fixes the table to the world instead.) |
| Solver room (`NJMAX` / `NCONMAX`) | 1024 / 512 | FlashCHORD trains with 512 constraint rows, which mesh-heavy objects overflow; rows that do not fit are dropped. |

## A steady hand base

Replanning at every control frame moves the wrist targets twenty times a second, and a stiff wrist drive answers
each move. With stiff gains at FlashCHORD's damping ratio (1600 N/m with 42 N s/m) the wrists moved 2.2 times as
fast as the demonstration at 3 Hz and above, with a bump near 5 Hz: the hand base trembled. The evaluation reports
this as `wrist_shake` (1 is as smooth as the demonstration). What was tried, on the CPU:

| Change | Wrist shake | Runs passing |
|---|---|---|
| Stiff gains at FlashCHORD's damping ratio | x2.2 | 118 of 120 |
| Replan every 0.1 s with 8 iterations, or every 0.2 s with 16 | x2.2, x2.3 | 56, 53 of 60 |
| Wrist noise 0.5 cm, 0.25 cm | x1.7, x1.5 | 55, 53 of 60 |
| One knot over the horizon | x1.8 | 59 of 60 |
| First-knot noise 0.25, 0.1 | x2.0, x2.0 | 53, 52 of 60 |
| Annealing to 0.3 | x1.8 | 53 of 60 |
| Twice the wrist damping | x1.5 | 117 of 120 |
| **Twice the damping, action filter 0.5 (the default)** | x1.2 | 114 of 120 |
| Twice the damping, filter 0.5, wrist noise 0.5 cm | x1.0 | 111 of 120 |

Making the planner search less (less noise, fewer knots, annealing, replanning less often) steadies the hand only
as far as it also stops the planner from correcting it. Damping the drive and filtering the action steady the hand
without taking anything from the search: twice the damping is free, the filter on top costs about three runs in a
hundred.

## ARCTIC references

References for [ARCTIC](https://arctic.is.tue.mpg.de) sequences are built with the ARCTIC pipeline of the toolkit
that ships FlashCHORD (its `docs/ARCTIC_SETUP.md`): the ARCTIC raw sequences and MANO need their own licences, and
the articulated URDFs come from [ArtiGrasp](https://github.com/zdchan/artigrasp), which has ten of ARCTIC's eleven
objects (no scissors). Five things were needed beyond that document when running the stages natively:

- `--sequence_pattern` is a regular expression searched in the sequence name, so `s02_microwave_use_01` also selects
  `s02_microwave_use_01_retake`: end the pattern with `$`.
- The loader needs NumPy 1 (chumpy, for the MANO pickles) and the retargeting needs Pinocchio 3, which needs NumPy 2:
  use one environment for the load stage and another for the retarget and support stages. The retargeting also needs
  a QP solver (`daqp`).
- Run the support-surface stage from the dataset root: the parquet stores mesh paths relative to it.
- ArtiGrasp caps the hinge of the phone and of the ketchup bottle at 2.5 rad, less than the ARCTIC motion opens them.
  A reference outside the joint range starts the simulation with a violated limit and throws the object off the
  table; widen the URDF limit to the demonstrated range. The GPU simulator warns when a reference leaves a joint's
  range.
- The support-surface stage writes one thin cylinder under the place where the object first rests, sized to the
  object's footprint (5 to 22 cm radius for the ARCTIC objects). The recordings were made on a table: objects are
  set down away from where they started, and an object that slips should land on the table, not on the floor a
  metre below. Raise the cylinder radius in the support `.usda` to table size (0.6 m was used here; the hands do not
  collide with the support). The evaluation reports `fallen_frames` when an object ends up below its support.

## Where it works and where it does not

The settings were run on twenty [ARCTIC](https://arctic.is.tue.mpg.de) clips: subject s01, ten articulated objects,
two takes each (`use_01`, `use_02`), 20 to 30 s, two Sharpa hands, nothing tuned per clip. A run passes when every
object body stays within 3 cm and 30 degrees of the reference on average, the hinge within 10 degrees, every
opening of the reference is followed at least half of the way, and the run was not stopped early.

| Simulator | Runs | Pass | Object on its path | Stopped early | Object error | Wrist shake | Time per clip |
|---|---|---|---|---|---|---|---|
| CPU (six seeds) | 120 | 99% | 99% | 1 | 1.0 cm | x1.2 | 9 min, twelve runs sharing 96 cores |
| GPU, one L40S (three runs per clip; going back only once the object is lost) | 60 | 92% | 93% | 2 | 1.5 cm | x1.3 | 5 min |
| CPU, all of ARCTIC (277 sequences, seed 0) | 277 | 96% | 97% | 5 | 1.1 cm | x1.4 | 11 min, twelve runs sharing 96 cores |
| ... looking 0.4 s ahead on the stretch planned again | 277 | 95% | 96% | 9 | 1.2 cm | x1.3 | 13 min, sixteen runs sharing 128 cores |
| ... going back only once the object is lost | 277 | 92% | 94% | 9 | 1.4 cm | x1.4 | 7 min, ten runs sharing 128 cores |

The last three rows are every ARCTIC sequence that has an object model: nine subjects, ten objects, 221 `use` and 56
`grab` sequences of 18 to 37 s (the 24 scissors sequences have none). With the defaults 267 pass; of the 10 that do
not, five stop with the object lost, four finish with the object off its path and one misses an opening of the
hinge. With seed 1, 267 pass as well (257 looking 0.4 s ahead: +13, -3). By object (seed 0): box 25 of 25, microwave
27 of 27, mixer 29 of 29, laptop 28 of 29, waffle iron 29 of 30, espresso machine 27 of 28, ketchup bottle 27 of 28,
notebook 24 of 25, capsule machine 25 of 27, phone 26 of 29. The scenes are the [example
dataset](workflow-cpu.md#prerequisites)'s `processed/chord`; its runs are those of the defaults.

On the CPU every clip passes in all six runs but the phone's first take, which loses the phone once (seed 4). With
the earlier rule (going back only once the object is lost) 114 of 120 passed: the ketchup bottle failed four runs of
twelve and the espresso machine's first take two of six. On the GPU, with that rule, five runs of sixty fail: the
ketchup bottle's second take twice, and one run each of the box's first take, the capsule machine's second and the
espresso machine's first.

Things to know when reading results:

- **Count runs, not seeds, on the GPU.** The physics is not repeatable once an object is in hand, and the object's
  collision pieces are decomposed anew on every machine.
- **Count enough runs on the CPU too.** A scene, a seed and the settings give one run, but seeds differ: the same
  settings passed 60 of 60 runs on seeds 0 to 2 and 58 of 60 on seeds 3 to 5. One or two runs in sixty is noise.
- **Compare settings on clips that include a small object.** The ketchup bottle and the phone are what separates
  settings; ten clips without them ranked annealing, guidance and 64 samples wrongly.
- **A run that goes back takes longer.** A run goes back 2.1 times on average; a clip with several returns can take
  twice as long.

## Tests

```bash
pytest spider/simulators/chord_export_test.py                           # the export, on synthetic scenes
CHORD_TEST_PARQUET=REFERENCE_DIR pytest spider/simulators/chord_test.py # GPU environment
```
