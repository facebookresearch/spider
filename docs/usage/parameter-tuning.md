# Parameter Tuning

This guide helps you tune SPIDER parameters for optimal performance on your tasks.

## Overview

Like reinforcement learning, different motion characteristics require different parameter settings. Start with defaults and iteratively refine based on results.

## Control Parameters

### Simulation Timestep (`sim_dt`)

**Default**: `0.01` (100 Hz)

The fundamental simulation timestep. Smaller values increase accuracy but slow down simulation. Start from a smaller one and gradually increase for faster simulation.

::: warning
All other timing parameters must be divisible by `sim_dt`
:::

### Control Interval (`ctrl_dt`)

**Default**: `0.4s` (2.5 Hz)

How frequently the controller updates. Affects:
- Optimization speed (larger = fewer optimizations)
- Responsiveness (smaller = more reactive)

**Tuning guidelines:**
- **Agile motions**: `0.05-0.2s`
- **Smooth motions**: `0.2-0.4s`

### Planning Horizon (`horizon`)

**Default**: `1.6s`

**Most important parameter**. How far ahead the optimizer looks.

**Impact:**
- ✅ Larger horizon: Better global plan, handles complex motions
- ❌ Larger horizon: Slower optimization, requires more samples
- ✅ Smaller horizon: Faster optimization
- ❌ Smaller horizon: Myopic behavior, may fail on complex tasks

**Tuning strategy:**
1. Start large (2.0s) for complex/unknown tasks
2. Gradually decrease until performance degrades
3. Monitor planned trajectories in Rerun viewer

```bash
# Complex manipulation
uv run examples/run_mjwp.py horizon=2.0

# Simple tracking
uv run examples/run_mjwp.py horizon=1.0
```

::: tip Visual Debugging
Use Rerun viewer to see the planned trajectory (blue/red traces). If the plan looks good but execution fails, the horizon might be too short.
:::

### Knot Spacing (`knot_dt`)

**Default**: `0.4s`

Temporal resolution of control parameterization. Actions are interpolated between knot points.

**Purpose**: Reduces search space by smoothing actions.

**Guidelines:**
- **Agile motions** (jumping, quick grasps): `0.1-0.2s`
- **Smooth motions** (walking, pouring): `0.2-0.4s`

```bash
# Fine control for agile task
uv run examples/run_mjwp.py knot_dt=0.1

# Smooth interpolation
uv run examples/run_mjwp.py knot_dt=0.3
```

## Sampling Parameters

### Number of Samples (`num_samples`)

**Default**: `1024`

Number of parallel trajectories to evaluate.

**Impact:**
- ✅ More samples: Better exploration, more robust
- ❌ More samples: Higher memory usage, slower per-iteration
- Scales linearly with GPU memory and compute

**Tuning:** try it out on your GPU and pick up the largest number that doesn't cause significant slow down.

### Temperature (`temperature`)

**Default**: `0.1`

Softmax temperature for sample weighting.

**Impact:**
- **Lower** (0.01-0.05): Sharper distribution, more optimal but can be unstable
- **Higher** (0.2-0.5): Smoother distribution, more stable but less optimal

**Tuning strategy:**
1. Start at `0.1`
2. If motion is shaky/unstable, increase to `0.15-0.2`
3. If motion is stable, decrease to `0.05-0.08` for better quality

Generally, this parameter is less important.

```bash
# Stable but less optimal
uv run examples/run_mjwp.py temperature=0.2

# Optimal but may be shaky
uv run examples/run_mjwp.py temperature=0.05
```

::: tip Monitoring
Check the `ctrl` panel in rerun viewer. If the action is shaky, increase the temperature.
:::

### Max Iterations (`max_num_iterations`)

**Default**: `32`

Maximum optimization iterations per control step.

**Tuning:**
1. Run with default and monitor `improvement` in Rerun
2. If `improvement` plateaus early (< 10 iters), decrease
3. If `improvement` is still high at end, increase

```bash
# Quick convergence
uv run examples/run_mjwp.py max_num_iterations=8

# Thorough optimization
uv run examples/run_mjwp.py max_num_iterations=64
```

### Early Stopping (`improvement_threshold`)

**Default**: `0.01`

Stop optimization if improvement falls below threshold.

**Usage:**
- Start at `0` (disabled) to see full convergence
- Gradually increase for speed: `0.01` → `0.02` → `0.05`

```bash
# No early stopping
uv run examples/run_mjwp.py improvement_threshold=0.0

# Aggressive early stopping
uv run examples/run_mjwp.py improvement_threshold=0.05
```

## Noise Scheduling

### Control Noise Scales

These control exploration throughout the horizon:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `first_ctrl_noise_scale` | `0.5` | Noise at start of horizon |
| `last_ctrl_noise_scale` | `1.0` | Noise at end of horizon |
| `final_noise_scale` | `0.1` | Global annealing factor |

**Tuning:**
- Motion is stable? Decrease `first_ctrl_noise_scale` to `0.01-0.1`
- Need more exploration? Increase `last_ctrl_noise_scale` to `0.1-1.0`
- High precision task? Decrease `final_noise_scale` to `0.05`

### Component-Specific Scales

For dexterous hands, noise is applied differently to each component:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `joint_noise_scale` | `0.3` | Robot joint angles |
| `pos_noise_scale` | `0.01` | Hand base position |
| `rot_noise_scale` | `0.03` | Hand base rotation |

**Tuning guidelines:**
- Base is stable? Decrease `pos_noise_scale` and `rot_noise_scale`
- Finger control needs more exploration? Increase `joint_noise_scale`

```bash
# Stable base, explore fingers
uv run examples/run_mjwp.py \
  pos_noise_scale=0.005 \
  rot_noise_scale=0.01 \
  joint_noise_scale=0.5
```

## Reward Scaling

Balance different tracking objectives:

### Position vs Rotation

```yaml
pos_rew_scale: 1.0     # End-effector position
rot_rew_scale: 0.3     # End-effector rotation
```

**Tuning:**
- Task requires precise orientation (e.g., key insertion)? Increase `rot_rew_scale`
- Position more important? Keep `rot_rew_scale` low

### Joint Tracking

```yaml
joint_rew_scale: 0.003
```

Lower weight since joint tracking is less critical than end-effector pose.

**Tuning:**
- Specific joint configuration needed? Increase to `0.01-0.03`
- Only care about end-effector? Keep very low (`0.001-0.003`)

### Velocity Regularization

```yaml
vel_rew_scale: 0.0001
```

Penalizes large velocities to smooth motion.

**Best practices:**
- ✅ Adding velocity tracking helps reduce latency
- ❌ Don't make it too large (causes sluggish motion)

```bash
# Reduce tracking latency
uv run examples/run_mjwp.py vel_rew_scale=0.001

# Disable velocity penalty
uv run examples/run_mjwp.py vel_rew_scale=0.0
```

## Common Issues

### Motion is Unstable/Shaky

**Symptoms**: Robot vibrates, erratic movements, or fails to complete the task

**Solutions:**
1. Iecrease `num_samples` (most effective)
2. Check if `horizon` is too short
3. Increase `knot_dt` for smoother actions or decrease it for more reactive motion

### Optimization Too Slow

**Symptoms**: RTR (realtime rate) < 0.1

**Solutions:**
1. Reduce `horizon` (most effective)
2. Increase `ctrl_dt` (also effective, make sure it doesn't cause instability)
