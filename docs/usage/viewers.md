# Viewers

SPIDER supports multiple viewer backends for visualization during retargeting. This guide covers setting up and using different viewers, including remote development workflows.

## Available Viewers

SPIDER supports the following viewer configurations:

| Viewer | Description | Use Case |
|--------|-------------|----------|
| `mujoco` | Native MuJoCo viewer | Fast local visualization |
| `rerun` | Rerun.io viewer | Remote development, rich logging |
| `viser` | Viser web viewer | Lightweight remote visualization |

To enable both, try `mujoco-rerun` or `mujoco-viser` viewer. The default is `viser`.

The Viser server of a retargeting run has no login, so it listens on the machine it runs on only
(`http://localhost:8080`). To watch a run on another machine, forward the port
(`ssh -L 8080:127.0.0.1:8080 HOST`), or set `SPIDER_VISER_HOST=0.0.0.0` on a network you trust;
`SPIDER_VISER_PORT` changes the port.

## Viser Viewer

For **saved dataset results**, use the dataset inspector:

```bash
uv run examples/inspect_dataset.py --dataset-dir example_datasets
```

It discovers standard processed trials and provides a trial selector, playback,
frame scrubbing, and an IK-reference overlay at `http://localhost:8080`. It runs on
CPU without starting retargeting. See [lifted datasets](lifted-datasets.md) for
conversion, download, and smoke-test instructions.

[Viser](https://github.com/nerfstudio-project/viser) provides a lightweight web-based viewer for 3D scenes.

### Local Usage

```bash
uv run examples/run_mjwp.py viewer=viser
```

This will:
1. Start a Viser server
2. Log simulation data in real-time
3. Print a URL to open in your browser

The scene follows the run while the timeline's frame slider is at its end, with the reference as a translucent
ghost and the sampled traces of the latest plan; move the slider back to look at an earlier frame. The CPU
workflow adds a panel that picks the task and the settings and starts runs
([details](../workflows/workflow-cpu.md#watching-a-run)).

::: warning
If you run headless (no X display), you may need to set a display before launching:

```bash
export DISPLAY=:1
# or
export MUJOCO_GL=egl
```
:::

::: info
If you see an import error, install the dependency:

```bash
pip install viser
```
:::

## Rerun Viewer

[Rerun](https://rerun.io/) provides a powerful logging and visualization system with time-travel capabilities, perfect for remote development.

### Local Usage

```bash
uv run examples/run_mjwp.py viewer=rerun
```

This will:
1. Automatically spawn a Rerun viewer
2. Log simulation data in real-time
3. Display 3D visualization and metrics

### Remote Development

For headless servers, use Rerun's web viewer:

#### Step 1: Start Rerun Server

On your remote server:

```bash
# Start Rerun web server
uv run rerun --serve-web --port 9876

# Or use rerun datapoints for more features
uv run rerun datapoints --serve-web --port 9876
```

#### Step 2: Run SPIDER

In another terminal on the remote server:

```bash
uv run examples/run_mjwp.py viewer=rerun
```

#### Step 3: Connect from Local Machine

Open your local browser to:

```
http://your-server:9876
```

Or connect via the Rerun client:

```bash
# On local machine
rerun --connect rerun+http://your-server:9876/proxy
```

::: warning
Make sure port 9876 is accessible through your firewall. You may need to configure SSH port forwarding:

```bash
ssh -L 9876:localhost:9876 user@remote-server
```
:::

### Logged Data

SPIDER logs the following data to Rerun:

- 3D scene visualization (robot and objects)
- Joint positions and velocities
- Optimization metrics (improvement, reward)
- Contact forces and positions
- Trace trajectories
