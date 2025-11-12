# Viewers

SPIDER supports multiple viewer backends for visualization during retargeting. This guide covers setting up and using different viewers, including remote development workflows.

## Available Viewers

SPIDER supports the following viewer configurations:

| Viewer | Description | Use Case |
|--------|-------------|----------|
| `mujoco` | Native MuJoCo viewer | Fast local visualization |
| `rerun` | Rerun.io viewer | Remote development, rich logging |

To enable both, try `mujoco-rerun` viewer.

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
