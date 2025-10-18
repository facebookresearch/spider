SPIDER is designed to work with any simulator out of the box.
In this document, we will walk through the process of porting a new simulator to SPIDER from an existing RL environment.

## Required Functions

At a high level, the following functions are needed for a new simulator (see `spider/simulators/hdmi.py` as a minimal example):

### Core Functions (Required)
1. **`setup_env(config, ref_data)`**: Initialize and return the environment instance
2. **`step_env(config, env, ctrl)`**: Step the simulator with control inputs
3. **`save_state(env)`**: Save the current state of all environments
4. **`load_state(env, state)`**: Load a saved state back into the environments
5. **`get_reward(config, env, ref)`**: Compute reward for current state
6. **`get_terminal_reward(config, env, ref_slice)`**: Compute terminal reward with scaling

### Helper Functions (Required for optimizer)
7. **`get_trace(config, env)`**: Return visualization trace points (hands, feet, objects)
8. **`save_env_params(config, env)`**: Save environment parameters for domain randomization
9. **`load_env_params(config, env, env_param)`**: Load environment parameters
10. **`sync_env(config, env)`**: Synchronize state across parallel environments (broadcast first env to all)

## Implementation Steps

1. **Create simulator module**: Create `spider/simulators/<simulator_name>.py` implementing all required functions
2. **Create config file**: Create `examples/config/<simulator_name>.yaml` with simulator-specific parameters
3. **Create launcher**: Create `examples/run_<simulator_name>.py` that:
   - Imports functions from your simulator module
   - Sets up the environment and gets action dimension (`env.num_actions`)
   - Creates the optimizer using `make_rollout_fn`, `make_optimize_once_fn`, `make_optimize_fn`
   - Implements the control loop with `optimize()`, `step_env()`, and `sync_env()`
   - Handles viewer updates and saves trajectory data

## Example Implementation

See the HDMI implementation as a reference:
- Simulator: `spider/simulators/hdmi.py`
- Config: `examples/config/hdmi.yaml`
- Launcher: `examples/run_hdmi.py`
- Shell script: `examples/run_hdmi.sh`

The HDMI example shows how to integrate with an external RL environment (mjlab/SimpleEnv) that has built-in reference motion and reward computation.
