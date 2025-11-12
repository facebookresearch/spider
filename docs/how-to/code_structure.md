## Code Structure

- `viewers`: visualizer and plotter, used to monitor all state of different simulators. Can render with mujoco render or log with rerun.
- `config.py`: config class and auto config setting, mainly for static parameter generation.
- `interp.py`: interpolation function for trajectory. in the repo, i will interpolate everything into simulation time, i.e. for 1s trajectory, with 20Hz control frequenc and 100Hz simulation frequency, the trajectory will have 100 points, each control has 5 point. Currently only support zero-order hold.
- `io.py`: load reference data.
- `optimizer.py`: zeroth-order optimization.
