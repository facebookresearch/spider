SPIDER is designed to work with any simulator out of the box.
In this document, we will walk through the process of porting a new simulator to SPIDER from a existing RL environment.

In the high-level, two things are needed for a new simulator:
1. A way to rollout the simulator given a control sequence and return cumulated cost. (e.g. `step_env` in `spider/simulators/mjwp.py`)
2. A way to reset the simulator to certain state (e.g. `load_state` in `spider/simulators/mjwp.py`)
