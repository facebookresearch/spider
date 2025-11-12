# Parameter tuning guide for sampling

Similar to RL, when handling every bad initial motion, we need to tweak the sampling parameters to get a good initial motion.

## Important control parameters

- `sim_dt`: simulation timestep. Default to 0.01.
- `ctrl_dt`: control interval, how often the control is updated.
- `ref_dt`: reference data interval, how fast is the reference motion.
- `horizon`: planning horizon, how far the robot is looking ahead. Every important. The better the initial motion is, the smaller horizon could be, the faster the optimization can be. Start from large, like 1.6, and then decrease it. Looking at rerun visualized planned trajectory to determine the horizon.
- `knot_dt`: knot point spacing, how smooth the action is. This is every useful to reduce the search space by interpolating the action. If the motion is agile, like jumping, use smaller knot_dt. If the motion is smooth, like walking, use larger knot_dt.

## Important sampling parameters

- `num_samples`: number of samples to generate, larger the better
- `temperature`: temperature for the sampling distribution. The smaller, the more optimal the motion will be, however, it also leads to larget variance (shaky motion). Recommendation: start from 0.1. If not unstable, make it larger. If stable, make it smaller.
- `max_num_iterations`: total number of iterations to optimize per step. The larger the better. See `improvement` log in `rerun` viewer. If improvement is high, increase number, otherwise decrease number.
- `improvement_threshold`: threshold for improvement for early return. Start from 0 and gradually increase for acceleration. Also refer to logged `improvement` in `rerun` viewer.
- `first_ctrl_noise_scale`: the sampling parameter used in DIAL-MPC. Set to 1.0 from start. Decrease it when motion is stable to optimize the speed.
- `last_ctrl_noise_scale`: the sampling parameter used in DIAL-MPC. Set to 1.0 from start. Control the final control noise level. No need to change it in most cases.
- `final_noise_scale`: the global annealing parameter. Set to 0.1 from start. Control final search range. Make it smaller when task requires high precision.
- `joint_noise_scale`: action scale for robot joints. Default to 1.0, depends on robot setting
- `pos_noise_scale`: only for dexterous hand, action scale for hand base movement.
- `rot_noise_scale`: only for dexterous hand, action scale for hand base rotation.

## Best practices

- *Reduce latency with velocity cost*: adding `qvel` tracking cost helps reduce the overall tracking latency.

- *Avoid negative cost*: removing `ctrl` panelty further improve sample efficiency -- the smoothness of the action should be controlled by the trajectory-level annealing factor and knots numbers. Use the `knot_dt` parameter to control the smoothness of the action.
