# Deployment to Real Robots

This guide covers deploying SPIDER-optimized trajectories to real-world robots, including coordinate system transformations and practical deployment considerations.

## Overview

SPIDER saves all trajectories in a **minimum coordinate system** defined in the corresponding scene XML file. While joint positions can be used directly, it's important to convert the robot base position and orientation to the world coordinate system for correct execution on real robots.

Please refer to `spider/postprocess/read_to_robot.py` for the conversion code. It is recommended to read from a site from the robot to get wrist orientation that align with your real world setup.

![Deployment Example](/figs/real/pick_cup_real.gif)
