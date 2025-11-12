# Deployment

SPIDER save everything in minimum coordinate system defined in the corresponding scene xml file.

While the joint position can be directly used, it is important to convert the robot base position and orientation to the world coordinate system which can be correctly executed by the real robot.

Please refer to `spider/postprocess/read_to_robot.py` for the conversion code. It is recommended to read from a site from the robot to get wrist orientation that align with your real world setup.
