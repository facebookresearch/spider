# Add new dataset

Prepare data: prepare the hand kinematics data and object poses data and put it under {dataset_dir}/raw/{dataset_name}/{data_id}.pkl
Please also put the mesh under the same folder since later when we need to compile the scene, the mesh is required.

Next, read the following key information from the data:
- hand kinematics data: qpos_finger_right, qpos_finger_left, qpos_wrist_right, qpos_wrist_left
- object poses data: object_pose_right, object_pose_left

Next, convert mesh to obj file that is compatible with mujoco.

Lastly, it is recommended to visualize the data in mujoco viewer to check if the data is correct.
