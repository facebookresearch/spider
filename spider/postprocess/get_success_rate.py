"""
Go through all info.csv and get the success rate of each task

Author: Chaoyi Pan
Date: 2025-07-31
"""

import os
import glob
import pandas as pd
import tyro
import numpy as np

from retarget.io import get_processed_data_dir, get_all_tasks


def quat_to_vel(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to angular velocity
    Args:
    quat: (..., 4) quaternion in (w, x, y, z) format
    Returns:
    vel: (..., 3) angular velocity
    """
    axis = quat[..., 1:4]  # Extract x, y, z components
    sin_a_2 = np.linalg.norm(axis, axis=-1, keepdims=True)

    # Handle zero case
    zero_mask = sin_a_2[..., 0] == 0.0
    result = np.zeros_like(axis)

    # Non-zero case
    non_zero_mask = ~zero_mask
    if np.any(non_zero_mask):
        speed = 2.0 * np.arctan2(sin_a_2[non_zero_mask, 0], quat[non_zero_mask, 0])
        # when axis-angle is larger than pi, rotation is in the opposite direction
        speed = np.where(speed > np.pi, speed - 2.0 * np.pi, speed)

        result[non_zero_mask] = (
            axis[non_zero_mask] * speed[..., np.newaxis] / sin_a_2[non_zero_mask]
        )

    return result


def mul_quat(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions
    Args:
    u: (..., 4) quaternion in (w, x, y, z) format
    v: (..., 4) quaternion in (w, x, y, z) format
    Returns:
    result: (..., 4) quaternion product
    """
    result = np.zeros_like(u)
    result[..., 0] = (
        u[..., 0] * v[..., 0]
        - u[..., 1] * v[..., 1]
        - u[..., 2] * v[..., 2]
        - u[..., 3] * v[..., 3]
    )
    result[..., 1] = (
        u[..., 0] * v[..., 1]
        + u[..., 1] * v[..., 0]
        + u[..., 2] * v[..., 3]
        - u[..., 3] * v[..., 2]
    )
    result[..., 2] = (
        u[..., 0] * v[..., 2]
        - u[..., 1] * v[..., 3]
        + u[..., 2] * v[..., 0]
        + u[..., 3] * v[..., 1]
    )
    result[..., 3] = (
        u[..., 0] * v[..., 3]
        + u[..., 1] * v[..., 2]
        - u[..., 2] * v[..., 1]
        + u[..., 3] * v[..., 0]
    )
    return result


def quat_sub(qa, qb):
    """
    Subtract quaternion qa from qb, in (w, x, y, z) format
    Args:
        qa: (..., 4) quaternion array
        qb: (..., 4) quaternion array
    Returns:
        quat_diff: (..., 3) angular difference
    """
    qneg = qb.copy()
    qneg[..., 1:] = -qneg[..., 1:]  # Negate x, y, z components
    qdif = mul_quat(qneg, qa)

    # convert to 3D velocity
    return quat_to_vel(qdif)


def main(
    dataset_dir: str = "../../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    hand_type: str = "bimanual",
    data_type: str = "ikrollout",
    pos_err_threshold: float = 0.1,
    quat_err_threshold: float = 0.5,
):
    # resolve paths using the new structure
    dataset_dir = os.path.abspath(dataset_dir)

    all_tasks = get_all_tasks(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        hand_type=hand_type,
    )

    print(f"Found tasks: {all_tasks}")

    # Read all info.csv files and collect them
    all_dataframes = []

    # Iterate through each task directory and find data_id subdirectories
    for task_name in all_tasks:
        # Get the task directory structure from processed data
        task_processed_dir = f"{dataset_dir}/processed/{dataset_name}/{robot_type}/{hand_type}/{task_name}"
        if not os.path.isdir(task_processed_dir):
            print(f"Warning: task directory not found: {task_processed_dir}")
            continue

        # Get all data_id directories within the task
        data_id_dirs = [
            d
            for d in os.listdir(task_processed_dir)
            if os.path.isdir(os.path.join(task_processed_dir, d)) and d.isdigit()
        ]

        for data_id_str in data_id_dirs:
            data_id = int(data_id_str)

            # Get the processed data directory for this task/data_id
            processed_dir_robot = get_processed_data_dir(
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                robot_type=robot_type,
                hand_type=hand_type,
                task=task_name,
                data_id=data_id,
            )

            # Check if trajectory files exist
            trajectory_file = f"{processed_dir_robot}/trajectory_{data_type}.npz"
            if not os.path.exists(trajectory_file):
                print(f"Warning: trajectory file not found: {trajectory_file}")
                continue

            try:
                # Load the trajectory data
                traj_data = np.load(trajectory_file)
                qpos_traj = traj_data["qpos"]
                if qpos_traj.ndim == 3:
                    qpos_traj = qpos_traj.reshape(-1, qpos_traj.shape[-1])
                elif qpos_traj.ndim == 2:
                    qpos_traj = qpos_traj
                else:
                    raise ValueError(f"Invalid trajectory dimension: {qpos_traj.ndim}")

                # For comparison, we need a reference trajectory (kinematic IK result)
                kinematic_file = f"{processed_dir_robot}/trajectory_kinematic.npz"
                if not os.path.exists(kinematic_file):
                    print(
                        f"Warning: kinematic reference file not found: {kinematic_file}"
                    )
                    continue

                kinematic_data = np.load(kinematic_file)
                qpos_kinematic = kinematic_data["qpos"]

                # Ensure both trajectories have the same length
                min_length = min(len(qpos_traj), len(qpos_kinematic))
                qpos_traj = qpos_traj[:min_length]
                qpos_kinematic = qpos_kinematic[:min_length]

                print(f"Comparing trajectories: {len(qpos_traj)} frames")

            except Exception as e:
                print(f"Error loading trajectory data for {task_name}/{data_id}: {e}")
                continue

            # compute object tracking error
            if hand_type == "bimanual":
                # get object qpos for both trajectories
                qpos_object_right_traj = qpos_traj[:, -14:-7]
                qpos_object_left_traj = qpos_traj[:, -7:]
                qpos_object_right_kinematic = qpos_kinematic[:, -14:-7]
                qpos_object_left_kinematic = qpos_kinematic[:, -7:]

                # get pos and quat
                pos_object_right_traj = qpos_object_right_traj[:, :3]
                quat_wxyz_object_right_traj = qpos_object_right_traj[:, 3:]
                pos_object_left_traj = qpos_object_left_traj[:, :3]
                quat_wxyz_object_left_traj = qpos_object_left_traj[:, 3:]

                pos_object_right_kinematic = qpos_object_right_kinematic[:, :3]
                quat_wxyz_object_right_kinematic = qpos_object_right_kinematic[:, 3:]
                pos_object_left_kinematic = qpos_object_left_kinematic[:, :3]
                quat_wxyz_object_left_kinematic = qpos_object_left_kinematic[:, 3:]

                # compute pos error
                pos_err_right = np.linalg.norm(
                    pos_object_right_traj - pos_object_right_kinematic, axis=1
                ).mean()
                pos_err_left = np.linalg.norm(
                    pos_object_left_traj - pos_object_left_kinematic, axis=1
                ).mean()
                quat_err_right = np.linalg.norm(
                    quat_sub(
                        quat_wxyz_object_right_traj, quat_wxyz_object_right_kinematic
                    ),
                    axis=1,
                ).mean()
                quat_err_left = np.linalg.norm(
                    quat_sub(
                        quat_wxyz_object_left_traj, quat_wxyz_object_left_kinematic
                    ),
                    axis=1,
                ).mean()

                # compute average pos and quat error (handle the case where the object is a place holder)
                # if pos_object_right_kinematic close to 0, then only use left
                # if pos_object_left_kinematic close to 0, then only use right
                # otherwise, use both
                left_mask = (
                    np.linalg.norm(
                        pos_object_left_kinematic - pos_object_left_kinematic[:1],
                        axis=1,
                    ).mean()
                    < 0.001
                )
                right_mask = (
                    np.linalg.norm(
                        pos_object_right_kinematic - pos_object_right_kinematic[:1],
                        axis=1,
                    ).mean()
                    < 0.001
                )

                if left_mask:
                    obj_pos_err = pos_err_right
                    obj_quat_err = quat_err_right
                elif right_mask:
                    obj_pos_err = pos_err_left
                    obj_quat_err = quat_err_left
                else:
                    obj_pos_err = (pos_err_right + pos_err_left) / 2
                    obj_quat_err = (quat_err_right + quat_err_left) / 2

            else:  # single hand case
                # get object qpos for single hand
                qpos_object_traj = qpos_traj[:, -7:]
                qpos_object_kinematic = qpos_kinematic[:, -7:]

                # get pos and quat
                pos_object_traj = qpos_object_traj[:, :3]
                quat_wxyz_object_traj = qpos_object_traj[:, 3:]
                pos_object_kinematic = qpos_object_kinematic[:, :3]
                quat_wxyz_object_kinematic = qpos_object_kinematic[:, 3:]

                # compute pos and quat error
                obj_pos_err = np.linalg.norm(
                    pos_object_traj - pos_object_kinematic, axis=1
                ).mean()
                obj_quat_err = np.linalg.norm(
                    quat_sub(quat_wxyz_object_traj, quat_wxyz_object_kinematic), axis=1
                ).mean()

                pos_err_right = obj_pos_err if hand_type == "right" else 0.0
                pos_err_left = obj_pos_err if hand_type == "left" else 0.0
                quat_err_right = obj_quat_err if hand_type == "right" else 0.0
                quat_err_left = obj_quat_err if hand_type == "left" else 0.0

            # compute success
            success = (obj_pos_err < pos_err_threshold) & (
                obj_quat_err < quat_err_threshold
            )

            # Create dataframe entry
            df_entry = {
                "dataset": dataset_name,
                "robot_type": robot_type,
                "hand_type": hand_type,
                "data_type": data_type,
                "task": task_name,
                "data_id": data_id,
                "directory": f"{task_name}/{data_id}",
                "obj_pos_err": obj_pos_err,
                "obj_quat_err": obj_quat_err,
                "obj_pos_err_right": pos_err_right,
                "obj_pos_err_left": pos_err_left,
                "obj_quat_err_right": quat_err_right,
                "obj_quat_err_left": quat_err_left,
                "success": success,
                "pos_err_threshold": pos_err_threshold,
                "quat_err_threshold": quat_err_threshold,
            }

            # Convert to single-row dataframe
            df = pd.DataFrame([df_entry])
            all_dataframes.append(df)
            print(
                f"Processed {task_name}/{data_id}: pos_err={obj_pos_err:.4f}, quat_err={obj_quat_err:.4f}, success={success}"
            )

    if not all_dataframes:
        print("No valid trajectory files found!")
        return

    # Merge all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nCombined dataframe shape: {combined_df.shape}")

    # Save the complete combined dataframe with all information
    processed_dir_robot = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        hand_type=hand_type,
        task=task_name,
        data_id=0,
    )
    complete_output_dir = f"{processed_dir_robot}/../../"
    os.makedirs(complete_output_dir, exist_ok=True)
    complete_output_file = os.path.join(complete_output_dir, "complete_data.csv")
    combined_df.to_csv(complete_output_file, index=False)
    print(f"Complete data saved to: {complete_output_file}")

    # Calculate success rates by task and data_id (individual entries)
    # Since we want data_id in summary, we'll use the complete data but select key columns
    success_stats = combined_df[
        [
            "dataset",
            "robot_type",
            "hand_type",
            "data_type",
            "task",
            "data_id",
            "obj_pos_err",
            "obj_quat_err",
            "success",
            "pos_err_threshold",
            "quat_err_threshold",
        ]
    ].copy()

    # Add some computed columns for compatibility with existing code
    success_stats["total_attempts"] = 1  # Each row represents one attempt
    success_stats["successful_attempts"] = success_stats["success"].astype(int)
    success_stats["success_rate"] = success_stats["success"].astype(float)
    success_stats["avg_pos_err"] = success_stats["obj_pos_err"]
    success_stats["std_pos_err"] = 0.0  # No std for individual entries
    success_stats["avg_quat_err"] = success_stats["obj_quat_err"]
    success_stats["std_quat_err"] = 0.0  # No std for individual entries
    success_stats["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\nSuccess Rate Summary:")
    print("=" * 80)
    print(
        success_stats[
            [
                "task",
                "data_id",
                "success",
                "obj_pos_err",
                "obj_quat_err",
            ]
        ].to_string(index=False)
    )

    # Overall success rate
    overall_success_rate = combined_df["success"].mean()
    print(
        f"\nOverall Success Rate: {overall_success_rate:.4f} ({overall_success_rate*100:.2f}%)"
    )

    # Overall tracking error
    print(f"\nOverall Tracking Error:")
    # position error
    pos_err_mean = combined_df["obj_pos_err"].mean()
    pos_err_std = combined_df["obj_pos_err"].std()
    quat_err_mean = combined_df["obj_quat_err"].mean()
    quat_err_std = combined_df["obj_quat_err"].std()
    print(rf"Position Error: {pos_err_mean:.3f} \pm {pos_err_std:.3f}")
    print(rf"Orientation Error: {quat_err_mean:.3f} \pm {quat_err_std:.3f}")

    # Handle summary.csv - check if it exists and update accordingly
    summary_file = os.path.join(complete_output_dir, "summary.csv")
    if os.path.exists(summary_file):
        print(f"\nFound existing summary.csv, checking for updates...")
        try:
            existing_summary = pd.read_csv(summary_file)

            # Check if entries with same parameters already exist
            mask = (
                (existing_summary["dataset"] == dataset_name)
                & (existing_summary["robot_type"] == robot_type)
                & (existing_summary["hand_type"] == hand_type)
                & (existing_summary["pos_err_threshold"] == pos_err_threshold)
                & (existing_summary["quat_err_threshold"] == quat_err_threshold)
            )

            existing_entries_count = mask.sum()
            if existing_entries_count > 0:
                print(
                    f"Found {existing_entries_count} existing entries with the same parameters - updating them"
                )
                # Remove existing entries with same parameters to avoid duplicates
                existing_summary = existing_summary[~mask]
                action = "updated"
            else:
                print(
                    f"No existing entries found with the same parameters - adding new entries"
                )
                action = "added"

            # Append new results
            updated_summary = pd.concat(
                [existing_summary, success_stats], ignore_index=True
            )
            updated_summary.to_csv(summary_file, index=False)

            if action == "updated":
                print(
                    f"Successfully updated {existing_entries_count} existing entries and replaced with {len(success_stats)} new entries"
                )
            else:
                print(
                    f"Successfully added {len(success_stats)} new entries to existing summary"
                )

        except Exception as e:
            print(f"Error updating existing summary: {e}")
            print("Creating new summary file...")
            success_stats.to_csv(summary_file, index=False)
    else:
        print(f"\nCreating new summary.csv...")
        success_stats.to_csv(summary_file, index=False)

    print(f"Summary saved to: {summary_file}")

    # print failed tasks and their directories
    failed_tasks = combined_df[~combined_df["success"]]
    if len(failed_tasks) > 0:
        print(f"\nFailed tasks ({len(failed_tasks)} total):")
        for index, row in failed_tasks.iterrows():
            print(
                f"  - {row['task']}/{row['data_id']}: pos_err={row['obj_pos_err']:.4f}, quat_err={row['obj_quat_err']:.4f}"
            )

    return combined_df, success_stats


if __name__ == "__main__":
    tyro.cli(main)
