"""
Read get success rate for a batch of tasks by calling get_success_rate.py's main function, save all results in a csv file.

Input:
- dataset_dir: str, the directory of the dataset
- dataset_name_list: list[str], the name of the dataset
- robot_type_list: list[str], the type of the robot
- hand_type_list: list[str], the type of the hand
- data_type_list: list[str], the type of the data
- pos_err_threshold: float, the threshold of the position error
- quat_err_threshold: float, the threshold of the quaternion error

Output:
- success_rate_df: pd.DataFrame, the success rate of the tasks with tracking errors
"""

import tyro
import pandas as pd
import os
from itertools import product
from retarget.postprocess.get_success_rate import main as get_success_rate_main


def main(
    dataset_dir: str = "../../example_datasets",
    dataset_name_list: list[str] = ["oakink", "gigahand"],
    robot_type_list: list[str] = ["allegro", "ability", "inspire", "schunk", "xhand"],
    hand_type_list: list[str] = ["bimanual"],
    data_type_list: list[str] = ["mjwpeq"],
    pos_err_threshold: float = 0.1,
    quat_err_threshold: float = 0.5,
):
    """
    Run get_success_rate for multiple parameter combinations and combine results.

    Args:
        dataset_dir: Directory containing the datasets
        dataset_name_list: List of dataset names to process
        robot_type_list: List of robot types to process
        hand_type_list: List of hand types to process
        data_type_list: List of data types to process
        pos_err_threshold: Position error threshold for success determination
        quat_err_threshold: Quaternion error threshold for success determination

    Returns:
        Combined DataFrame with all results
    """
    # Resolve dataset directory path
    dataset_dir = os.path.abspath(dataset_dir)

    # Generate all combinations of parameters
    param_combinations = list(
        product(dataset_name_list, robot_type_list, hand_type_list, data_type_list)
    )

    print(f"Processing {len(param_combinations)} parameter combinations...")

    all_summary_dataframes = []
    all_complete_dataframes = []

    # Process each combination
    for i, (dataset_name, robot_type, hand_type, data_type) in enumerate(
        param_combinations
    ):
        print(f"\n{'='*80}")
        print(f"Processing combination {i+1}/{len(param_combinations)}:")
        print(f"  Dataset: {dataset_name}")
        print(f"  Robot Type: {robot_type}")
        print(f"  Hand Type: {hand_type}")
        print(f"  Data Type: {data_type}")
        print(f"{'='*80}")

        try:
            # Call the main function from get_success_rate.py
            complete_df, summary_df = get_success_rate_main(
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                robot_type=robot_type,
                hand_type=hand_type,
                data_type=data_type,
                pos_err_threshold=pos_err_threshold,
                quat_err_threshold=quat_err_threshold,
            )

            # Note: data_type and other metadata are now included by get_success_rate.main()

            # Collect the results
            all_complete_dataframes.append(complete_df)
            all_summary_dataframes.append(summary_df)

            print(
                f"✓ Successfully processed {dataset_name}/{robot_type}/{hand_type}/{data_type}"
            )

        except Exception as e:
            print(
                f"✗ Error processing {dataset_name}/{robot_type}/{hand_type}/{data_type}: {e}"
            )
            continue

    if not all_summary_dataframes:
        print("\nNo successful results to combine!")
        return None

    # Combine all results
    print(f"\n{'='*80}")
    print("Combining results...")

    combined_complete_df = pd.concat(all_complete_dataframes, ignore_index=True)
    combined_summary_df = pd.concat(all_summary_dataframes, ignore_index=True)

    print(f"Combined complete data shape: {combined_complete_df.shape}")
    print(f"Combined summary data shape: {combined_summary_df.shape}")

    # Save combined results to dataset_dir
    output_dir = os.path.join(dataset_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Save combined summary data
    summary_output_file = os.path.join(output_dir, f"summary.csv")
    combined_summary_df.to_csv(summary_output_file, index=False)
    print(f"Combined summary data saved to: {summary_output_file}")

    # Print overall statistics
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")

    print(f"Total parameter combinations processed: {len(all_summary_dataframes)}")
    print(f"Total tasks analyzed: {len(combined_complete_df)}")

    # Overall success rate across all combinations
    overall_success_rate = combined_complete_df["success"].mean()
    print(
        f"Overall success rate across all combinations: {overall_success_rate:.4f} ({overall_success_rate*100:.2f}%)"
    )

    # Success rate by dataset
    if len(dataset_name_list) > 1:
        print(f"\nSuccess rate by dataset:")
        dataset_stats = (
            combined_complete_df.groupby("dataset")["success"]
            .agg(["count", "mean"])
            .round(4)
        )
        dataset_stats.columns = ["total_tasks", "success_rate"]
        print(dataset_stats.to_string())

    # Success rate by robot type
    if len(robot_type_list) > 1:
        print(f"\nSuccess rate by robot type:")
        robot_stats = (
            combined_complete_df.groupby("robot_type")["success"]
            .agg(["count", "mean"])
            .round(4)
        )
        robot_stats.columns = ["total_tasks", "success_rate"]
        print(robot_stats.to_string())

    # Success rate by hand type
    if len(hand_type_list) > 1:
        print(f"\nSuccess rate by hand type:")
        hand_stats = (
            combined_complete_df.groupby("hand_type")["success"]
            .agg(["count", "mean"])
            .round(4)
        )
        hand_stats.columns = ["total_tasks", "success_rate"]
        print(hand_stats.to_string())

    # Success rate by data type
    if len(data_type_list) > 1:
        print(f"\nSuccess rate by data type:")
        data_type_stats = (
            combined_complete_df.groupby("data_type")["success"]
            .agg(["count", "mean"])
            .round(4)
        )
        data_type_stats.columns = ["total_tasks", "success_rate"]
        print(data_type_stats.to_string())

    # Overall tracking errors
    pos_err_mean = combined_complete_df["obj_pos_err"].mean()
    pos_err_std = combined_complete_df["obj_pos_err"].std()
    quat_err_mean = combined_complete_df["obj_quat_err"].mean()
    quat_err_std = combined_complete_df["obj_quat_err"].std()

    print(f"\nOverall tracking errors across all combinations:")
    print(f"Position Error: {pos_err_mean:.3f} ± {pos_err_std:.3f}")
    print(f"Orientation Error: {quat_err_mean:.3f} ± {quat_err_std:.3f}")

    print(f"\n{'='*80}")
    print("Batch processing completed successfully!")
    print(f"{'='*80}")

    return combined_complete_df, combined_summary_df


if __name__ == "__main__":
    tyro.cli(main)
