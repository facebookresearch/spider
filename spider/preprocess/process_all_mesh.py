"""A standalone script to process all meshes in the assets/objects folder in parallel.
Author: Chaoyi Pan
Date: 2025-07-15
"""

import glob
import os
import traceback
from multiprocessing import Pool, cpu_count

import loguru
import tyro

# add ../ to path
from decompose import main as decompose_main


def get_available_tasks_oakink(embodiment_type: str):
    data_dir = "../../datasets/raw/oakink/"
    # list all pkl files in data_dir
    pkl_files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
    # filter out files that don't contain embodiment_type
    pkl_files = [f for f in pkl_files if embodiment_type in f]
    # filter out files whose npz file already exists
    loguru.logger.info(f"Found {len(pkl_files)} pkl files to process")
    # get task names
    task_names = [f.split("_")[0] for f in pkl_files]
    return task_names


def get_available_tasks_hot3d(embodiment_type: str):
    data_dir = f"../../datasets/raw/hot3d/{embodiment_type}"
    # list all folders in data_dir
    task_names = [
        f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))
    ]
    return task_names


def process_single_task(args):
    """Worker function to process a single task."""
    embodiment_type, task = args
    try:
        loguru.logger.info(
            f"Starting processing task: {task} with embodiment_type: {embodiment_type}"
        )
        decompose_main(embodiment_type=embodiment_type, task=task)
        loguru.logger.info(f"Successfully processed task: {task}")
    except Exception as e:
        error_msg = f"Error processing task {task}: {str(e)}\n{traceback.format_exc()}"
        loguru.logger.error(error_msg)
        raise


def main(
    dataset: str = "oakink",
    embodiment_type: str = "bimanual",
    num_workers: int = 24,
):
    if embodiment_type == "bimanual":
        hands = ["right", "left"]
    else:
        hands = [embodiment_type]

    if dataset == "oakink":
        task_names = get_available_tasks_oakink(embodiment_type)
    elif dataset == "hot3d":
        task_names = get_available_tasks_hot3d(embodiment_type)
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    processed_tasks = []
    unprocessed_tasks = []

    for i, task in enumerate(task_names):
        processed = True
        for hand in hands:
            output_dir = f"../assets/objects/{hand}_{task}"
            input_files = glob.glob(f"{output_dir}/*_{hand}.obj")
            output_file = f"{output_dir}/0.obj"
            if input_files and not os.path.exists(output_file):
                processed = False
                break

        if processed:
            processed_tasks.append(task)
        else:
            unprocessed_tasks.append(task)

    loguru.logger.info(f"Processed {len(processed_tasks)} tasks")
    loguru.logger.info(f"Unprocessed {len(unprocessed_tasks)} tasks")

    if len(unprocessed_tasks) == 0:
        loguru.logger.info("No tasks to process. Exiting.")
        return

    # Set number of workers if not specified
    if num_workers is None:
        num_workers = min(cpu_count(), len(unprocessed_tasks))
    else:
        num_workers = min(num_workers, len(unprocessed_tasks))

    loguru.logger.info(f"Using {num_workers} workers for parallel processing")

    # Prepare arguments for parallel processing
    task_args = [(embodiment_type, task) for task in unprocessed_tasks]

    # Process tasks in parallel
    with Pool(processes=num_workers) as pool:
        try:
            # Process all tasks
            pool.map(process_single_task, task_args)
            loguru.logger.info(
                f"Processing complete! All {len(unprocessed_tasks)} tasks processed."
            )
        except Exception as e:
            loguru.logger.error(f"Error during parallel processing: {e}")


if __name__ == "__main__":
    tyro.cli(main)
