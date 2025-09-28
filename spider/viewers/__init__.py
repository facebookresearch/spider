"""Define the viewers for the retargeting.

Author: Chaoyi Pan
Date: 2025-08-10
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import loguru
import mujoco
import mujoco.viewer
import numpy as np

from spider.config import Config
from spider.viewers.rerun_viewer import (
    build_and_log_scene,
    init_rerun,
    log_frame,
    log_reward_samples_by_iter,
    log_scene_from_npz,
    log_stage_improvements,
    log_traces_from_info,
)


def setup_viewer(config: Config, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
    """Setup the viewer for the retargeting."""
    if "rerun" in config.viewer:
        # setup rerun viewer
        init_rerun(app_name="retarget", spawn=config.rerun_spawn)
        if mj_model is not None:
            # check if python <= 3.8, if so, use log_scene_from_npz, this is a temporary fix for python 3.8 required by isaacgym
            if sys.version_info.major <= 3 and sys.version_info.minor <= 8:
                npz_path = Path(config.model_path).with_suffix(".npz")
                config.viewer_body_entity_and_ids = log_scene_from_npz(npz_path)
                loguru.logger.warning(
                    "viewer is set to rerun, but python <= 3.8 is detected, load from npz file instead"
                )
            else:
                _, _, config.viewer_body_entity_and_ids = build_and_log_scene(
                    Path(config.model_path)
                )
                loguru.logger.info(
                    "viewer is set to rerun, build and log scene from xml file"
                )
        else:
            loguru.logger.warning(
                "viewer is set to rerun, but mj_model is not provided"
            )
    # create mujoco viewer
    if "mujoco" in config.viewer:
        run_viewer = lambda: mujoco.viewer.launch_passive(mj_model, mj_data)
        loguru.logger.info("viewer is set to mujoco, launch passive viewer")
    else:

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {"is_running": lambda: True, "sync": lambda: None, "user_scn": None},
            )

        loguru.logger.info("viewer is disabled, launch dummy viewer")

    return run_viewer


# define logging function
def update_viewer(
    config: Config,
    viewer: mujoco.viewer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    mj_data_ref: mujoco.MjData,
    info: dict,
):
    # update mujoco scene if a viewer is provided
    if "mujoco" in config.viewer:
        mujoco.mj_kinematics(mj_model, mj_data)
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
        mujoco.mj_forward(mj_model, mj_data_ref)
        mujoco.mjv_updateScene(
            mj_model,
            mj_data_ref,
            vopt,
            pert,
            viewer.cam,
            catmask,
            getattr(viewer, "user_scn", None),
        )
        if hasattr(viewer, "sync"):
            viewer.sync()

    # update rerun scene
    if "rerun" in config.viewer:
        if "sim_step" not in info:
            frame_idx = int(mj_data.time / config.sim_dt)
        else:
            frame_idx = info["sim_step"]
        # Per-body transforms
        if mj_data is not None:
            log_frame(
                mj_data,
                frame_idx=frame_idx,
                time_seconds=mj_data.time,
                viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
            )

        # Traces (any keys starting with 'trace_')
        if "trace_sample" in info:
            log_traces_from_info(info["trace_sample"], plan_step=frame_idx)

        # Improvements curve
        if "improvement" in info:
            log_stage_improvements(info["improvement"], plan_step=frame_idx)

        # Rewards per iteration if provided as a mapping {iter_index: rewards}
        if "rew_sample" in info:
            I = int(info["rew_sample"].shape[0])
            for it in range(I):
                vals = info["rew_sample"][it]
                log_reward_samples_by_iter(vals, iter_index=it, plan_step=frame_idx)
    return


def setup_renderer(config: Config, mj_model: mujoco.MjModel):
    mj_model.vis.global_.offwidth = 720
    mj_model.vis.global_.offheight = 480
    renderer = (
        mujoco.Renderer(mj_model, height=480, width=720) if config.save_video else None
    )
    return renderer


def render_image(
    config: Config,
    renderer: mujoco.Renderer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    mj_data_ref: mujoco.MjData,
):
    # render sim
    mujoco.mj_kinematics(mj_model, mj_data)
    renderer.update_scene(mj_data, "front")
    sim_image = renderer.render()
    # add text named "sim"
    cv2.putText(
        sim_image,
        "sim",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    # render ref
    mujoco.mj_forward(mj_model, mj_data_ref)
    renderer.update_scene(mj_data_ref, "front")
    ref_image = renderer.render()
    # add text named "ref"
    cv2.putText(
        ref_image,
        "ref",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    image = np.concatenate([ref_image, sim_image], axis=1)
    return image
