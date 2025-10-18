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
import rerun as rr

from spider.config import Config
from spider.viewers.rerun_viewer import (
    build_and_log_scene,
    build_and_log_scene_from_spec,
    init_rerun,
    log_frame,
    # log_reward_samples_by_iter,
    log_scene_from_npz,
    # log_stage_improvements,
    log_traces_from_info,
    # RerunRealtimeUpdater,
)


def setup_viewer(config: Config, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
    """Setup the viewer for the retargeting."""
    if "rerun" in config.viewer:
        # setup rerun viewer
        init_rerun(app_name="spider", spawn=config.rerun_spawn)
        if mj_model is not None and config.model_path is not None:
            # Check if scene is already built (from spec)
            if config.model_path == "hdmi_scene_from_spec":
                loguru.logger.info(
                    "Rerun scene already built from HDMI spec, skipping XML loading"
                )
            # check if python <= 3.8, if so, use log_scene_from_npz
            elif sys.version_info.major <= 3 and sys.version_info.minor <= 8:
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
                "Rerun enabled but 3D scene not available (no model_path). Trajectory logging only."
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
        # Per-body transforms
        if mj_data is not None:
            log_frame(
                mj_data,
                sim_time=mj_data.time,
                viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
            )

        # Traces (any keys starting with 'trace_')
        if "trace_sample" in info:
            log_traces_from_info(info["trace_sample"], sim_time=mj_data.time)

        # Log scalar metrics (improvement, rew_max, rew_min, rew_median) as continuous time series
        for k, v in info.items():
            # skip trace related metrics
            if "trace" in k:
                continue
            if not isinstance(v, np.ndarray):
                continue
            # show scalar metrics
            if v.shape == (config.max_num_iterations,):
                # Extract metric base name (e.g., "rew" from "rew_max")
                metric_base = k.rsplit("_", 1)[0] if "_" in k else k

                for it in range(config.max_num_iterations):
                    start_sim_time = mj_data.time
                    end_sim_time = mj_data.time + config.ctrl_dt
                    plan_time = start_sim_time + (it / config.max_num_iterations) * (
                        end_sim_time - start_sim_time
                    )
                    rr.set_time("sim_time", timestamp=plan_time)
                    rr.log(f"metrics/{metric_base}/{k}", rr.Scalars([float(v[it])]))
            # show state metrics
            if v.shape[0] == config.ctrl_steps and v.ndim == 2:
                rr.set_time("sim_time", timestamp=mj_data.time)
                for dim_idx in range(v.shape[1]):
                    rr.log(
                        f"metrics/{k}/dim_{dim_idx}",
                        rr.Scalars([float(v[-1, dim_idx])]),
                    )


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
    try:
        renderer.update_scene(mj_data, "front")
    except Exception:
        renderer.update_scene(mj_data, 0)
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
    try:
        renderer.update_scene(mj_data_ref, "front")
    except Exception:
        renderer.update_scene(mj_data_ref, 0)
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
