# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Define the viewers for the retargeting.

Author: Chaoyi Pan
Date: 2025-08-10
"""

import os
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
from spider.viewers import rerun_viewer as rerun_viewer
from spider.viewers import viser_viewer as viser_viewer


def setup_viewer(config: Config, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
    """Setup the viewer for the retargeting."""
    viewer_str = (config.viewer or "").lower()
    if not config.show_viewer:
        viewer_str = ""
    use_rerun = "rerun" in viewer_str
    use_viser = "viser" in viewer_str
    if use_rerun and use_viser:
        loguru.logger.warning("Both rerun and viser requested; defaulting to rerun.")
        use_viser = False

    if use_rerun:
        # setup rerun viewer
        rerun_viewer.init_rerun(app_name="spider", spawn=config.rerun_spawn)
        if config.save_rerun:
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            if os.path.exists("tmp/spider.rbl"):
                # Load blueprint from .rbl file
                # Note: Blueprint.from_file() doesn't exist in rerun 0.25.1
                # The .rbl file can be loaded as an archive, but extracting the blueprint
                # requires version 0.26.0+ features. For now, we'll load it and send it
                # via the bindings if possible.
                try:
                    import rerun_bindings as bindings

                    # Load the blueprint archive - this makes the blueprint available
                    # to the viewer when the .rrd file is opened
                    bindings.load_archive("tmp/spider.rbl")
                    loguru.logger.info("Loaded blueprint archive from tmp/spider.rbl")
                    # Note: The blueprint will be available when viewing the .rrd file
                    # but we can't easily convert it to a Blueprint object in this version
                except Exception as e:
                    loguru.logger.warning(
                        f"Failed to load blueprint from tmp/spider.rbl: {e}. "
                        "Continuing without custom blueprint."
                    )
            rr.save("tmp/rerun_data.rrd")
        if mj_model is not None and config.model_path is not None:
            # Check if scene is already built (from spec)
            if config.model_path == "hdmi_scene_from_spec":
                loguru.logger.info(
                    "Rerun scene already built from HDMI spec, skipping XML loading"
                )
            # check if python <= 3.8, if so, use log_scene_from_npz
            elif sys.version_info.major <= 3 and sys.version_info.minor <= 8:
                npz_path = Path(config.model_path).with_suffix(".npz")
                config.viewer_body_entity_and_ids = rerun_viewer.log_scene_from_npz(
                    npz_path
                )
                loguru.logger.warning(
                    "viewer is set to rerun, but python <= 3.8 is detected, load from npz file instead"
                )
            else:
                _, _, config.viewer_body_entity_and_ids = (
                    rerun_viewer.build_and_log_scene(Path(config.model_path))
                )
                loguru.logger.info(
                    "viewer is set to rerun, build and log scene from xml file"
                )
        else:
            loguru.logger.warning(
                "Rerun enabled but 3D scene not available (no model_path). Trajectory logging only."
            )

    if use_viser:
        viser_viewer.init_viser(app_name="spider")
        if config.save_rerun:
            loguru.logger.warning("save_rerun is set, but Viser does not save .rrd.")
        if mj_model is not None and config.model_path is not None:
            if config.model_path == "hdmi_scene_from_spec":
                loguru.logger.info(
                    "Viser scene already built from HDMI spec, skipping XML loading"
                )
            else:
                _, _, config.viewer_body_entity_and_ids = (
                    viser_viewer.build_and_log_scene(Path(config.model_path))
                )
                if mj_data is not None:  # the starting pose, until frames are logged
                    mujoco.mj_kinematics(mj_model, mj_data)
                    viser_viewer.update_frame(
                        mj_data, config.viewer_body_entity_and_ids, data_ref=mj_data
                    )
                loguru.logger.info(
                    "viewer is set to viser, build and log scene from xml file"
                )
        else:
            loguru.logger.warning(
                "Viser enabled but 3D scene not available (no model_path). Trajectory logging only."
            )
    # create mujoco viewer
    if "mujoco" in viewer_str:
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
    viewer_str = (config.viewer or "").lower()
    if not config.show_viewer:
        viewer_str = ""
    use_rerun = "rerun" in viewer_str
    use_viser = "viser" in viewer_str and not use_rerun

    # update mujoco scene if a viewer is provided
    if "mujoco" in viewer_str:
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
    if use_rerun:
        # Per-body transforms
        if mj_data is not None:
            rerun_viewer.log_frame(
                mj_data,
                sim_time=mj_data.time,
                viewer_body_entity_and_ids=config.viewer_body_entity_and_ids,
            )

        # Traces (any keys starting with 'trace_')
        if "trace_sample" in info:
            rerun_viewer.log_traces_from_info(
                info["trace_sample"], sim_time=mj_data.time
            )

        # Log scalar metrics (improvement, rew_max, rew_min, rew_median) as continuous time series
        if config.save_metrics:
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
                        plan_time = start_sim_time + (
                            it / config.max_num_iterations
                        ) * (end_sim_time - start_sim_time)
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

    # update viser scene
    if use_viser:
        if "trace_sample" in info:
            opt_steps = info.get("opt_steps")
            num_iters = int(opt_steps[0]) if opt_steps is not None else None
            viser_viewer.log_traces_from_info(
                info["trace_sample"],
                trace_ref=info.get("trace_ref"),
                trace_cost=info.get("trace_cost"),
                sim_time=mj_data.time,
                num_iters=num_iters,
            )


def log_frame(
    data: mujoco.MjData,
    sim_time: float,
    viewer_body_entity_and_ids: list,
    data_ref: mujoco.MjData | None = None,
) -> None:
    if not viewer_body_entity_and_ids:
        return
    first = viewer_body_entity_and_ids[0][0]
    if isinstance(first, str):
        rerun_viewer.log_frame(
            data,
            sim_time=sim_time,
            viewer_body_entity_and_ids=viewer_body_entity_and_ids,
        )
    else:
        viser_viewer.log_frame(
            data,
            sim_time=sim_time,
            viewer_body_entity_and_ids=viewer_body_entity_and_ids,
            data_ref=data_ref,
        )


def setup_renderer(config: Config, mj_model: mujoco.MjModel):
    mj_model.vis.global_.offwidth = 720
    mj_model.vis.global_.offheight = 480
    renderer = (
        mujoco.Renderer(mj_model, height=480, width=720) if config.save_video else None
    )
    return renderer


# Colour of the reference motion drawn over the simulated one.
REFERENCE_RGBA = (0.10, 0.85, 0.35, 0.32)


def render_image(
    config: Config,
    renderer: mujoco.Renderer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    mj_data_ref: mujoco.MjData,
    camera: mujoco.MjvCamera | str | int | None = None,
):
    """Render the simulated state with the reference drawn over it as a green ghost.

    Args:
        config: Unused; kept for the callers' common signature.
        renderer: Offscreen renderer of ``mj_model``.
        mj_model: The scene.
        mj_data: The simulated state (solid).
        mj_data_ref: The reference state (translucent green).
        camera: Camera to render from. None uses the model's ``front`` camera,
            its first camera, or a free camera on the whole scene.

    Returns:
        The image, (height, width, 3) uint8.
    """
    del config
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    if camera is None:
        front = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        if front >= 0 or mj_model.ncam > 0:
            camera = max(front, 0)
        else:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(mj_model, camera)

    mujoco.mj_forward(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data_ref)
    renderer.update_scene(mj_data, camera, options)
    scene = renderer.scene
    first = scene.ngeom
    mujoco.mjv_addGeoms(
        mj_model,
        mj_data_ref,
        mujoco.MjvOption(),
        mujoco.MjvPerturb(),
        int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
        scene,
    )
    for g in range(first, scene.ngeom):
        geom = scene.geoms[g]
        rgba = REFERENCE_RGBA
        if (
            geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
            and 0 <= geom.objid < mj_model.ngeom
        ):
            body = mj_model.geom_bodyid[geom.objid]
            # A body exactly where the reference has it (a fixture) needs no ghost.
            if np.array_equal(mj_data.xpos[body], mj_data_ref.xpos[body]) and (
                np.array_equal(mj_data.xquat[body], mj_data_ref.xquat[body])
            ):
                rgba = (0.0, 0.0, 0.0, 0.0)
        geom.rgba[:] = rgba
    image = renderer.render()
    # White on a black box: readable on a light floor and on a dark background.
    text, font = "solid: sim   green: ref", cv2.FONT_HERSHEY_SIMPLEX
    x, y = 10, image.shape[0] - 12
    (width, height), baseline = cv2.getTextSize(text, font, 0.5, 1)
    cv2.rectangle(
        image, (x - 5, y - height - 5), (x + width + 5, y + baseline + 3), (0, 0, 0), -1
    )
    cv2.putText(image, text, (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image
