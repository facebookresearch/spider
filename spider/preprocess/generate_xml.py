import json
import os

import loguru
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from spider import ROOT
from spider.io import get_processed_data_dir


def main(
    dataset_dir: str = f"{ROOT}/../example_datasets",
    dataset_name: str = "oakink",
    robot_type: str = "allegro",
    embodiment_type: str = "bimanual",
    task: str = "pick_spoon_bowl",
    data_id: int = 0,
    hand_floor_collision: bool = False,
    object_density: float = 3000,
    object_armature: float = 0.001,
    object_frictionloss: float = 0.01,
    show_viewer: bool = True,
):
    dataset_dir = os.path.abspath(dataset_dir)
    processed_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir, exist_ok=True)

    # choose robot XML based on embodiment_type
    robots_assets_dir = (
        f"{dataset_dir}/processed/{dataset_name}/assets/robots/{robot_type}"
    )
    robot_xml_name = (
        "bimanual.xml" if embodiment_type == "bimanual" else f"{embodiment_type}.xml"
    )
    robot_xml_path = f"{robots_assets_dir}/{robot_xml_name}"
    if not os.path.exists(robot_xml_path):
        raise FileNotFoundError(f"Robot XML not found: {robot_xml_path}")

    # load robot xml as base scene
    mj_spec = mujoco.MjSpec.from_file(robot_xml_path)

    # Configure compiler for robust relative mesh loading
    assets_root_dir = f"{dataset_dir}/processed/{dataset_name}/assets"
    # get relative dir of assets_root_dir and scene.xml
    assets_root_dir_rel = os.path.relpath(assets_root_dir, f"{processed_dir}/..")
    # Set meshdir relative to where scene.xml will be saved
    original_meshdir = mj_spec.meshdir
    mj_spec.meshdir = assets_root_dir_rel

    # Rewrite robot mesh file paths to be relative to the assets directory
    robots_dir_abs = f"{assets_root_dir}/robots/{robot_type}"
    for mesh in getattr(mj_spec, "meshes", []):
        original = mesh.file
        # Determine absolute location of the original reference
        if os.path.isabs(original):
            candidate_abs = original
        else:
            # Prefer resolving relative to the robot's assets directory
            candidate_abs = os.path.normpath(
                os.path.join(robots_dir_abs, original_meshdir, original)
            )
        # Compute path relative to assets root
        try:
            file_rel_to_assets = os.path.relpath(candidate_abs, assets_root_dir)
        except ValueError:
            # Fallback: keep original if relpath fails (e.g., different drives)
            file_rel_to_assets = original
        mesh.file = file_rel_to_assets

    # load contact info for placing contact sites (optional)
    keypoint_data_dir = get_processed_data_dir(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    contact_npz_path = f"{keypoint_data_dir}/trajectory_keypoints.npz"
    loaded_data = np.load(contact_npz_path)
    try:
        contact_left = loaded_data["contact_left"]
        contact_pos_left = loaded_data["contact_pos_left"]
        contact_right = loaded_data["contact_right"]
        contact_pos_right = loaded_data["contact_pos_right"]
    except KeyError:
        loguru.logger.warning(
            f"No contact data found at {contact_npz_path}; falling back to zeros"
        )
        contact_left = np.zeros((10, 5))
        contact_pos_left = np.zeros((5, 3))
        contact_right = np.zeros((10, 5))
        contact_pos_right = np.zeros((5, 3))
    finger_names = [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ]
    if robot_type in ["allegro", "metahand"]:
        finger_names = finger_names[:4]

    # add assets
    mj_spec.add_texture(
        name="skybox",
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0, 0, 0],
        width=512,
        height=3072,
    )
    mj_spec.add_texture(
        name="groundplane",
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    mj_spec.add_material(
        name="groundplane",
        textures=["groundplane"],
        texuniform=True,
        texrepeat=[5, 5],
        reflectance=0.2,
    )

    # add floor
    if embodiment_type in ["right", "bimanual"]:
        material_name = "right_groundplane"
    else:
        material_name = "left_groundplane"
    mj_spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, 0.0],
        material=material_name,
    )

    # Load object convex meshes from task_info.json
    task_info_path = f"{keypoint_data_dir}/../task_info.json"
    task_info = {}
    with open(task_info_path) as f:
        task_info = json.load(f)
    right_convex_dir = task_info.get("right_object_convex_dir")
    left_convex_dir = task_info.get("left_object_convex_dir")
    right_mesh_dir = task_info.get("right_object_mesh_dir")
    left_mesh_dir = task_info.get("left_object_mesh_dir")

    # Right object meshes
    if (
        embodiment_type in ["right", "bimanual"]
        and right_convex_dir
        and os.path.isdir(right_convex_dir)
    ):
        right_object_files = sorted(
            [f for f in os.listdir(right_convex_dir) if f.endswith(".obj")]
        )
        for f in right_object_files:
            suffix = f.split(".")[0]
            file_abs = f"{right_convex_dir}/{f}"
            file_rel_to_meshdir = os.path.relpath(file_abs, assets_root_dir)
            mj_spec.add_mesh(name=f"right_{suffix}", file=file_rel_to_meshdir)
    else:
        right_object_files = []

    # Left object meshes
    if (
        embodiment_type in ["left", "bimanual"]
        and left_convex_dir
        and os.path.isdir(left_convex_dir)
    ):
        left_object_files = sorted(
            [f for f in os.listdir(left_convex_dir) if f.endswith(".obj")]
        )
        for f in left_object_files:
            suffix = f.split(".")[0]
            file_abs = f"{left_convex_dir}/{f}"
            file_rel_to_meshdir = os.path.relpath(file_abs, assets_root_dir)
            mj_spec.add_mesh(name=f"left_{suffix}", file=file_rel_to_meshdir)
    else:
        left_object_files = []

    # Visual meshes (non-colliding)
    right_visual_file = f"{right_mesh_dir}/visual.obj" if right_mesh_dir else None
    left_visual_file = f"{left_mesh_dir}/visual.obj" if left_mesh_dir else None
    if (
        embodiment_type in ["right", "bimanual"]
        and right_visual_file
        and os.path.exists(right_visual_file)
    ):
        file_rel_to_meshdir = os.path.relpath(right_visual_file, assets_root_dir)
        mj_spec.add_mesh(name="right_visual", file=file_rel_to_meshdir)
    if (
        embodiment_type in ["left", "bimanual"]
        and left_visual_file
        and os.path.exists(left_visual_file)
    ):
        file_rel_to_meshdir = os.path.relpath(left_visual_file, assets_root_dir)
        mj_spec.add_mesh(name="left_visual", file=file_rel_to_meshdir)

    # add object to model
    right_object_collision_names = []
    if embodiment_type in ["right", "bimanual"]:
        right_object_handle = mj_spec.worldbody.add_body(
            name="right_object",
            mocap=False,
        )
        right_object_handle.add_joint(
            name="right_object_joint",
            type=mujoco.mjtJoint.mjJNT_FREE,
            armature=object_armature,
            frictionloss=object_frictionloss,
        )
        # add geom to object
        for obj_file in right_object_files:
            suffix = obj_file.split(".")[0]
            if suffix.isdigit():
                rgba = [0, 1, 0, 1]
                density = object_density
                right_object_collision_names.append(f"right_object_{suffix}")
                group = 3
            else:
                rgba = [1, 1, 1, 1]
                density = 0
                group = 0
            right_object_handle.add_geom(
                name=f"right_object_{suffix}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"right_{suffix}",
                pos=[0, 0, 0],
                conaffinity=0,
                contype=0,
                rgba=rgba,
                density=density,
                group=group,
            )
        # add site to object
        right_object_handle.add_site(
            name="right_object",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.02, 0.03],
            pos=[0, 0, 0],
            rgba=[1, 0, 0, 1],
            group=3,
        )
        # add visual mesh (non-colliding)
        if "right_visual" in [m.name for m in mj_spec.meshes]:
            right_object_handle.add_geom(
                name="right_object_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="right_visual",
                pos=[0, 0, 0],
                conaffinity=0,
                contype=0,
                rgba=[1, 1, 1, 1],
                density=0,
                group=0,
            )
        # add trace site to the object (for visualization)
        right_object_handle.add_site(
            name="trace_right_object",
            pos=[0, 0, 0],
            size=[0.01, 0.01, 0.01],
            rgba=[0, 1, 0, 1],
            group=4,
        )
        # add contact site to the object (for virtual constraint)
        for i, finger_name in enumerate(finger_names):
            right_object_handle.add_site(
                name=f"track_object_right_{finger_name}",
                pos=contact_pos_right[i],
                size=[0.01, 0.01, 0.01],
                rgba=[0, 1, 0, 1],
                group=4,
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_object_right_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_object_right_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_hand_right_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_hand_right_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )

    left_object_collision_names = []
    if embodiment_type in ["left", "bimanual"]:
        left_object_handle = mj_spec.worldbody.add_body(
            name="left_object",
            mocap=False,
            gravcomp=(
                1 if len(left_object_files) == 0 else 0
            ),  # if left object is not present, set gravcomp to 1 to avoid gravity
        )
        left_joint_handle = left_object_handle.add_joint(
            name="left_object_joint",
            type=mujoco.mjtJoint.mjJNT_FREE,
            armature=object_armature,
            frictionloss=object_frictionloss,
        )
        # add geom to object
        for obj_file in left_object_files:
            suffix = obj_file.split(".")[0]
            if suffix.isdigit():
                rgba = [0, 1, 0, 1]
                density = object_density
                left_object_collision_names.append(f"left_object_{suffix}")
                group = 3
            else:
                rgba = [1, 1, 1, 1]
                density = 0
                group = 0
            left_object_handle.add_geom(
                name=f"left_object_{suffix}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"left_{suffix}",
                pos=[0, 0, 0],
                conaffinity=0,
                contype=0,
                rgba=rgba,
                density=density,
                group=group,
            )
        # add mass to object if there is no left object
        if len(left_object_files) == 0:
            left_object_handle.add_geom(
                name="left_object_mass",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[0.5, 0.5, 0.5],  # put it far away to avoid collision
                size=[0.1, 0.1, 0.1],
                density=10,
                group=3,
            )
            left_joint_handle.frictionloss = 1.0
            left_joint_handle.armature = 1.0
        # add site to object
        left_object_handle.add_site(
            name="left_object",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.02, 0.03],
            pos=[0, 0, 0],
            rgba=[1, 0, 0, 1],
            group=3,
        )
        # add visual mesh (non-colliding)
        if "left_visual" in [m.name for m in mj_spec.meshes]:
            left_object_handle.add_geom(
                name="left_object_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="left_visual",
                pos=[0, 0, 0],
                conaffinity=0,
                contype=0,
                rgba=[1, 1, 1, 1],
                density=0,
                group=0,
            )
        # add trace site to the object
        left_object_handle.add_site(
            name="trace_left_object",
            pos=[0, 0, 0],
            size=[0.01, 0.01, 0.01],
            rgba=[0, 1, 0, 1],
            group=4,
        )
        # add contact site to the object
        for i, finger_name in enumerate(finger_names):
            # if left object is not present, add contact site to the right object
            if len(left_object_files) == 0:
                handle = right_object_handle
            else:
                handle = left_object_handle
            handle.add_site(
                name=f"track_object_left_{finger_name}",
                pos=contact_pos_left[i],
                size=[0.01, 0.01, 0.01],
                rgba=[0, 1, 0, 1],
                group=4,
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_object_left_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_object_left_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_hand_left_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_hand_left_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )

    object_collision_names = right_object_collision_names + left_object_collision_names
    loguru.logger.info(f"Added {len(object_collision_names)} objects to model")

    # add contact pairs
    default_solref = [0.02, 1]
    default_friction = [1.0, 1.0, 0.1, 0.0, 0.0]
    small_friction = [0.01, 0.01, 0.0001, 0.0, 0.0]
    # [thumb, index intermediate, index, middle, ring, pinky + floor] <-> object
    hand_collision_names = []
    for geom_id in range(len(mj_spec.geoms)):
        geom = mj_spec.geoms[geom_id]
        if geom.name.startswith("collision_hand_"):
            hand_collision_names.append(geom.name)
    # if embodiment_type in ["right", "bimanual"]:
    #     hand_collision_names.extend(
    #         [
    #             "right_palm_collision",
    #             "right_thumb_collision",
    #             "right_thumb_distal_collision",
    #             "right_thumb_intermediate_collision",
    #             "right_thumb_proximal_collision",
    #             "right_index_collision",
    #             "right_index_intermediate_collision",
    #             "right_index_proximal_collision",
    #             "right_middle_collision",
    #             "right_middle_intermediate_collision",
    #             "right_middle_proximal_collision",
    #             "right_ring_collision",
    #             "right_ring_intermediate_collision",
    #             "right_ring_proximal_collision",
    #             "right_pinky_collision",
    #         ]
    #     )
    # if embodiment_type in ["left", "bimanual"]:
    #     hand_collision_names.extend(
    #         [
    #             "left_palm_collision",
    #             "left_thumb_collision",
    #             "left_thumb_distal_collision",
    #             "left_thumb_intermediate_collision",
    #             "left_thumb_proximal_collision",
    #             "left_index_collision",
    #             "left_index_intermediate_collision",
    #             "left_index_proximal_collision",
    #             "left_middle_collision",
    #             "left_middle_intermediate_collision",
    #             "left_middle_proximal_collision",
    #             "left_ring_collision",
    #             "left_ring_intermediate_collision",
    #             "left_ring_base_collision",
    #             "left_pinky_collision",
    #         ]
    #     )
    hand_collision_names.append("floor")

    object_names = []
    if embodiment_type in ["left", "bimanual"]:
        object_names.append("left_object")
    if embodiment_type in ["right", "bimanual"]:
        object_names.append("right_object")

    # if robot_type in ["allegro", "metahand"]:
    #     # remove pinky collision
    #     if embodiment_type == "right":
    #         hand_collision_names.remove("right_pinky_collision")
    #     elif embodiment_type == "left":
    #         hand_collision_names.remove("left_pinky_collision")
    #     elif embodiment_type == "bimanual":
    #         hand_collision_names.remove("right_pinky_collision")
    #         hand_collision_names.remove("left_pinky_collision")
    #     else:
    #         raise ValueError(f"Invalid hand type: {embodiment_type}")
    contact_cnt = 0

    # hand <-> object collision
    for object_collision_name in object_collision_names:
        for hand_collision_name in hand_collision_names:
            if "thumb" in hand_collision_name or "index" in hand_collision_name:
                condim = 4
            else:
                condim = 3
            # if "left" in hand_collision_name and "right" in object_collision_name:
            #     friction = small_friction
            # elif "right" in hand_collision_name and "left" in object_collision_name:
            #     friction = small_friction
            # else:
            friction = default_friction
            mj_spec.add_pair(
                name=f"{hand_collision_name}_{object_collision_name}",
                geomname1=hand_collision_name,
                geomname2=object_collision_name,
                solref=default_solref,
                friction=friction,
                condim=condim,
            )
            contact_cnt += 1

    # object <-> object collision
    if (
        embodiment_type == "bimanual"
        and len(right_object_collision_names) > 0
        and len(left_object_collision_names) > 0
    ):
        for right_object_collision_name in right_object_collision_names:
            for left_object_collision_name in left_object_collision_names:
                mj_spec.add_pair(
                    name=f"{right_object_collision_name}_{left_object_collision_name}",
                    geomname1=right_object_collision_name,
                    geomname2=left_object_collision_name,
                    solref=default_solref,
                    friction=small_friction,
                    condim=3,
                )
    # hand <-> floor collision
    if hand_floor_collision:
        for hand_collision_name in hand_collision_names:
            mj_spec.add_pair(
                name=f"{hand_collision_name}_floor",
                geomname1=hand_collision_name,
                geomname2="floor",
                solref=default_solref,
                friction=default_friction,
                condim=3,
            )
            contact_cnt += 1
    # hand self collision
    # thumb <-> index intermediate, index
    hand_collision_pairs = []
    for collision_name in hand_collision_names:
        if "0" in collision_name:  # currently only add finger tip collision
            # hand <-> hand collision
            hand_side = collision_name.split("_")[2]
            if embodiment_type == "bimanual":
                another_hand_side = "right" if hand_side == "left" else "left"
                for another_collision_name in hand_collision_names:
                    if (
                        another_hand_side in another_collision_name
                        and another_collision_name != collision_name
                        and (
                            "0" in another_collision_name
                            or "1" in another_collision_name
                        )
                        and (collision_name, another_collision_name)
                        not in hand_collision_pairs
                        and (another_collision_name, collision_name)
                        not in hand_collision_pairs
                    ):
                        mj_spec.add_pair(
                            name=f"{collision_name}_{another_collision_name}",
                            geomname1=collision_name,
                            geomname2=another_collision_name,
                            solref=default_solref,
                            friction=default_friction,
                            condim=3,
                        )
                        hand_collision_pairs.append(
                            (collision_name, another_collision_name)
                        )
                        contact_cnt += 1
            # hand self collision
            for another_collision_name in hand_collision_names:
                if (
                    hand_side in another_collision_name
                    and another_collision_name != collision_name
                    and "0" in another_collision_name
                    and (collision_name, another_collision_name)
                    not in hand_collision_pairs
                    and (another_collision_name, collision_name)
                    not in hand_collision_pairs
                ):
                    mj_spec.add_pair(
                        name=f"{collision_name}_{another_collision_name}",
                        geomname1=collision_name,
                        geomname2=another_collision_name,
                        solref=default_solref,
                        friction=default_friction,
                        condim=3,
                    )
                    hand_collision_pairs.append(
                        (collision_name, another_collision_name)
                    )
                    contact_cnt += 1

    # sides = []
    # if embodiment_type in ["right", "bimanual"]:
    #     sides.append("right")
    # if embodiment_type in ["left", "bimanual"]:
    #     sides.append("left")
    # for side in sides:
    #     if f"{side}_thumb_collision" in hand_collision_names:
    #         if f"{side}_index_intermediate_collision" in hand_collision_names:
    #             mj_spec.add_pair(
    #                 name=f"{side}_thumb_collision_{side}_index_intermediate_collision",
    #                 geomname1=f"{side}_thumb_collision",
    #                 geomname2=f"{side}_index_intermediate_collision",
    #                 solref=default_solref,
    #                 friction=default_friction,
    #                 condim=3,
    #             )
    #         if f"{side}_index_collision" in hand_collision_names:
    #             mj_spec.add_pair(
    #                 name=f"{side}_thumb_collision_{side}_index_collision",
    #                 geomname1=f"{side}_thumb_collision",
    #                 geomname2=f"{side}_index_collision",
    #                 solref=default_solref,
    #                 friction=default_friction,
    #                 condim=3,
    #             )
    # hand <-> hand collision
    # if embodiment_type == "bimanual":
    #     for finger_name in ["thumb", "index", "index_intermediate"]:
    #         mj_spec.add_pair(
    #             name=f"left_{finger_name}_collision_right_{finger_name}_collision",
    #             geomname1=f"left_{finger_name}_collision",
    #             geomname2=f"right_{finger_name}_collision",
    #             solref=default_solref,
    #             friction=default_friction,
    #         )

    loguru.logger.info(f"Added {contact_cnt} contact pairs")

    # add camera
    mj_spec.worldbody.add_camera(
        name="front",
        pos=[0.031, 0.941, 0.844],
        xyaxes=[-0.999, 0.033, -0.000, -0.022, -0.667, 0.745],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    )

    # add reference sites for both hands and objects
    # for side in ["right", "left"]:
    #     for finger_name in finger_names:
    #         mocap_handle = mj_spec.worldbody.add_body(
    #             name=f"ref_{side}_{finger_name}",
    #             pos=[0, 0, 0],
    #             quat=[1, 0, 0, 0],
    #             mocap=True,
    #         )
    #         mocap_handle.add_site(
    #             name=f"ref_{side}_{finger_name}",
    #             pos=[0, 0, 0],
    #             size=[0.02, 0.02, 0.02],
    #             group=4,
    #             rgba=[0, 1, 0, 1],
    #         )

    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)

    # save model in processed dir, use a stable name
    xml_file = mj_spec.to_xml()
    export_file_path = f"{processed_dir}/../scene.xml"
    with open(export_file_path, "w") as f:
        f.write(xml_file)
    loguru.logger.info(f"Saved model to {export_file_path}")

    # save another model with has equality constraints between track site and ref site
    for sid in range(mj_model.nsite):
        site_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if "track" in site_name:
            # get ref site name by replacing "track" with "ref"
            ref_site_name = site_name.replace("track", "ref")
            e = mj_spec.add_equality(
                name=f"{site_name}_equality_constraint",
                type=mujoco.mjtEq.mjEQ_CONNECT,
                name1=site_name,
                name2=ref_site_name,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                data=np.zeros(11),
            )
            e.solref = [0.02, 1.0]
            # disable the constraint when the distance is large
            e.solimp = [0.0, 1.0, 100.0, 0.5, 2.0]
    mj_model_eq = mj_spec.compile()
    xml_file_eq = mj_spec.to_xml()
    export_file_path_eq = f"{processed_dir}/../scene_eq.xml"
    with open(export_file_path_eq, "w") as f:
        f.write(xml_file_eq)
    loguru.logger.info(
        f"Saved model with equality constraints to {export_file_path_eq}"
    )

    # save task info
    task_info["robot_type"] = robot_type
    with open(f"{processed_dir}/../task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)

    # visualize model
    if show_viewer:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            rate_limiter = RateLimiter(1 / mj_model.opt.timestep)
            while viewer.is_running():
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                rate_limiter.sleep()


if __name__ == "__main__":
    tyro.cli(main)
