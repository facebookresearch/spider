"""MuJoCo XML Rerun visualizer.

Inputs:
- MuJoCo XML scene file (MJCF)
- Trajectory npz containing at least `qpos` (like `mjcpu_viewer.py`)

Behavior:
- Loads the XML with MuJoCo, compiles to a model, and builds a Rerun scene graph.
- Logs static geometry (primitives & meshes) per-geom under each body with local transforms.
- Replays trajectory by logging per-body world transforms for each frame.

Notes:
- Mesh loading & scaling follows the approach in `judo/visualizers/model.py` (using trimesh).
- Rerun logging patterns follow the URDF loader example and `retarget_example/rerun_vis/visualize_rerun.py`.
"""

from __future__ import annotations

import ast
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import loguru
import mujoco
import numpy as np
import rerun as rr
import trimesh
import tyro

# -----------------------------
# Trace visualization defaults
# -----------------------------

DEFAULT_TRACE_COLOR = [204, 26, 204]  # ~ (0.8, 0.1, 0.8)
DEFAULT_LEFT_OBJECT_TRACE_COLOR = "#0081FB"
DEFAULT_RIGHT_OBJECT_TRACE_COLOR = "#FB7A00"
DEFAULT_OBJECT_TRACE_COLOR = [187, 220, 229]  # Default object color
DEFAULT_FLOOR_COLOR = [211, 211, 211]  # light grey
DEFAULT_OBJECT_COLOR = [100, 149, 237]  # slightly darker blue
DEFAULT_HAND_COLOR = [200, 200, 200]  # grey
DEFAULT_TRACE_RADIUS = 0.0003


# -----------------------------
# XML mesh asset parsing helpers
# -----------------------------


def _parse_mesh_assets(xml_path: Path) -> dict[str, dict]:
    """Parse MJCF to resolve mesh assets.

    Returns mapping: mesh_name -> {"file": Path, "scale": np.ndarray | None}
    """
    mesh_map: dict[str, dict] = {}
    try:
        tree = ET.parse(str(xml_path))
    except Exception:
        return mesh_map

    root = tree.getroot()

    # Get meshdir from compiler element
    mesh_dir = ""
    compiler_nodes = root.findall("compiler")
    for compiler in compiler_nodes:
        meshdir_attr = compiler.get("meshdir")
        if meshdir_attr:
            mesh_dir = meshdir_attr
            break

    asset_nodes = root.findall("asset")
    for asset in asset_nodes:
        for mesh in asset.findall("mesh"):
            name = mesh.get("name")
            file_attr = mesh.get("file")
            if not name or not file_attr:
                loguru.logger.warning(
                    f"Skipping mesh with missing name or file: {mesh}"
                )
                continue
            scale_attr = mesh.get("scale")
            scale = None
            if scale_attr is not None:
                try:
                    scale_vals = [float(x) for x in scale_attr.strip().split()]
                    if len(scale_vals) == 1:
                        scale = np.array([scale_vals[0]] * 3, dtype=np.float32)
                    elif len(scale_vals) == 3:
                        scale = np.array(scale_vals, dtype=np.float32)
                except Exception:
                    scale = None

            # Construct full path using meshdir and file_attr
            if mesh_dir:
                full_path = (xml_path.parent / mesh_dir / file_attr).resolve()
            else:
                full_path = (xml_path.parent / file_attr).resolve()
            mesh_map[name] = {"file": full_path, "scale": scale}
    return mesh_map


# -----------------------------
# Geometry creation (trimesh)
# -----------------------------


def _mujoco_mesh_to_trimesh(
    model: mujoco.MjModel, geom_id: int, verbose: bool = False
) -> trimesh.Trimesh | None:
    """Convert a MuJoCo mesh geometry to a trimesh with textures if available.

    Args:
        model: MuJoCo model object
        geom_id: Index of the geometry in the model
        verbose: If True, print debug information during conversion

    Returns:
        A trimesh object with texture/material applied if available
    """
    # Get the mesh ID for this geometry
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        return None

    # Get mesh data ranges from MuJoCo
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    # Extract vertices and faces
    vertices = model.mesh_vert[vert_start : vert_start + vert_count]
    faces = model.mesh_face[face_start : face_start + face_count]

    # Check if this mesh has texture coordinates
    texcoord_adr = model.mesh_texcoordadr[mesh_id]
    texcoord_num = model.mesh_texcoordnum[mesh_id]

    if texcoord_num > 0:
        # This mesh has UV coordinates
        if verbose:
            loguru.logger.debug(f"Mesh has {texcoord_num} texture coordinates")

        # Extract texture coordinates
        texcoords_flat = model.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_num * 2]
        texcoords = texcoords_flat.reshape(-1, 2)

        # Get per-face texture coordinate indices
        face_texcoord_idx = model.mesh_facetexcoord[
            face_start * 3 : (face_start + face_count) * 3
        ].reshape(face_count, 3)

        # Duplicate vertices for each face reference (to support different UVs per face)
        new_vertices = vertices[faces.flatten()]
        new_uvs = texcoords[face_texcoord_idx.flatten()]
        new_faces = np.arange(face_count * 3).reshape(-1, 3)

        # Create the mesh
        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        # Handle material and texture
        matid = model.geom_matid[geom_id]
        if matid >= 0 and matid < model.nmat:
            rgba = model.mat_rgba[matid]
            texid = model.mat_texid[matid]

            if texid >= 0 and texid < model.ntex:
                # Extract texture data
                tex_width = model.tex_width[texid]
                tex_height = model.tex_height[texid]
                tex_nchannel = model.tex_nchannel[texid]
                tex_adr = model.tex_adr[texid]
                tex_size = tex_width * tex_height * tex_nchannel
                tex_data = model.tex_data[tex_adr : tex_adr + tex_size]

                # Create PIL image based on channels
                try:
                    from PIL import Image

                    if tex_nchannel == 1:
                        tex_array = tex_data.reshape(tex_height, tex_width)
                        image = Image.fromarray(tex_array.astype(np.uint8), mode="L")
                    elif tex_nchannel == 3:
                        tex_array = tex_data.reshape(tex_height, tex_width, 3)
                        image = Image.fromarray(tex_array.astype(np.uint8), mode="RGB")
                    elif tex_nchannel == 4:
                        tex_array = tex_data.reshape(tex_height, tex_width, 4)
                        image = Image.fromarray(tex_array.astype(np.uint8), mode="RGBA")
                    else:
                        image = None

                    if image is not None:
                        material = trimesh.visual.material.PBRMaterial(
                            baseColorFactor=rgba, baseColorTexture=image
                        )
                        mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
                        if verbose:
                            loguru.logger.debug(
                                f"Applied texture: {tex_width}x{tex_height}, {tex_nchannel} channels"
                            )
                    else:
                        rgba_255 = (rgba * 255).astype(np.uint8)
                        mesh.visual = trimesh.visual.ColorVisuals(
                            vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
                        )
                except ImportError:
                    loguru.logger.warning("PIL not available, skipping texture loading")
                    rgba_255 = (rgba * 255).astype(np.uint8)
                    mesh.visual = trimesh.visual.ColorVisuals(
                        vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
                    )
            else:
                # Material but no texture
                rgba_255 = (rgba * 255).astype(np.uint8)
                mesh.visual = trimesh.visual.ColorVisuals(
                    vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
                )
        else:
            # No material - use default color
            color = _get_entity_color("mesh_geom")
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(color, (len(new_vertices), 1))
            )
    else:
        # No texture coordinates - simpler case
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Apply material color if available
        matid = model.geom_matid[geom_id]
        if matid >= 0 and matid < model.nmat:
            rgba = model.mat_rgba[matid]
            rgba_255 = (rgba * 255).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1))
            )
        else:
            # Default color
            color = _get_entity_color("mesh_geom")
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(color, (len(mesh.vertices), 1))
            )

    return mesh


def _trimesh_from_primitive(geom_type: int, size: np.ndarray, rgba: np.ndarray | None = None) -> trimesh.Trimesh | None:
    """Create a trimesh mesh for a MuJoCo primitive geom.

    - sphere: size[0] radius
    - capsule: size[0] radius, size[1] half-length (total length = 2*size[1])
    - cylinder: size[0] radius, size[1] half-length
    - box: size[0:3] half extents
    - plane: creates a large flat box
    """
    t = mujoco.mjtGeom
    if geom_type == t.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    elif geom_type == t.mjGEOM_CAPSULE:
        radius = float(size[0])
        length = float(2.0 * size[1])
        mesh = trimesh.creation.capsule(radius=radius, height=length)
    elif geom_type == t.mjGEOM_CYLINDER:
        radius = float(size[0])
        height = float(2.0 * size[1])
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
    elif geom_type == t.mjGEOM_BOX:
        # MuJoCo stores half-sizes; trimesh box extents are full lengths
        extents = 2.0 * np.asarray(size[:3], dtype=np.float32)
        mesh = trimesh.creation.box(extents=extents)
    elif geom_type == t.mjGEOM_PLANE:
        # Create a large flat box for plane visualization
        mesh = trimesh.creation.box(extents=[20.0, 20.0, 0.01])
    else:
        return None

    # Apply material if RGBA is provided
    if rgba is not None:
        rgba_arr = np.asarray(rgba, dtype=np.float32)
        if rgba_arr.size >= 4:
            material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=rgba_arr[:4],
                metallicFactor=0.5,
                roughnessFactor=0.5,
            )
            mesh.visual = trimesh.visual.TextureVisuals(material=material)

    return mesh


def _xyzw_from_wxyz(wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion wxyz -> xyzw."""
    assert wxyz.shape[-1] == 4
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)


## Removed: no longer baking geom transforms into vertices; using node-local rr.Transform3D instead.


def _get_mesh_group_path(geom_name: str, entity_root: str) -> str:
    """Determine the group path for a mesh based on its name.

    If 'collision' is in the geom name, group under 'collision', otherwise 'visual'.

    Args:
        geom_name: The name of the geometry
        entity_root: The root entity path (e.g., 'mujoco')

    Returns:
        Group path like 'mujoco/collision' or 'mujoco/visual'
    """
    if (
        "collision" in geom_name.lower()
        or (
            geom_name.lower().startswith("left_object_")
            and geom_name.split("_")[-1].isdigit()
        )
        or (
            geom_name.lower().startswith("right_object_")
            and geom_name.split("_")[-1].isdigit()
        )
    ):
        return f"{entity_root}/collision"
    else:
        return f"{entity_root}/visual"


def _get_entity_color(entity_name: str) -> np.ndarray:
    """Get appropriate color based on entity name (hand vs object).

    Returns RGB color as uint8 array.
    """
    entity_lower = entity_name.lower()

    if (
        "hand" in entity_lower
        or "thumb" in entity_lower
        or "index" in entity_lower
        or "middle" in entity_lower
        or "ring" in entity_lower
        or "pinky" in entity_lower
    ):
        # Convert hex color to RGB
        return np.array(DEFAULT_HAND_COLOR, dtype=np.uint8)
    elif "object" in entity_lower:
        # Use default object trace color
        return np.array(DEFAULT_OBJECT_COLOR, dtype=np.uint8)
    else:
        # Default fallback color (gray)
        return np.array([128, 128, 128], dtype=np.uint8)


def _vertex_colors_from_rgba(
    mesh: trimesh.Trimesh, rgba: np.ndarray | None
) -> np.ndarray | None:
    """Broadcast a single RGBA color to per-vertex RGB(A).

    Rerun Mesh3D expects vertex_colors as Nx3 or Nx4; use Nx4 if RGBA given.
    """
    if rgba is None:
        return None
    rgba = np.asarray(rgba, dtype=np.float32)
    if rgba.size not in (3, 4):
        return None
    num_v = len(mesh.vertices)
    color = rgba[:4] if rgba.size == 4 else np.concatenate([rgba[:3], [1.0]], axis=0)
    colors = np.tile(color[None, :], (num_v, 1))
    return colors


# -----------------------------
# Scene construction & logging
# -----------------------------


def build_and_log_scene_from_spec(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    xml_path: Path | None = None,
    entity_root: str = "mujoco",
) -> list[tuple[str, int]]:
    """Build and log scene directly from MjSpec and MjModel (no XML file needed).

    Returns body entity info for subsequent animation.
    """
    # Parse mesh assets from XML if available
    mesh_assets = _parse_mesh_assets(xml_path) if xml_path is not None else {}

    # Give default names to unnamed bodies (but DON'T rename geoms - we'll skip them)
    body_placeholder_idx = 0
    for body in spec.bodies[1:]:
        if not body.name:
            body.name = f"RERUN_BODY_{body_placeholder_idx}"
            body_placeholder_idx += 1

    # Create a frame for the world root
    world_entity = f"{entity_root}/world"
    rr.log(world_entity, rr.Transform3D(translation=[0.0, 0.0, 0.0]))

    # Create collision and visual group nodes
    rr.log(f"{entity_root}/collision", rr.Transform3D(), static=True)
    rr.log(f"{entity_root}/visual", rr.Transform3D(), static=True)

    # Add a grey floor to the ground (in a fixed position)
    rr.log(
        f"{entity_root}/floor",
        rr.Boxes3D(
            half_sizes=[[0.3, 0.3, 0.001]],
            colors=DEFAULT_FLOOR_COLOR,
            fill_mode=3,
        ),
        static=True,
    )

    # Build mapping from spec geom to compiled model geom index
    # Try two strategies: by name and by (body_id, geom_index_in_body)
    geom_name_to_id = {}
    for geom_id in range(model.ngeom):
        try:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name:
                geom_name_to_id[geom_name] = geom_id
        except Exception:
            pass

    # Build alternative mapping by body and geom index for unnamed geoms
    geom_by_body_index = {}  # (body_id, geom_index) -> geom_id
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        # Count how many geoms we've seen for this body so far
        geom_index = sum(1 for gid in range(geom_id) if model.geom_bodyid[gid] == body_id)
        geom_by_body_index[(body_id, geom_index)] = geom_id

    # Iterate bodies and geoms
    body_entity_and_ids = []
    geom_count = 0
    mesh_count = 0
    primitive_count = 0
    skipped_count = 0

    for body in spec.bodies:  # includes worldbody at index 0
        body_name = body.name if body.name else f"body_{body.id}"

        # Add visual body entity for position tracking
        visual_body_entity = f"{entity_root}/visual/{body_name}"
        body_entity_and_ids.append((visual_body_entity, body.id))
        rr.log(visual_body_entity, rr.Transform3D())

        for geom_idx, geom in enumerate(body.geoms):
            geom_count += 1
            # Generate or use existing geom name
            if not geom.name:
                geom_name = f"unnamed_geom_{abs(hash((body_name, id(geom)))) % 10_000}"
            else:
                geom_name = geom.name

            # Find compiled model geom for this spec geom
            # Try by name first, then by body+index
            model_geom_id = geom_name_to_id.get(geom_name, -1)
            if model_geom_id < 0:
                # Try alternative lookup by (body_id, geom_index)
                model_geom_id = geom_by_body_index.get((body.id, geom_idx), -1)

            # Skip unnamed geoms that don't exist in the compiled model
            # (these are likely invalid or placeholder geoms)
            if not geom.name and model_geom_id < 0:
                skipped_count += 1
                loguru.logger.debug(f"Skipping unnamed geom {geom_name} (no model geom found)")
                continue

            # Group ALL geoms by collision vs visual based on name
            group_path = _get_mesh_group_path(geom_name, entity_root)
            geom_entity = f"{group_path}/{body_name}/{geom_name}"

            # Log geom's local transform relative to its parent body (use spec's local pose)
            try:
                local_pos = np.asarray(geom.pos, dtype=np.float32)
            except Exception:
                local_pos = np.zeros(3, dtype=np.float32)
            try:
                local_quat_wxyz = np.asarray(geom.quat, dtype=np.float32)
            except Exception:
                local_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            local_quat_xyzw = _xyzw_from_wxyz(local_quat_wxyz)
            rr.log(
                geom_entity,
                rr.Transform3D(translation=local_pos, quaternion=local_quat_xyzw),
                static=True,
            )

            # Create mesh using the compiled model for better material/texture support
            tm = None
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_count += 1
                # Try using compiled model's mesh data (better texture support)
                if model_geom_id >= 0:
                    try:
                        tm = _mujoco_mesh_to_trimesh(model, model_geom_id, verbose=False)
                    except Exception as e:
                        loguru.logger.debug(
                            f"Failed to convert mesh geom {geom_name} using model: {e}"
                        )

                # Fallback to XML-based mesh loading
                if tm is None and xml_path is not None:
                    mesh_info = None
                    try:
                        mesh_name = geom.meshname
                        if mesh_name in mesh_assets:
                            mesh_info = mesh_assets[mesh_name]
                    except Exception as e:
                        loguru.logger.debug(
                            f"Failed to get mesh info for {geom.meshname}: {e}"
                        )

                    if mesh_info is not None:
                        mesh_file: Path = mesh_info["file"]
                        if not mesh_file.exists():
                            alt_path = Path.cwd() / mesh_file.name
                            if alt_path.exists():
                                mesh_file = alt_path

                        if mesh_file.exists():
                            try:
                                tm = trimesh.load(str(mesh_file), force="mesh")
                                if isinstance(tm, trimesh.Scene):
                                    tm = tm.to_mesh()
                                # Apply asset scale if present
                                scale = mesh_info.get("scale")
                                if scale is not None:
                                    tm.apply_scale(scale)
                            except Exception as e:
                                loguru.logger.debug(f"Failed to load mesh {mesh_file}: {e}")

                if tm is None:
                    # Skip mesh geoms that couldn't be loaded (instead of warning)
                    continue

            else:
                primitive_count += 1
                # Primitives: sphere, box, capsule, cylinder, plane
                # Get RGBA from compiled model if available
                rgba = None
                if model_geom_id >= 0:
                    try:
                        rgba = model.geom_rgba[model_geom_id]
                    except Exception:
                        pass

                # Use spec geom size (or from compiled model if spec size is invalid)
                geom_size = geom.size
                if model_geom_id >= 0:
                    try:
                        model_size = model.geom_size[model_geom_id]
                        if np.any(geom_size == 0) or np.any(np.isnan(geom_size)):
                            geom_size = model_size
                    except Exception:
                        pass

                tm = _trimesh_from_primitive(geom.type, geom_size, rgba=rgba)
                if tm is None:
                    continue

            # Log the trimesh to rerun
            if tm is not None:
                _log_trimesh_entity(geom_entity, tm, None)

    loguru.logger.info(
        f"Scene build complete: {geom_count} total geoms ({skipped_count} skipped), {mesh_count} mesh geoms, {primitive_count} primitive geoms"
    )
    return body_entity_and_ids


def build_and_log_scene(
    xml_path: Path, entity_root: str = "mujoco"
) -> tuple[mujoco.MjSpec, mujoco.MjModel, list[tuple[str, int]]]:
    """Load the MJCF, create static geometry, and log it to Rerun.

    Returns spec, compiled model, and body entity info for subsequent animation.
    """
    # Load spec (support multiple MuJoCo versions)
    spec = mujoco.MjSpec.from_file(str(xml_path))

    # Give default names to unnamed bodies and geoms (mirrors judo implementation approach)
    geom_placeholder_idx = 0
    body_placeholder_idx = 0
    for body in spec.bodies[1:]:
        if not body.name:
            body.name = f"RERUN_BODY_{body_placeholder_idx}"
            body_placeholder_idx += 1
        for geom in body.geoms:
            if not geom.name:
                geom.name = f"RERUN_GEOM_{geom_placeholder_idx}"
                geom_placeholder_idx += 1

    # Compile model
    model = spec.compile()

    # Parse mesh assets from XML
    mesh_assets = _parse_mesh_assets(xml_path)

    # Create a frame for the world root
    world_entity = f"{entity_root}/world"
    rr.log(world_entity, rr.Transform3D(translation=[0.0, 0.0, 0.0]))

    # Create collision and visual group nodes
    rr.log(f"{entity_root}/collision", rr.Transform3D(), static=True)
    rr.log(f"{entity_root}/visual", rr.Transform3D(), static=True)

    # Add a grey floor to the ground (in a fixed position)
    rr.log(
        f"{entity_root}/floor",
        rr.Boxes3D(
            half_sizes=[[0.3, 0.3, 0.001]],
            colors=DEFAULT_FLOOR_COLOR,
            fill_mode=3,
        ),
        static=True,
    )

    # Iterate bodies and geoms
    body_entity_and_ids = []
    for body in spec.bodies[1:]:  # skip implicit worldbody at index 0
        body_name = body.name
        # Add both collision and visual body entities for position tracking
        collision_body_entity = f"{entity_root}/collision/{body_name}"
        visual_body_entity = f"{entity_root}/visual/{body_name}"
        body_entity_and_ids.append((collision_body_entity, body.id))
        body_entity_and_ids.append((visual_body_entity, body.id))
        # Initialize both body nodes (no transform yet; per-frame logging will update them)
        rr.log(collision_body_entity, rr.Transform3D())
        rr.log(visual_body_entity, rr.Transform3D())

        for geom in body.geoms:
            geom_name = (
                geom.name
                if geom.name
                else f"geom_{abs(hash((body_name, id(geom)))) % 10_000}"
            )

            # Group ALL geoms by collision vs visual based on name
            group_path = _get_mesh_group_path(geom_name, entity_root)
            geom_entity = f"{group_path}/{body_name}/{geom_name}"

            # Use compiled geom for consistent sizes and defaulted values
            model_geom = model.geom(geom.name)

            # Log geom's local transform relative to its parent body (use spec's local pose)
            try:
                local_pos = np.asarray(geom.pos, dtype=np.float32)
            except Exception:
                local_pos = np.zeros(3, dtype=np.float32)
            try:
                local_quat_wxyz = np.asarray(geom.quat, dtype=np.float32)
            except Exception:
                local_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            local_quat_xyzw = _xyzw_from_wxyz(local_quat_wxyz)
            rr.log(
                geom_entity,
                rr.Transform3D(translation=local_pos, quaternion=local_quat_xyzw),
                static=True,
            )

            # Create mesh for primitive or load file for mesh geoms
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_info = None
                try:
                    # geom.mesh is the asset name
                    mesh_name = geom.meshname
                    if mesh_name in mesh_assets:
                        mesh_info = mesh_assets[mesh_name]
                except Exception as e:
                    loguru.logger.warning(
                        f"Failed to get mesh info for {geom.meshname}: {e}"
                    )
                    mesh_info = None

                if mesh_info is None:
                    # Fallback: cannot resolve mesh file; skip this geom
                    loguru.logger.warning(
                        f"Skipping mesh with missing name or file: {geom.meshname}"
                    )
                    continue

                mesh_file: Path = mesh_info["file"]
                if not mesh_file.exists():
                    # Try relative to working dir just in case
                    alt_path = Path.cwd() / mesh_file.name
                    if alt_path.exists():
                        mesh_file = alt_path
                    else:
                        loguru.logger.warning(f"Mesh file not found: {mesh_file}")
                        continue

                try:
                    # DEBUG: check if visual.obj is in name, if so, replace it with cat.glb
                    # if "visual.obj" in mesh_file.name:
                    #     mesh_file = mesh_file.parent / "cat.glb"
                    tm = trimesh.load(str(mesh_file), force="mesh")
                except Exception:
                    loguru.logger.warning(f"Failed to load mesh: {mesh_file}")
                    continue

                if not isinstance(tm, trimesh.Trimesh):
                    # Flatten scenes to a single mesh with node transforms baked
                    if isinstance(tm, trimesh.Scene):
                        try:
                            tm = tm.to_mesh()
                        except Exception:
                            loguru.logger.warning(
                                f"Failed to flatten scene mesh: {mesh_file}"
                            )
                            continue
                    else:
                        continue

                # Apply asset scale if present
                scale = mesh_info.get("scale")
                if scale is not None:
                    tm.apply_scale(scale)
                _log_trimesh_entity(geom_entity, tm, model_geom)

            else:
                # Primitives: sphere, box, capsule, cylinder, plane (skip plane)
                tm = _trimesh_from_primitive(geom.type, model_geom.size)
                if tm is None:
                    continue
                _log_trimesh_entity(geom_entity, tm, model_geom)

    return spec, model, body_entity_and_ids


def export_scene_to_npz(
    xml_path: Path, out_path: Path, entity_root: str = "mujoco"
) -> None:
    """Export static scene (geometry + local transforms) to an NPZ that can be loaded without MuJoCo.

    The NPZ contains:
      - body_entity_and_ids: (B,) array of object strings
      - geom_body_index: (K,) int mapping each geom to parent body index (>=1 to skip world)
      - geom_names: (K,) array of object strings
      - geom_local_pos: (K, 3) float32
      - geom_local_quat_xyzw: (K, 4) float32
      - verts_flat: (sum_V, 3) float32
      - verts_offsets: (K+1,) int64
      - faces_flat: (sum_F, 3) uint32
      - faces_offsets: (K+1,) int64
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))
    model = spec.compile()
    mesh_assets = _parse_mesh_assets(xml_path)

    # Save body entities (paths) and corresponding model body ids separately to avoid stringifying tuples
    body_entities: list[str] = []
    body_model_ids: list[int] = []
    geom_names: list[str] = []
    geom_types: list[
        int
    ] = []  # Store geom types to enable proper grouping in log_scene_from_npz
    geom_body_index: list[int] = []
    geom_local_pos: list[np.ndarray] = []
    geom_local_quat_xyzw: list[np.ndarray] = []

    verts_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    verts_offsets = [0]
    faces_offsets = [0]

    # bodies[0] is world; we store body names for indices 1..B-1
    for i, body in enumerate(spec.bodies[1:], start=1):
        nm = body.name if body.name else f"body_{i}"
        body_entity = f"{entity_root}/{nm}"
        # Store entity path and the compiled model body id for later realtime updates
        body_entities.append(body_entity)
        try:
            body_model_ids.append(int(body.id))
        except Exception:
            # Fallback if spec body.id is unavailable; keep a placeholder that callers should not use
            body_model_ids.append(-1)
        for geom in body.geoms:
            gname = (
                geom.name if geom.name else f"geom_{abs(hash((nm, id(geom)))) % 10_000}"
            )
            geom_names.append(gname)
            geom_types.append(int(geom.type))  # Store geom type for grouping
            geom_body_index.append(i)
            # local pose from spec
            try:
                local_pos = np.asarray(geom.pos, dtype=np.float32)
            except Exception:
                local_pos = np.zeros(3, dtype=np.float32)
            try:
                local_quat_wxyz = np.asarray(geom.quat, dtype=np.float32)
            except Exception:
                local_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            local_quat_xyzw = _xyzw_from_wxyz(local_quat_wxyz)
            geom_local_pos.append(local_pos)
            geom_local_quat_xyzw.append(local_quat_xyzw)

            # mesh
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_info = None
                try:
                    mesh_name = geom.meshname
                    if mesh_name in mesh_assets:
                        mesh_info = mesh_assets[mesh_name]
                except Exception:
                    mesh_info = None
                if mesh_info is None:
                    continue
                mesh_file: Path = mesh_info["file"]
                if not mesh_file.exists():
                    alt_path = Path.cwd() / mesh_file.name
                    if alt_path.exists():
                        mesh_file = alt_path
                    else:
                        continue
                try:
                    tm = trimesh.load(str(mesh_file), force="mesh")
                except Exception:
                    continue
                if not isinstance(tm, trimesh.Trimesh):
                    if isinstance(tm, trimesh.Scene):
                        try:
                            tm = tm.to_mesh()
                        except Exception:
                            continue
                    else:
                        continue
                scale = mesh_info.get("scale")
                if scale is not None:
                    tm.apply_scale(scale)
                v = np.asarray(tm.vertices, dtype=np.float32)
                f = np.asarray(tm.faces, dtype=np.uint32)
            else:
                # Robustly fetch size for primitives. Some geoms may be unnamed in spec;
                # use the compiled model's per-body geom list to find a matching geom by type.
                size_arr = None
                try:
                    if geom.name:
                        size_arr = model.geom(geom.name).size
                except Exception:
                    size_arr = None
                if size_arr is None:
                    # Fallback: iterate over model's geoms on this body and take the first matching type
                    try:
                        # Prefer compiled spec id if available instead of name lookup (handles unnamed bodies)
                        body_id = int(body.id)
                    except Exception:
                        body_id = -1
                    if body_id >= 0:
                        try:
                            for gid in range(model.ngeom):
                                if int(model.geom_bodyid[gid]) != body_id:
                                    continue
                                if int(model.geom_type[gid]) != int(geom.type):
                                    continue
                                size_arr = model.geom_size[gid]
                                break
                        except Exception:
                            size_arr = None
                tm = _trimesh_from_primitive(geom.type, size_arr)
                if tm is None:
                    # skip planes or unsupported
                    v = np.zeros((0, 3), dtype=np.float32)
                    f = np.zeros((0, 3), dtype=np.uint32)
                else:
                    v = np.asarray(tm.vertices, dtype=np.float32)
                    f = np.asarray(tm.faces, dtype=np.uint32)

            verts_list.append(v)
            faces_list.append(f)
            verts_offsets.append(verts_offsets[-1] + v.shape[0])
            faces_offsets.append(faces_offsets[-1] + f.shape[0])

    if not verts_list:
        verts_flat = np.zeros((0, 3), dtype=np.float32)
        faces_flat = np.zeros((0, 3), dtype=np.uint32)
    else:
        verts_flat = (
            np.concatenate(verts_list, axis=0) if len(verts_list) > 1 else verts_list[0]
        )
        faces_flat = (
            np.concatenate(faces_list, axis=0) if len(faces_list) > 1 else faces_list[0]
        )

    # Save using non-object dtypes for cross-version compatibility
    np.savez(
        str(out_path),
        # New explicit arrays to avoid ambiguous stringified tuples
        body_entities=np.array(body_entities, dtype="U"),
        body_model_ids=np.array(body_model_ids, dtype=np.int32),
        # Geom bookkeeping
        geom_body_index=np.array(geom_body_index, dtype=np.int32),
        geom_names=np.array(geom_names, dtype="U"),
        geom_types=np.array(geom_types, dtype=np.int32),
        geom_local_pos=(
            np.stack(geom_local_pos, axis=0)
            if geom_local_pos
            else np.zeros((0, 3), dtype=np.float32)
        ),
        geom_local_quat_xyzw=(
            np.stack(geom_local_quat_xyzw, axis=0)
            if geom_local_quat_xyzw
            else np.zeros((0, 4), dtype=np.float32)
        ),
        # Geometry buffers
        verts_flat=verts_flat,
        verts_offsets=np.array(verts_offsets, dtype=np.int64),
        faces_flat=faces_flat,
        faces_offsets=np.array(faces_offsets, dtype=np.int64),
        # Backward-compatibility: also store the previous combined field as strings for older loaders
        body_entity_and_ids=np.array(
            [
                f"({repr(e)}, {int(i)})"
                for e, i in zip(body_entities, body_model_ids, strict=False)
            ],
            dtype="U",
        ),
    )


def log_scene_from_npz(
    npz_path: Path, entity_root: str = "mujoco"
) -> list[tuple[str, int]]:
    """Log a pre-baked static scene NPZ created by export_scene_to_npz.

    Returns the list of body names to help upstream code log transforms to
    paths like f"{entity_root}/{body_name}".
    """
    data = np.load(str(npz_path), allow_pickle=False)  # strict load

    # body_names / geom_names may be unicode or bytes; normalize to python str
    def _norm_str_array(arr: np.ndarray) -> list[str]:
        if np.issubdtype(arr.dtype, np.bytes_):
            return [x.decode("utf-8", errors="ignore") for x in arr.tolist()]
        return [str(x) for x in arr.tolist()]

    # Prefer new explicit fields; fall back to legacy combined string field if needed
    body_entity_and_ids: list[tuple[str, int]] = []
    if "body_entities" in data.files and "body_model_ids" in data.files:
        body_entities = _norm_str_array(data["body_entities"])  # type: ignore[index]
        body_model_ids = data["body_model_ids"].astype(int)  # type: ignore[index]
        body_entity_and_ids = list(
            zip(body_entities, body_model_ids.tolist(), strict=False)
        )
    else:
        # Legacy path: parse stringified tuples/lists safely
        try:
            legacy = _norm_str_array(data["body_entity_and_ids"])  # type: ignore[index]
        except Exception:
            legacy = []
        for item in legacy:
            try:
                parsed_item = ast.literal_eval(item)
                if isinstance(parsed_item, (list, tuple)) and len(parsed_item) >= 2:
                    body_entity_and_ids.append(
                        (str(parsed_item[0]), int(parsed_item[1]))
                    )
                else:
                    body_entity_and_ids.append((str(parsed_item), 0))
            except Exception:
                body_entity_and_ids.append((item, 0))
    geom_body_index = data["geom_body_index"].astype(int)
    geom_names = _norm_str_array(data["geom_names"])
    # Load geom types for grouping (backward compatibility: default to non-mesh if not present)
    geom_types = (
        data["geom_types"].astype(int)
        if "geom_types" in data.files
        else np.zeros(len(geom_names), dtype=int)
    )
    geom_local_pos = data["geom_local_pos"].astype(np.float32)
    geom_local_quat_xyzw = data["geom_local_quat_xyzw"].astype(np.float32)
    verts_flat = data["verts_flat"].astype(np.float32)
    faces_flat = data["faces_flat"].astype(np.uint32)
    v_off = data["verts_offsets"].astype(int)
    f_off = data["faces_offsets"].astype(int)

    # Create world and body frames
    rr.log(
        f"{entity_root}/world",
        rr.Transform3D(translation=np.array([0.0, 0.0, 0.0], dtype=np.float32)),
    )
    # Create collision and visual group nodes
    rr.log(f"{entity_root}/collision", rr.Transform3D(), static=True)
    rr.log(f"{entity_root}/visual", rr.Transform3D(), static=True)

    # Create both collision and visual body nodes for each body
    updated_body_entity_and_ids = []
    for body_entity, bid in body_entity_and_ids:
        # Extract body name from the original body entity
        body_name = body_entity.split("/")[-1]
        collision_body_entity = f"{entity_root}/collision/{body_name}"
        visual_body_entity = f"{entity_root}/visual/{body_name}"

        rr.log(collision_body_entity, rr.Transform3D())
        rr.log(visual_body_entity, rr.Transform3D())

        # Add both to the updated list for position tracking
        updated_body_entity_and_ids.append((collision_body_entity, bid))
        updated_body_entity_and_ids.append((visual_body_entity, bid))

    # Use the updated list
    body_entity_and_ids = updated_body_entity_and_ids

    # Log geoms under bodies
    for gi in range(len(geom_names)):
        bi = geom_body_index[gi]
        # Since body_entity_and_ids now has double entries (collision + visual),
        # we need to get the original body info differently
        if bi <= 0 or (bi - 1) * 2 >= len(body_entity_and_ids):
            continue
        # Get the collision body entity (first of the pair) to extract body name
        collision_body_entity, bid = body_entity_and_ids[(bi - 1) * 2]
        geom_name = geom_names[gi]
        geom_type = geom_types[gi]

        # Group ALL geoms by collision vs visual based on name
        group_path = _get_mesh_group_path(geom_name, entity_root)
        # Extract body name from collision_body_entity
        body_name = collision_body_entity.split("/")[-1]
        geom_entity = f"{group_path}/{body_name}/{geom_name}"

        rr.log(
            geom_entity,
            rr.Transform3D(
                translation=geom_local_pos[gi], quaternion=geom_local_quat_xyzw[gi]
            ),
            static=True,
        )
        vs = verts_flat[v_off[gi] : v_off[gi + 1]]
        fs = faces_flat[f_off[gi] : f_off[gi + 1]]
        if vs.shape[0] == 0 or fs.shape[0] == 0:
            continue
        # Use default color based on entity name
        entity_color = _get_entity_color(geom_entity)
        rr.log(
            geom_entity,
            rr.Mesh3D(
                vertex_positions=vs.astype(np.float32),
                triangle_indices=fs.astype(np.uint32),
                vertex_colors=entity_color,
                albedo_factor=entity_color,
            ),
            static=True,
        )

    return body_entity_and_ids


def init_rerun(app_name: str = "mujoco_xml_viewer", spawn: bool = False) -> None:
    """Initialize Rerun and optionally spawn a viewer window."""
    rr.init(app_name)
    if spawn:
        rr.spawn()


def log_frame(
    data: mujoco.MjData,
    sim_time: float,
    viewer_body_entity_and_ids: list[tuple[str, int]] = [],
) -> None:
    rr.set_time("sim_time", timestamp=sim_time)
    # Log per-body transforms
    for entity, bid in viewer_body_entity_and_ids:
        pos = np.asarray(data.xpos[bid], dtype=np.float32)
        quat_wxyz = np.asarray(data.xquat[bid], dtype=np.float32)
        quat_xyzw = _xyzw_from_wxyz(quat_wxyz)
        rr.log(entity, rr.Transform3D(translation=pos, quaternion=quat_xyzw))


def log_traces_from_info(traces: np.ndarray, sim_time: float) -> None:
    """Visualize trace arrays contained in an optimize() info dict.

    Expected per-key shapes:
    - (I, N, P, K, 3): iterations x traces x points x trace site number x 3
    Keys must start with 'trace_'. Colors vary by iteration as in offline visualizer.
    """
    rr.set_time("sim_time", timestamp=sim_time)
    rr.log("/traces", rr.Transform3D(), static=True)
    a = np.asarray(traces, dtype=np.float64)
    I, N, P, K, _ = a.shape
    # move K dimension forward
    a = a.transpose(0, 1, 3, 2, 4)  # (I, N, K, P, 3)
    # create colors
    colors = np.zeros([I, N, K, 3])
    white = np.array([255, 255, 255])
    red = np.array([255, 0, 0])
    blue = np.array([0, 0, 255])

    for i in range(I):
        for k in range(K):
            if I == 1:
                # When there's only one iteration, use full color
                if k < 1:
                    colors[i, :, k, :] = red
                else:
                    colors[i, :, k, :] = blue
            else:
                if k < 1:
                    colors[i, :, k, :] = (1 - i / (I - 1)) * white + (i / (I - 1)) * red
                else:
                    colors[i, :, k, :] = (1 - i / (I - 1)) * white + (
                        i / (I - 1)
                    ) * blue
    colors = colors.reshape(I * N * K, 3).astype(np.uint8)
    strips = a.reshape(I * N * K, P, 3)
    rr.log(
        "/traces",
        rr.LineStrips3D(strips, colors=colors, radii=DEFAULT_TRACE_RADIUS),
    )


def log_planning_traces(
    traces: dict[str, np.ndarray],
    entity_root: str,
    plan_step: int | None = None,
    radius: float = DEFAULT_TRACE_RADIUS,
) -> None:
    """Log planned trajectory line strips to Rerun.

    Accepts values per key with shapes:
    - (P, 3): a single strip
    - (N, P, 3): multiple strips
    - (I, N, P, 3): multiple iterations x strips
    In all cases, P are points.
    """
    if plan_step is not None:
        rr.set_time_sequence("plan", int(plan_step))
    # Ensure traces root exists
    rr.log(f"{entity_root}/traces", rr.Transform3D(), static=True)
    for name, arr in traces.items():
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[-1] == 3:
            strips = a[None, :, :]  # (1, P, 3)
            rr.log(
                f"{entity_root}/traces/{name}", rr.LineStrips3D(strips, radii=radius)
            )
        elif a.ndim == 3 and a.shape[-1] == 3:
            # (N, P, 3)
            rr.log(f"{entity_root}/traces/{name}", rr.LineStrips3D(a, radii=radius))
        elif a.ndim == 4 and a.shape[-1] == 3:
            # (I, N, P, 3)
            I, N, P, _ = a.shape
            strips = a.reshape(I * N, P, 3)
            rr.log(
                f"{entity_root}/traces/{name}", rr.LineStrips3D(strips, radii=radius)
            )
        else:
            loguru.logger.warning(
                f"log_planning_traces: skip '{name}' with incompatible shape {a.shape}"
            )


## Removed: we now flatten scenes using trimesh.Scene.to_mesh().


def _log_trimesh_entity(
    entity_path: str, mesh: trimesh.Trimesh, model_geom: any
) -> None:
    """Log a trimesh as rr.Mesh3D with proper visual data from the mesh."""
    vertex_positions = np.asarray(mesh.vertices, dtype=np.float32)
    triangle_indices = np.asarray(mesh.faces, dtype=np.uint32)

    # Validate mesh has data
    if len(vertex_positions) == 0 or len(triangle_indices) == 0:
        loguru.logger.warning(f"Skipping {entity_path}: empty mesh (verts={len(vertex_positions)}, faces={len(triangle_indices)})")
        return

    vertex_normals = (
        np.asarray(mesh.vertex_normals, dtype=np.float32)
        if mesh.vertex_normals is not None
        else None
    )

    # Extract color/texture information from the mesh's visual data
    vertex_colors = None
    albedo_factor = None

    # Check if mesh has visual data
    if hasattr(mesh, 'visual') and mesh.visual is not None:
        # Try to get vertex colors from the mesh visual
        if isinstance(mesh.visual, trimesh.visual.ColorVisuals):
            # ColorVisuals has vertex_colors attribute
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vc = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
                if vc.shape[0] == len(vertex_positions):
                    vertex_colors = vc
        elif isinstance(mesh.visual, trimesh.visual.TextureVisuals):
            # TextureVisuals might have a material with baseColorFactor
            if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                material = mesh.visual.material
                if hasattr(material, 'baseColorFactor'):
                    base_color = np.asarray(material.baseColorFactor, dtype=np.float32)
                    if base_color.size >= 3:
                        # Convert float [0,1] to uint8 [0,255]
                        albedo_factor = (base_color[:4] * 255).astype(np.uint8) if base_color.size >= 4 else np.concatenate([base_color[:3] * 255, [255]]).astype(np.uint8)
                        # Create per-vertex colors
                        vertex_colors = np.tile(albedo_factor, (len(vertex_positions), 1))

    # Fallback to entity-based default color if no visual data found
    if vertex_colors is None and albedo_factor is None:
        entity_color = _get_entity_color(entity_path)
        vertex_colors = np.tile(entity_color, (len(vertex_positions), 1))
        albedo_factor = entity_color

    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=vertex_positions,
            triangle_indices=triangle_indices,
            vertex_normals=vertex_normals,
            vertex_colors=vertex_colors,
            albedo_factor=albedo_factor if albedo_factor is not None else vertex_colors[0] if vertex_colors is not None else None,
        ),
        static=True,
    )


# -----------------------------
# Trajectory playback
# -----------------------------


def play_trajectory(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    npz_path: Path,
    entity_root: str = "mujoco",
    fps: float = 50.0,
) -> None:
    """Load trajectory from npz and log per-body world transforms over time."""
    data = mujoco.MjData(model)

    traj_data = np.load(str(npz_path))
    qpos_list = traj_data["qpos"].reshape(-1, model.nq)

    # Extract trace data: arrays with keys starting with "trace_".
    # Preferred shape: (D, I, N, P, 3) where D=downsampled_frames, I=iterations, N=traces, P=points.
    # We normalize to time-major shapes aligned with num_frames.
    num_frames = qpos_list.shape[0]
    traces: dict[str, dict] = {}

    def _normalize_trace_array_5d(
        arr: np.ndarray,
    ) -> tuple[np.ndarray, int, int] | None:
        """Return array shaped (T, I, N, P, 3) or None if incompatible.

        Repeats the first dimension to match T if needed.
        Returns (array_TINP3, I, N).
        """
        a = np.asarray(arr)
        if a.ndim != 5 or a.shape[-1] != 3:
            return None
        D, I, N, P, C = a.shape
        # Repeat D along time to match T
        if D <= 0:
            return None
        repeats = int(np.ceil(num_frames / D))
        a_rep = np.repeat(a, repeats, axis=0)[:num_frames]
        return a_rep.astype(np.float32, copy=False), I, N

    # Legacy 2-point segment traces are not supported anymore.

    def _generate_iteration_colors(
        num_iterations: int, base_rgb: list[int]
    ) -> list[list[int]]:
        """Generate light-to-deep colors for iteration groups.

        Smaller iteration index -> lighter color; larger -> deeper (closer to base).
        """
        if num_iterations <= 0:
            return []
        white = np.array([255, 255, 255], dtype=np.float32)
        base = np.array(base_rgb, dtype=np.float32)
        if num_iterations == 1:
            alphas = [0.35]  # modest lightening
        else:
            # alpha controls mix with white; 0 -> base, 1 -> white
            # Start lighter (higher alpha) for small iteration, end deeper (lower alpha)
            alphas = np.linspace(0.7, 0.0, num_iterations)
        colors = []
        for a in alphas:
            c = np.clip(base * (1.0 - a) + white * a, 0.0, 255.0)
            colors.append(c.astype(np.uint8).tolist())
        return colors

    for key in traj_data.files:
        if not key.startswith("trace_"):
            continue
        raw = traj_data[key]
        # Priority: try new 5D format first
        norm5 = _normalize_trace_array_5d(raw)
        if norm5 is not None:
            arr5, iters, ntraces = norm5
            if "object" in key:
                iter_colors = _generate_iteration_colors(
                    iters, DEFAULT_OBJECT_TRACE_COLOR
                )
            else:
                iter_colors = _generate_iteration_colors(iters, DEFAULT_TRACE_COLOR)
            traces[key] = {
                "points5": arr5,  # (T, I, N, P, 3)
                "iters": iters,
                "ntraces": ntraces,
                "iter_colors": iter_colors,
                "radius": DEFAULT_TRACE_RADIUS,
            }
            continue
        loguru.logger.warning(
            f"Skipping trace '{key}': incompatible shape {np.shape(raw)}"
        )

    # Create a traces root node for organization if any traces found
    if traces:
        rr.log(f"{entity_root}/traces", rr.Transform3D(), static=True)
        for name in traces:
            rr.log(f"{entity_root}/traces/{name}", rr.Transform3D(), static=True)

    # Initial forward for FK
    data.qpos[:] = qpos_list[0]
    mujoco.mj_forward(model, data)

    bodies = spec.bodies
    dt = 1.0 / fps if fps > 0 else 0.0
    for frame_idx in range(qpos_list.shape[0]):
        rr.set_time_sequence("frame", frame_idx)

        data.qpos[:] = qpos_list[frame_idx]
        mujoco.mj_forward(model, data)

        # Log world transforms for each body under both collision and visual groups
        for i in range(1, len(bodies)):
            body_name = bodies[i].name if bodies[i].name else f"body_{i}"

            data_idx = bodies[i].id  # align spec body ordering with model indices
            pos = np.asarray(data.xpos[data_idx], dtype=np.float32)
            quat_wxyz = np.asarray(data.xquat[data_idx], dtype=np.float32)
            quat_xyzw = _xyzw_from_wxyz(quat_wxyz)

            # Log body transform to both collision and visual groups
            collision_body_entity = f"{entity_root}/collision/{body_name}"
            visual_body_entity = f"{entity_root}/visual/{body_name}"

            rr.log(
                collision_body_entity,
                rr.Transform3D(translation=pos, quaternion=quat_xyzw),
            )
            rr.log(
                visual_body_entity,
                rr.Transform3D(translation=pos, quaternion=quat_xyzw),
            )

        # Log trace segments for this frame, if any
        if traces:
            for name, info in traces.items():
                # New format: multiple iterations x traces, polyline per trace
                if "points5" in info:
                    arr5_t = info["points5"][frame_idx]  # (I, N, P, 3)
                    iters = info["iters"]
                    ntraces = info["ntraces"]
                    colors_by_iter = info["iter_colors"]  # len I
                    radius = info["radius"]

                    # Reshape to (I*N, P, 3)
                    strips = arr5_t.reshape(iters * ntraces, arr5_t.shape[2], 3)
                    # Colors per strip: repeat each iteration color N times
                    colors = np.repeat(
                        np.asarray(colors_by_iter, dtype=np.uint8),
                        repeats=ntraces,
                        axis=0,
                    )
                    rr.log(
                        f"{entity_root}/traces/{name}",
                        rr.LineStrips3D(strips, colors=colors, radii=radius),
                    )
                # No legacy fallback

        if dt > 0:
            time.sleep(dt)


# -----------------------------
# CLI
# -----------------------------


def main(
    xml: str = "../../example_datasets/processed/gigahand/schunk/bimanual/p36-tea/scene.xml",
    npz: str = "../../example_datasets/processed/gigahand/schunk/bimanual/p36-tea/10/trajectory_mjwpeq.npz",
    entity_root: str = "mujoco",
    fps: float = 50.0,
    spawn: bool = False,
    save_rrd: bool = False,
) -> None:
    xml_path = Path(xml).resolve()
    npz_path = Path(npz).resolve()

    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    rr.init("mujoco_xml_viewer")
    if spawn:
        rr.spawn()

    # bake scene for other usecase
    # export_scene_to_npz(xml_path, xml_path.with_suffix(".npz"))

    spec, model, body_entity_and_ids = build_and_log_scene(
        xml_path, entity_root=entity_root
    )
    play_trajectory(spec, model, npz_path, entity_root=entity_root, fps=fps)

    # save as rrd file
    if save_rrd:
        rr.save(f"{npz_path.with_suffix('.rrd')}")


if __name__ == "__main__":
    tyro.cli(main)
