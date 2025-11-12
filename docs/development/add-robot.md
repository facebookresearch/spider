# Add New Robot

This guide explains how to add a new robot to SPIDER, covering both dexterous hands and humanoid robots.

## Overview

Adding a new robot involves:
1. Converting/preparing robot model (URDF/MJCF)
2. Configuring default settings
3. Setting up collision geometry
4. Adding trace and tracking sites
5. Configuring actuators
6. Testing the integration

## Prerequisites

- Robot URDF or MJCF model
- Basic understanding of MuJoCo XML format
- Robot specifications (joint limits, actuator parameters)

## Add Dexterous Hand

### Step 1: Convert URDF to MJCF (Optional)

If you have a URDF file, convert it to MuJoCo XML:

```xml
<!-- Add this snippet to your URDF before conversion -->
<mujoco>
  <compiler meshdir="assets" balanceinertia="true" discardvisual="false"/>
</mujoco>
```

Then use MuJoCo's compiler:

```bash
# Using MuJoCo command-line tool
python -c "import mujoco; mujoco.MjModel.from_xml_path('robot.urdf').save('robot.xml')"
```

### Step 2: Add Default Settings

Add default simulation parameters to your XML:

```xml
<default>
  <!-- Disable all contacts by default, use explicit collision pairs -->
  <geom density="800" condim="1" contype="0" conaffinity="0" />

  <!-- Position controller settings (response time ~1s) -->
  <position kp="300" dampratio="1.0" inheritrange="1" />

  <!-- Joint dynamics -->
  <joint damping="0.0" armature="1.0" frictionloss="0.0" />

  <!-- Trace sites for visualization -->
  <site size="0.01" type="sphere" rgba="1 0 0 1" group="3" />
</default>
```

**Parameter explanations:**
- `density="800"`: Material density (kg/m³)
- `kp="300"`: Position controller stiffness
- `dampratio="1.0"`: Critical damping
- `armature="1.0"`: Joint inertia

### Step 3: Add Assets

Include textures and materials:

```xml
<asset>
  <!-- Skybox texture -->
  <texture type="skybox" builtin="gradient"
           rgb1="0.3 0.5 0.7" rgb2="0 0 0"
           width="512" height="3072" />

  <!-- Ground plane texture -->
  <texture type="2d" name="left_groundplane" builtin="checker"
           mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
           markrgb="0.8 0.8 0.8" width="300" height="300" />

  <!-- Ground material -->
  <material name="left_groundplane" texture="left_groundplane"
            texuniform="true" texrepeat="5 5" reflectance="0.2" />
</asset>

<!-- Camera for tracking -->
<camera name="left_track" pos="0.868 -0.348 -0.175"
        xyaxes="0.037 0.999 -0.000 0.011 -0.000 1.000" />
```

### Step 4: Clean Up Joint Properties

Remove incompatible joint properties from the URDF-converted file:

```xml
<!-- Remove these attributes from joints: -->
<!-- actuatorfrcrange, damping, frictionloss, armature, etc. -->

<!-- Before: -->
<joint name="finger_joint" damping="0.1" armature="0.01" ... />

<!-- After: -->
<joint name="finger_joint" ... />
<!-- Dynamics are now inherited from <default> section -->
```

### Step 5: Rename Bodies and Geoms

Change mesh names to avoid conflicts (especially for bimanual setups):

```xml
<!-- Left hand bodies should start with "left_" -->
<body name="left_palm">
  <geom name="left_palm_visual" type="mesh" mesh="left_palm_mesh" />
</body>

<!-- Right hand bodies should start with "right_" -->
<body name="right_palm">
  <geom name="right_palm_visual" type="mesh" mesh="right_palm_mesh" />
</body>
```

::: tip Naming Convention
- Bodies: `{side}_{finger}_{part}` (e.g., `left_index_proximal`)
- Visual geoms: `{side}_{name}_visual`
- Collision geoms: `collision_hand_{side}_{finger}_{part}`
:::

### Step 6: Add Collision Geometry

Create simplified collision geometry using primitives:

```xml
<body name="left_index_tip">
  <!-- Visual mesh -->
  <geom name="left_index_tip_visual" type="mesh" mesh="index_tip"
        contype="0" conaffinity="0" group="0" />

  <!-- Collision geometry (capsule primitive) -->
  <geom name="collision_hand_left_index_0" type="capsule"
        size="0.008 0.015" pos="0.015 0 0" quat="0.707 0 0.707 0"
        contype="1" conaffinity="1" group="3" rgba="0 1 0 1" />
</body>
```

**Naming convention for collision geoms:**
```
collision_hand_{side}_{finger}_{part}
```

Where:
- `side`: `left` or `right`
- `finger`: `thumb`, `index`, `middle`, `ring`, `pinky`, `palm`
- `part`: `0`, `1`, `2`, `3`, ... (from tip to base)

**Collision geom guidelines:**
- Use primitives (`box`, `sphere`, `cylinder`, `capsule`) instead of meshes
- Set `group="3"` for collision geoms
- Set `rgba="0 1 0 1"` (green) for visibility
- Keep collision geoms simple for performance

### Step 7: Add Trace Sites

Add trace sites for fingertip visualization:

```xml
<body name="left_thumb_tip">
  <!-- Trace site (for trajectory recording) -->
  <site name="trace_hand_left_thumb_tip" pos="0.02 0 0"
        size="0.005" type="sphere" rgba="1 0 0 1" group="3" />
</body>

<body name="left_index_tip">
  <site name="trace_hand_left_index_tip" pos="0.02 0 0"
        size="0.005" type="sphere" rgba="1 0 0 1" group="3" />
</body>

<!-- Repeat for middle, ring, pinky -->
```

**Naming convention:** `trace_hand_{side}_{finger}_tip`

### Step 8: Add Tracking Sites

Add tracking sites for end-effector control:

```xml
<body name="left_palm">
  <site name="left_palm" pos="0 0 0"
        euler="0 0 0" size="0.01" type="sphere" rgba="1 1 0 1" group="3" />
</body>

<body name="left_thumb_tip">
  <site name="left_thumb_tip" pos="0.02 0 0"
        size="0.01" type="sphere" rgba="0 1 1 1" group="3" />
</body>

<!-- Repeat for all fingertips -->
```

**Required sites:**
- `{side}_palm`: Palm center with proper orientation
- `{side}_thumb_tip`, `{side}_index_tip`, `{side}_middle_tip`
- `{side}_ring_tip`, `{side}_pinky_tip`

**Palm site orientation:**
- Z-axis: pointing forward (finger direction)
- X-axis: pointing down (toward ground)

![Base Definition](/figs/base_def.png)

### Step 9: Add Actuators

Define position actuators for all joints:

```xml
<actuator>
  <!-- Base DOFs (for free-floating hand) -->
  <position name="left_pos_x_position" joint="left_pos_x" kp="1000" />
  <position name="left_pos_y_position" joint="left_pos_y" kp="1000" />
  <position name="left_pos_z_position" joint="left_pos_z" kp="1000" />
  <position name="left_rot_x_position" joint="left_rot_x" kp="1000" />
  <position name="left_rot_y_position" joint="left_rot_y" kp="1000" />
  <position name="left_rot_z_position" joint="left_rot_z" kp="1000" />

  <!-- Thumb joints -->
  <position name="left_thumb_proximal_yaw_position"
            joint="left_thumb_proximal_yaw_joint" />
  <position name="left_thumb_proximal_pitch_position"
            joint="left_thumb_proximal_pitch_joint" />
  <position name="left_thumb_intermediate_position"
            joint="left_thumb_intermediate_joint" />
  <position name="left_thumb_distal_position"
            joint="left_thumb_distal_joint" />

  <!-- Index finger joints -->
  <position name="left_index_proximal_position"
            joint="left_index_proximal_joint" />
  <position name="left_index_intermediate_position"
            joint="left_index_intermediate_joint" />

  <!-- Repeat for middle, ring, pinky -->
</actuator>
```

**Important notes:**
- Base DOFs use higher `kp` (e.g., 1000) for stability
- Finger joints inherit `kp` from `<default>` section
- **Actuator order must match joint order** (SPIDER uses joint positions as control)

### Step 10: Example Complete Hand XML

Here's a minimal example structure:

```xml
<mujoco model="my_hand">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <option timestep="0.01" iterations="6" solver="CG" jacobian="auto">
    <flag eulerdamp="disable" />
  </option>

  <default>
    <geom density="800" condim="1" contype="0" conaffinity="0" />
    <position kp="300" dampratio="1.0" inheritrange="1" />
    <joint damping="0.0" armature="1.0" frictionloss="0.0" />
    <site size="0.01" type="sphere" rgba="1 0 0 1" group="3" />
  </default>

  <asset>
    <!-- Textures and materials -->
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <material name="metal" rgba="0.7 0.7 0.7 1" />

    <!-- Meshes -->
    <mesh name="left_palm_mesh" file="meshes/palm.obj" />
    <mesh name="left_thumb_tip_mesh" file="meshes/thumb_tip.obj" />
    <!-- ... more meshes -->
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>

    <!-- Hand body -->
    <body name="left_hand" pos="0 0 0.2">
      <!-- Base free joint for 6-DOF -->
      <freejoint name="left_base_joint"/>

      <!-- Palm -->
      <body name="left_palm">
        <geom name="left_palm_visual" type="mesh" mesh="left_palm_mesh" />
        <geom name="collision_hand_left_palm_0" type="box" size="0.04 0.03 0.01" />
        <site name="left_palm" pos="0 0 0" euler="0 0 0"/>

        <!-- Thumb -->
        <body name="left_thumb_proximal">
          <joint name="left_thumb_proximal_yaw_joint" axis="0 0 1" range="-0.5 0.5"/>
          <joint name="left_thumb_proximal_pitch_joint" axis="0 1 0" range="-0.2 1.0"/>
          <!-- ... thumb links ... -->
          <body name="left_thumb_tip">
            <site name="left_thumb_tip" pos="0.02 0 0"/>
            <site name="trace_hand_left_thumb_tip" pos="0.02 0 0"/>
          </body>
        </body>

        <!-- Index, middle, ring, pinky similar structure -->
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="left_pos_x_position" joint="left_pos_x" kp="1000"/>
    <!-- ... all actuators ... -->
  </actuator>

  <contact>
    <!-- Define collision pairs if needed -->
  </contact>
</mujoco>
```

## Add Humanoid Robot

Humanoid robots have similar requirements but with additional considerations.

### Step 1: Performance Optimization

Reduce solver iterations for faster simulation:

```xml
<option timestep="0.02" iterations="2" ls_iterations="10">
  <flag eulerdamp="disable" />
</option>

<visual>
  <global azimuth="135" elevation="-25" offwidth="1920" offheight="1080" />
  <quality shadowsize="8192" />
  <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0 0 0" />
  <scale forcewidth="0.25" contactwidth="0.4" contactheight="0.15"
         framelength="5" framewidth="0.3" />
  <rgba haze="0.15 0.25 0.35 1" force="1 0 0 1" />
</visual>

<custom>
  <numeric data="15" name="max_contact_points" />
  <numeric data="15" name="max_geom_pairs" />
</custom>
```

### Step 2: Default Settings

```xml
<default>
  <!-- Disable all contacts by default -->
  <geom contype="0" conaffinity="0" />

  <!-- Position controller (stronger for humanoid) -->
  <position kp="500" dampratio="1" inheritrange="1" />

  <!-- Joint dynamics -->
  <joint damping="0.0" armature="1.0" frictionloss="0.0" />
</default>
```

### Step 3: Add Collision Geoms for Feet

Add collision spheres to feet (4 contact points per foot):

```xml
<body name="left_foot">
  <!-- Visual mesh -->
  <geom name="left_foot_visual" type="mesh" mesh="left_foot_mesh"
        contype="0" conaffinity="0"/>

  <!-- Collision spheres -->
  <geom name="lf0" type="sphere" size="0.02" pos="-0.05 0.025 -0.03"
        priority="1" friction="2.0" condim="3" />
  <geom name="lf1" type="sphere" size="0.02" pos="-0.05 -0.025 -0.03"
        priority="1" friction="2.0" condim="3" />
  <geom name="lf2" type="sphere" size="0.02" pos="0.12 0.03 -0.03"
        priority="1" friction="2.0" condim="3" />
  <geom name="lf3" type="sphere" size="0.02" pos="0.12 -0.03 -0.03"
        priority="1" friction="2.0" condim="3" />
</body>

<!-- Similar for right foot: rf0, rf1, rf2, rf3 -->
```

### Step 4: Add Collision Geoms for Hands

```xml
<body name="left_hand">
  <!-- Visual mesh -->
  <geom name="left_hand_visual" type="mesh" mesh="left_hand_mesh"
        contype="0" conaffinity="0"/>

  <!-- Collision sphere -->
  <geom name="lh" type="sphere" size="0.05" pos="0.1 0.0 0.0" />
</body>

<!-- Similar for right hand: rh -->
```

### Step 5: Add Trace Sites

```xml
<body name="pelvis">
  <site name="trace_pelvis" size="0.01" pos="0 0 0" />
</body>

<body name="left_foot">
  <site name="trace_left_foot" size="0.01" pos="0 0 0" />
</body>

<body name="right_foot">
  <site name="trace_right_foot" size="0.01" pos="0 0 0" />
</body>

<body name="left_hand">
  <site name="trace_left_hand" size="0.01" pos="0 0 0" />
</body>

<body name="right_hand">
  <site name="trace_right_hand" size="0.01" pos="0 0 0" />
</body>
```

### Step 6: Add Contact Pairs

Define explicit contact pairs:

```xml
<contact>
  <!-- Left foot contacts -->
  <pair name="left_foot_floor0" geom1="lf0" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />
  <pair name="left_foot_floor1" geom1="lf1" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />
  <pair name="left_foot_floor2" geom1="lf2" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />
  <pair name="left_foot_floor3" geom1="lf3" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />

  <!-- Right foot contacts -->
  <pair name="right_foot_floor0" geom1="rf0" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />
  <pair name="right_foot_floor1" geom1="rf1" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />
  <pair name="right_foot_floor2" geom1="rf2" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />
  <pair name="right_foot_floor3" geom1="rf3" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />

  <!-- Hand contacts -->
  <pair name="left_hand_floor" geom1="lh" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />
  <pair name="right_hand_floor" geom1="rh" geom2="floor"
        solref="0.008 1" friction="1 1" condim="3" />
</contact>
```

### Step 7: Use Position Actuators

```xml
<actuator>
  <!-- Legs -->
  <position name="left_hip_pitch" joint="left_hip_pitch_joint" />
  <position name="left_hip_roll" joint="left_hip_roll_joint" />
  <position name="left_hip_yaw" joint="left_hip_yaw_joint" />
  <position name="left_knee" joint="left_knee_joint" />
  <position name="left_ankle_pitch" joint="left_ankle_pitch_joint" />
  <position name="left_ankle_roll" joint="left_ankle_roll_joint" />

  <!-- Arms -->
  <position name="left_shoulder_pitch" joint="left_shoulder_pitch_joint" />
  <position name="left_shoulder_roll" joint="left_shoulder_roll_joint" />
  <position name="left_elbow" joint="left_elbow_joint" />
  <!-- ... -->

  <!-- Repeat for right side -->
</actuator>
```

::: warning Important
Actuator order must match joint order in the model definition!
:::

### Step 8: Setup Scene

Add white background and tracking camera:

```xml
<statistic center="1.0 0.7 1.0" extent="0.8"/>

<asset>
  <!-- White skybox -->
  <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1"
           width="800" height="800"/>

  <!-- White groundplane -->
  <texture type="2d" name="groundplane" builtin="checker" mark="edge"
           rgb1="1 1 1" rgb2="1 1 1" markrgb="1 1 1"
           width="300" height="300"/>
  <material name="groundplane" texture="groundplane" texuniform="true"
            texrepeat="5 5" reflectance="0"/>
</asset>

<worldbody>
  <!-- Floor -->
  <geom name="floor" size="0 0 0.01" type="plane" material="groundplane"
        contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>

  <!-- Directional light -->
  <light pos="0 0 5" dir="0 0 -1" type="directional"/>

  <!-- Tracking camera -->
  <camera name="track" pos="1.734 -1.135 .35"
          xyaxes="0.552 0.834 -0.000 -0.170 0.112 0.979"
          mode="trackcom"/>
</worldbody>
```

## Testing Your Robot

### Step 1: Visualize in MuJoCo

```python
import mujoco
import mujoco.viewer

# Load model
model = mujoco.MjModel.from_xml_path('my_robot.xml')
data = mujoco.MjData(model)

# Launch viewer
mujoco.viewer.launch(model, data)
```

Check:
- [ ] Model loads without errors
- [ ] All joints move freely
- [ ] Collision geometry looks correct (green geoms)
- [ ] Trace sites are visible
- [ ] Tracking sites are positioned correctly

### Step 2: Test with SPIDER

Create a simple test scene:

```bash
# Test IK (for hands)
uv run spider/preprocess/ik.py \
  --robot-type=my_hand \
  --task=test_task \
  --dataset-name=test \
  --data-id=0 \
  --embodiment-type=right

# Test physics optimization
uv run examples/run_mjwp.py \
  robot_type=my_hand \
  task=test_task \
  max_sim_steps=100
```

### Step 3: Verify Output

Check that:
- [ ] Trajectory is saved successfully
- [ ] Joint positions are within limits
- [ ] No collision warnings
- [ ] Video renders correctly

## Common Issues

### Issue 1: Model Doesn't Load

**Error:** `XML parse error`

**Solutions:**
- Check XML syntax
- Validate with MuJoCo's `simulate` tool
- Remove incompatible URDF attributes

### Issue 2: Unstable Simulation

**Symptoms:** Robot falls through floor, explodes

**Solutions:**
```xml
<!-- Increase solver iterations -->
<option iterations="10" ls_iterations="20"/>

<!-- Add more collision points -->
<!-- Adjust contact parameters -->
<pair solref="0.01 1" friction="2 2"/>

<!-- Check mass distribution -->
<!-- Ensure reasonable inertia values -->
```

### Issue 3: Actuators Don't Work

**Symptoms:** Robot doesn't move

**Solutions:**
- Check actuator names match joint names
- Verify `kp` values are reasonable (100-1000)
- Ensure joints have proper ranges
- Check actuator order matches joint order

### Issue 4: Poor IK Results

**Symptoms:** IK fails or produces bad poses

**Solutions:**
- Check tracking site orientations (especially palm)
- Verify joint limits are correct
- Add more trace sites for debugging
- Tune IK weights in config

## Robot Configuration Files

### Directory Structure

```
spider/assets/robots/
├── my_hand/
│   ├── left.xml          # Left hand only
│   ├── right.xml         # Right hand only
│   ├── bimanual.xml      # Both hands
│   ├── meshes/
│   │   ├── palm.obj
│   │   ├── thumb_tip.obj
│   │   └── ...
│   └── textures/
│       └── hand_texture.png
```

### Register Robot

Add robot to SPIDER's config:

```python
# In spider/config.py or appropriate config file

ROBOT_TYPES = {
    "allegro": "spider/assets/robots/allegro",
    "inspire": "spider/assets/robots/inspire",
    "my_hand": "spider/assets/robots/my_hand",  # Add your robot
    # ...
}
```

## Next Steps

After adding your robot:

1. **Test thoroughly** with multiple tasks
2. **Document** robot-specific parameters
3. **Tune** IK and optimization parameters
4. **Share** your robot configuration (PR welcome!)
