# Port new robot

## Add humanoid

This guide explains how to optimize a humanoid robot MJCF file for performance in MuJoCo simulations.

### 1. Performance optimization

Reduce solver iterations for faster simulation:

```xml
<!-- performance options -->
<option timestep="0.02" iterations="2" ls_iterations="10">
  <flag eulerdamp="disable" />
</option>

<visual>
  <global azimuth="135" elevation="-25" offwidth="1920" offheight="1080" />
  <quality shadowsize="8192" />
  <headlight ambient="0.3 0.3 0.3" diffuse="0.6 0.6 0.6" specular="0 0 0" />
  <scale forcewidth="0.25" contactwidth="0.4" contactheight="0.15" framelength="5"
    framewidth="0.3" />
  <rgba haze="0.15 0.25 0.35 1" force="1 0 0 1" />
</visual>

<custom>
  <numeric data="15" name="max_contact_points" />
  <numeric data="15" name="max_geom_pairs" />
</custom>
```

### 2. Contact speed up and actuator defaults

Disable all contacts by default (especially for visual meshes) and enable only specific collision pairs. Also set up position controller defaults:

```xml
<default>
  <geom contype="0" conaffinity="0" />
  <position kp="500" dampratio="1" inheritrange="1" />
  <joint damping="0.0" armature="1.0" frictionloss="0.0" />
</default>
```

**Important:**
- All mesh geoms will inherit `contype="0" conaffinity="0"` from this default, which disables their collision by default. Only collision geoms that explicitly enable contact will participate in collisions.
- The `<position kp="500" dampratio="1" inheritrange="1" />` sets default parameters for position actuators (PD controller with kp=500, damping ratio=1).

### 3. Add collision geoms

Add collision spheres to feet (4 contact points per foot for stability):

```xml
<!-- Left foot collision geoms -->
<geom name="lf0" type="sphere" size="0.02" pos="-0.05 0.025 -0.03" priority="1" friction="2.0" condim="3" />
<geom name="lf1" type="sphere" size="0.02" pos="-0.05 -0.025 -0.03" priority="1" friction="2.0" condim="3" />
<geom name="lf2" type="sphere" size="0.02" pos="0.12 0.03 -0.03" priority="1" friction="2.0" condim="3" />
<geom name="lf3" type="sphere" size="0.02" pos="0.12 -0.03 -0.03" priority="1" friction="2.0" condim="3" />

<!-- Right foot collision geoms -->
<geom name="rf0" type="sphere" size="0.02" pos="-0.05 0.025 -0.03" priority="1" friction="2.0" condim="3" />
<geom name="rf1" type="sphere" size="0.02" pos="-0.05 -0.025 -0.03" priority="1" friction="2.0" condim="3" />
<geom name="rf2" type="sphere" size="0.02" pos="0.12 0.03 -0.03" priority="1" friction="2.0" condim="3" />
<geom name="rf3" type="sphere" size="0.02" pos="0.12 -0.03 -0.03" priority="1" friction="2.0" condim="3" />
```

Add collision spheres to hands (single contact point per hand):

```xml
<!-- Left hand collision geom -->
<geom name="lh" type="sphere" size="0.05" pos="0.1 0.0 0.0" />

<!-- Right hand collision geom -->
<geom name="rh" type="sphere" size="0.05" pos="0.1 0.0 0.0" />
```

**Note:** Collision geoms do not specify `contype` or `conaffinity`, so they inherit the default values. However, they do specify contact parameters like `priority`, `friction`, and `condim` to enable contact.

### 4. Add trace sites

Add trace sites for tracking key body parts during simulation (useful for visualization and debugging):

```xml
<!-- Pelvis trace site -->
<site name="trace_pelvis" size="0.01" pos="0 0 0" />

<!-- Foot trace sites -->
<site name="trace_left_foot" size="0.01" pos="0 0 0" />
<site name="trace_right_foot" size="0.01" pos="0 0 0" />

<!-- Hand trace sites -->
<site name="trace_left_hand" size="0.01" pos="0 0 0" />
<site name="trace_right_hand" size="0.01" pos="0 0 0" />
```

**Tip:** Use `pos="0 0 0"` for trace sites as the position is relative to the parent body.

### 5. Add contact pairs

Define explicit contact pairs between collision geoms and the floor:

```xml
<contact>
  <!-- Left foot contacts -->
  <pair name="left_foot_floor0" geom1="lf0" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />
  <pair name="left_foot_floor1" geom1="lf1" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />
  <pair name="left_foot_floor2" geom1="lf2" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />
  <pair name="left_foot_floor3" geom1="lf3" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />

  <!-- Right foot contacts -->
  <pair name="right_foot_floor0" geom1="rf0" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />
  <pair name="right_foot_floor1" geom1="rf1" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />
  <pair name="right_foot_floor2" geom1="rf2" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />
  <pair name="right_foot_floor3" geom1="rf3" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />

  <!-- Hand contacts -->
  <pair name="left_hand_floor" geom1="lh" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />
  <pair name="right_hand_floor" geom1="rh" geom2="floor" solref="0.008 1" friction="1 1" condim="3" />
</contact>
```

### 6. Use position actuators

Use position actuators (not motor actuators) for all joints. The position actuator parameters are inherited from the default setting:

```xml
<actuator>
  <position name="left_hip_pitch" joint="left_hip_pitch_joint" />
  <position name="left_hip_roll" joint="left_hip_roll_joint" />
  <position name="left_knee" joint="left_knee_joint" />
  <!-- ... other joints ... -->
</actuator>
```

**Note:** Position actuators use the default `kp="500" dampratio="1"` settings from the `<default>` section, providing PD control for joint positions.

**Note:** make sure motor definition follow the same order as the joint definition since we are using joint position as initial control guess.

### 7. Setup scene with white background

Add scene setup with white skybox, white checker groundplane, directional light, and tracking camera:

```xml
<!-- setup scene -->
<statistic center="1.0 0.7 1.0" extent="0.8"/>
<asset>
  <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
  <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="1 1 1" width="300" height="300"/>
  <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
</asset>
<worldbody>
  <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
  <light pos="0 0 5" dir="0 0 -1" type="directional"/>
  <camera name="track" pos="1.734 -1.135 .35" xyaxes="0.552 0.834 -0.000 -0.170 0.112 0.979" mode="trackcom"/>
</worldbody>
```

**Note:** This creates a clean white background with directional lighting from above and a tracking camera that follows the robot's center of mass.

### Summary

The key optimizations are:
1. **Reduce solver iterations** (iterations=2, ls_iterations=10)
2. **Disable all collisions by default** using `<default><geom contype="0" conaffinity="0" /></default>` and set position controller defaults
3. **Add explicit collision geoms** for feet (4 spheres per foot) and hands (1 sphere per hand)
4. **Add trace sites** for tracking key body parts
5. **Define explicit contact pairs** between collision geoms and the floor
6. **Use position actuators** for all joints with PD control
7. **Setup clean scene** with white background, directional light, and tracking camera

This approach significantly speeds up simulation by minimizing unnecessary collision checks while maintaining accurate contact simulation where needed.
