# AirSim -> ROS -> 3DGS Coordinate Guide

This document explains exactly how the current project converts AirSim data into the camera and point-cloud format expected by 3DGS / FastGS.

## 1. What each system means

AirSim camera pose is read from `simGetImages()`.

- AirSim world uses NED:
  - `+X`: north
  - `+Y`: east
  - `+Z`: down

ROS mapping in this project uses ENU world and ROS optical camera axes:

- ROS `world` uses ENU:
  - `+X`: east / forward in the map
  - `+Y`: north / left in the map
  - `+Z`: up
- ROS `camera_optical_frame` uses the standard optical convention:
  - `+X`: right in the image
  - `+Y`: down in the image
  - `+Z`: forward from the camera

COLMAP and 3DGS also expect the camera frame to be:

- `+X` right
- `+Y` down
- `+Z` forward

That means `camera_optical_frame` is already the correct camera coordinate system for COLMAP-style extrinsics and for 3DGS initialization.

## 2. The real transform chain in this repository

The chain comes from two places:

- `scripts/airsim_frontend.py`
- `launch/airsim_mapping.launch`

### 2.1 AirSim NED -> ROS ENU

In `airsim_frontend.py`, the position is converted as:

```python
pos_enu = (p.x_val, -p.y_val, -p.z_val)
ori_enu = (o.x_val, -o.y_val, -o.z_val, o.w_val)
```

This is the matrix form:

```text
R_ned_to_enu =
[ 1  0  0 ]
[ 0 -1  0 ]
[ 0  0 -1 ]
```

So for a point in AirSim world:

```text
p_world_enu = R_ned_to_enu * p_world_ned
```

For orientation, the same basis change is applied:

```text
R_world_enu_base = R_ned_to_enu * R_world_ned_base * R_ned_to_enu^T
```

The quaternion component change in code,

```text
(x, y, z, w) -> (x, -y, -z, w)
```

is exactly this basis change.

### 2.2 ROS `base_link` -> `camera_optical_frame`

In `airsim_mapping.launch` there is a static transform:

```xml
<node pkg="tf2_ros" type="static_transform_publisher"
      args="0 0 0 -1.5708 0 -1.5708 base_link camera_optical_frame" />
```

In matrix form, this is:

```text
R_base_to_camera =
[ 0  0  1 ]
[-1  0  0 ]
[ 0 -1  0 ]
```

This means:

- camera right = `-base_link y`
- camera down = `-base_link z`
- camera forward = ` base_link x`

This is the standard ROS optical camera rotation.

### 2.3 Why the position does not change from `base_link` to camera

The static transform has zero translation:

```text
t_base_to_camera = [0, 0, 0]
```

So in the current project:

- camera center and `base_link` center are treated as the same point
- only the axes are rotated

This is important: if you later add a real camera offset on the drone, you must also add that translation into the export script.

## 3. The one pose that 3DGS actually needs

3DGS needs the camera pose in the world, or equivalently the world-to-camera extrinsic.

The correct camera pose is:

```text
R_world_camera = R_world_base * R_base_to_camera
t_world_camera = t_world_base
```

Because translation is zero in the current launch file.

If you build a camera-to-world matrix:

```text
C2W =
[ R_world_camera  t_world_camera ]
[      0               1        ]
```

then COLMAP-style world-to-camera is:

```text
W2C = inverse(C2W)
R_w2c = R_world_camera^T
t_w2c = -R_world_camera^T * t_world_camera
```

This `R_w2c` / `t_w2c` is exactly what goes into `images.txt`.

## 4. The point-cloud rule used in this project

In the current workflow, the exporter assumes `/airsim/point_cloud` is already in ROS `world`.

That means:

- the point cloud is accumulated directly as a world-space colored point cloud
- no extra camera-to-world transform is applied to the cloud during export

This matches the intended usage in this repository:

- images need a camera pose for 3DGS
- the collected point cloud is already the initialization point cloud in the global map frame

If a future pipeline publishes `/airsim/point_cloud` in camera coordinates instead, the exporter must be updated accordingly. The current exporter does not do that conversion.

## 5. The most common mistake

The most common error is to mix up:

- the world-space point cloud
- the image camera pose
- the body pose

The exporter therefore treats them separately:

1. `/airsim/point_cloud` is collected directly as `world` points
2. the image pose is exported independently
3. by default the image pose is built from `/airsim/pose` plus the fixed `base_link -> camera_optical_frame` rotation
4. if your `/airsim/pose` is already the optical camera pose, the exporter can be switched to use it directly

## 6. What the export script writes

The exporter added for this workflow writes:

- `images/*.png`
- `sparse/0/cameras.txt`
- `sparse/0/images.txt`
- `sparse/0/points3D.txt`
- `input.ply`
- `cameras.json`

So the output can be consumed either as:

- a COLMAP text model
- a FastGS-style camera list plus a colored PLY point cloud

## 7. Practical summary

If you remember only one thing, remember this:

The point cloud and the image pose do not need to be produced the same way.

In this repository, the practical rule is:

- collect `/airsim/point_cloud` directly as world-space geometry
- export image poses in the camera convention expected by 3DGS

The provided exporter supports both pose interpretations:

- `/airsim/pose = world -> base_link`
- `/airsim/pose = world -> camera_optical_frame`

The choice is controlled by a ROS parameter so you do not need to rewrite the script when the upstream publisher changes.
