#!/usr/bin/env python3
"""
Convert recorded data to COLMAP format.

This script converts the recorded RGB images, poses, and point clouds
to COLMAP-compatible format for 3D reconstruction.

Usage:
    # Read paths from yaml config (recommended)
    python3 convert_to_colmap.py --config /path/to/factory.yaml

    # Or specify paths manually
    python3 convert_to_colmap.py --input /path/to/recorded_data --output /path/to/colmap_output

Input directory structure:
    input_dir/
        images/
            000000.png
            000001.png
            ...
        pointcloud_colored/
            000000.pcd
            000001.pcd
            ...
        pointcloud_nocolor/
            000000.pcd
            ...
        poses.txt
        camera_intrinsics.txt

Output directory structure (COLMAP format):
    output_dir/
        images/
            000000.png
            ...
        sparse/0/
            cameras.txt (or cameras.bin)
            images.txt (or images.bin)
            points3D.txt (or points3D.bin)
        pointcloud/
            merged_colored.ply
            merged_nocolor.ply
"""

import argparse
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
import struct
import yaml


def read_camera_intrinsics(intrinsics_file):
    """Read camera intrinsics from file."""
    with open(intrinsics_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) >= 6:
            fx, fy, cx, cy = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            width, height = int(parts[4]), int(parts[5])
            return fx, fy, cx, cy, width, height

    raise ValueError("Could not parse camera intrinsics file")


def read_poses(poses_file):
    """Read poses from file.

    Format: frame_id tx ty tz qx qy qz qw timestamp
    Returns: dict of {frame_id: (position, quaternion, timestamp)}
    """
    poses = {}
    with open(poses_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                frame_id = parts[0]
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                timestamp = float(parts[8]) if len(parts) > 8 else 0.0
                poses[frame_id] = {
                    'position': np.array([tx, ty, tz]),
                    'quaternion': np.array([qx, qy, qz, qw]),  # xyzw format
                    'timestamp': timestamp
                }
    return poses


def quaternion_to_rotation_matrix(q):
    """Convert quaternion (xyzw) to rotation matrix."""
    r = R.from_quat(q)  # scipy uses xyzw format
    return r.as_matrix()


def get_ros_to_colmap_transform():
    """Get transformation matrix from ROS coordinate system to COLMAP/OpenCV coordinate system.

    ROS Convention (for cameras, typically):
        - X: forward (optical axis)
        - Y: left
        - Z: up
        - Right-handed coordinate system

    COLMAP/OpenCV Convention:
        - X: right
        - Y: down
        - Z: forward (optical axis)
        - Right-handed coordinate system

    Transformation:
        X_colmap = Y_ros    (ROS left → COLMAP right requires negation)
        Y_colmap = Z_ros    (ROS up → COLMAP down requires negation)
        Z_colmap = X_ros    (ROS forward → COLMAP forward)

    Actually, standard ROS camera frame is:
        - X: right
        - Y: down
        - Z: forward
    Which is already the same as OpenCV/COLMAP!

    BUT if your odometry is in a different frame (e.g., body frame with Z-up),
    you need to transform it first.

    Common ROS body frame (e.g., quadrotor):
        - X: forward
        - Y: left
        - Z: up

    Returns:
        3x3 rotation matrix to transform from ROS body frame to COLMAP camera frame
    """
    # Transformation from ROS body frame (X:forward, Y:left, Z:up)
    # to COLMAP camera frame (X:right, Y:down, Z:forward)
    #
    # [ X_colmap ]   [  0  -1   0 ] [ X_ros ]
    # [ Y_colmap ] = [  0   0  -1 ] [ Y_ros ]
    # [ Z_colmap ]   [  1   0   0 ] [ Z_ros ]

    T_ros_to_colmap = np.array([
        [ 0, -1,  0],  # X_colmap = -Y_ros (left becomes right)
        [ 0,  0, -1],  # Y_colmap = -Z_ros (up becomes down)
        [ 1,  0,  0]   # Z_colmap = X_ros (forward stays forward)
    ])

    return T_ros_to_colmap


def world_to_camera_pose(position, quaternion, apply_coordinate_transform=True):
    """Convert ROS odometry pose to COLMAP camera pose.

    CRITICAL UNDERSTANDING:
    =======================

    Step 1: Coordinate System Transformation
    -----------------------------------------
    ROS Body Frame (common for robots/drones):
        - X: forward, Y: left, Z: up

    COLMAP/OpenCV Camera Frame:
        - X: right, Y: down, Z: forward

    We need to transform the camera pose from ROS body convention
    to COLMAP camera convention.

    Step 2: Pose Inversion
    ----------------------
    ROS Odometry:
        - position: Camera center in world frame (C_world)
        - quaternion: Body-to-world rotation (R_body_to_world)
        - Meaning: p_world = R_body_to_world * p_body + C_world

    COLMAP images.txt:
        - Stores world-to-camera transformation
        - QW QX QY QZ TX TY TZ
        - Meaning: p_camera = R_world_to_camera * p_world + T_world_to_camera
        - Camera center: C_world = -R_world_to_camera^T * T_world_to_camera

    Conversion Algorithm:
    ---------------------
    Given: (C_world, R_body_to_world) in ROS body frame

    1. Get coordinate transformation matrix T_body_to_cam
    2. Transform rotation: R_cam_to_world = R_body_to_world * T_body_to_cam^T
    3. Invert rotation: R_world_to_cam = R_cam_to_world^T
    4. Compute translation: T_world_to_cam = -R_world_to_cam * C_world
    5. Convert to quaternion (WXYZ format)

    Args:
        position: Camera center in world frame [x, y, z] from ROS
        quaternion: Body-to-world quaternion [qx, qy, qz, qw] from ROS
        apply_coordinate_transform: If True, apply ROS→COLMAP coordinate transform

    Returns:
        (q_wxyz, t_vec): COLMAP format quaternion and translation
    """
    # Step 1: Get rotation matrix from ROS quaternion (body-to-world)
    R_body_to_world = quaternion_to_rotation_matrix(quaternion)
    C_world = position

    if apply_coordinate_transform:
        # Get transformation from ROS body frame to COLMAP camera frame
        T_body_to_cam = get_ros_to_colmap_transform()

        # Transform the orientation to camera frame
        # R_cam_to_world = R_body_to_world * T_body_to_cam^T
        R_cam_to_world = R_body_to_world @ T_body_to_cam.T
    else:
        # No coordinate transformation (assume frames match)
        R_cam_to_world = R_body_to_world

    # Step 2: Invert to get world-to-camera rotation (required by COLMAP)
    R_world_to_cam = R_cam_to_world.T

    # Step 3: Compute world-to-camera translation
    # From COLMAP formula: C_world = -R_world_to_cam^T * T_world_to_cam
    # We get: T_world_to_cam = -R_world_to_cam * C_world
    T_world_to_cam = -R_world_to_cam @ C_world

    # Step 4: Convert rotation to quaternion in WXYZ format
    r = R.from_matrix(R_world_to_cam)
    q_xyzw = r.as_quat()  # scipy returns [qx, qy, qz, qw]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    return q_wxyz, T_world_to_cam


def write_cameras_txt(output_path, fx, fy, cx, cy, width, height):
    """Write cameras.txt in COLMAP format."""
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        # Using PINHOLE model: fx, fy, cx, cy
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")


def write_images_txt(output_path, poses, image_dir, verbose=False, apply_coord_transform=True):
    """Write images.txt in COLMAP format.

    Args:
        output_path: Path to output images.txt
        poses: Dictionary of poses
        image_dir: Directory containing images
        verbose: If True, print debug information for coordinate transformation
        apply_coord_transform: If True, apply ROS→COLMAP coordinate transformation
    """
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}\n")

        for idx, (frame_id, pose_data) in enumerate(sorted(poses.items()), start=1):
            image_name = f"{frame_id}.png"
            image_path = os.path.join(image_dir, image_name)

            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping")
                continue

            # Convert pose to COLMAP format
            q_colmap, t_colmap = world_to_camera_pose(
                pose_data['position'],
                pose_data['quaternion'],
                apply_coordinate_transform=apply_coord_transform
            )

            # Debug output for first few frames
            if verbose and idx <= 3:
                print(f"\n--- Frame {frame_id} (ID={idx}) ---")
                print(f"  ROS Input:")
                print(f"    Position (world):   [{pose_data['position'][0]:.4f}, {pose_data['position'][1]:.4f}, {pose_data['position'][2]:.4f}]")
                print(f"    Quaternion (xyzw):  [{pose_data['quaternion'][0]:.4f}, {pose_data['quaternion'][1]:.4f}, {pose_data['quaternion'][2]:.4f}, {pose_data['quaternion'][3]:.4f}]")
                print(f"  COLMAP Output (coord_transform={apply_coord_transform}):")
                print(f"    Quaternion (wxyz):  [{q_colmap[0]:.4f}, {q_colmap[1]:.4f}, {q_colmap[2]:.4f}, {q_colmap[3]:.4f}]")
                print(f"    Translation:        [{t_colmap[0]:.4f}, {t_colmap[1]:.4f}, {t_colmap[2]:.4f}]")
                # Verify by computing camera center
                R_w2c = quaternion_to_rotation_matrix([q_colmap[1], q_colmap[2], q_colmap[3], q_colmap[0]])
                C_computed = -R_w2c.T @ t_colmap
                print(f"    Camera center (verify): [{C_computed[0]:.4f}, {C_computed[1]:.4f}, {C_computed[2]:.4f}]")

            # Write image line
            f.write(f"{idx} {q_colmap[0]} {q_colmap[1]} {q_colmap[2]} {q_colmap[3]} ")
            f.write(f"{t_colmap[0]} {t_colmap[1]} {t_colmap[2]} 1 {image_name}\n")
            # Empty line for 2D points (we don't have feature matches)
            f.write("\n")


def write_points3d_txt(output_path, points=None, colors=None):
    """Write points3D.txt in COLMAP format.

    Args:
        output_path: Path to output file
        points: Nx3 numpy array of 3D points (optional)
        colors: Nx3 numpy array of RGB colors in [0,1] range (optional)
    """
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        if points is None or len(points) == 0:
            f.write("# Number of points: 0\n")
            return

        f.write(f"# Number of points: {len(points)}\n")

        for i, pt in enumerate(points):
            point_id = i + 1
            x, y, z = pt[0], pt[1], pt[2]

            # Get color (default to gray if no colors)
            if colors is not None and i < len(colors):
                r = int(colors[i][0] * 255)
                g = int(colors[i][1] * 255)
                b = int(colors[i][2] * 255)
            else:
                r, g, b = 128, 128, 128

            # Error is set to 0, no track information
            error = 0.0
            f.write(f"{point_id} {x} {y} {z} {r} {g} {b} {error}\n")


def read_pcd_file(pcd_path):
    """Read PCD file and return points as numpy array.

    Returns: (points, colors) where colors may be None for XYZ-only point clouds
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        return points, colors
    except ImportError:
        # Fallback: simple PCD parser for binary/ascii
        return read_pcd_simple(pcd_path)


def read_pcd_simple(pcd_path):
    """Simple PCD file reader (fallback if Open3D not available)."""
    points = []
    colors = []
    has_rgb = False
    data_type = 'ascii'
    num_points = 0

    with open(pcd_path, 'rb') as f:
        # Read header
        while True:
            line = f.readline().decode('utf-8').strip()
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
                has_rgb = 'rgb' in fields or ('r' in fields and 'g' in fields and 'b' in fields)
            elif line.startswith('POINTS'):
                num_points = int(line.split()[1])
            elif line.startswith('DATA'):
                data_type = line.split()[1]
                break

        if data_type == 'ascii':
            for _ in range(num_points):
                line = f.readline().decode('utf-8').strip()
                parts = line.split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    if has_rgb and len(parts) >= 6:
                        colors.append([float(parts[3])/255, float(parts[4])/255, float(parts[5])/255])
        else:
            # Binary format - more complex, skip for now
            print(f"Warning: Binary PCD not fully supported, skipping {pcd_path}")
            return np.array([]), None

    return np.array(points), np.array(colors) if colors else None


def merge_point_clouds(pcd_dir, output_ply_path, colored=True):
    """Merge all PCD files in directory and save as PLY."""
    try:
        import open3d as o3d

        all_points = []
        all_colors = []

        pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])

        for pcd_file in pcd_files:
            pcd_path = os.path.join(pcd_dir, pcd_file)
            pcd = o3d.io.read_point_cloud(pcd_path)

            if len(pcd.points) > 0:
                all_points.append(np.asarray(pcd.points))
                if colored and pcd.has_colors():
                    all_colors.append(np.asarray(pcd.colors))

        if all_points:
            merged_points = np.vstack(all_points)
            merged_pcd = o3d.geometry.PointCloud()
            merged_pcd.points = o3d.utility.Vector3dVector(merged_points)

            if colored and all_colors:
                merged_colors = np.vstack(all_colors)
                merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

            # Downsample to reduce size
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.01)

            o3d.io.write_point_cloud(output_ply_path, merged_pcd)
            print(f"Saved merged point cloud: {output_ply_path} ({len(merged_pcd.points)} points)")

    except ImportError:
        print("Warning: Open3D not available, skipping point cloud merging")


def merge_colored_pointclouds_for_colmap(pcd_dir, voxel_size=0.05):
    """Merge all colored PCD files and downsample for COLMAP points3D.txt.

    Args:
        pcd_dir: Directory containing PCD files
        voxel_size: Voxel size for downsampling (default 0.05m)

    Returns:
        (points, colors): Tuple of numpy arrays, or (None, None) if failed
    """
    try:
        import open3d as o3d

        if not os.path.exists(pcd_dir):
            print(f"Warning: Point cloud directory {pcd_dir} does not exist")
            return None, None

        pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
        if not pcd_files:
            print(f"Warning: No PCD files found in {pcd_dir}")
            return None, None

        all_points = []
        all_colors = []

        print(f"Merging {len(pcd_files)} colored point cloud files...")
        for pcd_file in pcd_files:
            pcd_path = os.path.join(pcd_dir, pcd_file)
            pcd = o3d.io.read_point_cloud(pcd_path)

            if len(pcd.points) > 0:
                all_points.append(np.asarray(pcd.points))
                if pcd.has_colors():
                    all_colors.append(np.asarray(pcd.colors))

        if not all_points:
            print("Warning: No valid points found in PCD files")
            return None, None

        # Merge all points
        merged_points = np.vstack(all_points)
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)

        if all_colors:
            merged_colors = np.vstack(all_colors)
            merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        print(f"Total points before downsampling: {len(merged_pcd.points)}")

        # Downsample
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)

        print(f"Points after downsampling (voxel_size={voxel_size}m): {len(merged_pcd.points)}")

        points = np.asarray(merged_pcd.points)
        colors = np.asarray(merged_pcd.colors) if merged_pcd.has_colors() else None

        return points, colors

    except ImportError:
        print("Warning: Open3D not available, cannot merge point clouds")
        return None, None


def write_ply_simple(output_path, points, colors=None):
    """Write PLY file without Open3D dependency."""
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i, pt in enumerate(points):
            if colors is not None:
                c = colors[i]
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}\n")
            else:
                f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")


def read_yaml_config(yaml_path):
    """Read configuration from yaml file.

    Returns: dict with keys 'save_path', 'cam_fx', 'cam_fy', 'cam_cx', 'cam_cy', 'cam_width', 'cam_height'
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    result = {
        'save_path': config.get('data_record/save_path', '/tmp/colmap_data'),
        'cam_fx': config.get('data_record/cam_fx', 157.05),
        'cam_fy': config.get('data_record/cam_fy', 157.05),
        'cam_cx': config.get('data_record/cam_cx', 272.0),
        'cam_cy': config.get('data_record/cam_cy', 272.0),
        'cam_width': config.get('data_record/cam_width', 544),
        'cam_height': config.get('data_record/cam_height', 544),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Convert recorded data to COLMAP format')
    parser.add_argument('--config', '-c', help='Path to yaml config file (e.g., factory.yaml)')
    parser.add_argument('--input', '-i', help='Input directory with recorded data (overrides config)')
    parser.add_argument('--output', '-o', help='Output directory for COLMAP format (default: input_dir/colmap)')
    parser.add_argument('--merge-pointcloud', action='store_true', help='Merge point clouds into single PLY')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print debug information for coordinate transformations')
    parser.add_argument('--no-coord-transform', action='store_true',
                       help='Disable ROS→COLMAP coordinate transformation (use if frames already match)')
    args = parser.parse_args()

    # Determine input directory
    input_dir = None
    yaml_config = None

    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file {args.config} does not exist")
            return
        yaml_config = read_yaml_config(args.config)
        input_dir = yaml_config['save_path']
        print(f"Read config from: {args.config}")
        print(f"  save_path: {input_dir}")

    if args.input:
        input_dir = args.input

    if not input_dir:
        # Try to find default config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_config = os.path.join(script_dir, '..', 'config', 'factory.yaml')
        if os.path.exists(default_config):
            yaml_config = read_yaml_config(default_config)
            input_dir = yaml_config['save_path']
            print(f"Using default config: {default_config}")
            print(f"  save_path: {input_dir}")
        else:
            print("Error: No input specified. Use --config or --input")
            return

    # Determine output directory
    output_dir = args.output if args.output else os.path.join(input_dir, 'colmap')

    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return

    required_files = ['poses.txt']
    required_dirs = ['images']

    for f in required_files:
        if not os.path.exists(os.path.join(input_dir, f)):
            print(f"Error: Required file {f} not found in {input_dir}")
            return

    for d in required_dirs:
        if not os.path.exists(os.path.join(input_dir, d)):
            print(f"Error: Required directory {d} not found in {input_dir}")
            return

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sparse', '0'), exist_ok=True)

    # Read camera intrinsics (from file or yaml config)
    print("Reading camera intrinsics...")
    intrinsics_file = os.path.join(input_dir, 'camera_intrinsics.txt')
    if os.path.exists(intrinsics_file):
        fx, fy, cx, cy, width, height = read_camera_intrinsics(intrinsics_file)
        print(f"  From file: fx={fx}, fy={fy}, cx={cx}, cy={cy}, {width}x{height}")
    elif yaml_config:
        fx = yaml_config['cam_fx']
        fy = yaml_config['cam_fy']
        cx = yaml_config['cam_cx']
        cy = yaml_config['cam_cy']
        width = yaml_config['cam_width']
        height = yaml_config['cam_height']
        print(f"  From yaml: fx={fx}, fy={fy}, cx={cx}, cy={cy}, {width}x{height}")
    else:
        print("Error: No camera intrinsics found (neither file nor yaml config)")
        return

    print("Reading poses...")
    poses = read_poses(os.path.join(input_dir, 'poses.txt'))
    print(f"  Found {len(poses)} poses")

    # Copy images
    print("Copying images...")
    src_image_dir = os.path.join(input_dir, 'images')
    dst_image_dir = os.path.join(output_dir, 'images')
    for img_file in os.listdir(src_image_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(
                os.path.join(src_image_dir, img_file),
                os.path.join(dst_image_dir, img_file)
            )

    # Write COLMAP files
    sparse_dir = os.path.join(output_dir, 'sparse', '0')

    print("Writing cameras.txt...")
    write_cameras_txt(os.path.join(sparse_dir, 'cameras.txt'), fx, fy, cx, cy, width, height)

    apply_coord_transform = not args.no_coord_transform
    if args.verbose:
        print(f"\nCoordinate transformation: {'ENABLED' if apply_coord_transform else 'DISABLED'}")
        if apply_coord_transform:
            print("  Converting from ROS body frame (X:forward, Y:left, Z:up)")
            print("  to COLMAP camera frame (X:right, Y:down, Z:forward)")

    print("Writing images.txt...")
    write_images_txt(os.path.join(sparse_dir, 'images.txt'), poses, dst_image_dir,
                     verbose=args.verbose, apply_coord_transform=apply_coord_transform)

    # Merge colored point clouds and write to points3D.txt
    colored_pcd_dir = os.path.join(input_dir, 'pointcloud_colored')
    print("Processing colored point clouds for points3D.txt...")
    points, colors = merge_colored_pointclouds_for_colmap(colored_pcd_dir, voxel_size=0.05)

    print("Writing points3D.txt...")
    write_points3d_txt(os.path.join(sparse_dir, 'points3D.txt'), points, colors)

    # Optionally merge point clouds to PLY files
    if args.merge_pointcloud:
        print("Merging point clouds to PLY...")
        pointcloud_dir = os.path.join(output_dir, 'pointcloud')
        os.makedirs(pointcloud_dir, exist_ok=True)

        nocolor_pcd_dir = os.path.join(input_dir, 'pointcloud_nocolor')

        if os.path.exists(colored_pcd_dir):
            merge_point_clouds(
                colored_pcd_dir,
                os.path.join(pointcloud_dir, 'merged_colored.ply'),
                colored=True
            )

        if os.path.exists(nocolor_pcd_dir):
            merge_point_clouds(
                nocolor_pcd_dir,
                os.path.join(pointcloud_dir, 'merged_nocolor.ply'),
                colored=False
            )

    print(f"\nConversion complete! Output saved to: {output_dir}")
    print("\nTo use with COLMAP:")
    print(f"  colmap feature_extractor --database_path {output_dir}/database.db --image_path {output_dir}/images")
    print(f"  colmap exhaustive_matcher --database_path {output_dir}/database.db")
    print(f"  # Or use the provided sparse model directly for downstream tasks")


if __name__ == '__main__':
    main()
