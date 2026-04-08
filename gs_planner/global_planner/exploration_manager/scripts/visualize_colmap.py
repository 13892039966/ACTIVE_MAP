#!/usr/bin/env python3
"""
Visualize COLMAP format data using Open3D.

This script visualizes:
1. Colored point cloud from points3D.txt
2. Camera poses from images.txt as camera frustums

Usage:
    python3 visualize_colmap.py --colmap_path /path/to/colmap_data
    python3 visualize_colmap.py --colmap_path /path/to/colmap_data --point_size 2.0 --camera_scale 0.1
"""

import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os


def read_cameras_txt(cameras_file):
    """Read cameras.txt from COLMAP sparse reconstruction.

    Returns: dict of {camera_id: camera_info}
    """
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def read_images_txt(images_file):
    """Read images.txt from COLMAP sparse reconstruction.

    Format:
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        POINTS2D[] (second line, ignored)

    Returns: list of image_info dicts
    """
    images = []
    with open(images_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or len(line) == 0:
            i += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]

        # Store world-to-camera transformation
        # q_wxyz is the quaternion for world-to-camera rotation
        # t is the world-to-camera translation
        images.append({
            'id': image_id,
            'q_wxyz': np.array([qw, qx, qy, qz]),
            't': np.array([tx, ty, tz]),
            'camera_id': camera_id,
            'name': name
        })

        i += 2  # Skip the next line (POINTS2D data)

    return images


def read_points3d_txt(points_file, skip_points=1):
    """Read points3D.txt from COLMAP sparse reconstruction.

    Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]

    Args:
        points_file: Path to points3D.txt
        skip_points: Skip every N points for performance (default=1, use larger for big clouds)

    Returns: (points, colors) numpy arrays
    """
    points_list = []
    colors_list = []

    with open(points_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue

            if skip_points > 1 and idx % skip_points != 0:
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])

            points_list.append([x, y, z])
            colors_list.append([r / 255.0, g / 255.0, b / 255.0])

    return np.array(points_list), np.array(colors_list)


def colmap_pose_to_camera_center(q_wxyz, t):
    """Convert COLMAP world-to-camera pose to camera center in world frame.

    COLMAP stores world-to-camera transformation:
        p_camera = R_w2c * p_world + t_w2c

    The camera center in world coordinates is:
        C_world = -R_w2c^T * t_w2c

    Args:
        q_wxyz: Quaternion [w, x, y, z] for world-to-camera rotation
        t: Translation vector [tx, ty, tz] for world-to-camera

    Returns:
        Camera center in world frame [x, y, z]
    """
    # Convert quaternion to rotation matrix
    # scipy uses [x, y, z, w] format, COLMAP uses [w, x, y, z]
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    R_w2c = R.from_quat(q_xyzw).as_matrix()

    # Compute camera center
    C_world = -R_w2c.T @ t

    return C_world


def colmap_pose_to_rotation_matrix(q_wxyz):
    """Convert COLMAP quaternion to rotation matrix.

    Args:
        q_wxyz: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    return R.from_quat(q_xyzw).as_matrix()


def create_camera_frustum(camera_center, rotation_matrix, fx, fy, cx, cy,
                          width, height, scale=0.1, color=[1, 0, 0]):
    """Create a camera frustum visualization.

    Args:
        camera_center: Camera center in world frame [x, y, z]
        rotation_matrix: World-to-camera rotation matrix (3x3)
        fx, fy: Focal lengths
        cx, cy: Principal points
        width, height: Image dimensions
        scale: Scale factor for frustum size
        color: RGB color for frustum

    Returns:
        Open3D LineSet representing the camera frustum
    """
    # Define frustum corners in camera frame
    # At depth = 1, the image corners are:
    # Top-left:    (-cx/fx, -(height-cy)/fy, 1)
    # Top-right:   ((width-cx)/fx, -(height-cy)/fy, 1)
    # Bottom-left: (-cx/fx, cy/fy, 1)
    # Bottom-right: ((width-cx)/fx, cy/fy, 1)

    x1 = -cx / fx
    x2 = (width - cx) / fx
    y1 = -cy / fy
    y2 = (height - cy) / fy

    # Frustum vertices in camera frame
    vertices_cam = np.array([
        [0, 0, 0],  # Camera center
        [x1, y1, 1],  # Top-left at depth 1
        [x2, y1, 1],  # Top-right at depth 1
        [x2, y2, 1],  # Bottom-right at depth 1
        [x1, y2, 1],  # Bottom-left at depth 1
    ]) * scale

    # Transform vertices to world frame
    # P_world = R_w2c^T * (P_cam - t_w2c) = R_c2w * P_cam + C_world
    R_c2w = rotation_matrix.T
    vertices_world = (R_c2w @ vertices_cam.T).T + camera_center

    # Define edges
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Rectangle at depth 1
    ]

    # Create LineSet
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(vertices_world)
    frustum.lines = o3d.utility.Vector2iVector(edges)
    frustum.colors = o3d.utility.Vector3dVector([color for _ in edges])

    return frustum


def create_camera_axis(camera_center, rotation_matrix, scale=0.1):
    """Create RGB axes for camera orientation.

    Args:
        camera_center: Camera center in world frame
        rotation_matrix: World-to-camera rotation matrix (3x3)
        scale: Axis length

    Returns:
        Open3D LineSet with X(Red), Y(Green), Z(Blue) axes
    """
    R_c2w = rotation_matrix.T

    # Camera frame axes in world coordinates
    # In camera frame: X=right, Y=down, Z=forward
    axis_origin = camera_center
    x_axis = camera_center + R_c2w @ np.array([scale, 0, 0])
    y_axis = camera_center + R_c2w @ np.array([0, scale, 0])
    z_axis = camera_center + R_c2w @ np.array([0, 0, scale])

    # Create LineSet
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector([
        axis_origin, x_axis,
        axis_origin, y_axis,
        axis_origin, z_axis,
    ])
    axis.lines = o3d.utility.Vector2iVector([
        [0, 1], [2, 3], [4, 5]
    ])
    axis.colors = o3d.utility.Vector3dVector([
        [1, 0, 0],  # X: Red
        [0, 1, 0],  # Y: Green
        [0, 0, 1],  # Z: Blue
    ])

    return axis


def create_trajectory_line(centers):
    """Create a line through camera centers.

    Args:
        centers: List of camera centers [N, 3]

    Returns:
        Open3D LineSet
    """
    n = len(centers)
    if n < 2:
        return None

    lines = []
    for i in range(n - 1):
        lines.append([i, i + 1])

    trajectory = o3d.geometry.LineSet()
    trajectory.points = o3d.utility.Vector3dVector(centers)
    trajectory.lines = o3d.utility.Vector2iVector(lines)
    trajectory.colors = o3d.utility.Vector3dVector([[1, 1, 0] for _ in range(n - 1)])  # Yellow

    return trajectory


def main():
    parser = argparse.ArgumentParser(description='Visualize COLMAP data using Open3D')
    parser.add_argument('--colmap_path', '-p', type=str,
                        default='/home/daisy/workspace/NARUTO/results/Replica/room1/NARUTO/run_0/colmap_room_naruto',
                        help='Path to COLMAP data directory')
    parser.add_argument('--sparse', '-s', type=str, default='sparse/0',
                        help='Relative path to sparse reconstruction (default: sparse/0)')
    parser.add_argument('--skip_points', type=int, default=1,
                        help='Skip every N points for performance (default=1)')
    parser.add_argument('--point_size', type=float, default=2.0,
                        help='Point cloud point size (default=2.0)')
    parser.add_argument('--camera_scale', type=float, default=0.1,
                        help='Scale for camera frustums (default=0.1)')
    parser.add_argument('--camera_step', type=int, default=5,
                        help='Show every Nth camera (default=5)')
    parser.add_argument('--show_cameras', action='store_true', default=True,
                        help='Show camera frustums (default: True)')
    parser.add_argument('--show_trajectory', action='store_true', default=True,
                        help='Show camera trajectory (default: True)')
    parser.add_argument('--show_axes', action='store_true',
                        help='Show camera coordinate axes')
    args = parser.parse_args()

    colmap_path = args.colmap_path
    sparse_dir = os.path.join(colmap_path, args.sparse)

    # Check if directory exists
    if not os.path.exists(sparse_dir):
        print(f"Error: Sparse directory {sparse_dir} does not exist")
        return

    print(f"Loading COLMAP data from: {colmap_path}")

    # Read cameras
    cameras_file = os.path.join(sparse_dir, 'cameras.txt')
    if os.path.exists(cameras_file):
        cameras = read_cameras_txt(cameras_file)
        print(f"  Loaded {len(cameras)} cameras")
    else:
        print(f"  Warning: {cameras_file} not found")
        cameras = {}

    # Read images (poses)
    images_file = os.path.join(sparse_dir, 'images.txt')
    if os.path.exists(images_file):
        images = read_images_txt(images_file)
        print(f"  Loaded {len(images)} camera poses")
    else:
        print(f"  Error: {images_file} not found")
        return

    # Read points
    points_file = os.path.join(sparse_dir, 'points3D.txt')
    if os.path.exists(points_file):
        points, colors = read_points3d_txt(points_file, skip_points=args.skip_points)
        print(f"  Loaded {len(points)} 3D points")
    else:
        print(f"  Warning: {points_file} not found")
        points, colors = np.array([]), np.array([])

    # Create point cloud
    geometries = []
    if len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)
        print(f"  Point cloud memory: {points.nbytes / 1024 / 1024:.2f} MB")

    # Compute camera centers
    camera_centers = []
    for img in images:
        q = img['q_wxyz']
        t = img['t']
        center = colmap_pose_to_camera_center(q, t)
        camera_centers.append(center)

    camera_centers = np.array(camera_centers)

    # Show trajectory
    if args.show_trajectory and len(camera_centers) > 1:
        traj_line = create_trajectory_line(camera_centers)
        if traj_line is not None:
            geometries.append(traj_line)
            print(f"  Added trajectory line ({len(camera_centers)} cameras)")

    # Show camera frustums
    if args.show_cameras:
        print(f"  Adding camera frustums (every {args.camera_step}th camera)...")
        for idx, img in enumerate(images[::args.camera_step]):
            cam_info = cameras.get(img['camera_id'], None)
            if cam_info is None:
                continue

            # Get camera intrinsics
            params = cam_info['params']
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
            width, height = cam_info['width'], cam_info['height']

            # Get pose
            q = img['q_wxyz']
            t = img['t']
            center = colmap_pose_to_camera_center(q, t)
            R_w2c = colmap_pose_to_rotation_matrix(q)

            # Create frustum
            frustum = create_camera_frustum(center, R_w2c, fx, fy, cx, cy,
                                           width, height, scale=args.camera_scale)
            geometries.append(frustum)

            # Create axes if requested
            if args.show_axes:
                axis = create_camera_axis(center, R_w2c, scale=args.camera_scale * 1.5)
                geometries.append(axis)

    # Visualize
    if len(geometries) == 0:
        print("No geometries to visualize")
        return

    print("\nStarting visualization...")
    print("  Controls:")
    print("    - Mouse drag: Rotate")
    print("    - Mouse wheel: Zoom")
    print("    - Shift + drag: Pan")
    print("    - Q: Exit")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="COLMAP Visualization",
        width=1280,
        height=720,
    )


if __name__ == '__main__':
    main()
