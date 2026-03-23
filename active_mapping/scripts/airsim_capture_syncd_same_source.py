#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os

import airsim
import cv2
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as SciRot


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def rotmat2qvec(rotation):
    matrix = np.asarray(rotation, dtype=np.float64)
    k = np.array([
        [matrix[0, 0] - matrix[1, 1] - matrix[2, 2], matrix[1, 0] + matrix[0, 1], matrix[2, 0] + matrix[0, 2], matrix[1, 2] - matrix[2, 1]],
        [matrix[1, 0] + matrix[0, 1], matrix[1, 1] - matrix[0, 0] - matrix[2, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 0] - matrix[0, 2]],
        [matrix[2, 0] + matrix[0, 2], matrix[2, 1] + matrix[1, 2], matrix[2, 2] - matrix[0, 0] - matrix[1, 1], matrix[0, 1] - matrix[1, 0]],
        [matrix[1, 2] - matrix[2, 1], matrix[2, 0] - matrix[0, 2], matrix[0, 1] - matrix[1, 0], matrix[0, 0] + matrix[1, 1] + matrix[2, 2]],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def voxel_downsample(points, colors, voxel_size):
    if voxel_size <= 0.0 or points.shape[0] == 0:
        return points, colors

    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices], colors[unique_indices]


def write_ply(path, points, colors):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def camera_axis_transform(name):
    transforms = {
        "identity": np.eye(3, dtype=np.float64),
        "flip_yz": np.diag([1.0, -1.0, -1.0]).astype(np.float64),
        "swap_yz": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        "swap_yz_flip_y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    }
    if name not in transforms:
        raise ValueError("Unsupported camera_axis_variant '{}'. Expected one of {}.".format(name, sorted(transforms)))
    return transforms[name]


def normalize_pose_rotation(name, rotation):
    semantics = str(name).lower()
    if semantics == "rwc":
        return rotation
    if semantics == "rcw":
        return rotation.T
    raise ValueError("Unsupported pose_rotation_semantics '{}'. Expected one of ['rcw', 'rwc'].".format(name))


class AirSimCaptureSyncdSameSource:
    def __init__(self):
        rospy.init_node("airsim_capture_syncd_same_source", anonymous=True)

        self.vehicle_name = rospy.get_param("~vehicle_name", "Drone1")
        self.cam_name = rospy.get_param("~cam_name", "front")
        self.capture_interval = float(rospy.get_param("~capture_interval", 0.3))
        self.camera_fov_deg = float(rospy.get_param("~camera_fov_deg", 90.0))
        self.airsim_host = rospy.get_param("~airsim_host", "127.0.0.1")
        self.airsim_port = int(rospy.get_param("~airsim_port", 41451))
        self.airsim_timeout_sec = float(rospy.get_param("~airsim_timeout_sec", 60.0))
        self.airsim_retry_interval = float(rospy.get_param("~airsim_retry_interval", 1.0))

        self.cloud_stride = max(1, int(rospy.get_param("~cloud_stride", 4)))
        self.cloud_voxel_size = float(rospy.get_param("~cloud_voxel_size", 0.03))
        self.max_cloud_depth = float(rospy.get_param("~max_cloud_depth", 30.0))
        self.max_frames = int(rospy.get_param("~max_frames", 0))
        self.min_translation = float(rospy.get_param("~min_translation", 0.02))
        self.min_rotation_deg = float(rospy.get_param("~min_rotation_deg", 1.0))
        self.camera_axis_variant = rospy.get_param("~camera_axis_variant", "identity")
        self.camera_axis_matrix = camera_axis_transform(self.camera_axis_variant)
        self.pose_rotation_semantics = rospy.get_param("~pose_rotation_semantics", "rwc")

        self.output_dir = os.path.expanduser(
            rospy.get_param(
                "~output_dir",
                "/home/xhy/mapping_GS/active_recon_ws/viewpoint_captures/mycolmap_same_source",
            )
        )
        self.image_dir = os.path.join(self.output_dir, "images")
        self.sparse_dir = os.path.join(self.output_dir, "sparse", "0")

        ensure_dir(self.image_dir)
        ensure_dir(self.sparse_dir)

        self.client = self._connect_airsim_with_retry()
        self.frames = []
        self.cloud_points = []
        self.cloud_colors = []
        self.finished = False
        self.last_saved_image = None
        self.last_saved_position = None
        self.last_saved_rotation = None

    def _connect_airsim_with_retry(self):
        deadline = None if self.airsim_timeout_sec <= 0.0 else (rospy.Time.now().to_sec() + self.airsim_timeout_sec)
        while not rospy.is_shutdown():
            try:
                client = airsim.MultirotorClient(ip=self.airsim_host, port=self.airsim_port)
                client.confirmConnection()
                rospy.loginfo("AirSim same-source capture connected to %s:%d", self.airsim_host, self.airsim_port)
                return client
            except Exception as exc:
                rospy.logwarn("AirSim same-source capture connect failed: %s. Retrying in %.1fs", str(exc), self.airsim_retry_interval)
                if deadline is not None and rospy.Time.now().to_sec() > deadline:
                    rospy.logerr("AirSim same-source capture connect timeout (%.1fs) reached.", self.airsim_timeout_sec)
                    raise
                rospy.sleep(self.airsim_retry_interval)
        raise rospy.ROSInterruptException("ROS shutdown before AirSim same-source capture connected")

    def compute_intrinsics(self, width, height):
        fov_rad = math.radians(self.camera_fov_deg)
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return fx, fy, cx, cy

    def airsim_pose_to_enu(self, response):
        p = response.camera_position
        o = response.camera_orientation
        pos_enu = np.array([p.x_val, -p.y_val, -p.z_val], dtype=np.float64)
        quat_enu = np.array([o.x_val, -o.y_val, -o.z_val, o.w_val], dtype=np.float64)
        rot_enu = SciRot.from_quat(quat_enu).as_matrix()
        rot_fixed = rot_enu.dot(self.camera_axis_matrix)
        return pos_enu, normalize_pose_rotation(self.pose_rotation_semantics, rot_fixed)

    def save_image(self, image, image_name):
        path = os.path.join(self.image_dir, image_name)
        if not cv2.imwrite(path, image):
            raise RuntimeError("Failed to write image to {}".format(path))

    def is_duplicate_image(self, image):
        if self.last_saved_image is None:
            return False
        return image.shape == self.last_saved_image.shape and np.array_equal(image, self.last_saved_image)

    def pose_changed_enough(self, position, rotation_matrix):
        if self.last_saved_position is None or self.last_saved_rotation is None:
            return True

        translation = np.linalg.norm(position - self.last_saved_position)
        current_rotation = SciRot.from_matrix(rotation_matrix)
        rotation_delta = self.last_saved_rotation.inv() * current_rotation
        rotation_deg = math.degrees(rotation_delta.magnitude())
        return translation >= self.min_translation or rotation_deg >= self.min_rotation_deg

    def backproject_depth_to_world(self, image_bgr, depth_response, r_world_camera, t_world_camera, fx, fy, cx, cy):
        width = int(depth_response.width)
        height = int(depth_response.height)
        depth = np.array(depth_response.image_data_float, dtype=np.float32).reshape(height, width)

        if image_bgr.shape[0] != height or image_bgr.shape[1] != width:
            raise RuntimeError(
                "RGB/depth shape mismatch: rgb={}x{} depth={}x{}".format(
                    image_bgr.shape[1], image_bgr.shape[0], width, height
                )
            )

        step = self.cloud_stride
        v_coords = np.arange(0, height, step, dtype=np.int32)
        u_coords = np.arange(0, width, step, dtype=np.int32)
        grid_u, grid_v = np.meshgrid(u_coords, v_coords)
        sampled_depth = depth[grid_v, grid_u].astype(np.float64)

        valid_mask = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
        if self.max_cloud_depth > 0.0:
            valid_mask &= sampled_depth <= self.max_cloud_depth

        if not np.any(valid_mask):
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

        u_valid = grid_u[valid_mask].astype(np.float64)
        v_valid = grid_v[valid_mask].astype(np.float64)
        z_valid = sampled_depth[valid_mask]

        x_valid = (u_valid - cx) * z_valid / fx
        y_valid = (v_valid - cy) * z_valid / fy
        points_camera = np.stack([x_valid, y_valid, z_valid], axis=1)
        points_world = points_camera.dot(r_world_camera.T) + t_world_camera.reshape(1, 3)

        colors_bgr = image_bgr[grid_v[valid_mask], grid_u[valid_mask]]
        colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)
        return points_world, colors_rgb

    def write_cameras_txt(self, width, height, fx, fy, cx, cy):
        with open(os.path.join(self.sparse_dir, "cameras.txt"), "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: 1\n")
            f.write(f"1 PINHOLE {width} {height} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n")

    def write_images_txt(self):
        with open(os.path.join(self.sparse_dir, "images.txt"), "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(self.frames)}\n")
            for frame in self.frames:
                r_world_camera = np.asarray(frame["rotation"], dtype=np.float64)
                t_world_camera = np.asarray(frame["position"], dtype=np.float64)
                r_w2c = r_world_camera.T
                t_w2c = -r_w2c.dot(t_world_camera)
                qvec = rotmat2qvec(r_w2c)
                qstr = " ".join(f"{value:.17g}" for value in qvec)
                tstr = " ".join(f"{value:.17g}" for value in t_w2c)
                f.write(f"{frame['id']} {qstr} {tstr} 1 {frame['img_name']}\n\n")

    def write_points3d_txt(self, points, colors):
        with open(os.path.join(self.sparse_dir, "points3D.txt"), "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {points.shape[0]}\n")
            for point_id, (point, color) in enumerate(zip(points, colors), start=1):
                f.write(
                    f"{point_id} "
                    f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
                )

    def write_metadata(self, width, height, fx, fy, cx, cy, points_count):
        cameras_json = []
        for frame in self.frames:
            cameras_json.append(
                {
                    "id": frame["id"],
                    "img_name": os.path.splitext(frame["img_name"])[0],
                    "width": width,
                    "height": height,
                    "position": frame["position"],
                    "rotation": frame["rotation"],
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                }
            )
        with open(os.path.join(self.output_dir, "cameras.json"), "w") as f:
            json.dump(cameras_json, f, indent=2)

        with open(os.path.join(self.output_dir, "dataset_summary.json"), "w") as f:
            json.dump(
                {
                    "frames": len(self.frames),
                    "points_after_voxel": int(points_count),
                    "capture_interval": self.capture_interval,
                    "depth_pixel_stride": self.cloud_stride,
                    "camera_pose_source": "AirSim direct query",
                    "point_cloud_source": "Same AirSim sample depth back-projection",
                    "camera_axis_variant": self.camera_axis_variant,
                    "pose_rotation_semantics": self.pose_rotation_semantics,
                },
                f,
                indent=2,
            )

    def finalize(self, width=None, height=None, fx=None, fy=None, cx=None, cy=None):
        if self.finished:
            return
        self.finished = True

        if not self.frames:
            rospy.logwarn("No frames captured. Nothing to export.")
            return

        points = np.concatenate(self.cloud_points, axis=0) if self.cloud_points else np.zeros((0, 3), dtype=np.float64)
        colors = np.concatenate(self.cloud_colors, axis=0) if self.cloud_colors else np.zeros((0, 3), dtype=np.uint8)
        points, colors = voxel_downsample(points, colors, self.cloud_voxel_size)

        self.write_cameras_txt(width, height, fx, fy, cx, cy)
        self.write_images_txt()
        self.write_points3d_txt(points, colors)
        write_ply(os.path.join(self.output_dir, "input.ply"), points, colors)
        self.write_metadata(width, height, fx, fy, cx, cy, points.shape[0])
        rospy.loginfo("Exported same-source dataset to %s with %d frames and %d points", self.output_dir, len(self.frames), points.shape[0])

    def run(self):
        width = None
        height = None
        fx = None
        fy = None
        cx = None
        cy = None

        rate = rospy.Rate(1.0 / self.capture_interval)
        while not rospy.is_shutdown():
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(self.cam_name, airsim.ImageType.Scene, False, True),
                    airsim.ImageRequest(self.cam_name, airsim.ImageType.DepthPlanar, True, False),
                ],
                vehicle_name=self.vehicle_name,
            )
            if len(responses) < 2:
                rate.sleep()
                continue

            rgb_response = responses[0]
            depth_response = responses[1]
            if rgb_response.width == 0 or rgb_response.height == 0:
                rate.sleep()
                continue

            if width is None:
                width = int(rgb_response.width)
                height = int(rgb_response.height)
                fx, fy, cx, cy = self.compute_intrinsics(width, height)

            raw_rgb_bytes = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
            image_bgr = cv2.imdecode(raw_rgb_bytes, cv2.IMREAD_COLOR)
            if image_bgr is None:
                rospy.logwarn("Failed to decode RGB frame from AirSim")
                rate.sleep()
                continue

            t_world_camera, r_world_camera = self.airsim_pose_to_enu(rgb_response)
            if self.is_duplicate_image(image_bgr):
                rospy.loginfo("Skip duplicate frame: RGB image identical to previous saved frame")
                rate.sleep()
                continue

            if not self.pose_changed_enough(t_world_camera, r_world_camera):
                rospy.loginfo("Skip frame: camera pose change below thresholds")
                rate.sleep()
                continue

            points_world, colors_rgb = self.backproject_depth_to_world(
                image_bgr=image_bgr,
                depth_response=depth_response,
                r_world_camera=r_world_camera,
                t_world_camera=t_world_camera,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )

            frame_id = len(self.frames) + 1
            image_name = f"frame_{frame_id:06d}.png"
            self.save_image(image_bgr, image_name)
            self.last_saved_image = image_bgr.copy()
            self.last_saved_position = t_world_camera.copy()
            self.last_saved_rotation = SciRot.from_matrix(r_world_camera)
            self.frames.append(
                {
                    "id": frame_id,
                    "img_name": image_name,
                    "position": t_world_camera.tolist(),
                    "rotation": r_world_camera.tolist(),
                }
            )

            if points_world.shape[0] > 0:
                self.cloud_points.append(points_world)
                self.cloud_colors.append(colors_rgb)

            rospy.loginfo(
                "Captured same-source frame %d: %s points=%d",
                frame_id,
                image_name,
                points_world.shape[0],
            )

            if self.max_frames > 0 and len(self.frames) >= self.max_frames:
                break
            rate.sleep()

        self.finalize(width, height, fx, fy, cx, cy)


if __name__ == "__main__":
    try:
        AirSimCaptureSyncdSameSource().run()
    except rospy.ROSInterruptException:
        pass
