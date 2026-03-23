#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import struct
import threading

import airsim
import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from scipy.spatial.transform import Rotation as SciRot
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


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


def unpack_point_color(point):
    if len(point) >= 6:
        return np.array([point[3], point[4], point[5]], dtype=np.uint8)
    if len(point) < 4:
        return np.array([255, 255, 255], dtype=np.uint8)

    rgb_raw = point[3]
    if isinstance(rgb_raw, float):
        rgb_raw = struct.unpack("I", struct.pack("f", rgb_raw))[0]
    else:
        rgb_raw = int(rgb_raw)
    return np.array([(rgb_raw >> 16) & 255, (rgb_raw >> 8) & 255, rgb_raw & 255], dtype=np.uint8)


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


class AirSimCaptureSyncd:
    def __init__(self):
        rospy.init_node("airsim_capture_syncd", anonymous=True)

        self.vehicle_name = rospy.get_param("~vehicle_name", "Drone1")
        self.cam_name = rospy.get_param("~cam_name", "front")
        self.capture_interval = float(rospy.get_param("~capture_interval", 0.3))
        self.camera_fov_deg = float(rospy.get_param("~camera_fov_deg", 90.0))
        self.airsim_host = rospy.get_param("~airsim_host", "127.0.0.1")
        self.airsim_port = int(rospy.get_param("~airsim_port", 41451))
        self.airsim_timeout_sec = float(rospy.get_param("~airsim_timeout_sec", 60.0))
        self.airsim_retry_interval = float(rospy.get_param("~airsim_retry_interval", 1.0))

        self.point_cloud_topic = rospy.get_param("~point_cloud_topic", "/airsim/point_cloud")
        self.global_frame_id = rospy.get_param("~global_frame_id", "world")
        self.cloud_collect_interval = float(rospy.get_param("~cloud_collect_interval", 0.3))
        self.cloud_stride = max(1, int(rospy.get_param("~cloud_stride", 4)))
        self.cloud_voxel_size = float(rospy.get_param("~cloud_voxel_size", 0.03))
        self.max_cloud_depth = float(rospy.get_param("~max_cloud_depth", 30.0))
        self.max_frames = int(rospy.get_param("~max_frames", 0))
        self.min_translation = float(rospy.get_param("~min_translation", 0.02))
        self.min_rotation_deg = float(rospy.get_param("~min_rotation_deg", 1.0))
        self.camera_axis_variant = rospy.get_param("~camera_axis_variant", "identity")
        self.camera_axis_matrix = camera_axis_transform(self.camera_axis_variant)
        self.pose_rotation_semantics = rospy.get_param("~pose_rotation_semantics", "rwc")

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.output_dir = os.path.expanduser(
            rospy.get_param(
                "~output_dir",
                "/home/xhy/mapping_GS/active_recon_ws/viewpoint_captures/mycolmap",
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
        self.last_cloud_pose_position = None
        self.last_cloud_pose_rotation = None

        self.latest_cloud_msg = None
        self.last_cloud_collect_time = rospy.Time(0)
        self.cloud_lock = threading.Lock()
        rospy.Subscriber(self.point_cloud_topic, PointCloud2, self.cloud_callback, queue_size=1)

    def _connect_airsim_with_retry(self):
        deadline = None if self.airsim_timeout_sec <= 0.0 else (rospy.Time.now().to_sec() + self.airsim_timeout_sec)
        while not rospy.is_shutdown():
            try:
                client = airsim.MultirotorClient(ip=self.airsim_host, port=self.airsim_port)
                client.confirmConnection()
                rospy.loginfo("AirSim capture connected to %s:%d", self.airsim_host, self.airsim_port)
                return client
            except Exception as exc:
                rospy.logwarn("AirSim capture connect failed: %s. Retrying in %.1fs", str(exc), self.airsim_retry_interval)
                if deadline is not None and rospy.Time.now().to_sec() > deadline:
                    rospy.logerr("AirSim capture connect timeout (%.1fs) reached.", self.airsim_timeout_sec)
                    raise
                rospy.sleep(self.airsim_retry_interval)
        raise rospy.ROSInterruptException("ROS shutdown before AirSim capture connected")

    def cloud_callback(self, msg):
        with self.cloud_lock:
            self.latest_cloud_msg = msg

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

    def maybe_collect_cloud(self):
        now = rospy.Time.now()
        if self.last_cloud_collect_time != rospy.Time(0):
            if (now - self.last_cloud_collect_time).to_sec() < self.cloud_collect_interval:
                return

        with self.cloud_lock:
            cloud_msg = self.latest_cloud_msg

        if cloud_msg is None:
            rospy.logwarn_throttle(2.0, "Cloud collection skipped: no point cloud received yet")
            return

        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.global_frame_id,
                cloud_msg.header.frame_id,
                cloud_msg.header.stamp,
                rospy.Duration(0.2),
            )
        except tf2_ros.TransformException as exc:
            rospy.logwarn_throttle(2.0, "Cloud collection skipped: TF lookup failed target=%s source=%s reason=%s", self.global_frame_id, cloud_msg.header.frame_id, str(exc))
            return

        cloud_world_msg = do_transform_cloud(cloud_msg, transform_stamped)

        points = []
        colors = []
        raw_count = 0
        stride_kept = 0
        depth_kept = 0
        for index, point in enumerate(pc2.read_points(cloud_world_msg, skip_nans=True)):
            raw_count += 1
            if index % self.cloud_stride != 0:
                continue
            stride_kept += 1
            xyz_world = np.array(point[:3], dtype=np.float64)
            if self.max_cloud_depth > 0.0:
                cx = transform_stamped.transform.translation.x
                cy = transform_stamped.transform.translation.y
                cz = transform_stamped.transform.translation.z
                if np.linalg.norm(xyz_world - np.array([cx, cy, cz], dtype=np.float64)) > self.max_cloud_depth:
                    continue
            depth_kept += 1
            points.append(xyz_world)
            colors.append(unpack_point_color(point))

        if points:
            self.cloud_points.append(np.asarray(points, dtype=np.float64))
            self.cloud_colors.append(np.asarray(colors, dtype=np.uint8))
            self.last_cloud_collect_time = now
            rospy.loginfo(
                "Collected cloud chunk: frame=%s raw=%d stride_kept=%d depth_kept=%d accumulated=%d",
                cloud_msg.header.frame_id,
                raw_count,
                stride_kept,
                depth_kept,
                len(points),
            )
        else:
            rospy.logwarn_throttle(
                2.0,
                "Cloud collection produced 0 points: frame=%s raw=%d stride_kept=%d depth_kept=%d",
                cloud_msg.header.frame_id,
                raw_count,
                stride_kept,
                depth_kept,
            )

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
                    "cloud_collect_interval": self.cloud_collect_interval,
                    "point_cloud_topic": self.point_cloud_topic,
                    "camera_pose_source": "AirSim direct query",
                    "point_cloud_source": "ROS nodelet output transformed to world via TF",
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
        rospy.loginfo("Exported dataset to %s with %d frames and %d points", self.output_dir, len(self.frames), points.shape[0])

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

            frame_id = len(self.frames) + 1
            image_name = f"frame_{frame_id:06d}.png"
            self.save_image(image_bgr, image_name)
            self.last_saved_image = image_bgr.copy()
            self.last_saved_position = t_world_camera.copy()
            self.last_saved_rotation = SciRot.from_matrix(r_world_camera)
            self.last_cloud_pose_position = t_world_camera.copy()
            self.last_cloud_pose_rotation = r_world_camera.copy()
            self.frames.append(
                {
                    "id": frame_id,
                    "img_name": image_name,
                    "position": t_world_camera.tolist(),
                    "rotation": r_world_camera.tolist(),
                }
            )
            rospy.loginfo("Captured frame %d: %s", frame_id, image_name)

            self.maybe_collect_cloud()

            if self.max_frames > 0 and len(self.frames) >= self.max_frames:
                break
            rate.sleep()

        self.finalize(width, height, fx, fy, cx, cy)


if __name__ == "__main__":
    try:
        AirSimCaptureSyncd().run()
    except rospy.ROSInterruptException:
        pass
