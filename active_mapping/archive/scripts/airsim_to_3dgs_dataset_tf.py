#!/usr/bin/env python3

import json
import math
import os
import struct

import cv2
import message_filters
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as SciRot
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


def rotation_angle_deg(rot_a, rot_b):
    delta = rot_a.inv() * rot_b
    return math.degrees(delta.magnitude())


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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def pack_rgb_fields(point):
    if len(point) >= 6:
        return np.array([point[3], point[4], point[5]], dtype=np.uint8)

    if len(point) < 4:
        return np.array([255, 255, 255], dtype=np.uint8)

    rgb_raw = point[3]
    if isinstance(rgb_raw, float):
        rgb_raw = struct.unpack("I", struct.pack("f", rgb_raw))[0]
    else:
        rgb_raw = int(rgb_raw)

    r = (rgb_raw >> 16) & 255
    g = (rgb_raw >> 8) & 255
    b = rgb_raw & 255
    return np.array([r, g, b], dtype=np.uint8)


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


class AirSimTo3DGSTfExporter:
    def __init__(self):
        rospy.init_node("airsim_to_3dgs_dataset_tf", anonymous=True)

        self.rgb_topic = rospy.get_param("~rgb_topic", "/airsim/rgb")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/airsim/camera_info")
        self.point_cloud_topic = rospy.get_param("~point_cloud_topic", "/airsim/point_cloud")
        self.output_dir = os.path.expanduser(rospy.get_param("~output_dir", "~/airsim_3dgs_dataset_tf"))
        self.image_dir = os.path.join(self.output_dir, "images")
        self.sparse_dir = os.path.join(self.output_dir, "sparse", "0")

        self.world_frame = rospy.get_param("~world_frame", "world")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_optical_frame")
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 0.05))
        self.tf_cache_time = float(rospy.get_param("~tf_cache_time", 30.0))

        self.min_translation = float(rospy.get_param("~min_translation", 0.10))
        self.min_rotation_deg = float(rospy.get_param("~min_rotation_deg", 5.0))
        self.min_frame_interval = float(rospy.get_param("~min_frame_interval", 0.0))
        self.max_frames = int(rospy.get_param("~max_frames", 0))

        self.cloud_stride = max(1, int(rospy.get_param("~cloud_stride", 4)))
        self.max_cloud_depth = float(rospy.get_param("~max_cloud_depth", 20.0))
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.03))
        self.cloud_collect_interval = float(rospy.get_param("~cloud_collect_interval", 0.3))
        self.max_cloud_age = float(rospy.get_param("~max_cloud_age", 0.20))

        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 20))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.02))

        ensure_dir(self.image_dir)
        ensure_dir(self.sparse_dir)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(self.tf_cache_time))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.frames = []
        self.cloud_points = []
        self.cloud_colors = []
        self.camera_models = {}
        self.finished = False
        self.last_saved_position = None
        self.last_saved_rotation = None
        self.last_saved_time = None
        self.last_cloud_collect_time = rospy.Time(0)

        rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        info_sub = message_filters.Subscriber(self.camera_info_topic, CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, info_sub],
            self.sync_queue_size,
            self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.image_callback)

        self.cloud_sub = rospy.Subscriber(self.point_cloud_topic, PointCloud2, self.cloud_callback, queue_size=10)

        rospy.on_shutdown(self.finalize)

        rospy.loginfo(
            "airsim_to_3dgs_dataset_tf listening on rgb=%s camera_info=%s cloud=%s",
            self.rgb_topic,
            self.camera_info_topic,
            self.point_cloud_topic,
        )
        rospy.loginfo(
            "Image poses are resolved from TF %s -> %s at RGB timestamps; clouds are accumulated independently.",
            self.world_frame,
            self.camera_frame,
        )

    def should_save_frame(self, stamp, position, rotation):
        if self.max_frames > 0 and len(self.frames) >= self.max_frames:
            return False

        if self.last_saved_position is None:
            return True

        dt = max(0.0, (stamp - self.last_saved_time).to_sec())
        translation = np.linalg.norm(position - self.last_saved_position)
        rotation_deg = rotation_angle_deg(self.last_saved_rotation, rotation)

        if dt < self.min_frame_interval:
            return False
        if translation < self.min_translation and rotation_deg < self.min_rotation_deg:
            return False
        return True

    def camera_key(self, camera_info):
        return (
            int(camera_info.width),
            int(camera_info.height),
            float(camera_info.K[0]),
            float(camera_info.K[4]),
            float(camera_info.K[2]),
            float(camera_info.K[5]),
        )

    def get_camera_id(self, camera_info):
        key = self.camera_key(camera_info)
        if key not in self.camera_models:
            self.camera_models[key] = len(self.camera_models) + 1
        return self.camera_models[key]

    def lookup_camera_pose(self, stamp):
        transform = self.tf_buffer.lookup_transform(
            self.world_frame,
            self.camera_frame,
            stamp,
            rospy.Duration(self.tf_timeout),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        t_world_camera = np.array(
            [translation.x, translation.y, translation.z],
            dtype=np.float64,
        )
        r_world_camera = SciRot.from_quat(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        )
        return t_world_camera, r_world_camera.as_matrix(), r_world_camera

    def save_image(self, image_msg, image_name):
        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("Failed to convert image: %s", str(exc))
            return False

        path = os.path.join(self.image_dir, image_name)
        if not cv2.imwrite(path, image):
            rospy.logwarn("Failed to write image to %s", path)
            return False
        return True

    def image_callback(self, rgb_msg, camera_info_msg):
        if self.finished:
            return

        stamp = rgb_msg.header.stamp if rgb_msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        try:
            t_world_camera, r_world_camera, r_world_camera_obj = self.lookup_camera_pose(stamp)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "Skipping image frame: TF lookup failed for %s -> %s at %.6f: %s",
                self.world_frame,
                self.camera_frame,
                stamp.toSec(),
                str(exc),
            )
            return

        if not self.should_save_frame(stamp, t_world_camera, r_world_camera_obj):
            return

        frame_index = len(self.frames) + 1
        image_name = f"frame_{frame_index:06d}.png"
        if not self.save_image(rgb_msg, image_name):
            return

        camera_id = self.get_camera_id(camera_info_msg)
        self.frames.append(
            {
                "image_id": frame_index,
                "image_name": image_name,
                "camera_id": camera_id,
                "stamp": {"secs": int(stamp.secs), "nsecs": int(stamp.nsecs)},
                "width": int(camera_info_msg.width),
                "height": int(camera_info_msg.height),
                "fx": float(camera_info_msg.K[0]),
                "fy": float(camera_info_msg.K[4]),
                "cx": float(camera_info_msg.K[2]),
                "cy": float(camera_info_msg.K[5]),
                "position": t_world_camera.tolist(),
                "rotation": r_world_camera.tolist(),
            }
        )
        self.last_saved_position = t_world_camera.copy()
        self.last_saved_rotation = r_world_camera_obj
        self.last_saved_time = stamp

        rospy.loginfo(
            "Saved frame %d from TF pose at stamp %.6f: image=%s",
            frame_index,
            stamp.toSec(),
            image_name,
        )

        if self.max_frames > 0 and len(self.frames) >= self.max_frames:
            rospy.loginfo("Reached max_frames=%d, exporting dataset.", self.max_frames)
            self.finalize()
            rospy.signal_shutdown("dataset export completed")

    def cloud_callback(self, cloud_msg):
        if self.finished:
            return

        now = rospy.Time.now()
        if self.last_cloud_collect_time != rospy.Time(0):
            if (now - self.last_cloud_collect_time).to_sec() < self.cloud_collect_interval:
                return

        stamp = cloud_msg.header.stamp if cloud_msg.header.stamp != rospy.Time(0) else now
        cloud_age = abs((now - stamp).to_sec())
        if self.max_cloud_age > 0.0 and cloud_age > self.max_cloud_age:
            rospy.logwarn_throttle(
                2.0,
                "Skipping cloud: age %.3fs exceeds max_cloud_age %.3fs",
                cloud_age,
                self.max_cloud_age,
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                cloud_msg.header.frame_id,
                stamp,
                rospy.Duration(self.tf_timeout),
            )
            cloud_world = do_transform_cloud(cloud_msg, transform)
            t_world_camera, _, _ = self.lookup_camera_pose(stamp)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Skipping cloud: transform failed: %s", str(exc))
            return

        points = []
        colors = []
        for index, point in enumerate(pc2.read_points(cloud_world, skip_nans=True)):
            if index % self.cloud_stride != 0:
                continue

            xyz = np.array(point[:3], dtype=np.float64)
            if self.max_cloud_depth > 0.0:
                depth = np.linalg.norm(xyz - t_world_camera)
                if depth > self.max_cloud_depth:
                    continue

            rgb = pack_rgb_fields(point)
            points.append(xyz)
            colors.append(rgb)

        if not points:
            return

        self.cloud_points.append(np.asarray(points, dtype=np.float64))
        self.cloud_colors.append(np.asarray(colors, dtype=np.uint8))
        self.last_cloud_collect_time = now

        rospy.loginfo(
            "Collected cloud chunk at %.6f: kept=%d accumulated_chunks=%d",
            stamp.toSec(),
            len(points),
            len(self.cloud_points),
        )

    def write_cameras_txt(self):
        with open(os.path.join(self.sparse_dir, "cameras.txt"), "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(self.camera_models)}\n")
            for key, camera_id in sorted(self.camera_models.items(), key=lambda item: item[1]):
                width, height, fx, fy, cx, cy = key
                f.write(f"{camera_id} PINHOLE {width} {height} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n")

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
                f.write(f"{frame['image_id']} {qstr} {tstr} {frame['camera_id']} {frame['image_name']}\n")
                f.write("\n")

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

    def write_metadata(self, points):
        with open(os.path.join(self.output_dir, "cameras.json"), "w") as f:
            json.dump(
                [
                    {
                        "id": frame["image_id"],
                        "img_name": os.path.splitext(frame["image_name"])[0],
                        "width": frame["width"],
                        "height": frame["height"],
                        "position": frame["position"],
                        "rotation": frame["rotation"],
                        "fx": frame["fx"],
                        "fy": frame["fy"],
                        "cx": frame["cx"],
                        "cy": frame["cy"],
                    }
                    for frame in self.frames
                ],
                f,
                indent=2,
            )

        with open(os.path.join(self.output_dir, "dataset_summary.json"), "w") as f:
            json.dump(
                {
                    "frames": len(self.frames),
                    "points_after_voxel": int(points.shape[0]),
                    "voxel_size": self.voxel_size,
                    "cloud_stride": self.cloud_stride,
                    "topics": {
                        "rgb": self.rgb_topic,
                        "camera_info": self.camera_info_topic,
                        "point_cloud": self.point_cloud_topic,
                    },
                    "pose_source": {
                        "type": "TF lookup at RGB timestamp",
                        "world_frame": self.world_frame,
                        "camera_frame": self.camera_frame,
                    },
                    "cloud_source": {
                        "type": "Independent PointCloud2 accumulation transformed to world via TF",
                        "max_cloud_age": self.max_cloud_age,
                        "cloud_collect_interval": self.cloud_collect_interval,
                    },
                },
                f,
                indent=2,
            )

    def finalize(self):
        if self.finished:
            return
        self.finished = True

        if not self.frames:
            rospy.logwarn("No frames were captured. Nothing to export.")
            return

        if self.cloud_points:
            points = np.concatenate(self.cloud_points, axis=0)
            colors = np.concatenate(self.cloud_colors, axis=0)
        else:
            points = np.zeros((0, 3), dtype=np.float64)
            colors = np.zeros((0, 3), dtype=np.uint8)

        points, colors = voxel_downsample(points, colors, self.voxel_size)

        self.write_cameras_txt()
        self.write_images_txt()
        self.write_points3d_txt(points, colors)
        write_ply(os.path.join(self.output_dir, "input.ply"), points, colors)
        self.write_metadata(points)

        rospy.loginfo(
            "Exported dataset to %s with %d frames and %d points",
            self.output_dir,
            len(self.frames),
            points.shape[0],
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        AirSimTo3DGSTfExporter().run()
    except rospy.ROSInterruptException:
        pass
