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
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as SciRot
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


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


class AirSimTo3DGSExporter:
    def __init__(self):
        rospy.init_node("airsim_to_3dgs_dataset", anonymous=True)

        self.rgb_topic = rospy.get_param("~rgb_topic", "/airsim/rgb")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/airsim/camera_info")
        self.point_cloud_topic = rospy.get_param("~point_cloud_topic", "/airsim/point_cloud")
        self.pose_topic = rospy.get_param("~pose_topic", "/airsim/pose")
        self.output_dir = os.path.expanduser(rospy.get_param("~output_dir", "~/airsim_3dgs_dataset"))
        self.image_dir = os.path.join(self.output_dir, "images")
        self.sparse_dir = os.path.join(self.output_dir, "sparse", "0")
        self.pose_frame = rospy.get_param("~pose_frame", "base_link")

        self.min_translation = float(rospy.get_param("~min_translation", 0.10))
        self.min_rotation_deg = float(rospy.get_param("~min_rotation_deg", 5.0))
        self.min_frame_interval = float(rospy.get_param("~min_frame_interval", 0.0))
        self.max_frames = int(rospy.get_param("~max_frames", 0))
        self.cloud_stride = max(1, int(rospy.get_param("~cloud_stride", 4)))
        self.max_depth = float(rospy.get_param("~max_depth", 20.0))
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.03))
        self.sync_queue_size = int(rospy.get_param("~sync_queue_size", 20))
        self.sync_slop = float(rospy.get_param("~sync_slop", 0.05))

        ensure_dir(self.image_dir)
        ensure_dir(self.sparse_dir)

        self.bridge = CvBridge()
        self.frames = []
        self.cloud_points = []
        self.cloud_colors = []
        self.last_saved_position = None
        self.last_saved_rotation = None
        self.last_saved_time = None
        self.camera_models = {}
        self.finished = False

        self.r_base_camera = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )

        rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        info_sub = message_filters.Subscriber(self.camera_info_topic, CameraInfo)
        cloud_sub = message_filters.Subscriber(self.point_cloud_topic, PointCloud2)
        pose_sub = message_filters.Subscriber(self.pose_topic, PoseStamped)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, info_sub, cloud_sub, pose_sub],
            self.sync_queue_size,
            self.sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.synced_callback)

        rospy.on_shutdown(self.finalize)

        rospy.loginfo("airsim_to_3dgs_dataset listening on %s %s %s %s",
                      self.rgb_topic, self.camera_info_topic, self.point_cloud_topic, self.pose_topic)
        rospy.loginfo("airsim_to_3dgs_dataset pose_frame=%s point_cloud is assumed to be world-frame",
                      self.pose_frame)

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

    def collect_world_cloud(self, cloud_msg):
        points = []
        colors = []
        for index, point in enumerate(pc2.read_points(cloud_msg, skip_nans=True)):
            if index % self.cloud_stride != 0:
                continue

            xyz = np.array(point[:3], dtype=np.float64)
            depth = np.linalg.norm(xyz)
            if self.max_depth > 0.0 and depth > self.max_depth:
                continue

            rgb = pack_rgb_fields(point)
            points.append(xyz)
            colors.append(rgb)

        if not points:
            return None, None
        return np.asarray(points, dtype=np.float64), np.asarray(colors, dtype=np.uint8)

    def resolve_camera_pose(self, pose_msg):
        t_world_pose = np.array(
            [
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z,
            ],
            dtype=np.float64,
        )
        r_world_pose = SciRot.from_quat(
            [
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w,
            ]
        )

        if self.pose_frame == "camera_optical_frame":
            return t_world_pose, r_world_pose.as_matrix(), SciRot.from_matrix(r_world_pose.as_matrix())

        if self.pose_frame == "base_link":
            r_world_camera = r_world_pose.as_matrix().dot(self.r_base_camera)
            return t_world_pose, r_world_camera, SciRot.from_matrix(r_world_camera)

        raise ValueError("Unsupported pose_frame '{}', expected 'base_link' or 'camera_optical_frame'".format(self.pose_frame))

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

    def synced_callback(self, rgb_msg, camera_info_msg, cloud_msg, pose_msg):
        if self.finished:
            return

        stamp = rgb_msg.header.stamp if rgb_msg.header.stamp != rospy.Time(0) else rospy.Time.now()
        t_world_camera, r_world_camera, r_world_camera_obj = self.resolve_camera_pose(pose_msg)

        if not self.should_save_frame(stamp, t_world_camera, r_world_camera_obj):
            return

        frame_index = len(self.frames) + 1
        image_name = f"frame_{frame_index:06d}.png"
        if not self.save_image(rgb_msg, image_name):
            return

        world_points, world_colors = self.collect_world_cloud(cloud_msg)
        if world_points is not None:
            self.cloud_points.append(world_points)
            self.cloud_colors.append(world_colors)

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
            "Saved frame %d: image=%s cloud_points=%d",
            frame_index,
            image_name,
            0 if world_points is None else world_points.shape[0],
        )

        if self.max_frames > 0 and len(self.frames) >= self.max_frames:
            rospy.loginfo("Reached max_frames=%d, exporting dataset.", self.max_frames)
            self.finalize()
            rospy.signal_shutdown("dataset export completed")

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

    def write_metadata(self, points, colors):
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
                        "pose": self.pose_topic,
                    },
                    "coordinate_chain": {
                        "world": "ROS ENU",
                        "point_cloud": "assumed already in world",
                        "pose_topic_interpretation": self.pose_frame,
                        "camera_pose_used_for_export": "world -> camera_optical_frame" if self.pose_frame == "base_link" else "world -> pose_topic_frame",
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
        self.write_metadata(points, colors)

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
        AirSimTo3DGSExporter().run()
    except rospy.ROSInterruptException:
        pass
