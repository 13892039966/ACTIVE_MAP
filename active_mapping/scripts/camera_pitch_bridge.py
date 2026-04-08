#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile


def configure_ros_log_dir():
    """Use a temp log directory by default to avoid permission issues in ~/.ros/log."""
    configured_dir = os.environ.get("ROS_LOG_DIR")
    if configured_dir:
        return

    fallback_dir = os.path.join(tempfile.gettempdir(), "active_mapping_ros_logs")
    os.makedirs(fallback_dir, exist_ok=True)
    os.environ["ROS_LOG_DIR"] = fallback_dir


configure_ros_log_dir()

import airsim
import numpy as np
import rospy
from std_msgs.msg import Float64


NED_TO_ENU = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
], dtype=np.float64)

FRD_TO_FLU = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
], dtype=np.float64)


def airsim_quat_to_rotation_matrix(quat):
    x, y, z, w = quat.x_val, quat.y_val, quat.z_val, quat.w_val
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def rotation_matrix_to_quaternion(rot):
    trace = np.trace(rot)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat


def rpy_to_rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float64)
    rot_y = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float64)
    rot_z = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return rot_z.dot(rot_y).dot(rot_x)


def camera_link_quat_to_rotation_matrix(quat_xyzw):
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def enu_flu_quat_to_ned_frd(quat_xyzw):
    rot_enu_flu = np.array(camera_link_quat_to_rotation_matrix(quat_xyzw), dtype=np.float64)
    rot_ned_frd = NED_TO_ENU.T.dot(rot_enu_flu).dot(FRD_TO_FLU.T)
    return rotation_matrix_to_quaternion(rot_ned_frd)


def ned_frd_quat_to_enu_flu(quat):
    rot_ned_frd = airsim_quat_to_rotation_matrix(quat)
    return NED_TO_ENU.dot(rot_ned_frd).dot(FRD_TO_FLU)


class CameraPitchBridge:
    def __init__(self):
        rospy.init_node("camera_pitch_bridge", anonymous=True)

        self.vehicle_name = rospy.get_param("~vehicle_name", "drone_1")
        self.camera_name = rospy.get_param("~camera_name", "front_center")
        self.pitch_topic = rospy.get_param("~pitch_topic", "/airsim/camera_pitch_cmd")
        self.airsim_host = rospy.get_param("~airsim_host", "127.0.0.1")
        self.airsim_port = int(rospy.get_param("~airsim_port", 41451))
        self.airsim_timeout_sec = float(rospy.get_param("~airsim_timeout_sec", 60.0))
        self.airsim_retry_interval = float(rospy.get_param("~airsim_retry_interval", 1.0))

        self.camera_mount_pos_ned = np.array([
            float(rospy.get_param("~camera_mount_x", 0.35)),
            float(rospy.get_param("~camera_mount_y", 0.0)),
            float(rospy.get_param("~camera_mount_z", -0.2)),
        ], dtype=np.float64)
        self.camera_mount_rpy_deg = np.array([
            float(rospy.get_param("~camera_mount_roll_deg", 0.0)),
            float(rospy.get_param("~camera_mount_pitch_deg", 0.0)),
            float(rospy.get_param("~camera_mount_yaw_deg", 0.0)),
        ], dtype=np.float64)
        self.pitch_min_deg = float(rospy.get_param("~pitch_min_deg", -89.0))
        self.pitch_max_deg = float(rospy.get_param("~pitch_max_deg", 89.0))
        self.pitch_sign = float(rospy.get_param("~pitch_sign", 1.0))
        self.baseline_from_current_pose = bool(rospy.get_param("~baseline_from_current_pose", True))
        self.pitch_speed_deg_per_sec = float(rospy.get_param("~pitch_speed_deg_per_sec", 20.0))
        self.update_rate_hz = float(rospy.get_param("~update_rate_hz", 30.0))

        self.client = self._connect_airsim_with_retry()
        self.baseline_pose = self._get_initial_baseline_pose()
        self.current_pitch_deg = 0.0
        self.target_pitch_deg = 0.0
        self.last_update_time = rospy.Time.now()
        self.pitch_sub = rospy.Subscriber(self.pitch_topic, Float64, self.pitch_callback, queue_size=10)
        self.update_timer = rospy.Timer(
            rospy.Duration.from_sec(1.0 / max(self.update_rate_hz, 1e-3)),
            self.update_pitch,
        )

        rospy.loginfo(
            "camera_pitch_bridge listening on %s for %s/%s (baseline_from_current_pose=%s, pitch_sign=%.1f, speed=%.2f deg/s, rate=%.2f Hz)",
            self.pitch_topic,
            self.vehicle_name,
            self.camera_name,
            str(self.baseline_from_current_pose),
            self.pitch_sign,
            self.pitch_speed_deg_per_sec,
            self.update_rate_hz,
        )

    def _connect_airsim_with_retry(self):
        deadline = None if self.airsim_timeout_sec <= 0.0 else (rospy.Time.now().to_sec() + self.airsim_timeout_sec)
        while not rospy.is_shutdown():
            try:
                client = airsim.MultirotorClient(ip=self.airsim_host, port=self.airsim_port)
                client.confirmConnection()
                rospy.loginfo(
                    "camera_pitch_bridge connected to AirSim at %s:%d",
                    self.airsim_host,
                    self.airsim_port,
                )
                return client
            except Exception as exc:
                rospy.logwarn(
                    "camera_pitch_bridge AirSim connect failed: %s. Retrying in %.1fs",
                    str(exc),
                    self.airsim_retry_interval,
                )
                if deadline is not None and rospy.Time.now().to_sec() > deadline:
                    rospy.logerr("camera_pitch_bridge connect timeout (%.1fs) reached.", self.airsim_timeout_sec)
                    raise
                rospy.sleep(self.airsim_retry_interval)
        raise rospy.ROSInterruptException("ROS shutdown before camera_pitch_bridge connected")

    def _get_initial_baseline_pose(self):
        if self.baseline_from_current_pose:
            try:
                camera_info = self.client.simGetCameraInfo(self.camera_name, vehicle_name=self.vehicle_name)
                rospy.loginfo("camera_pitch_bridge using current AirSim camera pose as baseline")
                return camera_info.pose
            except Exception as exc:
                rospy.logwarn(
                    "Failed to read current camera pose from AirSim, falling back to configured mount pose: %s",
                    str(exc),
                )

        fallback_quat_ros = rotation_matrix_to_quaternion(
            rpy_to_rotation_matrix(
                np.deg2rad(self.camera_mount_rpy_deg[0]),
                np.deg2rad(self.camera_mount_rpy_deg[1]),
                np.deg2rad(self.camera_mount_rpy_deg[2]),
            )
        )
        fallback_quat_ned = enu_flu_quat_to_ned_frd(fallback_quat_ros)
        return airsim.Pose(
            airsim.Vector3r(
                float(self.camera_mount_pos_ned[0]),
                float(self.camera_mount_pos_ned[1]),
                float(self.camera_mount_pos_ned[2]),
            ),
            airsim.Quaternionr(
                float(fallback_quat_ned[0]),
                float(fallback_quat_ned[1]),
                float(fallback_quat_ned[2]),
                float(fallback_quat_ned[3]),
            ),
        )

    def pitch_callback(self, msg):
        # Interpret the input as a ROS camera_link pitch in radians.
        pitch_deg = np.rad2deg(float(msg.data)) * self.pitch_sign
        self.target_pitch_deg = float(np.clip(pitch_deg, self.pitch_min_deg, self.pitch_max_deg))

    def update_pitch(self, event):
        now = event.current_real if event.current_real != rospy.Time() else rospy.Time.now()
        dt = max((now - self.last_update_time).to_sec(), 0.0)
        self.last_update_time = now
        if dt <= 0.0:
            return

        max_step = self.pitch_speed_deg_per_sec * dt
        error = self.target_pitch_deg - self.current_pitch_deg
        if abs(error) < 1e-6:
            return

        step = float(np.clip(error, -max_step, max_step))
        self.current_pitch_deg += step
        self.apply_pitch_deg(self.current_pitch_deg)

    def apply_pitch_deg(self, pitch_deg):
        clamped_pitch_deg = float(np.clip(pitch_deg, self.pitch_min_deg, self.pitch_max_deg))
        if clamped_pitch_deg != pitch_deg:
            rospy.logwarn_throttle(
                1.0,
                "Clamped camera pitch from %.2f deg to %.2f deg",
                pitch_deg,
                clamped_pitch_deg,
            )

        baseline_rot_ros = ned_frd_quat_to_enu_flu(self.baseline_pose.orientation)
        delta_rot_ros = rpy_to_rotation_matrix(0.0, np.deg2rad(clamped_pitch_deg), 0.0)
        target_rot_ros = baseline_rot_ros.dot(delta_rot_ros)
        camera_quat_ned = rotation_matrix_to_quaternion(
            NED_TO_ENU.T.dot(target_rot_ros).dot(FRD_TO_FLU.T)
        )

        camera_pose = airsim.Pose(
            airsim.Vector3r(
                float(self.baseline_pose.position.x_val),
                float(self.baseline_pose.position.y_val),
                float(self.baseline_pose.position.z_val),
            ),
            airsim.Quaternionr(
                float(camera_quat_ned[0]),
                float(camera_quat_ned[1]),
                float(camera_quat_ned[2]),
                float(camera_quat_ned[3]),
            ),
        )

        try:
            self.client.simSetCameraPose(self.camera_name, camera_pose, vehicle_name=self.vehicle_name)
        except Exception as exc:
            rospy.logwarn_throttle(
                1.0,
                "Failed to apply camera pitch %.2f deg: %s",
                clamped_pitch_deg,
                str(exc),
            )


if __name__ == "__main__":
    try:
        CameraPitchBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
