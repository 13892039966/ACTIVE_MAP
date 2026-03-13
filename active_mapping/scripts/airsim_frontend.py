#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import threading

def configure_ros_log_dir():
    """Use a temp log directory by default to avoid permission issues in ~/.ros/log."""
    configured_dir = os.environ.get("ROS_LOG_DIR")
    if configured_dir:
        return

    fallback_dir = os.path.join(tempfile.gettempdir(), "active_mapping_ros_logs")
    os.makedirs(fallback_dir, exist_ok=True)
    os.environ["ROS_LOG_DIR"] = fallback_dir

configure_ros_log_dir()

import rospy
import airsim
import numpy as np
import tf2_ros
import cv2  # 🌟 新增：用于解码压缩后的图像流
from cv_bridge import CvBridge

# ROS 消息类型
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header


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

# ROS optical frame: x right, y down, z forward.
FRD_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

FLU_FROM_OPTICAL = FRD_TO_FLU.dot(FRD_FROM_OPTICAL)


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


def ned_vector_to_enu(vec):
    return NED_TO_ENU.dot(np.array([vec.x_val, vec.y_val, vec.z_val], dtype=np.float64))


def ned_quat_to_enu_flu(quat):
    rot_ned_frd = airsim_quat_to_rotation_matrix(quat)
    rot_enu_flu = NED_TO_ENU.dot(rot_ned_frd).dot(FRD_TO_FLU)
    return rotation_matrix_to_quaternion(rot_enu_flu)


def ned_quat_to_enu_optical(quat):
    rot_ned_frd = airsim_quat_to_rotation_matrix(quat)
    rot_enu_optical = NED_TO_ENU.dot(rot_ned_frd).dot(FRD_FROM_OPTICAL)
    return rotation_matrix_to_quaternion(rot_enu_optical)


def optical_quat_from_camera_link_quat(camera_link_quat):
    rot_world_flu = np.array(camera_link_quat_to_rotation_matrix(camera_link_quat), dtype=np.float64)
    rot_world_optical = rot_world_flu.dot(FLU_FROM_OPTICAL)
    return rotation_matrix_to_quaternion(rot_world_optical)


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


def fill_transform(msg, stamp, parent_frame, child_frame, translation, quaternion):
    msg.header.stamp = stamp
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x = float(translation[0])
    msg.transform.translation.y = float(translation[1])
    msg.transform.translation.z = float(translation[2])
    msg.transform.rotation.x = float(quaternion[0])
    msg.transform.rotation.y = float(quaternion[1])
    msg.transform.rotation.z = float(quaternion[2])
    msg.transform.rotation.w = float(quaternion[3])


class AirSimFrontend:
    def __init__(self):
        rospy.init_node('airsim_frontend_node', anonymous=True)

        # 1. 基础配置
        self.vehicle_name = rospy.get_param("~vehicle_name", "drone_1")
        self.cam_name = rospy.get_param("~camera_name", "front_center")
        self.image_freq = float(rospy.get_param("~image_freq", 15.0))
        self.odom_freq = float(rospy.get_param("~odom_freq", 200.0))
        self.airsim_host = rospy.get_param("~airsim_host", "127.0.0.1")
        self.airsim_port = int(rospy.get_param("~airsim_port", 41451))
        self.airsim_timeout_sec = float(rospy.get_param("~airsim_timeout_sec", 60.0))
        self.airsim_retry_interval = float(rospy.get_param("~airsim_retry_interval", 1.0))
        self.max_valid_depth = float(rospy.get_param("~max_valid_depth", 20.0))
        
        # 2. 相机内参计算 (基于 AirSim 默认的 90度 FOV)
        self.width, self.height = 640, 480
        # fx = width / (2 * tan(FOV/2)) = 640 / (2 * tan(pi/4)) = 320
        self.fx = self.width / (2.0 * np.tan(np.pi / 4.0))
        self.fy = self.fx
        self.cx, self.cy = self.width / 2.0, self.height / 2.0

        # 3. ROS 工具初始化
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # 4. 发布器定义
        # 注意：队列(queue_size)设为 5 即可，强制丢弃过时数据，保证绝对实时
        self.rgb_pub = rospy.Publisher('/airsim/rgb', Image, queue_size=5)
        self.depth_pub = rospy.Publisher('/airsim/depth', Image, queue_size=5)
        self.cam_info_pub = rospy.Publisher('/airsim/camera_info', CameraInfo, queue_size=5) 
        
        self.pose_pub = rospy.Publisher('/airsim/pose', PoseStamped, queue_size=5)
        self.odom_pub = rospy.Publisher('/airsim/odom', Odometry, queue_size=5)

        # 5. 连接 AirSim 仿真器
        self.state_client = self._connect_airsim_with_retry("state")
        self.image_client = self._connect_airsim_with_retry("image")
        # rospy.loginfo("🚀 主动重建前端 (AirSim Frontend) 已就绪！开始高速泵送数据...")

    def _connect_airsim_with_retry(self, client_label):
        """Connect to AirSim with retries so roslaunch won't exit when AirSim is late."""
        deadline = None if self.airsim_timeout_sec <= 0.0 else (rospy.Time.now().to_sec() + self.airsim_timeout_sec)
        while not rospy.is_shutdown():
            try:
                client = airsim.MultirotorClient(ip=self.airsim_host, port=self.airsim_port)
                client.confirmConnection()
                rospy.loginfo(
                    "AirSim frontend %s client connected to %s:%d",
                    client_label,
                    self.airsim_host,
                    self.airsim_port,
                )
                return client
            except Exception as e:
                rospy.logwarn(
                    "AirSim frontend %s client connect failed: %s. Retrying in %.1fs",
                    client_label,
                    str(e),
                    self.airsim_retry_interval,
                )
                if deadline is not None and rospy.Time.now().to_sec() > deadline:
                    rospy.logerr("AirSim frontend connect timeout (%.1fs) reached.", self.airsim_timeout_sec)
                    raise
                rospy.sleep(self.airsim_retry_interval)
        raise rospy.ROSInterruptException("ROS shutdown before AirSim frontend connected")

    def generate_camera_info(self, header):
        """生成 ROS 标准的 CameraInfo 消息"""
        cam_info = CameraInfo()
        cam_info.header = header
        cam_info.width = self.width
        cam_info.height = self.height
        cam_info.distortion_model = "plumb_bob" # 无畸变模型
        cam_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 内参矩阵 K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        cam_info.K = [self.fx, 0.0, self.cx, 
                      0.0, self.fy, self.cy, 
                      0.0, 0.0, 1.0]
        
        # 旋转矩阵 R (单位阵)
        cam_info.R = [1.0, 0.0, 0.0, 
                      0.0, 1.0, 0.0, 
                      0.0, 0.0, 1.0]
        
        # 投影矩阵 P: [fx, 0, cx, 0,  0, fy, cy, 0,  0, 0, 1, 0]
        cam_info.P = [self.fx, 0.0, self.cx, 0.0, 
                      0.0, self.fy, self.cy, 0.0, 
                      0.0, 0.0, 1.0, 0.0]
        return cam_info

    def _get_vehicle_kinematics(self):
        vehicle_state = self.state_client.getMultirotorState(vehicle_name=self.vehicle_name)
        vehicle_kinematics = vehicle_state.kinematics_estimated

        body_pos_enu = ned_vector_to_enu(vehicle_kinematics.position)
        body_ori_enu = ned_quat_to_enu_flu(vehicle_kinematics.orientation)
        linear_vel_enu = ned_vector_to_enu(vehicle_kinematics.linear_velocity)
        angular_vel_flu = FRD_TO_FLU.dot(np.array([
            vehicle_kinematics.angular_velocity.x_val,
            vehicle_kinematics.angular_velocity.y_val,
            vehicle_kinematics.angular_velocity.z_val,
        ], dtype=np.float64))
        return body_pos_enu, body_ori_enu, linear_vel_enu, angular_vel_flu

    def _publish_body_state(self, stamp, body_pos_enu, body_ori_enu, linear_vel_enu, angular_vel_flu):
        body_tf = TransformStamped()
        fill_transform(body_tf, stamp, "world", "base_link", body_pos_enu, body_ori_enu)
        self.tf_broadcaster.sendTransform(body_tf)

        body_header = Header(stamp=stamp, frame_id="world")

        odom = Odometry()
        odom.header = body_header
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(body_pos_enu[0])
        odom.pose.pose.position.y = float(body_pos_enu[1])
        odom.pose.pose.position.z = float(body_pos_enu[2])
        odom.pose.pose.orientation.x = float(body_ori_enu[0])
        odom.pose.pose.orientation.y = float(body_ori_enu[1])
        odom.pose.pose.orientation.z = float(body_ori_enu[2])
        odom.pose.pose.orientation.w = float(body_ori_enu[3])
        odom.twist.twist.linear.x = float(linear_vel_enu[0])
        odom.twist.twist.linear.y = float(linear_vel_enu[1])
        odom.twist.twist.linear.z = float(linear_vel_enu[2])
        odom.twist.twist.angular.x = float(angular_vel_flu[0])
        odom.twist.twist.angular.y = float(angular_vel_flu[1])
        odom.twist.twist.angular.z = float(angular_vel_flu[2])
        self.odom_pub.publish(odom)

    def _image_loop(self):
        image_period = rospy.Duration.from_sec(1.0 / self.image_freq) if self.image_freq > 0.0 else None
        if image_period is None:
            return

        next_image_time = rospy.Time(0)

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if now < next_image_time:
                remaining = (next_image_time - now).to_sec()
                if remaining > 0.0:
                    rospy.sleep(min(remaining, 0.001))
                continue

            # 第一个请求 (RGB) 的第4个参数 (compress) 设为 True，极大降低 RPC 带宽和引擎内存占用
            # 第二个请求 (Depth) 保持 float 格式无压缩，以保证 3DGS 重建和建图的物理精度
            responses = self.image_client.simGetImages([
                airsim.ImageRequest(self.cam_name, airsim.ImageType.Scene, False, True),
                airsim.ImageRequest(self.cam_name, airsim.ImageType.DepthPlanar, True, False)
            ], vehicle_name=self.vehicle_name)

            if len(responses) >= 2:
                camera_pos_enu = ned_vector_to_enu(responses[0].camera_position)
                camera_link_ori_enu = ned_quat_to_enu_flu(responses[0].camera_orientation)
                camera_ori_enu = optical_quat_from_camera_link_quat(camera_link_ori_enu)

                camera_link_tf = TransformStamped()
                fill_transform(
                    camera_link_tf,
                    now,
                    "world",
                    "camera_link",
                    camera_pos_enu,
                    camera_link_ori_enu,
                )

                camera_tf = TransformStamped()
                fill_transform(
                    camera_tf,
                    now,
                    "camera_link",
                    "camera_optical_frame",
                    np.zeros(3, dtype=np.float64),
                    rotation_matrix_to_quaternion(FLU_FROM_OPTICAL),
                )

                self.tf_broadcaster.sendTransform([camera_link_tf, camera_tf])

                img_header = Header(stamp=now, frame_id="camera_optical_frame")

                raw_rgb_bytes = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = cv2.imdecode(raw_rgb_bytes, cv2.IMREAD_COLOR)

                img_depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(self.height, self.width)
                invalid_mask = (~np.isfinite(img_depth)) | (img_depth <= 0.0) | (img_depth >= self.max_valid_depth)
                img_depth[invalid_mask] = np.nan

                rgb_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="bgr8")
                depth_msg = self.bridge.cv2_to_imgmsg(img_depth, encoding="32FC1")

                rgb_msg.header = img_header
                depth_msg.header = img_header
                cam_info_msg = self.generate_camera_info(img_header)

                body_header = Header(stamp=now, frame_id="world")
                pose_msg = PoseStamped()
                pose_msg.header = body_header
                pose_msg.pose.position.x = float(camera_pos_enu[0])
                pose_msg.pose.position.y = float(camera_pos_enu[1])
                pose_msg.pose.position.z = float(camera_pos_enu[2])
                pose_msg.pose.orientation.x = float(camera_ori_enu[0])
                pose_msg.pose.orientation.y = float(camera_ori_enu[1])
                pose_msg.pose.orientation.z = float(camera_ori_enu[2])
                pose_msg.pose.orientation.w = float(camera_ori_enu[3])

                self.pose_pub.publish(pose_msg)
                self.rgb_pub.publish(rgb_msg)
                self.depth_pub.publish(depth_msg)
                self.cam_info_pub.publish(cam_info_msg)

                del raw_rgb_bytes, img_rgb, img_depth, responses

            next_image_time = now + image_period

    def run(self):
        image_thread = None
        if self.image_freq > 0.0:
            image_thread = threading.Thread(target=self._image_loop, name="airsim_image_loop", daemon=True)
            image_thread.start()

        odom_rate = rospy.Rate(self.odom_freq)

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            body_pos_enu, body_ori_enu, linear_vel_enu, angular_vel_flu = self._get_vehicle_kinematics()
            self._publish_body_state(now, body_pos_enu, body_ori_enu, linear_vel_enu, angular_vel_flu)

            odom_rate.sleep()

        if image_thread is not None:
            image_thread.join(timeout=1.0)

if __name__ == '__main__':
    try:
        AirSimFrontend().run()
    except rospy.ROSInterruptException:
        pass
