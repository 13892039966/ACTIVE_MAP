#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

class AirSimFrontend:
    def __init__(self):
        rospy.init_node('airsim_frontend_node', anonymous=True)

        # 1. 基础配置
        self.vehicle_name = "Drone1"
        self.cam_name = "front"
        self.freq = 15.0  # 频率控制在 15Hz，兼顾流畅度与性能
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
        self.client = self._connect_airsim_with_retry()
        # rospy.loginfo("🚀 主动重建前端 (AirSim Frontend) 已就绪！开始高速泵送数据...")

    def _connect_airsim_with_retry(self):
        """Connect to AirSim with retries so roslaunch won't exit when AirSim is late."""
        deadline = None if self.airsim_timeout_sec <= 0.0 else (rospy.Time.now().to_sec() + self.airsim_timeout_sec)
        while not rospy.is_shutdown():
            try:
                client = airsim.MultirotorClient(ip=self.airsim_host, port=self.airsim_port)
                client.confirmConnection()
                rospy.loginfo("AirSim frontend connected to %s:%d", self.airsim_host, self.airsim_port)
                return client
            except Exception as e:
                rospy.logwarn("AirSim frontend connect failed: %s. Retrying in %.1fs", str(e), self.airsim_retry_interval)
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

    def run(self):
        rate = rospy.Rate(self.freq)
        while not rospy.is_shutdown():
            # ================= 1. 从 AirSim 拉取数据 =================
            # 🌟 核心内存优化：
            # 第一个请求 (RGB) 的第4个参数 (compress) 设为 True，极大降低 RPC 带宽和引擎内存占用
            # 第二个请求 (Depth) 保持 float 格式无压缩，以保证 3DGS 重建和建图的物理精度
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.cam_name, airsim.ImageType.Scene, False, True),
                airsim.ImageRequest(self.cam_name, airsim.ImageType.DepthPlanar, True, False)
            ], vehicle_name=self.vehicle_name)

            if len(responses) < 2: 
                continue

            # 🌟 核心：在这一刻锁定统一的时间戳
            now = rospy.Time.now()

            p = responses[0].camera_position
            o = responses[0].camera_orientation

            # AirSim NED坐标系 (北东地) -> ROS ENU坐标系 (前左上/东北天)
            pos_enu = (p.x_val, -p.y_val, -p.z_val)
            ori_enu = (o.x_val, -o.y_val, -o.z_val, o.w_val)

            # ================= 2. 广播动态 TF =================
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = "world"          # 父坐标系
            t.child_frame_id = "base_link"       # 子坐标系 (无人机中心)

            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos_enu
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = ori_enu
            
            self.tf_broadcaster.sendTransform(t) 

            # ================= 3. 封装消息 =================
            # A. 视觉相关消息 (绑定相机光心坐标系)
            img_header = Header(stamp=now, frame_id="camera_optical_frame")
            
            # 🌟 核心内存优化：使用 cv2 解码压缩后的二进制图像流 (默认解码为 BGR)
            raw_rgb_bytes = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = cv2.imdecode(raw_rgb_bytes, cv2.IMREAD_COLOR)
            
            # 深度图维持原样 (float32 解析)
            img_depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(self.height, self.width)
            # Treat no-return and saturated depths as invalid so they do not become a fake far plane.
            invalid_mask = (~np.isfinite(img_depth)) | (img_depth <= 0.0) | (img_depth >= self.max_valid_depth)
            img_depth[invalid_mask] = np.nan

            # CvBridge 打包，注意 OpenCV 解码出来是 BGR，所以 encoding 用 "bgr8"
            rgb_msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="bgr8")
            depth_msg = self.bridge.cv2_to_imgmsg(img_depth, encoding="32FC1")
            
            rgb_msg.header = img_header
            depth_msg.header = img_header
            cam_info_msg = self.generate_camera_info(img_header)

            # B. 本体运动消息 (绑定世界坐标系)
            body_header = Header(stamp=now, frame_id="world")

            pose_msg = PoseStamped()
            pose_msg.header = body_header
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = pos_enu
            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = ori_enu

            odom = Odometry()
            odom.header = body_header
            odom.child_frame_id = "base_link"
            odom.pose.pose = pose_msg.pose

            # ================= 4. 极速发布 =================
            self.pose_pub.publish(pose_msg)
            self.odom_pub.publish(odom)
            
            self.rgb_pub.publish(rgb_msg)
            self.depth_pub.publish(depth_msg)
            self.cam_info_pub.publish(cam_info_msg)

            # 显式清理大对象，协助垃圾回收，延缓内存碎片化
            del raw_rgb_bytes, img_rgb, img_depth, responses
            
            rate.sleep()

if __name__ == '__main__':
    try:
        AirSimFrontend().run()
    except rospy.ROSInterruptException:
        pass
