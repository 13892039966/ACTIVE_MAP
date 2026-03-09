#!/usr/bin/env python3
# scripts/airsim_teleop.py
# -*- coding: utf-8 -*-

import rospy
import airsim
import math
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R

class AirSimTeleop:
    def __init__(self):
        rospy.init_node('airsim_teleop', anonymous=True)
        self.airsim_host = rospy.get_param("~airsim_host", "127.0.0.1")
        self.airsim_port = int(rospy.get_param("~airsim_port", 41451))
        self.airsim_timeout_sec = float(rospy.get_param("~airsim_timeout_sec", 60.0))
        self.airsim_retry_interval = float(rospy.get_param("~airsim_retry_interval", 1.0))
        self.client = self._connect_airsim_with_retry()
        
        # 获取相机的初始出生点和姿态
        init_pose = self.client.simGetVehiclePose()
        self.x = init_pose.position.x_val
        self.y = init_pose.position.y_val
        self.z = init_pose.position.z_val
        
        q = init_pose.orientation
        r = R.from_quat([q.x_val, q.y_val, q.z_val, q.w_val])
        euler = r.as_euler('zyx') 
        self.yaw = euler[0]
        # 🌟 记录初始的 Pitch 角
        self.pitch = euler[1]
        
        # 初始化所有速度变量
        self.vx = self.vy = self.vz = self.vyaw = self.vpitch = 0.0
        
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_callback)
        rospy.loginfo("🎮 AirSim Teleop 驱动已就绪 (支持 6DoF 飞行)！")

    def _connect_airsim_with_retry(self):
        """Connect to AirSim with retries so roslaunch won't exit when AirSim is late."""
        deadline = None if self.airsim_timeout_sec <= 0.0 else (rospy.Time.now().to_sec() + self.airsim_timeout_sec)
        while not rospy.is_shutdown():
            try:
                client = airsim.MultirotorClient(ip=self.airsim_host, port=self.airsim_port)
                client.confirmConnection()
                rospy.loginfo("AirSim teleop connected to %s:%d", self.airsim_host, self.airsim_port)
                return client
            except Exception as e:
                rospy.logwarn("AirSim teleop connect failed: %s. Retrying in %.1fs", str(e), self.airsim_retry_interval)
                if deadline is not None and rospy.Time.now().to_sec() > deadline:
                    rospy.logerr("AirSim teleop connect timeout (%.1fs) reached.", self.airsim_timeout_sec)
                    raise
                rospy.sleep(self.airsim_retry_interval)
        raise rospy.ROSInterruptException("ROS shutdown before AirSim teleop connected")

    def cmd_callback(self, msg):
        self.vx = msg.linear.x      
        self.vy = msg.linear.y      
        self.vz = msg.linear.z      
        self.vyaw = msg.angular.z   
        # 🌟 接收 Pitch 速度指令
        self.vpitch = msg.angular.y 

    def run(self):
        rate = rospy.Rate(30)
        dt = 1.0 / 30.0
        
        # 定义最大低头和抬头角度 (防万向节死锁，限制在正负 85 度左右)
        max_pitch = math.radians(85)
        
        while not rospy.is_shutdown():
            if self.vx != 0 or self.vy != 0 or self.vz != 0 or self.vyaw != 0 or self.vpitch != 0:
                v_forward = self.vx
                v_right = -self.vy
                v_down = -self.vz
                yaw_rate = -self.vyaw
                
                # 在 AirSim 的 NED 坐标系下，正 Pitch 代表低头。
                # 由于键盘上是 W(正角速度) 代表抬头，所以这里取反
                pitch_rate = -self.vpitch 
                
                self.yaw += yaw_rate * dt
                self.pitch += pitch_rate * dt
                
                # 🌟 限制 Pitch 角度，防止视角翻转 (非常重要)
                self.pitch = max(-max_pitch, min(max_pitch, self.pitch))
                
                # 结合偏航角计算绝对 XY 坐标的位移 (让你总是朝着相机正前方水平飞行)
                self.x += (v_forward * math.cos(self.yaw) - v_right * math.sin(self.yaw)) * dt
                self.y += (v_forward * math.sin(self.yaw) + v_right * math.cos(self.yaw)) * dt
                self.z += v_down * dt
                
                pose = airsim.Pose()
                pose.position.x_val = self.x
                pose.position.y_val = self.y
                pose.position.z_val = self.z
                
                # 🌟 重新打包带有 Pitch 的四元数 (ZYX 顺序：Yaw, Pitch, Roll)
                q = R.from_euler('zyx', [self.yaw, self.pitch, 0.0]).as_quat()
                pose.orientation.x_val = q[0]
                pose.orientation.y_val = q[1]
                pose.orientation.z_val = q[2]
                pose.orientation.w_val = q[3]
                
                self.client.simSetVehiclePose(pose, True)
                
            rate.sleep()

if __name__ == '__main__':
    try:
        AirSimTeleop().run()
    except rospy.ROSInterruptException:
        pass
