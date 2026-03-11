#!/usr/bin/env python3

import math

import airsim
import rospy
from quadrotor_msgs.msg import PositionCommand
from scipy.spatial.transform import Rotation as R


class PositionCmdToAirSim:
    def __init__(self):
        rospy.init_node("position_cmd_to_airsim", anonymous=True)

        self.airsim_host = rospy.get_param("~airsim_host", "127.0.0.1")
        self.airsim_port = int(rospy.get_param("~airsim_port", 41451))
        self.cmd_topic = rospy.get_param("~cmd_topic", "planning/pos_cmd")
        self.timeout_sec = float(rospy.get_param("~airsim_timeout_sec", 60.0))
        self.retry_interval = float(rospy.get_param("~airsim_retry_interval", 1.0))
        self.control_rate = float(rospy.get_param("~control_rate", 30.0))
        self.cmd_timeout = float(rospy.get_param("~cmd_timeout", 0.2))

        self.client = self._connect_airsim_with_retry()
        self.latest_cmd = None
        self.latest_cmd_time = rospy.Time(0)

        rospy.Subscriber(self.cmd_topic, PositionCommand, self.cmd_callback, queue_size=20)

    def _connect_airsim_with_retry(self):
        deadline = None if self.timeout_sec <= 0.0 else (rospy.Time.now().to_sec() + self.timeout_sec)
        while not rospy.is_shutdown():
            try:
                client = airsim.MultirotorClient(ip=self.airsim_host, port=self.airsim_port)
                client.confirmConnection()
                rospy.loginfo("PositionCommand bridge connected to %s:%d", self.airsim_host,
                              self.airsim_port)
                return client
            except Exception as exc:
                rospy.logwarn("AirSim bridge connect failed: %s. Retrying in %.1fs", str(exc),
                              self.retry_interval)
                if deadline is not None and rospy.Time.now().to_sec() > deadline:
                    rospy.logerr("AirSim bridge connect timeout (%.1fs) reached.", self.timeout_sec)
                    raise
                rospy.sleep(self.retry_interval)
        raise rospy.ROSInterruptException("ROS shutdown before AirSim bridge connected")

    def cmd_callback(self, msg):
        self.latest_cmd = msg
        self.latest_cmd_time = rospy.Time.now()
        rospy.loginfo_throttle(
            1.0,
            "PositionCommand bridge cmd: vel=(%.2f, %.2f, %.2f) acc=(%.2f, %.2f, %.2f) yaw=%.2f",
            msg.velocity.x,
            msg.velocity.y,
            msg.velocity.z,
            msg.acceleration.x,
            msg.acceleration.y,
            msg.acceleration.z,
            msg.yaw,
        )

    def _publish_pose(self, cmd):
        pose = airsim.Pose()
        pose.position.x_val = cmd.position.x
        pose.position.y_val = -cmd.position.y
        pose.position.z_val = -cmd.position.z

        q = R.from_euler("zyx", [-cmd.yaw, 0.0, 0.0]).as_quat()
        pose.orientation.x_val = q[0]
        pose.orientation.y_val = q[1]
        pose.orientation.z_val = q[2]
        pose.orientation.w_val = q[3]

        self.client.simSetVehiclePose(pose, True)

    def run(self):
        rate = rospy.Rate(self.control_rate)

        while not rospy.is_shutdown():
            if self.latest_cmd is None:
                rate.sleep()
                continue

            age = (rospy.Time.now() - self.latest_cmd_time).to_sec()
            if age > self.cmd_timeout:
                rate.sleep()
                continue

            self._publish_pose(self.latest_cmd)
            rate.sleep()


if __name__ == "__main__":
    try:
        PositionCmdToAirSim().run()
    except rospy.ROSInterruptException:
        pass
