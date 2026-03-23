#!/usr/bin/env python3

import rospy
import tf2_ros

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry


class ExternalPoseTFBroadcaster:
    def __init__(self):
        rospy.init_node("external_pose_tf_broadcaster", anonymous=True)

        self.pose_topic = rospy.get_param("~pose_topic", "/external/camera_pose")
        self.pose_type = rospy.get_param("~pose_type", "PoseStamped")
        self.parent_frame = rospy.get_param("~parent_frame", "world")
        self.child_frame = rospy.get_param("~child_frame", "camera_optical_frame")

        self.broadcaster = tf2_ros.TransformBroadcaster()

        pose_type_norm = self.pose_type.strip().lower()
        if pose_type_norm == "posestamped":
            self.sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_stamped_callback, queue_size=10)
        elif pose_type_norm == "odometry":
            self.sub = rospy.Subscriber(self.pose_topic, Odometry, self.odometry_callback, queue_size=10)
        else:
            raise ValueError("Unsupported pose_type '{}', expected PoseStamped or Odometry".format(self.pose_type))

        rospy.loginfo(
            "external_pose_tf_broadcaster listening on %s (%s), publishing %s -> %s",
            self.pose_topic,
            self.pose_type,
            self.parent_frame,
            self.child_frame,
        )

    def publish_tf(self, stamp, pose):
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.parent_frame
        msg.child_frame_id = self.child_frame
        msg.transform.translation.x = pose.position.x
        msg.transform.translation.y = pose.position.y
        msg.transform.translation.z = pose.position.z
        msg.transform.rotation = pose.orientation
        self.broadcaster.sendTransform(msg)

    def pose_stamped_callback(self, msg):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        self.publish_tf(stamp, msg.pose)

    def odometry_callback(self, msg):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        self.publish_tf(stamp, msg.pose.pose)


if __name__ == "__main__":
    try:
        ExternalPoseTFBroadcaster()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
