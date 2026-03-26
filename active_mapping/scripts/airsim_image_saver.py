#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
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

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class AirSimImageSaver:
    def __init__(self):
        rospy.init_node("airsim_image_saver", anonymous=False)

        self.image_topic = rospy.get_param("~image_topic", "/airsim/rgb")
        self.output_dir = rospy.get_param("~output_dir", "/tmp/airsim_rgb_frames")
        self.save_period = float(rospy.get_param("~save_period", 0.2))
        self.image_prefix = rospy.get_param("~image_prefix", "frame")

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_stamp = None
        self.last_saved_stamp = None
        self.frame_index = 0

        self._reset_output_dir(self.output_dir)

        self.sub = rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1)
        rospy.Timer(rospy.Duration.from_sec(self.save_period), self._save_timer_cb)

        rospy.loginfo(
            "AirSim image saver started. topic=%s, period=%.3fs, output_dir=%s",
            self.image_topic,
            self.save_period,
            self.output_dir,
        )

    def _reset_output_dir(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for entry in os.listdir(output_dir):
            path = os.path.join(output_dir, entry)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(5.0, "Failed to convert image from %s: %s", self.image_topic, str(exc))
            return

        with self.lock:
            self.latest_image = cv_image.copy()
            self.latest_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

    def _save_timer_cb(self, _event):
        with self.lock:
            if self.latest_image is None or self.latest_stamp is None:
                return
            if self.last_saved_stamp is not None and self.latest_stamp == self.last_saved_stamp:
                return

            image_to_save = self.latest_image.copy()
            stamp = self.latest_stamp
            self.last_saved_stamp = stamp
            self.frame_index += 1
            frame_index = self.frame_index

        timestamp_ns = stamp.to_nsec()
        filename = "{}_{:06d}_{}.png".format(self.image_prefix, frame_index, timestamp_ns)
        output_path = os.path.join(self.output_dir, filename)

        if not cv2.imwrite(output_path, image_to_save):
            rospy.logwarn("Failed to save image to %s", output_path)
            return

        rospy.loginfo_throttle(5.0, "Saved AirSim images to %s", self.output_dir)


if __name__ == "__main__":
    saver = AirSimImageSaver()
    rospy.spin()
