#!/usr/bin/env python3

import os
import threading

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class CaptureOnViewpoint:
    def __init__(self):
        rospy.init_node("capture_on_viewpoint", anonymous=True)

        self.image_topic = rospy.get_param("~image_topic", "/airsim/rgb")
        self.output_dir = rospy.get_param(
            "~output_dir", os.path.expanduser("~/viewpoint_captures")
        )
        self.save_interval = float(rospy.get_param("~save_interval", 0.3))
        self.max_image_age = float(rospy.get_param("~max_image_age", 1.0))
        self.skip_duplicate_images = bool(
            rospy.get_param("~skip_duplicate_images", True)
        )
        self.duplicate_preview_width = int(
            rospy.get_param("~duplicate_preview_width", 64)
        )
        self.duplicate_diff_threshold = float(
            rospy.get_param("~duplicate_diff_threshold", 2.0)
        )

        os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_image_msg = None
        self.latest_image_receive_time = rospy.Time(0)
        self.saved_count = 0
        self.last_save_time = rospy.Time(0)
        self.last_saved_preview = None

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=5)
        rospy.Timer(rospy.Duration(0.05), self.timer_callback)

    def image_callback(self, msg):
        with self.lock:
            self.latest_image_msg = msg
            self.latest_image_receive_time = rospy.Time.now()

    def timer_callback(self, _event):
        now = rospy.Time.now()
        with self.lock:
            image_msg = self.latest_image_msg
            image_receive_time = self.latest_image_receive_time

        if image_msg is None:
            return

        since_last = (now - self.last_save_time).to_sec()
        if since_last < self.save_interval:
            return

        if image_receive_time != rospy.Time(0):
            receive_age = (now - image_receive_time).to_sec()
            if receive_age > self.max_image_age:
                rospy.logwarn_throttle(
                    2.0,
                    "CaptureOnViewpoint skip stale image by receive age: %.3fs > %.3fs",
                    receive_age,
                    self.max_image_age,
                )
                return

        image_stamp = image_msg.header.stamp
        if image_stamp != rospy.Time(0):
            image_age = (now - image_stamp).to_sec()
            if image_age > self.max_image_age:
                rospy.logwarn_throttle(
                    2.0,
                    "CaptureOnViewpoint skip stale image by stamp age: %.3fs > %.3fs",
                    image_age,
                    self.max_image_age,
                )
                return

        self.save_current_image(image_msg, now)

    def save_current_image(self, image_msg, now):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn("CaptureOnViewpoint failed to convert image: %s", str(exc))
            return

        preview = self.build_preview(cv_image)
        if self.skip_duplicate_images and self.last_saved_preview is not None:
            mean_diff = float(cv2.mean(cv2.absdiff(preview, self.last_saved_preview))[0])
            if mean_diff <= self.duplicate_diff_threshold:
                rospy.loginfo_throttle(
                    2.0,
                    "CaptureOnViewpoint skip duplicate RGB frame (mean diff %.3f <= %.3f)",
                    mean_diff,
                    self.duplicate_diff_threshold,
                )
                self.last_save_time = now
                return

        stamp = image_msg.header.stamp if image_msg.header.stamp != rospy.Time(0) else now
        ts = "{:010d}_{:09d}".format(stamp.secs, stamp.nsecs)
        filename = "rgb_{:06d}_{}.png".format(self.saved_count, ts)
        path = os.path.join(self.output_dir, filename)
        if not cv2.imwrite(path, cv_image):
            rospy.logwarn("CaptureOnViewpoint failed to save image to %s", path)
            return

        self.saved_count += 1
        self.last_save_time = now
        self.last_saved_preview = preview
        rospy.loginfo(
            "CaptureOnViewpoint saved %s (%d images total)",
            path,
            self.saved_count,
        )

    def build_preview(self, cv_image):
        height, width = cv_image.shape[:2]
        preview_width = max(8, min(self.duplicate_preview_width, width))
        preview_height = max(1, int(round(height * float(preview_width) / float(width))))
        preview = cv2.resize(
            cv_image,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        CaptureOnViewpoint().run()
    except rospy.ROSInterruptException:
        pass
