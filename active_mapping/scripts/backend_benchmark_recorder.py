#!/usr/bin/env python3

import csv
import json
import os
from datetime import datetime

import rospy

from active_mapping.msg import BackendBenchmark


class BackendBenchmarkRecorder:
    FIELD_NAMES = [
        "stamp",
        "window_duration_sec",
        "processed_count",
        "queue_drop_count",
        "cloud_age_mean",
        "cloud_age_max",
        "support_ms",
        "search_ms",
        "score_ms",
        "total_ms",
        "raw_neighbors_mean",
        "zeroed_ratio",
        "edge_mean",
        "texture_mean",
    ]

    def __init__(self):
        rospy.init_node("backend_benchmark_recorder", anonymous=True)
        self.topic = rospy.get_param("~topic", "/active_mapping/backend_benchmark")
        self.output_root = os.path.expanduser(rospy.get_param("~output_root", "~/active_mapping_benchmarks"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_root, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "timeseries.csv")
        self.summary_path = os.path.join(self.output_dir, "summary.json")
        self.samples = []
        self.start_time = None
        self.end_time = None

        rospy.Subscriber(self.topic, BackendBenchmark, self.callback, queue_size=50)
        rospy.on_shutdown(self.write_outputs)
        rospy.loginfo("Backend benchmark recorder writing to %s", self.output_dir)

    def callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = stamp
        self.end_time = stamp
        self.samples.append({
            "stamp": stamp,
            "window_duration_sec": msg.window_duration_sec,
            "processed_count": int(msg.processed_count),
            "queue_drop_count": int(msg.queue_drop_count),
            "cloud_age_mean": msg.cloud_age_mean,
            "cloud_age_max": msg.cloud_age_max,
            "support_ms": msg.support_ms,
            "search_ms": msg.search_ms,
            "score_ms": msg.score_ms,
            "total_ms": msg.total_ms,
            "raw_neighbors_mean": msg.raw_neighbors_mean,
            "zeroed_ratio": msg.zeroed_ratio,
            "edge_mean": msg.edge_mean,
            "texture_mean": msg.texture_mean,
        })

    def aggregate_metric(self, key):
        values = [sample[key] for sample in self.samples]
        if not values:
            return None
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }

    def write_outputs(self):
        if not self.samples:
            rospy.loginfo("Backend benchmark recorder: no samples captured, nothing to write.")
            return

        with open(self.csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELD_NAMES)
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(sample)

        summary = {
            "sample_count": len(self.samples),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": (self.end_time - self.start_time) if self.start_time is not None and self.end_time is not None else 0.0,
            "metrics": {
                key: self.aggregate_metric(key)
                for key in self.FIELD_NAMES
                if key != "stamp"
            },
            "csv_path": self.csv_path,
            "summary_path": self.summary_path,
        }

        with open(self.summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2, sort_keys=True)

        rospy.loginfo("Backend benchmark summary written to %s", self.summary_path)
        rospy.loginfo("Backend benchmark timeseries written to %s", self.csv_path)


if __name__ == "__main__":
    try:
        BackendBenchmarkRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
