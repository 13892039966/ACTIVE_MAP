#!/usr/bin/env python3

import math
import struct
import threading

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class VoxelCell:
    __slots__ = (
        "count",
        "sum_xyz",
        "sum_rgb",
        "centroid",
        "mean_rgb",
        "entropy",
        "dirty",
    )

    def __init__(self):
        self.count = 0
        self.sum_xyz = np.zeros(3, dtype=np.float64)
        self.sum_rgb = np.zeros(3, dtype=np.float64)
        self.centroid = np.zeros(3, dtype=np.float64)
        self.mean_rgb = np.zeros(3, dtype=np.float64)
        self.entropy = 0.0
        self.dirty = True

    def add_point(self, xyz, rgb):
        self.count += 1
        self.sum_xyz += xyz
        self.sum_rgb += rgb
        inv = 1.0 / float(self.count)
        self.centroid = self.sum_xyz * inv
        self.mean_rgb = self.sum_rgb * inv
        self.dirty = True


class StructureTensorMapNode:
    def __init__(self):
        rospy.init_node("structure_tensor_map", anonymous=True)

        self.input_topic = rospy.get_param("~input_topic", "/point")
        self.output_topic = rospy.get_param("~output_topic", "/global_map")
        self.output_intensity_topic = rospy.get_param("~output_intensity_topic", "/global_map_intensity")
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.15))
        self.max_depth = float(rospy.get_param("~max_depth", 30.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))
        self.neighbor_radius = int(rospy.get_param("~neighbor_radius", 1))
        self.knn_k = int(rospy.get_param("~knn_k", 12))
        self.min_neighbors = int(rospy.get_param("~min_neighbors", 4))
        self.max_points_per_cloud = int(rospy.get_param("~max_points_per_cloud", 250000))
        self.max_publish_voxels = int(rospy.get_param("~max_publish_voxels", 200000))
        self.entropy_gamma = float(rospy.get_param("~entropy_gamma", 0.8))
        self.knn_sigma_scale = float(rospy.get_param("~knn_sigma_scale", 1.5))

        self.cells = {}
        self.dirty_keys = set()
        self.frame_id = None
        self.last_stamp = rospy.Time(0)
        self.state_lock = threading.Lock()

        self.cloud_sub = rospy.Subscriber(self.input_topic, PointCloud2, self.cloud_callback, queue_size=1)
        self.map_pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.intensity_pub = rospy.Publisher(self.output_intensity_topic, PointCloud2, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.publish_rate, 1e-3)), self.on_timer)

        rospy.loginfo(
            "structure_tensor_map subscribed to %s, publishing %s and %s",
            self.input_topic,
            self.output_topic,
            self.output_intensity_topic,
        )

    def voxel_key(self, xyz):
        return (
            int(math.floor(xyz[0] / self.voxel_size)),
            int(math.floor(xyz[1] / self.voxel_size)),
            int(math.floor(xyz[2] / self.voxel_size)),
        )

    def neighboring_keys(self, key):
        kr = self.neighbor_radius
        for dx in range(-kr, kr + 1):
            for dy in range(-kr, kr + 1):
                for dz in range(-kr, kr + 1):
                    yield (key[0] + dx, key[1] + dy, key[2] + dz)

    def unpack_rgb(self, point, field_names):
        if "r" in field_names and "g" in field_names and "b" in field_names:
            return np.array([float(point["r"]), float(point["g"]), float(point["b"])], dtype=np.float64)

        rgb_value = None
        if "rgb" in field_names:
            rgb_value = point["rgb"]
        elif "rgba" in field_names:
            rgb_value = point["rgba"]

        if rgb_value is None:
            return None

        if isinstance(rgb_value, float):
            packed = struct.unpack("I", struct.pack("f", rgb_value))[0]
        else:
            packed = int(rgb_value)

        r = (packed >> 16) & 255
        g = (packed >> 8) & 255
        b = packed & 255
        return np.array([float(r), float(g), float(b)], dtype=np.float64)

    def cloud_callback(self, msg):
        with self.state_lock:
            self.frame_id = msg.header.frame_id or self.frame_id or "map"
            self.last_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

        field_names = [field.name for field in msg.fields]
        updated_keys = set()
        processed = 0
        skipped = 0

        for raw_point in pc2.read_points(msg, field_names=field_names, skip_nans=True):
            point = dict(zip(field_names, raw_point))
            xyz = np.array([point["x"], point["y"], point["z"]], dtype=np.float64)
            if not np.all(np.isfinite(xyz)):
                skipped += 1
                continue
            if np.dot(xyz, xyz) > self.max_depth * self.max_depth:
                skipped += 1
                continue

            rgb = self.unpack_rgb(point, field_names)
            if rgb is None:
                skipped += 1
                continue

            key = self.voxel_key(xyz)
            with self.state_lock:
                cell = self.cells.get(key)
                if cell is None:
                    cell = VoxelCell()
                    self.cells[key] = cell
                cell.add_point(xyz, rgb)
            updated_keys.add(key)

            processed += 1
            if self.max_points_per_cloud > 0 and processed >= self.max_points_per_cloud:
                break

        impacted = set()
        for key in updated_keys:
            impacted.add(key)
            impacted.update(self.neighboring_keys(key))

        with self.state_lock:
            for key in impacted:
                if key in self.cells:
                    self.cells[key].dirty = True
                    self.dirty_keys.add(key)

            voxel_count = len(self.cells)
            dirty_count = len(self.dirty_keys)

        rospy.loginfo_throttle(
            2.0,
            "structure_tensor_map cloud processed=%d skipped=%d voxels=%d dirty=%d",
            processed,
            skipped,
            voxel_count,
            dirty_count,
        )

    def compute_residual_entropy(self, key, center_cell):
        neighbors = []
        for nkey in self.neighboring_keys(key):
            neighbor = self.cells_snapshot.get(nkey)
            if neighbor is None or neighbor.count == 0:
                continue
            dist = float(np.linalg.norm(neighbor.centroid - center_cell.centroid))
            neighbors.append((dist, neighbor.mean_rgb))

        if len(neighbors) < self.min_neighbors:
            return 0.0

        neighbors.sort(key=lambda item: item[0])
        knn = neighbors[: max(self.knn_k, self.min_neighbors)]

        dists = np.asarray([item[0] for item in knn], dtype=np.float64)
        colors = np.asarray([item[1] for item in knn], dtype=np.float64) / 255.0

        sigma = max(self.voxel_size * self.knn_sigma_scale, 1e-6)
        weights = np.exp(-(dists * dists) / (2.0 * sigma * sigma))
        weights_sum = float(np.sum(weights))
        if weights_sum <= 1e-12:
            return 0.0
        weights /= weights_sum

        color_center = np.sum(colors * weights[:, None], axis=0)
        residuals = colors - color_center[None, :]

        cov = np.zeros((3, 3), dtype=np.float64)
        for idx in range(residuals.shape[0]):
            cov += weights[idx] * np.outer(residuals[idx], residuals[idx])

        reg = 1e-6 * np.eye(3, dtype=np.float64)
        det_cov = float(np.linalg.det(cov + reg))
        det_cov = max(det_cov, 1e-18)

        entropy = 0.5 * math.log(((2.0 * math.pi * math.e) ** 3) * det_cov)
        entropy = max(entropy, 0.0)
        entropy = 1.0 - math.exp(-entropy)
        if self.entropy_gamma > 0.0:
            entropy = math.pow(max(0.0, min(1.0, entropy)), self.entropy_gamma)
        return max(0.0, min(1.0, entropy))

    def recompute_dirty_cells(self):
        with self.state_lock:
            if not self.dirty_keys:
                return
            dirty_snapshot = list(self.dirty_keys)
            self.dirty_keys.clear()
            self.cells_snapshot = dict(self.cells)

        for key in dirty_snapshot:
            cell = self.cells_snapshot.get(key)
            if cell is None or cell.count == 0:
                continue

            cell.entropy = self.compute_residual_entropy(key, cell)
            cell.dirty = False

        with self.state_lock:
            for key in dirty_snapshot:
                live_cell = self.cells.get(key)
                snap_cell = self.cells_snapshot.get(key)
                if live_cell is None or snap_cell is None:
                    continue
                live_cell.entropy = snap_cell.entropy
                live_cell.dirty = False

    def pack_rgb(self, rgb_vec):
        rgb = np.clip(np.round(rgb_vec), 0.0, 255.0).astype(np.uint8)
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        rgb_uint32 = (r << 16) | (g << 8) | b
        rgb_float = struct.unpack("f", struct.pack("I", rgb_uint32))[0]
        return rgb_float

    def build_publish_clouds(self):
        with self.state_lock:
            items = [(key, cell) for key, cell in self.cells.items() if cell.count > 0]
            stamp = self.last_stamp if self.last_stamp != rospy.Time() else rospy.Time.now()
            frame_id = self.frame_id or "map"
        if not items:
            return None, None

        items.sort(key=lambda item: item[1].entropy, reverse=True)
        if self.max_publish_voxels > 0:
            items = items[: self.max_publish_voxels]

        color_points = []
        intensity_points = []
        for _, cell in items:
            x, y, z = cell.centroid.tolist()
            rgb = self.pack_rgb(cell.mean_rgb)
            color_points.append((x, y, z, rgb, float(cell.entropy)))
            intensity_points.append((x, y, z, float(cell.entropy)))

        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        color_fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
            PointField("intensity", 16, PointField.FLOAT32, 1),
        ]
        intensity_fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
        ]

        color_msg = pc2.create_cloud(header, color_fields, color_points)
        intensity_msg = pc2.create_cloud(header, intensity_fields, intensity_points)
        return color_msg, intensity_msg

    def on_timer(self, _event):
        self.recompute_dirty_cells()
        color_msg, intensity_msg = self.build_publish_clouds()
        if color_msg is None:
            return
        self.map_pub.publish(color_msg)
        self.intensity_pub.publish(intensity_msg)


if __name__ == "__main__":
    try:
        StructureTensorMapNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
