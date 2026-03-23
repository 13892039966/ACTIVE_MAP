#!/usr/bin/env python3

import math
import struct

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def pack_rgb(r, g, b):
    return struct.unpack("f", struct.pack("I", (r << 16) | (g << 8) | b))[0]


class SyntheticTextureScenePublisher:
    def __init__(self):
        rospy.init_node("synthetic_texture_scene_publisher", anonymous=True)

        self.topic = rospy.get_param("~topic", "/synthetic/point_cloud")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.publish_rate = float(rospy.get_param("~publish_rate", 1.0))
        self.scene_type = rospy.get_param("~scene_type", "edge_plane")
        self.grid_step = float(rospy.get_param("~grid_step", 0.04))
        self.width = float(rospy.get_param("~width", 2.0))
        self.height = float(rospy.get_param("~height", 1.2))
        self.sparse_keep_ratio = float(rospy.get_param("~sparse_keep_ratio", 0.18))
        self.depth_offset = float(rospy.get_param("~depth_offset", 0.35))
        self.plane_gap = float(rospy.get_param("~plane_gap", 0.06))
        self.tilt_angle_deg = float(rospy.get_param("~tilt_angle_deg", 25.0))
        self.secondary_tilt_angle_deg = float(rospy.get_param("~secondary_tilt_angle_deg", -20.0))
        self.checker_size = float(rospy.get_param("~checker_size", 0.16))
        self.stripe_width = float(rospy.get_param("~stripe_width", 0.16))

        self.publisher = rospy.Publisher(self.topic, PointCloud2, queue_size=1, latch=True)
        self.fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]
        self.points = self.build_scene(self.scene_type)

        rospy.loginfo("Synthetic texture scene publisher ready: scene=%s points=%d topic=%s frame=%s",
                      self.scene_type, len(self.points), self.topic, self.frame_id)
        rospy.loginfo(
            "Synthetic scene params: grid_step=%.3f width=%.3f height=%.3f tilt_angle_deg=%.1f secondary_tilt_angle_deg=%.1f depth_offset=%.3f plane_gap=%.3f checker_size=%.3f stripe_width=%.3f sparse_keep_ratio=%.3f",
            self.grid_step,
            self.width,
            self.height,
            self.tilt_angle_deg,
            self.secondary_tilt_angle_deg,
            self.depth_offset,
            self.plane_gap,
            self.checker_size,
            self.stripe_width,
            self.sparse_keep_ratio,
        )

    def build_scene(self, scene_type):
        builders = {
            "edge_plane": self.build_edge_plane,
            "striped_plane": self.build_striped_plane,
            "sparse_edge": self.build_sparse_edge,
            "angled_dual_plane": self.build_angled_dual_plane,
            "layered_overlap_planes": self.build_layered_overlap_planes,
            "tilted_checkerboard_planes": self.build_tilted_checkerboard_planes,
            "corner_textured_room": self.build_corner_textured_room,
        }
        if scene_type not in builders:
            raise ValueError("Unsupported scene_type '{}'. Use one of: {}".format(
                scene_type, ", ".join(sorted(builders.keys()))))
        return builders[scene_type]()

    def iter_grid(self):
        nx = max(2, int(round(self.width / self.grid_step)) + 1)
        ny = max(2, int(round(self.height / self.grid_step)) + 1)
        x0 = -0.5 * self.width
        y0 = -0.5 * self.height
        for iy in range(ny):
            for ix in range(nx):
                yield (
                    x0 + ix * self.grid_step,
                    y0 + iy * self.grid_step,
                    ix,
                    iy,
                    nx,
                    ny,
                )

    def build_edge_plane(self):
        points = []
        split_x = 0.0
        for x, y, _, _, _, _ in self.iter_grid():
            if x < split_x:
                rgb = pack_rgb(210, 210, 210)
            else:
                rgb = pack_rgb(40, 110, 220)
            points.append((x, y, 2.0, rgb))

        # Add a background slab behind the right half so the color edge has cross-surface interference.
        for x, y, _, _, _, _ in self.iter_grid():
            if x < 0.12:
                continue
            rgb = pack_rgb(220, 60, 60)
            points.append((x, y, 2.0 + self.depth_offset, rgb))
        return points

    def build_striped_plane(self):
        points = []
        stripe_period = max(2, int(round(self.stripe_width / self.grid_step)))
        for x, y, ix, _, _, _ in self.iter_grid():
            stripe_id = (ix // stripe_period) % 2
            if stripe_id == 0:
                rgb = pack_rgb(235, 235, 235)
            else:
                rgb = pack_rgb(30, 30, 30)
            points.append((x, y, 2.0, rgb))
        return points

    def build_sparse_edge(self):
        points = []
        keep_mod = max(2, int(round(1.0 / max(self.sparse_keep_ratio, 1e-3))))
        for x, y, ix, iy, _, _ in self.iter_grid():
            if ((ix + 2 * iy) % keep_mod) != 0:
                continue
            rgb = pack_rgb(210, 210, 210) if x < 0.0 else pack_rgb(30, 120, 220)
            points.append((x, y, 2.0, rgb))

        # A few farther points make the edge area easy to confuse when radius/self filtering is weak.
        for x, y, ix, iy, _, _ in self.iter_grid():
            if x < 0.08 or ((ix + iy) % (keep_mod + 1)) != 0:
                continue
            points.append((x, y, 2.0 + self.depth_offset, pack_rgb(220, 70, 70)))
        return points

    def rotate_x(self, y, z, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return y * c - z * s, y * s + z * c

    def rotate_y(self, x, z, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return x * c + z * s, -x * s + z * c

    def rotate_z(self, x, y, angle_rad):
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return x * c - y * s, x * s + y * c

    def checker_rgb(self, x, y, dark_rgb=(35, 35, 35), light_rgb=(235, 235, 235)):
        cell_x = int(math.floor((x + 0.5 * self.checker_size) / self.checker_size))
        cell_y = int(math.floor((y + 0.5 * self.checker_size) / self.checker_size))
        return pack_rgb(*(light_rgb if ((cell_x + cell_y) % 2 == 0) else dark_rgb))

    def stripe_rgb(self, x, light_rgb=(235, 235, 235), dark_rgb=(30, 30, 30)):
        stripe_id = int(math.floor((x + 0.5 * self.stripe_width) / self.stripe_width))
        return pack_rgb(*(light_rgb if (stripe_id % 2 == 0) else dark_rgb))

    def append_plane_points(self, points, center, basis_u, basis_v, half_u, half_v, color_fn):
        nu = max(2, int(round((2.0 * half_u) / self.grid_step)) + 1)
        nv = max(2, int(round((2.0 * half_v) / self.grid_step)) + 1)
        for iv in range(nv):
            for iu in range(nu):
                u = -half_u + iu * (2.0 * half_u / max(1, nu - 1))
                v = -half_v + iv * (2.0 * half_v / max(1, nv - 1))
                x = center[0] + u * basis_u[0] + v * basis_v[0]
                y = center[1] + u * basis_u[1] + v * basis_v[1]
                z = center[2] + u * basis_u[2] + v * basis_v[2]
                points.append((x, y, z, color_fn(u, v)))

    def build_angled_dual_plane(self):
        points = []
        angle_a = math.radians(self.tilt_angle_deg)
        angle_b = math.radians(self.secondary_tilt_angle_deg)

        basis_u_a = (1.0, 0.0, 0.0)
        _, by_a, bz_a = self.rotate_x(1.0, 0.0, angle_a)
        basis_v_a = (0.0, by_a, bz_a)
        self.append_plane_points(
            points,
            center=(-0.25, 0.0, 2.0),
            basis_u=basis_u_a,
            basis_v=basis_v_a,
            half_u=0.85,
            half_v=0.55,
            color_fn=lambda u, v: self.stripe_rgb(u),
        )

        bx_b, _, bz_b = self.rotate_y(1.0, 0.0, angle_b)
        basis_u_b = (bx_b, 0.0, bz_b)
        basis_v_b = (0.0, 1.0, 0.0)
        self.append_plane_points(
            points,
            center=(0.45, 0.0, 2.12),
            basis_u=basis_u_b,
            basis_v=basis_v_b,
            half_u=0.75,
            half_v=0.5,
            color_fn=lambda u, v: pack_rgb(70, 170, 220),
        )
        return points

    def build_layered_overlap_planes(self):
        points = []
        self.append_plane_points(
            points,
            center=(0.0, 0.0, 1.92),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 1.0, 0.0),
            half_u=0.72,
            half_v=0.48,
            color_fn=lambda u, v: self.checker_rgb(u, v),
        )
        self.append_plane_points(
            points,
            center=(0.12, 0.0, 1.92 + self.plane_gap),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 1.0, 0.0),
            half_u=0.95,
            half_v=0.60,
            color_fn=lambda u, v: pack_rgb(70, 145, 215),
        )
        return points

    def build_tilted_checkerboard_planes(self):
        points = []
        angle_x = math.radians(self.tilt_angle_deg)
        angle_y = math.radians(self.secondary_tilt_angle_deg)
        basis_u_a = self.rotate_y(1.0, 0.0, angle_y) + (0.0,)
        basis_u_a = (basis_u_a[0], 0.0, basis_u_a[1])
        _, by_a, bz_a = self.rotate_x(1.0, 0.0, angle_x)
        basis_v_a = (0.0, by_a, bz_a)
        self.append_plane_points(
            points,
            center=(-0.15, 0.0, 2.0),
            basis_u=basis_u_a,
            basis_v=basis_v_a,
            half_u=0.78,
            half_v=0.55,
            color_fn=lambda u, v: self.checker_rgb(u, v),
        )

        bx_b, by_b = self.rotate_z(1.0, 0.0, math.radians(35.0))
        basis_u_b = (bx_b, by_b, 0.0)
        _, by2_b, bz2_b = self.rotate_x(1.0, 0.0, math.radians(-18.0))
        basis_v_b = (0.0, by2_b, bz2_b)
        self.append_plane_points(
            points,
            center=(0.58, 0.05, 2.08),
            basis_u=basis_u_b,
            basis_v=basis_v_b,
            half_u=0.55,
            half_v=0.5,
            color_fn=lambda u, v: pack_rgb(210, 115, 70),
        )
        return points

    def build_corner_textured_room(self):
        points = []
        half_u = 0.9
        half_v = 0.65

        # floor
        self.append_plane_points(
            points,
            center=(0.0, 0.0, 2.4),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 1.0, 0.0),
            half_u=half_u,
            half_v=half_v,
            color_fn=lambda u, v: pack_rgb(170, 170, 170),
        )
        # wall XZ, textured
        self.append_plane_points(
            points,
            center=(0.0, -half_v, 2.4 + half_v),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 0.0, 1.0),
            half_u=half_u,
            half_v=half_v,
            color_fn=lambda u, v: self.stripe_rgb(u, light_rgb=(240, 240, 240), dark_rgb=(45, 45, 45)),
        )
        # wall YZ, pure
        self.append_plane_points(
            points,
            center=(-half_u, 0.0, 2.4 + half_v),
            basis_u=(0.0, 1.0, 0.0),
            basis_v=(0.0, 0.0, 1.0),
            half_u=half_v,
            half_v=half_v,
            color_fn=lambda u, v: pack_rgb(80, 150, 210),
        )
        return points

    def publish_once(self):
        msg = PointCloud2()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        msg.height = 1
        msg.width = len(self.points)
        msg.fields = self.fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = b"".join(struct.pack("ffff", *point) for point in self.points)
        self.publisher.publish(msg)

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.publish_once()
            rate.sleep()


if __name__ == "__main__":
    try:
        SyntheticTextureScenePublisher().run()
    except rospy.ROSInterruptException:
        pass
