"""cloud_viz: light colored point clouds of the Femtos for RViz, rebuilt from the official driver's compressed streams
(the driver's own clouds are ~17 MB per frame and are disabled in cameras_config.yaml).

Per camera <name> (topics <topic_prefix>/...): depth/image_raw/compressedDepth (lossless PNG, depth registered to
color by the driver, already filtered by its depth filters) + color/image_raw/compressed + color/camera_info.
Each depth image is cleaned (flying pixels, max_depth, isolated points, grazing points), subsampled by `stride`,
colored:
  ~/<name>/points   in the color optical frame (fuse: false)
  ~/fused/points    all cameras in base_frame (TF), cropped to [crop_min, crop_max], merged per voxel (fuse: true)
at most rate_hz. Display only: the recorded / perception streams are untouched.

  ros2 run camera_rig cloud_viz --ros-args -p fuse:=true
"""

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

FIELDS = [
    PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
    for i, n in enumerate("xyz")
] + [PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)]


def decode_depth(data):
    """compressedDepth 16UC1: 12-byte config header then a 16-bit PNG (mm)"""
    return cv2.imdecode(np.frombuffer(data, np.uint8)[12:], cv2.IMREAD_UNCHANGED)


def remove_flying_pixels(z, rel, win):
    """zero the pixels whose win x win depth range > rel * depth (mixed foreground / background at depth edges).
    The registered depth is the 640x576 depth upsampled to 1280x720 by the driver: an edge ramp spans ~2 pixels, use
    win 5. Holes (0) are not an edge."""
    kernel, valid = np.ones((win, win), np.uint8), z > 0
    z_min = cv2.erode(np.where(valid, z, np.inf).astype(np.float32), kernel)
    out = z.copy()
    out[valid & (cv2.dilate(z, kernel) - z_min > rel * z)] = 0
    return out


def remove_isolated(z, win, min_count):
    """zero the pixels with fewer than min_count valid neighbours in the win x win window: speckles and thin slivers
    left by the depth filters and the edge removal (a straight border keeps ~half of its neighbours)"""
    valid = (z > 0).astype(np.float32)
    count = (
        cv2.boxFilter(
            valid, -1, (win, win), normalize=False, borderType=cv2.BORDER_CONSTANT
        )
        - valid
    )
    out = z.copy()
    out[(valid > 0) & (count < min_count)] = 0
    return out


def remove_grazing(z, rx, ry, max_deg, step):
    """zero the pixels whose surface is almost parallel to the view ray (normal from the neighbours `step` pixels away
    vs ray > max_deg): D2C fill-in / multipath streaks. Pixels without their 4 neighbours are kept."""
    P, s = np.dstack([rx * z, ry * z, z]), step
    du, dv = np.zeros_like(P), np.zeros_like(P)
    du[:, s:-s], dv[s:-s] = P[:, 2 * s :] - P[:, : -2 * s], P[2 * s :] - P[: -2 * s]
    valid = z > 0
    nb = np.zeros_like(valid)
    nb[s:-s, s:-s] = (
        valid[s:-s, 2 * s :]
        & valid[s:-s, : -2 * s]
        & valid[2 * s :, s:-s]
        & valid[: -2 * s, s:-s]
    )
    n = np.cross(du, dv)
    cos = np.abs((n * P).sum(2)) / (
        np.linalg.norm(n, axis=2) * np.linalg.norm(P, axis=2) + 1e-12
    )
    out = z.copy()
    out[valid & nb & (cos < np.cos(np.radians(max_deg)))] = 0
    return out


def voxel_merge(pts, voxel):
    """pts Nx4 (x, y, z, packed rgb float): one point per voxel cell, mean position and mean color"""
    ijk = (
        np.floor(pts[:, :3] / voxel).astype(np.int64) + 2**20
    )  # 1D key, faster than unique(axis=0)
    _, inv, cnt = np.unique(
        (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2],
        return_inverse=True,
        return_counts=True,
    )
    inv = inv.ravel()
    out = np.zeros((len(cnt), 4), np.float32)
    for k in range(3):
        out[:, k] = np.bincount(inv, pts[:, k]) / cnt
    rgb = pts[:, 3].view(np.uint32)
    mean = [np.bincount(inv, (rgb >> s) & 0xFF) / cnt for s in (16, 8, 0)]
    out[:, 3] = (
        (np.uint32(mean[0]) << 16) | (np.uint32(mean[1]) << 8) | np.uint32(mean[2])
    ).view(np.float32)
    return out


def quat_matrix(x, y, z, w):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def cloud_msg(header, pts):
    return PointCloud2(
        header=header,
        height=1,
        width=len(pts),
        fields=FIELDS,
        is_bigendian=False,
        point_step=16,
        row_step=16 * len(pts),
        is_dense=True,
        data=pts.tobytes(),
    )


class CloudViz(Node):
    def __init__(self):
        super().__init__("cloud_viz")
        p = self.declare_parameter
        p("cameras", ["golf", "foxtrot", "echo"])
        p(
            "topic_prefix", "/camera/{}_camera"
        )  # official driver namespace of each camera
        p("stride", 2)  # one pixel out of stride in each direction
        p("rate_hz", 2.0)
        p("max_depth", 1.3)  # m along the optical axis (walls, floor)
        p(
            "flying_rel", 0.02
        )  # flying pixels: depth range / depth in the window, 0 = off
        p("flying_win", 5)
        p("grazing_deg", 75.0)  # grazing points: max angle normal / view ray, 0 = off
        p(
            "grazing_step", 2
        )  # normal from neighbours >= grazing_step full-res pixels away
        p(
            "support_win", 5
        )  # isolated points: < support_min valid neighbours in the window
        p("support_min", 12)  # (out of support_win^2 - 1, full-res pixels), 0 = off
        p("fuse", False)
        p("base_frame", "prl_ur5_base")
        p(
            "crop_min", [-1.0, -0.5, -0.01]
        )  # fused cloud box in base_frame (m): table footprint, above its top
        p("crop_max", [0.7, 0.55, 0.8])
        p(
            "voxel", 0.005
        )  # fused cloud: points of all cameras merged per voxel (m), 0 = off
        self.period = 1.0 / self.param("rate_hz")
        self.info, self.color, self.last, self.rays, self.fused = {}, {}, {}, {}, {}
        if self.param("fuse"):
            self.tf = Buffer()
            TransformListener(self.tf, self)
            self.fused_pub = self.create_publisher(PointCloud2, "~/fused/points", 1)
            self.create_timer(self.period, self.publish_fused)
        for c in self.param("cameras"):
            base = self.param("topic_prefix").format(c)
            pub = (
                None
                if self.param("fuse")
                else self.create_publisher(PointCloud2, f"~/{c}/points", 1)
            )
            self.create_subscription(
                CameraInfo,
                f"{base}/color/camera_info",
                lambda m, c=c: self.info.__setitem__(c, m),
                1,
            )
            self.create_subscription(
                CompressedImage,
                f"{base}/color/image_raw/compressed",
                lambda m, c=c: self.color.__setitem__(c, m),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CompressedImage,
                f"{base}/depth/image_raw/compressedDepth",
                lambda m, c=c, pub=pub: self.on_depth(c, pub, m),
                qos_profile_sensor_data,
            )
        self.get_logger().info(
            f"cameras {self.param('cameras')}, "
            f"{'fused in ' + self.param('base_frame') if self.param('fuse') else 'per camera'}"
        )

    def param(self, name):
        return self.get_parameter(name).value

    def points(self, cam, m):
        """Nx4 float32 (x, y, z, rgb) in the color optical frame of a compressedDepth message"""
        depth = decode_depth(m.data).astype(np.float32) * 1e-3
        H, W = depth.shape
        if self.rays.get(cam, (None,))[0] != (H, W):
            # pinhole K only (no distortion), as the driver's own cloud
            K = np.array(self.info[cam].k).reshape(3, 3)
            v, u = np.mgrid[0:H, 0:W].astype(np.float32)
            self.rays[cam] = ((H, W), (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1])
        _, rx, ry = self.rays[cam]
        if self.param("flying_rel") > 0:
            depth = remove_flying_pixels(
                depth, self.param("flying_rel"), self.param("flying_win")
            )
        depth[depth > self.param("max_depth")] = 0
        if self.param("support_min") > 0:
            depth = remove_isolated(
                depth, self.param("support_win"), self.param("support_min")
            )
        s = self.param("stride")
        depth, rx, ry = depth[::s, ::s], rx[::s, ::s], ry[::s, ::s]
        if self.param("grazing_deg") > 0:  # on the subsampled grid: stride^2 cheaper
            depth = remove_grazing(
                depth,
                rx,
                ry,
                self.param("grazing_deg"),
                max(1, self.param("grazing_step") // s),
            )
        ok = depth > 0
        pts = np.zeros((int(ok.sum()), 4), np.float32)
        pts[:, 0], pts[:, 1], pts[:, 2] = (
            rx[ok] * depth[ok],
            ry[ok] * depth[ok],
            depth[ok],
        )
        if cam in self.color:
            bgr = cv2.imdecode(
                np.frombuffer(self.color[cam].data, np.uint8), cv2.IMREAD_COLOR
            )
            bgr = cv2.resize(bgr, (W, H))[::s, ::s][ok].astype(np.uint32)
            pts[:, 3] = ((bgr[:, 2] << 16) | (bgr[:, 1] << 8) | bgr[:, 0]).view(
                np.float32
            )
        return pts

    def on_depth(self, cam, pub, m):
        now = time.monotonic()
        if now - self.last.get(cam, 0.0) < self.period or cam not in self.info:
            return
        self.last[cam] = now
        pts = self.points(cam, m)
        if pub is not None:
            pub.publish(cloud_msg(m.header, pts))
            return
        try:
            tf = self.tf.lookup_transform(
                self.param("base_frame"), m.header.frame_id, Time()
            ).transform
        except TransformException as e:
            self.get_logger().warn(f"{cam}: {e}", throttle_duration_sec=5.0)
            return
        r, t = tf.rotation, tf.translation
        pts[:, :3] = pts[:, :3] @ quat_matrix(r.x, r.y, r.z, r.w).T.astype(
            np.float32
        ) + np.float32([t.x, t.y, t.z])
        lo, hi = np.float32(self.param("crop_min")), np.float32(self.param("crop_max"))
        self.fused[cam] = (
            m.header.stamp,
            pts[((pts[:, :3] >= lo) & (pts[:, :3] <= hi)).all(1)],
        )

    def publish_fused(self):
        if not self.fused:
            return
        items = list(self.fused.values())
        header = PointCloud2().header
        header.frame_id = self.param("base_frame")
        header.stamp = max((s for s, _ in items), key=lambda s: (s.sec, s.nanosec))
        pts = np.vstack([p for _, p in items])
        self.fused_pub.publish(
            cloud_msg(
                header,
                voxel_merge(pts, self.param("voxel"))
                if self.param("voxel") > 0
                else pts,
            )
        )


def main():
    rclpy.init()
    node = CloudViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
