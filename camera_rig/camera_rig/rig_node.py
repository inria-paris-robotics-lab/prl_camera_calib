"""Camera rig of the calibration: capture service, session files and preview, for two sources.

orbbec_rig: the Femto Mega opened with pyorbbecsdk inside the node, so that the 4K color never goes through DDS.
IP, port, sync mode and fps come from `cameras_config` (prl_ur5_robot_configuration); the primary is started last.
The sync mode stored in each device is only compared to the file, never written.

realsense_rig: cameras published by realsense2_camera, read from <topic_prefix>/color/image_raw/compressed and
color/camera_info (topic_cameras.py). Color only, no hardware sync.

`names` gives the MC-Calib order (Cam_001, Cam_002...).

Service ~/capture (inria_calib_interfaces/srv/Capture):
  1. waits for color frames received after request time + settle_s;
  2. picks one color frame per camera with timestamps within sync_slop_s;
  3. optionally keeps the median of the next depth_median_frames depth frames (orbbec);
  4. writes <session>/Cam_00X/<frame>.png (+ raw/Cam_00X/<frame>_depth16.png), and on the first capture
     factory_intrinsics.yml (MC-Calib initialization) and cameras.txt;
  5. returns the number of ChArUco corners per camera.

orbbec_rig also publishes reduced streams (publish_rate, only when subscribed): ~/<name>/color/image_raw,
~/<name>/color/camera_info, ~/<name>/depth/image_raw, ~/<name>/depth/camera_info, and the static TF color -> depth
optical frames (publish_depth_tf).
~/preview: mosaic with the detected ChArUco corners and the stream status.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

from inria_calib_interfaces.srv import Capture

from . import cameras_config
from .charuco import CharucoCounter, read_board_config
from .femto import open_rig
from .topic_cameras import TopicCamera
from .frames import (
    decode_color,
    mccalib_paths,
    median_depth,
    read_calibrated_intrinsics,
    reduce_for_width,
    select_synced,
    write_intrinsics,
)
from .preview import GREEN, ORANGE, RED, WHITE, blank_tile, make_tile, mosaic


def quaternion_xyzw(R):
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q[:] = [
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
            s / 4,
        ]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q[i], q[j], q[k] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    return q / np.linalg.norm(q)


def read_depth_extrinsics(path):
    """{serial or IP: (R, t in m)} from a calibrated_depth_to_color.yml"""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    out = {}
    for k in range(int(fs.getNode("nb_device").real())):
        n = fs.getNode(f"device_{k}")
        R, t = (
            n.getNode("R_depth_to_color").mat(),
            n.getNode("t_depth_to_color_mm").mat().ravel() / 1000.0,
        )
        out[n.getNode("serial").string()] = (R, t)
        out[n.getNode("spec").string().split(":", 1)[-1].split()[0]] = (R, t)
    return out


DEFAULT_NAMES = {
    "orbbec": ["golf_camera", "foxtrot_camera", "echo_camera"],
    "realsense": ["alpha_camera", "bravo_camera", "charlie_camera"],
}


class RigNode(Node):
    def __init__(self, source):
        """source: "orbbec" (Femto Mega opened here with pyorbbecsdk) or "realsense" (topics of realsense2_camera)"""
        super().__init__(f"{source}_rig")
        p = self.declare_parameter
        p(
            "cameras_config", ""
        )  # file name in prl_ur5_robot_configuration/config/fixed_cameras, a path, or "none"
        p("names", DEFAULT_NAMES[source])  # MC-Calib order, keys of cameras_config
        p("topic_prefix", "/camera/{}")  # realsense
        p("color_transport", "compressed")  # realsense: compressed | raw
        p(
            "cameras", ["192.168.1.11", "192.168.1.12", "192.168.1.10"]
        )  # only without cameras_config
        p("primary", "192.168.1.10")  # only without cameras_config
        p("color_width", 3840)
        p("color_height", 2160)
        p("fps", 0)  # orbbec: 0 = color_fps of cameras_config
        p("enable_depth", True)
        p("depth_width", 640)
        p("depth_height", 576)
        p("sync_slop_s", 0.02)  # max timestamp difference between cameras
        p("timeout_s", 5.0)
        p("color_buffer", 5)
        p("depth_buffer", 40)  # >= the largest depth_median_frames
        p(
            "board_config", ""
        )  # MC-Calib config for the ChArUco corner count, "" = no detection
        p("detect_scale", 1.0)
        p("stale_warn_s", 2.0)
        p("publish_rate", 5.0)  # Hz, 0 = off
        p("publish_color_width", 960)
        p(
            "calibration_dir", ""
        )  # MC-Calib session: calibrated intrinsics in camera_info
        p("depth_extrinsics_file", "")  # "" = factory depth -> color
        p("publish_depth_tf", True)  # False when the URDF already has the depth frames
        p("preview_rate", 2.0)  # Hz, 0 = off
        p("preview_tile_width", 640)
        p("preview_detect", True)

        self.names = list(self.param("names"))
        expected_sync = self.open_orbbec() if source == "orbbec" else self.open_topics()
        self.counter = (
            CharucoCounter(read_board_config(self.param("board_config")))
            if self.param("board_config")
            else None
        )
        self.pool = ThreadPoolExecutor(max_workers=len(self.cams))
        self.light_pool = ThreadPoolExecutor(
            max_workers=len(self.cams)
        )  # streams / preview, apart from the captures
        self.last_capture = None

        for cam in self.cams:
            expected = expected_sync.get(cam.ip)
            if expected and cam.sync_mode.upper() != expected:
                self.get_logger().warn(
                    f"{cam.ip}: sync mode stored in the device {cam.sync_mode} != {expected} in "
                    "cameras_config (set it with OrbbecViewer or the official driver)"
                )
            err = cam.geometry.get("R_depth_to_color_error", 0)
            if err > 1e-4:
                self.get_logger().info(
                    f"{cam.ip}: factory depth->color rotation not orthonormal ({err:.4f}), "
                    "closest rotation used"
                )

        light_cb = MutuallyExclusiveCallbackGroup()
        self.color_info = self.make_color_infos()
        if self.param("publish_rate") > 0:
            self.pubs = [
                {
                    "color": self.create_publisher(
                        Image, f"~/{n}/color/image_raw", qos_profile_sensor_data
                    ),
                    "color_info": self.create_publisher(
                        CameraInfo, f"~/{n}/color/camera_info", 10
                    ),
                    "depth": self.create_publisher(
                        Image, f"~/{n}/depth/image_raw", qos_profile_sensor_data
                    ),
                    "depth_info": self.create_publisher(
                        CameraInfo, f"~/{n}/depth/camera_info", 10
                    ),
                }
                for n in self.names
            ]
            self.last_published = [None] * len(self.cams)
            self.create_timer(
                1.0 / self.param("publish_rate"),
                self.publish_streams,
                callback_group=light_cb,
            )
        if (
            source == "orbbec"
            and self.param("enable_depth")
            and self.param("publish_depth_tf")
        ):
            self.publish_depth_tf()
        if self.param("preview_rate") > 0:
            self.preview_pub = self.create_publisher(Image, "~/preview", 1)
            self.preview_stats = [(c.color.count, time.monotonic()) for c in self.cams]
            self.create_timer(
                1.0 / self.param("preview_rate"),
                self.publish_preview,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
        self.create_service(
            Capture,
            "~/capture",
            self.on_capture,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.create_timer(
            5.0, self.check_streams, callback_group=ReentrantCallbackGroup()
        )
        self.get_logger().info(
            f"cameras (MC-Calib order) {[(n, c.ip) for n, c in zip(self.names, self.cams)]}, "
            "service ~/capture, "
            f"ChArUco count {'on' if self.counter else 'off'}"
        )

    def param(self, name):
        return self.get_parameter(name).value

    def open_orbbec(self):
        """Femto Mega opened with pyorbbecsdk -> self.cams; returns {ip: sync mode expected by cameras_config}"""
        fps, port, expected_sync = self.param("fps") or 15, 8090, {}
        setup = (
            ""
            if self.param("cameras_config") == "none"
            else cameras_config.resolve(self.param("cameras_config"))
        )
        if setup:
            try:
                femtos = cameras_config.load_femtos(setup, self.names)
            except (OSError, ValueError) as e:
                raise SystemExit(f"cameras_config: {e}")
            self.names, self.ips = (
                [c["name"] for c in femtos],
                [c["ip"] for c in femtos],
            )
            primary, fps = (
                cameras_config.primary_ip(femtos),
                self.param("fps") or femtos[0]["fps"],
            )
            if len({c["port"] for c in femtos}) > 1:
                raise SystemExit(
                    f"cameras_config: different ports {[c['port'] for c in femtos]} not supported"
                )
            port, expected_sync = (
                femtos[0]["port"],
                {c["ip"]: c["sync_mode"] for c in femtos},
            )
        else:
            self.ips, primary = list(self.param("cameras")), self.param("primary")
            if len(self.ips) != len(self.names):
                raise SystemExit(
                    "parameters cameras and names must have the same length"
                )
        self.cams = open_rig(
            self.ips,
            primary,
            log=self.get_logger().info,
            port=port,
            color_size=(self.param("color_width"), self.param("color_height")),
            fps=fps,
            depth_size=(self.param("depth_width"), self.param("depth_height")),
            enable_depth=self.param("enable_depth"),
            color_buffer=self.param("color_buffer"),
            depth_buffer=self.param("depth_buffer"),
        )
        return expected_sync

    def open_topics(self):
        """cameras published by realsense2_camera -> self.cams; returns {} (no sync mode to check)"""
        serials = {}
        setup = (
            ""
            if self.param("cameras_config") == "none"
            else cameras_config.resolve(self.param("cameras_config"))
        )
        if setup:
            try:
                serials = {
                    c["name"]: c["serial"]
                    for c in cameras_config.load_realsenses(setup, self.names)
                }
            except (OSError, ValueError) as e:
                raise SystemExit(f"cameras_config: {e}")
        group, self.cams = (
            ReentrantCallbackGroup(),
            [],
        )  # frames keep coming while a capture waits for them
        for name in self.names:
            prefix = self.param("topic_prefix").format(name)
            ok, info = wait_for_message(
                CameraInfo,
                self,
                f"{prefix}/color/camera_info",
                time_to_wait=self.param("timeout_s"),
            )
            if not ok:
                raise SystemExit(
                    f"no {prefix}/color/camera_info within {self.param('timeout_s')} s (driver running?)"
                )
            self.cams.append(
                TopicCamera(
                    self,
                    name,
                    "realsense",
                    serials.get(name, ""),
                    prefix,
                    info,
                    self.param("color_buffer"),
                    transport=self.param("color_transport"),
                    callback_group=group,
                )
            )
            self.get_logger().info(
                f"{name}: {prefix}/color ({self.param('color_transport')}) "
                f"{info.width}x{info.height}"
            )
        return {}

    def frame_id(self, i, kind):
        return f"{self.names[i]}_{kind}_optical_frame"

    # ------------------------------------------------------------------ geometry
    def make_color_infos(self):
        """[(K, dist, width, height, source)] full-resolution color model per camera"""
        calibrated = {}
        if self.param("calibration_dir"):
            calibrated = read_calibrated_intrinsics(self.param("calibration_dir"))
        out = []
        for cam in self.cams:
            g = cam.geometry
            model = calibrated.get(cam.ip) or calibrated.get(cam.serial)
            if model and tuple(model[2:4]) == tuple(g["color_size"]):
                out.append((*model, "calibrated"))
            else:
                if model:
                    self.get_logger().warn(
                        f"{cam.ip}: calibration is {model[2]}x{model[3]}, stream "
                        f"{g['color_size'][0]}x{g['color_size'][1]}: factory intrinsics used"
                    )
                out.append((g["color_K"], g["color_dist"], *g["color_size"], "factory"))
            self.get_logger().info(
                f"{cam.ip}: color camera_info from {out[-1][4]} intrinsics"
            )
        return out

    def camera_info(self, i, kind, width, height, stamp):
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = stamp, self.frame_id(i, kind)
        if kind == "color":
            K, dist, w, h, _ = self.color_info[i]
            info.distortion_model = "plumb_bob"
        else:
            g = self.cams[i].geometry
            K, dist, (w, h) = g["depth_K"], g["depth_dist"], g["depth_size"]
            info.distortion_model = "rational_polynomial"
        S = np.diag([width / w, height / h, 1.0])
        Ks = S @ np.asarray(K, np.float64)
        info.width, info.height = int(width), int(height)
        info.k = Ks.ravel().tolist()
        info.d = [float(x) for x in np.ravel(dist)]
        info.r = np.eye(3).ravel().tolist()
        info.p = np.hstack([Ks, np.zeros((3, 1))]).ravel().tolist()
        return info

    def publish_depth_tf(self):
        calibrated = (
            read_depth_extrinsics(self.param("depth_extrinsics_file"))
            if self.param("depth_extrinsics_file")
            else {}
        )
        msgs = []
        for i, cam in enumerate(self.cams):
            g = cam.geometry
            R, t = (
                calibrated.get(cam.serial)
                or calibrated.get(cam.ip)
                or (g["R_depth_to_color"], g["t_depth_to_color_m"])
            )
            m = TransformStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id, m.child_frame_id = (
                self.frame_id(i, "color"),
                self.frame_id(i, "depth"),
            )
            (
                m.transform.translation.x,
                m.transform.translation.y,
                m.transform.translation.z,
            ) = map(float, t)
            q = quaternion_xyzw(R)
            (
                m.transform.rotation.x,
                m.transform.rotation.y,
                m.transform.rotation.z,
                m.transform.rotation.w,
            ) = map(float, q)
            msgs.append(m)
        self.tf_static = StaticTransformBroadcaster(self)
        self.tf_static.sendTransform(msgs)

    # ------------------------------------------------------------------ light streams
    @staticmethod
    def image_msg(img, encoding, stamp, frame_id):
        msg = Image()
        msg.header.stamp, msg.header.frame_id = stamp, frame_id
        msg.height, msg.width = img.shape[:2]
        msg.encoding = encoding
        msg.step = img.shape[1] * img.itemsize * (img.shape[2] if img.ndim == 3 else 1)
        msg.data = img.tobytes()
        return msg

    def publish_camera(self, i):
        item = self.cams[i].color.latest()
        if item is None or item[0] == self.last_published[i]:
            return
        self.last_published[i] = item[0]
        stamp = Time(nanoseconds=item[1]).to_msg()  # arrival, host clock = ROS time
        pubs = self.pubs[i]
        if (
            pubs["color"].get_subscription_count()
            or pubs["color_info"].get_subscription_count()
        ):
            raw = item[2]
            img = decode_color(
                raw, reduce_for_width(raw.width, self.param("publish_color_width"))
            )
            pubs["color"].publish(
                self.image_msg(img, "bgr8", stamp, self.frame_id(i, "color"))
            )
            pubs["color_info"].publish(
                self.camera_info(i, "color", img.shape[1], img.shape[0], stamp)
            )
        d = self.cams[i].depth.latest()
        if d is not None and (
            pubs["depth"].get_subscription_count()
            or pubs["depth_info"].get_subscription_count()
        ):
            dstamp = Time(nanoseconds=d[1]).to_msg()
            pubs["depth"].publish(
                self.image_msg(d[2], "16UC1", dstamp, self.frame_id(i, "depth"))
            )
            pubs["depth_info"].publish(
                self.camera_info(i, "depth", d[2].shape[1], d[2].shape[0], dstamp)
            )

    def publish_streams(self):
        try:
            list(self.light_pool.map(self.publish_camera, range(len(self.cams))))
        except Exception as e:  # noqa: BLE001 - a corrupted frame must not kill the timer
            self.get_logger().warn(f"streams: {e}", throttle_duration_sec=5.0)

    def check_streams(self):
        limit = time.time_ns() - int(self.param("stale_warn_s") * 1e9)
        stale = [
            n for n, c in zip(self.names, self.cams) if c.color.last_arrival_ns < limit
        ]
        if stale:
            self.get_logger().warn(
                f"no frame for {self.param('stale_warn_s')} s from {stale} "
                "(secondary without trigger? primary started last?)"
            )

    # ------------------------------------------------------------------ capture
    def wait_color_set(self, after_ns, deadline):
        slop = int(self.param("sync_slop_s") * 1e9)
        while time.monotonic() < deadline:
            chosen = select_synced([c.color.fresh(after_ns) for c in self.cams], slop)
            if chosen:
                return chosen, ""
            time.sleep(0.01)
        missing = [
            n for n, c in zip(self.names, self.cams) if not c.color.fresh(after_ns)
        ]
        return None, (
            f"no fresh color frame from {missing}"
            if missing
            else f"no color set within sync_slop_s={self.param('sync_slop_s')} s (hardware sync?)"
        )

    def wait_depth(self, after_ns, n, deadline):
        if n > self.param("depth_buffer"):
            return (
                None,
                f"depth_median_frames {n} > depth_buffer {self.param('depth_buffer')}",
            )
        while time.monotonic() < deadline:
            fresh = [
                sorted(c.depth.fresh(after_ns), key=lambda it: it[0]) for c in self.cams
            ]
            if all(len(f) >= n for f in fresh):
                return [[it[2] for it in f[:n]] for f in fresh], ""
            time.sleep(0.01)
        counts = [len(c.depth.fresh(after_ns)) for c in self.cams]
        return None, f"not enough depth frames {dict(zip(self.names, counts))} for {n}"

    def save_camera(self, i, raw, depth_frames, session_dir, frame_name):
        color_path, depth_path = mccalib_paths(session_dir, i, frame_name)
        img = decode_color(raw)
        color_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(color_path), img):
            raise IOError(f"cannot write {color_path}")
        paths = [str(color_path)]
        if depth_frames:
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(depth_path), median_depth(depth_frames))
            paths.append(str(depth_path))
        corners = (
            self.counter.count(img, self.param("detect_scale")) if self.counter else -1
        )
        return paths, corners

    def write_session_files(self, session):
        intr = session / "factory_intrinsics.yml"
        if not intr.exists():
            write_intrinsics(
                intr,
                [
                    (
                        c.geometry["color_K"],
                        c.geometry["color_dist"],
                        *c.geometry["color_size"],
                    )
                    for c in self.cams
                ],
            )
        txt = session / "cameras.txt"
        if not txt.exists():
            txt.write_text(
                "".join(
                    f"Cam_{i + 1:03d}: {c.kind}:{c.ip} {c.geometry['color_size'][0]}x"
                    f"{c.geometry['color_size'][1]}\n"
                    for i, c in enumerate(self.cams)
                )
            )

    def fail(self, res, frame, message):
        res.success, res.message = False, message
        self.get_logger().error(f"capture {frame}: {message}")
        self.last_capture = (frame, False, message, [])
        return res

    def on_capture(self, req, res):
        t0 = time.monotonic()
        after_ns = time.time_ns() + int(req.settle_s * 1e9)
        deadline = t0 + req.settle_s + self.param("timeout_s")
        res.camera_names = [
            n.removesuffix("_camera") for n in self.names
        ]  # golf, foxtrot, echo
        if not req.session_dir or not req.frame_name:
            return self.fail(
                res, req.frame_name, "session_dir and frame_name are required"
            )

        chosen, msg = self.wait_color_set(after_ns, deadline)
        if chosen is None:
            return self.fail(res, req.frame_name, msg)
        depth = [[] for _ in self.cams]
        if req.depth_median_frames > 0:
            if not self.param("enable_depth"):
                return self.fail(
                    res, req.frame_name, "depth requested but enable_depth is false"
                )
            depth, msg = self.wait_depth(after_ns, req.depth_median_frames, deadline)
            if depth is None:
                return self.fail(res, req.frame_name, msg)

        session = Path(req.session_dir)
        try:
            jobs = [
                self.pool.submit(
                    self.save_camera, i, chosen[i][2], depth[i], session, req.frame_name
                )
                for i in range(len(self.cams))
            ]
            out = [j.result() for j in jobs]
            self.write_session_files(session)
        except Exception as e:  # noqa: BLE001 - reported through the service response
            return self.fail(res, req.frame_name, f"write failed: {e}")

        res.image_paths = [p for o in out for p in o[0]]
        res.charuco_corners = [int(o[1]) for o in out]
        res.stamp = Time(nanoseconds=chosen[0][1]).to_msg()
        res.success = True
        spread_ms = (max(c[0] for c in chosen) - min(c[0] for c in chosen)) * 1e-6
        res.message = f"ok ({spread_ms:.1f} ms between cameras)"
        self.last_capture = (
            req.frame_name,
            True,
            res.message,
            list(res.charuco_corners),
        )
        self.get_logger().info(
            f"capture {req.frame_name}: corners {list(res.charuco_corners)}, "
            f"stamp spread {spread_ms:.1f} ms, {time.monotonic() - t0:.1f} s"
        )
        return res

    # ------------------------------------------------------------------ preview
    def preview_tile(self, i, ref_stamp_ns, now_ns, fps):
        tw = self.param("preview_tile_width")
        item = self.cams[i].color.latest()
        name = self.names[i].removesuffix("_camera")
        if item is None:
            return blank_tile(tw, tw * 9 // 16, f"{name}: no frame")
        stamp_ns, arrival_ns, raw = item
        img = decode_color(raw, reduce_for_width(raw.width, 2 * tw))
        boards = (
            self.counter.detect(img)
            if self.counter and self.param("preview_detect")
            else []
        )
        n_corners = sum(len(c) for _, c in boards)
        age_ms = (now_ns - arrival_ns) * 1e-6
        dt_ms = (stamp_ns - ref_stamp_ns) * 1e-6 if ref_stamp_ns is not None else 0.0
        slop_ms = self.param("sync_slop_s") * 1e3
        lines = [
            (f"{name}  {raw.width}x{raw.height}  {self.cams[i].sync_mode}", WHITE),
            (
                f"{fps:.1f} fps  age {age_ms:.0f} ms",
                GREEN if age_ms < 1000 * self.param("stale_warn_s") else RED,
            ),
            (f"sync {dt_ms:+.1f} ms", GREEN if abs(dt_ms) <= slop_ms else ORANGE),
        ]
        if self.counter and self.param("preview_detect"):
            lines.append(
                (
                    f"corners {n_corners} ({len(boards)} boards)",
                    GREEN if n_corners >= 6 else RED,
                )
            )
        d = self.cams[i].depth.latest()
        return make_tile(img, boards, lines, tw, d[2] if d is not None else None)

    def publish_preview(self):
        if self.preview_pub.get_subscription_count() == 0:
            return
        now_ns, t = time.time_ns(), time.monotonic()
        fps = []
        for k, c in enumerate(self.cams):
            c0, t0 = self.preview_stats[k]
            fps.append((c.color.count - c0) / max(t - t0, 1e-3))
            self.preview_stats[k] = (c.color.count, t)
        ref = self.cams[0].color.latest()
        try:
            tiles = list(
                self.light_pool.map(
                    lambda k: self.preview_tile(
                        k, ref[0] if ref else None, now_ns, fps[k]
                    ),
                    range(len(self.cams)),
                )
            )
        except Exception as e:  # noqa: BLE001 - a bad frame must not kill the preview timer
            self.get_logger().warn(f"preview: {e}", throttle_duration_sec=5.0)
            return
        footer = []
        if self.last_capture:
            frame, ok, message, corners = self.last_capture
            footer.append(
                (
                    f"last capture {frame}: {'OK' if ok else 'FAILED'}",
                    GREEN if ok else RED,
                )
            )
            footer.append((message[:60], WHITE))
            if corners:
                footer.append((f"corners {corners}", WHITE))
        self.preview_pub.publish(
            self.image_msg(
                mosaic(tiles, footer), "bgr8", self.get_clock().now().to_msg(), ""
            )
        )

    def shutdown(self):
        for cam in self.cams:
            cam.stop()
        self.pool.shutdown()
        self.light_pool.shutdown()


def main(source="orbbec"):
    rclpy.init()
    node = RigNode(source)
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


def main_orbbec():
    main("orbbec")


def main_realsense():
    main("realsense")


if __name__ == "__main__":
    main()
