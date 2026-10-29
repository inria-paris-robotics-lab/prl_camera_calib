"""Femto Mega over Ethernet with pyorbbecsdk.

One thread per camera reads color + depth framesets (frame sync, full frame required) and keeps
the most recent ones in ring buffers. The color is kept as the RAW MJPG bytes (~1-2 MB in 4K
instead of 25 MB decoded): it is decoded at full resolution only for a capture, and at reduced
resolution (JPEG DCT scaling, cheap) for the published streams and the preview.

Hardware sync (sync hub): the sync mode is stored in each device (OrbbecViewer); the secondaries
must be streaming before the primary starts: open_rig() starts the primary last.
No ROS dependency.
"""

import threading
import time

import numpy as np

from .frames import ColorRaw, FrameBuffer


def closest_rotation(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def _fmt_name(fmt):
    return getattr(fmt, "name", str(fmt).split(".")[-1])


def import_pyorbbecsdk():
    """pyorbbecsdk loaded with ITS OWN libOrbbecSDK.so.2 (shipped in the package folder).

    The ROS driver workspace (orbbec_camera) puts an older libOrbbecSDK.so.2 on LD_LIBRARY_PATH, which wins
    over the wheel's RUNPATH ($ORIGIN): 'undefined symbol: ob_application_config_set_struct'. Loading the
    bundled library first (RTLD_GLOBAL) makes the extension module bind to it, whatever LD_LIBRARY_PATH says.
    """
    import ctypes
    import importlib.util
    import os

    spec = importlib.util.find_spec("pyorbbecsdk")
    if spec is not None and spec.submodule_search_locations:
        lib = os.path.join(
            list(spec.submodule_search_locations)[0], "libOrbbecSDK.so.2"
        )
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    import pyorbbecsdk

    return pyorbbecsdk


class Femto(threading.Thread):
    kind = "orbbec"  # cameras.txt / MC-Calib session: "Cam_001: orbbec:<ip> WxH"

    def __init__(
        self,
        ip,
        color_size=(3840, 2160),
        fps=15,
        depth_size=(640, 576),
        enable_depth=True,
        color_buffer=5,
        depth_buffer=40,
        port=8090,
        log=print,
    ):
        super().__init__(daemon=True, name=f"femto_{ip}")
        sdk = import_pyorbbecsdk()
        Config, Context, OBFormat, Pipeline = (
            sdk.Config,
            sdk.Context,
            sdk.OBFormat,
            sdk.Pipeline,
        )
        OBFrameAggregateOutputMode, OBSensorType = (
            sdk.OBFrameAggregateOutputMode,
            sdk.OBSensorType,
        )

        self.ip, self.log = ip, log
        self.ctx = Context()  # keep a reference: the device belongs to this context
        self.device = self.ctx.create_net_device(ip, port)
        info = self.device.get_device_info()
        self.serial = info.get_serial_number()
        self.sync_mode = self._sync_mode()
        self.pipeline = Pipeline(self.device)

        profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None
        for fmt in (OBFormat.MJPG, OBFormat.RGB, OBFormat.YUYV):
            try:
                color_profile = profiles.get_video_stream_profile(*color_size, fmt, fps)
                break
            except Exception:  # noqa: BLE001 - format not available at this resolution
                pass
        if color_profile is None:
            raise RuntimeError(
                f"{ip}: no color profile {color_size[0]}x{color_size[1]}@{fps}"
            )
        self.config = Config()
        self.config.enable_stream(color_profile)
        depth_profile = None
        if enable_depth:
            depth_profile = self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            ).get_video_stream_profile(*depth_size, OBFormat.Y16, fps)
            self.config.enable_stream(depth_profile)
            self.config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )
            self.pipeline.enable_frame_sync()  # color and depth of the same instant
        self.geometry = self._geometry(color_profile, depth_profile)

        self.global_ts = False
        try:
            if self.device.is_global_timestamp_supported():
                self.device.enable_global_timestamp(True)
                self.global_ts = True
        except Exception:  # noqa: BLE001 - older firmware / SDK
            pass

        self.color = FrameBuffer(color_buffer)  # (stamp ns, arrival ns, ColorRaw)
        self.depth = FrameBuffer(
            depth_buffer
        )  # (stamp ns, arrival ns, uint16 depth in mm)
        self.running, self.missed = False, 0
        self.log(
            f"Femto {ip} serial {self.serial} sync {self.sync_mode}: color "
            f"{color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()} "
            f"{_fmt_name(color_profile.get_format())}"
            + (
                f", depth {depth_profile.get_width()}x{depth_profile.get_height()}"
                if depth_profile
                else ""
            )
            + (", global timestamps" if self.global_ts else ", host timestamps")
        )

    def _sync_mode(self):
        try:
            return _fmt_name(self.device.get_multi_device_sync_config().mode)
        except Exception:  # noqa: BLE001
            return "unknown"

    @staticmethod
    def _geometry(color_profile, depth_profile):
        ci, cd = color_profile.get_intrinsic(), color_profile.get_distortion()
        g = {
            "color_K": np.array([[ci.fx, 0, ci.cx], [0, ci.fy, ci.cy], [0, 0, 1]]),
            "color_dist": np.array([cd.k1, cd.k2, cd.p1, cd.p2, cd.k3]),
            "color_size": (color_profile.get_width(), color_profile.get_height()),
        }
        if depth_profile is not None:
            di, dd = depth_profile.get_intrinsic(), depth_profile.get_distortion()
            ext = depth_profile.get_extrinsic_to(color_profile)
            R_raw = np.array(ext.rot, dtype=np.float64).reshape(3, 3)
            g.update(
                {
                    "depth_K": np.array(
                        [[di.fx, 0, di.cx], [0, di.fy, di.cy], [0, 0, 1]]
                    ),
                    "depth_dist": np.array(
                        [dd.k1, dd.k2, dd.p1, dd.p2, dd.k3, dd.k4, dd.k5, dd.k6]
                    ),  # rational
                    "depth_size": (
                        depth_profile.get_width(),
                        depth_profile.get_height(),
                    ),
                    # the SDK factory rotation is not orthonormal (~1 %): closest rotation
                    "R_depth_to_color": closest_rotation(R_raw),
                    "R_depth_to_color_error": float(
                        np.abs(R_raw.T @ R_raw - np.eye(3)).max()
                    ),
                    "t_depth_to_color_m": np.array(ext.transform, dtype=np.float64)
                    / 1000.0,
                }
            )
        return g

    def _stamp_ns(self, frame):
        if self.global_ts:
            ts = frame.get_global_timestamp_us()
            if ts:
                return int(ts) * 1000
        return int(frame.get_system_timestamp_us()) * 1000

    def start_stream(self):
        self.pipeline.start(self.config)
        self.running = True
        self.start()

    def run(self):
        while self.running:
            try:
                fs = self.pipeline.wait_for_frames(1000)
            except Exception as e:  # noqa: BLE001 - keep the thread alive, reported by the node
                self.log(f"Femto {self.ip}: {e}")
                fs = None
            if fs is None:
                self.missed += 1
                continue
            arrival = time.time_ns()
            c = fs.get_color_frame()
            if c is None:
                continue
            stamp = self._stamp_ns(c)
            self.color.push(
                stamp,
                arrival,
                ColorRaw(
                    np.asanyarray(c.get_data()).copy(),
                    _fmt_name(c.get_format()),
                    c.get_width(),
                    c.get_height(),
                ),
            )
            d = fs.get_depth_frame()
            if d is not None:
                z = np.frombuffer(d.get_data(), np.uint16).reshape(
                    d.get_height(), d.get_width()
                )
                scale = d.get_depth_scale()  # mm per unit
                z = (
                    z.copy()
                    if scale == 1.0
                    else (z.astype(np.float32) * scale).round().astype(np.uint16)
                )
                self.depth.push(stamp, arrival, z)

    def stop(self):
        self.running = False
        if self.is_alive():
            self.join(timeout=3.0)
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass


def open_rig(ips, primary, log=print, **kwargs):
    """Femto objects in the given (MC-Calib) order; streams started secondaries first, primary last."""
    cams = [Femto(ip, log=log, **kwargs) for ip in ips]
    order = sorted(range(len(cams)), key=lambda k: ips[k] == primary)
    for k in order:
        if ips[k] == primary:
            time.sleep(1.0)  # secondaries armed before the first trigger
        cams[k].start_stream()
    return cams
