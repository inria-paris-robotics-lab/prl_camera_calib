"""Cameras read from the topics of their ROS driver (realsense2_camera), with the same interface as femto.Femto:
color FrameBuffer of (stamp ns, arrival ns, ColorRaw), geometry from camera_info, ip / serial / kind / sync_mode,
stop(). Color only.

transport "compressed" (default) reads color/image_raw/compressed: large raw images published best effort are often
lost (one lost UDP fragment drops the whole image). "raw" reads color/image_raw."""

import time

import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from .frames import ColorRaw, FrameBuffer

ENCODINGS = {"rgb8": "RGB", "bgr8": "BGR"}


def geometry_from_camera_info(info):
    """color_K, color_dist (OpenCV order, plumb_bob / first 5 coefficients) and color_size of a CameraInfo"""
    return {
        "color_K": np.array(info.k, np.float64).reshape(3, 3),
        "color_dist": np.array(
            list(info.d)[:5] + [0.0] * (5 - min(5, len(info.d))), np.float64
        ),
        "color_size": (int(info.width), int(info.height)),
    }


def color_raw(msg):
    """ColorRaw of an rgb8 / bgr8 Image (row padding removed)"""
    fmt = ENCODINGS.get(msg.encoding)
    if fmt is None:
        raise ValueError(f"unsupported color encoding {msg.encoding} (rgb8 / bgr8)")
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[
        :, : msg.width * 3
    ]
    return ColorRaw(np.ascontiguousarray(rows).ravel(), fmt, msg.width, msg.height)


class TopicCamera:
    def __init__(
        self,
        node,
        name,
        kind,
        device,
        topic_prefix,
        info,
        color_buffer=5,
        sync_mode="",
        transport="compressed",
        callback_group=None,
    ):
        """kind: orbbec / realsense, device: IP / serial; both written to cameras.txt ("Cam_001: orbbec:<ip> WxH")"""
        self.name, self.kind, self.ip, self.serial = name, kind, device or name, device
        self.sync_mode = sync_mode or "-"
        self.geometry = geometry_from_camera_info(info)
        self.color = FrameBuffer(color_buffer)
        self.depth = FrameBuffer(1)  # stays empty: color only
        if transport == "compressed":
            node.create_subscription(
                CompressedImage,
                f"{topic_prefix}/color/image_raw/compressed",
                self.on_compressed,
                qos_profile_sensor_data,
                callback_group=callback_group,
            )
        elif transport == "raw":
            node.create_subscription(
                Image,
                f"{topic_prefix}/color/image_raw",
                self.on_color,
                qos_profile_sensor_data,
                callback_group=callback_group,
            )
        else:
            raise ValueError(
                f"color transport {transport!r}: expected compressed or raw"
            )

    @staticmethod
    def stamp_ns(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def on_color(self, msg):
        self.color.push(self.stamp_ns(msg), time.time_ns(), color_raw(msg))

    def on_compressed(self, msg):
        w, h = self.geometry["color_size"]  # decoded later by frames.decode_color
        self.color.push(
            self.stamp_ns(msg),
            time.time_ns(),
            ColorRaw(np.frombuffer(msg.data, np.uint8), "MJPG", w, h),
        )

    def stop(self):
        pass
