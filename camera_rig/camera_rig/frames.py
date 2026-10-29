"""Frame buffers, synchronized set selection, color decoding and MC-Calib files. No ROS dependency."""

import threading
import warnings
from collections import deque, namedtuple
from pathlib import Path

import cv2
import numpy as np

ColorRaw = namedtuple(
    "ColorRaw", "data fmt width height"
)  # data: MJPG bytes or raw pixels (uint8)

_REDUCED = {
    1: cv2.IMREAD_COLOR,
    2: cv2.IMREAD_REDUCED_COLOR_2,
    4: cv2.IMREAD_REDUCED_COLOR_4,
    8: cv2.IMREAD_REDUCED_COLOR_8,
}


def decode_color(raw, reduce=1):
    """ColorRaw -> BGR image, size divided by reduce (1, 2, 4, 8); MJPG is decoded directly at the reduced size"""
    fmt = raw.fmt.upper()
    if fmt in ("MJPG", "MJPEG"):
        img = cv2.imdecode(raw.data, _REDUCED[reduce])
        if img is None:
            raise ValueError("corrupted MJPG frame")
        return img
    if fmt == "RGB":
        img = cv2.cvtColor(
            raw.data.reshape(raw.height, raw.width, 3), cv2.COLOR_RGB2BGR
        )
    elif fmt == "BGR":
        img = raw.data.reshape(raw.height, raw.width, 3)
    elif fmt == "YUYV":
        img = cv2.cvtColor(
            raw.data.reshape(raw.height, raw.width, 2), cv2.COLOR_YUV2BGR_YUYV
        )
    else:
        raise ValueError(f"unsupported color format {raw.fmt}")
    return (
        img
        if reduce == 1
        else cv2.resize(
            img,
            (raw.width // reduce, raw.height // reduce),
            interpolation=cv2.INTER_AREA,
        )
    )


def reduce_for_width(full_width, target_width):
    """largest JPEG reduction factor (1, 2, 4, 8) keeping at least target_width pixels"""
    for r in (8, 4, 2):
        if full_width // r >= target_width:
            return r
    return 1


def median_depth(frames):
    """per-pixel median of uint16 depth frames, ignoring 0 (no measure); 0 where never measured"""
    stack = np.stack(frames).astype(np.float32)
    stack[stack == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
        med = np.nanmedian(stack, axis=0)
    return np.nan_to_num(med, nan=0).round().astype(np.uint16)


class FrameBuffer:
    """Thread-safe ring buffer of (header stamp ns, arrival ns, msg)."""

    def __init__(self, maxlen):
        self.items = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.last_arrival_ns = 0
        self.count = 0  # total frames received (frame rate)

    def push(self, stamp_ns, arrival_ns, msg):
        with self.lock:
            self.items.append((stamp_ns, arrival_ns, msg))
            self.last_arrival_ns = arrival_ns
            self.count += 1

    def latest(self):
        with self.lock:
            return self.items[-1] if self.items else None

    def fresh(self, after_ns):
        """items received after after_ns (arrival time: valid whatever the driver time domain)"""
        with self.lock:
            return [it for it in self.items if it[1] > after_ns]


def select_synced(fresh_per_camera, slop_ns):
    """One item per camera with header stamps within slop_ns of the first camera's.

    fresh_per_camera: [[(stamp_ns, arrival_ns, msg)]] per camera. Tries the most recent
    reference frames first. Returns the list of items or None.
    """
    if any(not f for f in fresh_per_camera):
        return None
    for ref in sorted(fresh_per_camera[0], key=lambda it: -it[0]):
        chosen = [ref]
        for others in fresh_per_camera[1:]:
            best = min(others, key=lambda it: abs(it[0] - ref[0]))
            if abs(best[0] - ref[0]) > slop_ns:
                break
            chosen.append(best)
        else:
            return chosen
    return None


def write_intrinsics(path, models):
    """MC-Calib cam_params file, models: [(K, dist, width, height)] (Brown model: first 5 coefficients)."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera", len(models))
    for i, (K, dist, w, h) in enumerate(models):
        d = np.zeros(5)
        d[: min(5, len(dist))] = np.ravel(dist)[:5]
        fs.startWriteStruct(f"camera_{i}", cv2.FILE_NODE_MAP)
        fs.write("camera_matrix", np.asarray(K, np.float64).reshape(3, 3))
        fs.write("distortion_vector", d.reshape(1, -1))
        fs.write("distortion_type", 0)
        fs.write("img_width", int(w))
        fs.write("img_height", int(h))
        fs.endWriteStruct()
    fs.release()


def read_calibrated_intrinsics(calib_dir):
    """{device (ip/serial): (K, dist, width, height)} from an MC-Calib result + its cameras.txt"""
    calib_dir = Path(calib_dir)
    fs = cv2.FileStorage(
        str(calib_dir / "calibrated_cameras_data.yml"), cv2.FILE_STORAGE_READ
    )
    if not fs.isOpened():
        raise FileNotFoundError(calib_dir / "calibrated_cameras_data.yml")
    out = {}
    for i, line in enumerate((calib_dir / "cameras.txt").read_text().splitlines()):
        device = (
            line.split(":", 1)[1].split(":", 1)[1].split()[0]
        )  # "Cam_001: orbbec:<ip> WxH"
        n = fs.getNode(f"camera_{i}")
        out[device] = (
            n.getNode("camera_matrix").mat(),
            n.getNode("distortion_vector").mat().ravel(),
            int(n.getNode("img_width").real()),
            int(n.getNode("img_height").real()),
        )
    return out


def mccalib_paths(session_dir, cam_index, frame_name):
    """(color png, depth png): Cam_00X/<frame>.png for MC-Calib, depth kept out of the Cam_ folders"""
    cam = f"Cam_{cam_index + 1:03d}"
    root = Path(session_dir)
    return (
        root / cam / f"{frame_name}.png",
        root / "raw" / cam / f"{frame_name}_depth16.png",
    )
