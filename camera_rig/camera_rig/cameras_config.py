"""Cameras of the robot setup (prl_ur5_robot_configuration/config/fixed_cameras/cameras_config.yaml):

cameras:
  echo_camera: {type: femto-mega, activate: true, ip: 192.168.1.10, port: "8090", sync_mode: primary, ...}
  alpha_camera: {type: realsense, activate: true, serial_no: "_145422071857", ...}

No ROS dependency (plain yaml).
"""

from pathlib import Path

import yaml

DEFAULT_PACKAGE = "prl_ur5_robot_configuration"
DEFAULT_DIR = "config/fixed_cameras"


def resolve(value):
    """path of a cameras file: a path as is, a bare file name (or "" = cameras_config.yaml) in the fixed_cameras folder
    of the installed robot configuration; "" when the package is not found"""
    if "/" in value:
        return value
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        return ""
    try:
        return str(
            Path(get_package_share_directory(DEFAULT_PACKAGE))
            / DEFAULT_DIR
            / (value or "cameras_config.yaml")
        )
    except PackageNotFoundError:
        return ""


def load_femtos(path, names=()):
    """[{name, ip, port, sync_mode, fps, color_size}] of the active femto-mega cameras, in the order of `names`
    (the MC-Calib order) or in file order when `names` is empty. Raises ValueError on a bad setup."""
    cameras = yaml.safe_load(Path(path).read_text()).get("cameras", {})
    femtos = {
        n: c
        for n, c in cameras.items()
        if c.get("type") == "femto-mega" and c.get("activate", True)
    }
    if names:
        unknown = [n for n in names if n not in femtos]
        if unknown:
            raise ValueError(
                f"{path}: {unknown} are not active femto-mega cameras (available {list(femtos)})"
            )
    else:
        names = list(femtos)
    out = [
        {
            "name": n,
            "ip": str(femtos[n]["ip"]),
            "port": int(femtos[n].get("port", 8090)),
            "sync_mode": str(femtos[n].get("sync_mode", "")).upper(),
            "fps": int(femtos[n].get("color_fps", 15)),
            # None when not written: the driver default is used
            "color_size": (
                int(femtos[n]["color_width"]),
                int(femtos[n]["color_height"]),
            )
            if "color_width" in femtos[n] and "color_height" in femtos[n]
            else None,
        }
        for n in names
    ]
    primaries = [c["ip"] for c in out if c["sync_mode"] == "PRIMARY"]
    if len(out) > 1 and len(primaries) != 1:
        raise ValueError(
            f"{path}: expected exactly one primary among {[c['name'] for c in out]}, got {primaries}"
        )
    if len({c["fps"] for c in out}) > 1:
        raise ValueError(
            f"{path}: synchronized cameras need the same color_fps, got {[c['fps'] for c in out]}"
        )
    return out


def primary_ip(femtos):
    return next((c["ip"] for c in femtos if c["sync_mode"] == "PRIMARY"), "")


def load_realsenses(path, names=()):
    """[{name, serial}] of the active realsense cameras, in the order of `names` (the MC-Calib order) or in file
    order when `names` is empty. Raises ValueError on a bad setup."""
    cameras = yaml.safe_load(Path(path).read_text()).get("cameras", {})
    found = {
        n: c
        for n, c in cameras.items()
        if c.get("type") == "realsense" and c.get("activate", True)
    }
    unknown = [n for n in names if n not in found]
    if unknown:
        raise ValueError(
            f"{path}: {unknown} are not active realsense cameras (available {list(found)})"
        )
    # serial_no "_145422071857": the leading underscore only keeps it a string for realsense2_camera
    return [
        {"name": n, "serial": str(found[n].get("serial_no", "")).lstrip("_")}
        for n in (names or list(found))
    ]
