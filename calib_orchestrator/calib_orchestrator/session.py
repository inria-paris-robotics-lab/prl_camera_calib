"""Calibration session on disk: robot poses recorded at each capture, replayable joint poses and
the resulting base_frame -> cameras extrinsics. No ROS dependency (plain yaml).

<session_dir>/
  Cam_00X/<frame>.png       written by the capture service (camera_rig: orbbec_rig or realsense_rig)
  robot_poses.yml           frame -> stamp, joints, T_base_tool, ChArUco corners per camera
  validated_poses.yml       joint configurations reached during this session (replay with poses_file)
  mc_calib.yml, mc_calib.log, MC-Calib outputs
  cams_extrinsics.yml       base_frame -> camera optical frames (+ intrinsics, pose generation visibility)
  fixed_cameras.yaml        same calibration in the prl_ur5_robot_configuration fixed_cameras format
  evaluation.txt / .json    quality report (evaluate.py), OK / WARN / FAIL per check
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml


class Session:
    def __init__(self, data_root, session_dir=""):
        self.dir = Path(
            session_dir
            or Path(data_root) / datetime.now().strftime("session_%Y%m%d_%H%M%S")
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        self.frames = []  # one dict per accepted capture, index = MC-Calib frame index
        self.joint_names = []

    def frame_name(self):
        """file stem of the next capture (MC-Calib matches frames across cameras by sorted file order)"""
        return f"{len(self.frames):06d}"

    def add_frame(self, stamp_s, joint_names, joints, T_base_tool, corners):
        self.joint_names = list(joint_names)
        self.frames.append(
            {
                "frame": self.frame_name(),
                "stamp": float(stamp_s),
                "joints": [float(j) for j in joints],
                "T_base_tool": np.asarray(T_base_tool).round(9).tolist(),
                "charuco_corners": [int(c) for c in corners],
            }
        )
        self.save()

    def save(self):
        with open(self.dir / "robot_poses.yml", "w") as f:
            yaml.safe_dump(
                {"joint_names": self.joint_names, "frames": self.frames},
                f,
                sort_keys=False,
            )
        with open(self.dir / "validated_poses.yml", "w") as f:
            yaml.safe_dump(
                {
                    "joint_names": self.joint_names,
                    "poses": [fr["joints"] for fr in self.frames],
                },
                f,
                sort_keys=False,
                default_flow_style=None,
            )

    def robot_poses(self):
        """{MC-Calib frame index: T_base_tool}"""
        return {k: np.array(fr["T_base_tool"]) for k, fr in enumerate(self.frames)}


def load_joint_poses(path):
    """(joint_names, [joint vectors]) from a validated_poses.yml"""
    data = yaml.safe_load(Path(path).read_text())
    return data["joint_names"], [list(map(float, p)) for p in data["poses"]]


def write_extrinsics(
    path,
    parent_frame,
    camera_frames,
    T_base_cams,
    cameras,
    T_tool_cube,
    residuals_mm_deg,
):
    """cams_extrinsics.yml: base -> camera optical frames, plus the intrinsics used by the pose generator."""
    entries = []
    for name, T, cam in zip(camera_frames, T_base_cams, cameras):
        entries.append(
            {
                "child_frame": name,
                "translation": T[:3, 3].round(6).tolist(),
                "rotation_xyzw": pin.Quaternion(T[:3, :3]).coeffs().round(9).tolist(),
                "camera_matrix": np.asarray(cam["K"]).round(4).tolist(),
                "image_size": [int(cam.get("width", 0)), int(cam.get("height", 0))],
            }
        )
    res = np.asarray(residuals_mm_deg)
    data = {
        "parent_frame": parent_frame,
        "date": datetime.now().isoformat(timespec="seconds"),
        "hand_eye_residual_mm": {
            "mean": float(res[:, 0].mean()),
            "max": float(res[:, 0].max()),
        },
        "hand_eye_residual_deg": {
            "mean": float(res[:, 1].mean()),
            "max": float(res[:, 1].max()),
        },
        "tool_to_cube": {
            "translation": T_tool_cube[:3, 3].round(6).tolist(),
            "rotation_xyzw": pin.Quaternion(T_tool_cube[:3, :3])
            .coeffs()
            .round(9)
            .tolist(),
        },
        "transforms": entries,
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)


def read_extrinsics(path):
    """(parent_frame, [(child_frame, T_parent_child, K, (w, h))])"""
    data = yaml.safe_load(Path(path).read_text())
    out = []
    for e in data["transforms"]:
        T = np.eye(4)
        T[:3, :3] = (
            pin.Quaternion(np.array(e["rotation_xyzw"], float)).normalized().matrix()
        )
        T[:3, 3] = e["translation"]
        out.append(
            (
                e["child_frame"],
                T,
                np.array(e.get("camera_matrix", np.eye(3))),
                tuple(e.get("image_size", (0, 0))),
            )
        )
    return data["parent_frame"], out


def write_fixed_cameras(template, path, poses):
    """Copy of a fixed_cameras yaml (list of {name_prefix, pose, ...}) with the calibrated poses.
    poses: {name_prefix: T_parent_<name_prefix>_camera_calibration_pose}; other cameras are kept as is."""
    cameras = yaml.safe_load(Path(template).read_text())
    missing = set(poses) - {c["name_prefix"] for c in cameras}
    if missing:
        raise ValueError(f"{template}: no camera with name_prefix {sorted(missing)}")
    for cam in cameras:
        T = poses.get(cam["name_prefix"])
        if T is not None:
            roll, pitch, yaw = map(float, pin.rpy.matrixToRpy(T[:3, :3]))
            cam["pose"] = {
                "x": round(float(T[0, 3]), 6),
                "y": round(float(T[1, 3]), 6),
                "z": round(float(T[2, 3]), 6),
                "roll": round(roll, 6),
                "pitch": round(pitch, 6),
                "yaw": round(yaw, 6),
            }
    with open(path, "w") as f:
        f.write(
            f"# calibrated on {datetime.now().isoformat(timespec='seconds')}, session {Path(path).parent}, "
            f"template {template}\n"
        )
        yaml.safe_dump(cameras, f, sort_keys=False)
