"""Run the MC-Calib `calibrate` binary on a capture session. No ROS dependency."""

import re
import subprocess
from pathlib import Path

import cv2
import numpy as np


def write_config(template, out, session_dir, cam_params_path=None):
    """Copy an MC-Calib config with root_path / save_path (and optionally cam_params_path) set to the session."""
    text = Path(template).read_text()
    values = {"root_path": session_dir, "save_path": session_dir}
    if cam_params_path:
        values["cam_params_path"] = cam_params_path
    for key, value in values.items():
        text, n = re.subn(
            rf'^(\s*{key}\s*:\s*)"[^"]*"', rf'\g<1>"{value}"', text, flags=re.M
        )
        if n != 1:
            raise ValueError(
                f"{template}: key {key} not found (expected one '{key}: \"...\"' line)"
            )
    Path(out).write_text(text)
    return Path(out)


def run(binary, config, log_path, timeout_s=3600):
    """Run calibrate, log stdout/stderr to log_path. Raises on failure."""
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [str(binary), str(config)],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            cwd=Path(config).parent,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"MC-Calib exited with code {proc.returncode}, see {log_path}"
        )


def mean_reprojection_px(session_dir):
    """mean reprojection error over all cameras from reprojection_error_data.yml (-1 if missing)"""
    path = Path(session_dir) / "reprojection_error_data.yml"
    if not path.exists():
        return -1.0
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    errors = []
    for gk in fs.root().keys():
        g = fs.getNode(gk)
        if not (gk.startswith("camera_group_") and g.isMap()):
            continue
        for fk in g.keys():
            f = g.getNode(fk)
            if not f.isMap():
                continue
            for ck in f.keys():
                n = f.getNode(ck)
                if n.isMap() and re.fullmatch(r"camera_\d+", ck):
                    errors.extend(n.getNode("error_list").mat().ravel())
    return float(np.mean(errors)) if errors else -1.0
