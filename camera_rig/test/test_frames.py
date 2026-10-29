from pathlib import Path

import cv2
import numpy as np
import pytest

from camera_rig.femto import ColorRaw, closest_rotation
from camera_rig.frames import (
    FrameBuffer,
    decode_color,
    median_depth,
    reduce_for_width,
    select_synced,
    write_intrinsics,
)


def test_decode_color_mjpg_reduced():
    rng = np.random.default_rng(0)
    img = cv2.resize(
        rng.integers(0, 255, (9, 16, 3), np.uint8),
        (1280, 720),
        interpolation=cv2.INTER_NEAREST,
    )
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    raw = ColorRaw(jpg.ravel(), "MJPG", 1280, 720)
    full = decode_color(raw)
    assert full.shape == (720, 1280, 3) and np.abs(full.astype(int) - img).mean() < 3
    assert decode_color(raw, 4).shape == (180, 320, 3)
    rgb = ColorRaw(img[..., ::-1].copy().ravel(), "RGB", 1280, 720)
    assert np.array_equal(decode_color(rgb), img)
    assert decode_color(rgb, 2).shape == (360, 640, 3)
    assert (
        reduce_for_width(3840, 960) == 4
        and reduce_for_width(3840, 1280) == 2
        and reduce_for_width(1280, 960) == 1
    )


def test_median_depth_ignores_zeros():
    a = np.array([[0, 100], [0, 0]], np.uint16)
    b = np.array([[200, 110], [0, 0]], np.uint16)
    c = np.array([[220, 0], [0, 50]], np.uint16)
    assert median_depth([a, b, c]).tolist() == [[210, 105], [0, 50]]


def test_select_synced():
    ms = 1_000_000
    cams = [
        [(100 * ms, 0, "a0"), (166 * ms, 0, "a1")],
        [(101 * ms, 0, "b0"), (167 * ms, 0, "b1")],
        [(99 * ms, 0, "c0")],
    ]  # third camera only has the older frame
    chosen = select_synced(cams, 5 * ms)
    assert [c[2] for c in chosen] == ["a0", "b0", "c0"]
    assert select_synced(cams, 0) is None
    assert select_synced([[], cams[1]], 5 * ms) is None


def test_frame_buffer_fresh():
    buf = FrameBuffer(3)
    for k in range(5):
        buf.push(k, 10 * k, k)
    assert [it[2] for it in buf.fresh(25)] == [3, 4]
    assert len(buf.items) == 3


def test_write_intrinsics(tmp_path):
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1.0]])
    dist = np.array([0.1, -0.2, 0.001, 0.002, 0.05, 0.3, 0.1, 0.0])
    write_intrinsics(tmp_path / "f.yml", [(K, dist, 1920, 1080)] * 2)
    fs = cv2.FileStorage(str(tmp_path / "f.yml"), cv2.FILE_STORAGE_READ)
    assert int(fs.getNode("nb_camera").real()) == 2
    assert fs.getNode("camera_1").getNode("distortion_vector").mat().shape == (1, 5)


def test_closest_rotation():
    R = np.array(
        [[0.9935, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )  # like the SDK factory matrix
    Rc = closest_rotation(R)
    assert np.allclose(Rc.T @ Rc, np.eye(3)) and np.isclose(np.linalg.det(Rc), 1.0)


def test_open_rig_starts_primary_last(monkeypatch):
    import camera_rig.femto as femto

    started = []

    class FakeFemto:
        def __init__(self, ip, log=print, **kwargs):
            self.ip = ip

        def start_stream(self):
            started.append(self.ip)

    monkeypatch.setattr(femto, "Femto", FakeFemto)
    monkeypatch.setattr(femto.time, "sleep", lambda s: None)
    cams = femto.open_rig(["a", "p", "b"], "p")
    assert [c.ip for c in cams] == ["a", "p", "b"]  # MC-Calib order kept
    assert started == ["a", "b", "p"]


ROBOT_CAMERAS = (
    Path(__file__).resolve().parents[3]
    / "prl_ur5_robot_configuration/config/fixed_cameras/cameras_config.yaml"
)


@pytest.mark.skipif(not ROBOT_CAMERAS.exists(), reason="no robot configuration")
def test_load_femtos_robot_setup():
    from camera_rig.cameras_config import load_femtos, primary_ip

    femtos = load_femtos(
        ROBOT_CAMERAS, ["golf_camera", "foxtrot_camera", "echo_camera"]
    )
    assert [c["ip"] for c in femtos] == ["192.168.1.11", "192.168.1.12", "192.168.1.10"]
    assert primary_ip(femtos) == "192.168.1.10" and {c["port"] for c in femtos} == {
        8090
    }
    assert [c["name"] for c in load_femtos(ROBOT_CAMERAS)] == [
        "echo_camera",
        "foxtrot_camera",
        "golf_camera",
    ]
    with pytest.raises(ValueError):
        load_femtos(ROBOT_CAMERAS, ["alpha_camera"])  # realsense, inactive


def test_load_femtos_needs_one_primary(tmp_path):
    from camera_rig.cameras_config import load_femtos

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "cameras:\n  a: {type: femto-mega, ip: 1.1.1.1, sync_mode: secondary_synced}\n"
        "  b: {type: femto-mega, ip: 1.1.1.2, sync_mode: secondary_synced}\n"
    )
    with pytest.raises(ValueError):
        load_femtos(cfg)
