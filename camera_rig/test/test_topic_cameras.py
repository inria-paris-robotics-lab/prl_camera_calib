import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image

from camera_rig.cameras_config import load_realsenses
from camera_rig.frames import ColorRaw, decode_color
from camera_rig.topic_cameras import color_raw, geometry_from_camera_info


def test_geometry_from_camera_info():
    info = CameraInfo(
        width=1920,
        height=1080,
        k=[1380.0, 0, 960, 0, 1381.0, 540, 0, 0, 1],
        d=[0.1, -0.2, 0.0, 0.0],
    )
    g = geometry_from_camera_info(info)
    assert (
        g["color_size"] == (1920, 1080)
        and g["color_K"][0, 0] == 1380.0
        and g["color_K"][1, 2] == 540
    )
    assert np.allclose(
        g["color_dist"], [0.1, -0.2, 0, 0, 0]
    )  # padded to the 5 MC-Calib coefficients


def test_color_raw_drops_row_padding():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    padded = np.zeros((2, 12), np.uint8)
    padded[:, :9] = rgb.reshape(2, 9)
    msg = Image(height=2, width=3, encoding="rgb8", step=12, data=padded.tobytes())
    assert np.array_equal(
        decode_color(color_raw(msg)), rgb[..., ::-1]
    )  # BGR for OpenCV
    with pytest.raises(ValueError):
        color_raw(Image(height=1, width=1, encoding="mono8", step=1, data=b"\x00"))


def test_load_realsenses(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        'cameras:\n  alpha_camera: {type: realsense, serial_no: "_145422071857", activate: true}\n'
        '  bravo_camera: {type: realsense, serial_no: "_031522071313", activate: false}\n'
        "  echo_camera: {type: femto-mega, ip: 1.1.1.1}\n"
    )
    assert load_realsenses(cfg) == [{"name": "alpha_camera", "serial": "145422071857"}]
    with pytest.raises(ValueError):
        load_realsenses(cfg, ["bravo_camera"])  # inactive


def test_compressed_frame_decodes():
    import cv2

    img = np.zeros((72, 128, 3), np.uint8)
    img[20:50, 30:90] = (40, 200, 90)
    jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].ravel()
    out = decode_color(
        ColorRaw(np.frombuffer(jpg.tobytes(), np.uint8), "MJPG", 128, 72)
    )  # as TopicCamera.on_compressed
    assert out.shape == img.shape and np.abs(out.astype(int) - img).mean() < 3
