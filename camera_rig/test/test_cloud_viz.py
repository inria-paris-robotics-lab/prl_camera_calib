import cv2
import numpy as np

from camera_rig.cloud_viz import (
    decode_depth,
    quat_matrix,
    remove_flying_pixels,
    remove_grazing,
    remove_isolated,
    voxel_merge,
)


def rays(h, w, f=500.0):
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    return (u - w / 2) / f, (v - h / 2) / f


def test_decode_depth_compressed_png():
    z = np.arange(12, dtype=np.uint16).reshape(3, 4) * 100
    data = np.concatenate([np.zeros(12, np.uint8), cv2.imencode(".png", z)[1].ravel()])
    assert np.array_equal(decode_depth(data.tobytes()), z)


def test_flying_pixels_at_edges_only():
    z = np.full((20, 20), 1.0, np.float32)
    z[:, 10:] = 2.0  # depth step
    z[5, 3] = 0.0  # a hole is not an edge
    out = remove_flying_pixels(z, 0.02, 5)
    assert (out[:, 8:12] == 0).all()  # both sides of the step dropped
    assert (out[:, :7][z[:, :7] > 0] == 1.0).all() and (out[:, 13:] == 2.0).all()
    assert out[5, 2] == 1.0 and out[5, 4] == 1.0  # neighbours of the hole kept


def test_grazing_drops_surfaces_parallel_to_rays():
    rx, ry = rays(40, 40)
    facing = np.full((40, 40), 1.0, np.float32)  # plane facing the camera
    assert (remove_grazing(facing, rx, ry, 75.0, 2) > 0).all()
    grazing = (1.0 + 20.0 * np.arange(40, dtype=np.float32) / 40)[None, :].repeat(
        40, 0
    )  # steep depth ramp
    out = remove_grazing(grazing, rx, ry, 75.0, 2)
    assert (out[2:-2, 2:-2] == 0).all() and (
        out[:, :2] > 0
    ).all()  # borders without 4 neighbours kept


def test_voxel_merge_mean_position_and_color():
    p = np.zeros((3, 4), np.float32)
    p[:, :3] = [[0.001, 0.001, 0.001], [0.002, 0.002, 0.002], [-0.1, 0.0, 0.0]]
    p[:, 3] = np.array([0xFF0000, 0x0000FF, 0x00FF00], np.uint32).view(np.float32)
    out = voxel_merge(p, 0.005)
    assert len(out) == 2
    merged = out[np.abs(out[:, 0] - 0.0015) < 1e-6][0]
    assert np.allclose(merged[:3], 0.0015) and merged[3:].view(np.uint32)[0] == 0x7F007F


def test_quat_matrix():
    R = quat_matrix(0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4))  # 90 deg about z
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_isolated_points_removed_borders_kept():
    z = np.zeros((30, 30), np.float32)
    z[5:25, 5:15] = 1.0  # surface: its straight border has 14 / 24 neighbours
    z[10, 22] = 1.0  # speckle
    z[2:4, 20:22] = 1.0  # 2x2 sliver
    out = remove_isolated(z, 5, 12)
    assert out[10, 22] == 0 and not out[2:4, 20:22].any()
    assert (
        out[7:23, 5:15] == 1.0
    ).all()  # interior and straight borders (2+ px from a corner) kept
    assert out[5, 5] == 0  # corner: 8 neighbours only
