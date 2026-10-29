"""Preview mosaic of the rig: one tile per camera (color + detected ChArUco corners + depth inset +
stream status). No ROS dependency.
"""

import math

import cv2
import numpy as np

GREEN, ORANGE, RED, WHITE = (80, 220, 80), (0, 165, 255), (60, 60, 255), (255, 255, 255)
BOARD_COLORS = [
    (255, 128, 0),
    (0, 200, 255),
    (200, 0, 255),
    (0, 255, 128),
    (255, 0, 128),
]


def draw_text_lines(img, lines, x=8, y0=8, scale=0.5, step=20):
    """lines: [(text, color)] on a dark box (outlined text is misaligned with OpenCV 5)"""
    if not lines:
        return
    w = max(
        cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for t, _ in lines
    )
    sub = img[y0 : y0 + step * len(lines) + 6, x : x + w + 12]
    sub[:] = (sub * 0.35).astype(np.uint8)
    for k, (text, color) in enumerate(lines):
        cv2.putText(
            img,
            text,
            (x + 6, y0 + step * (k + 1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )


def depth_colormap(depth_mm, max_mm=2500):
    d = np.clip(depth_mm.astype(np.float32) / max_mm * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(255 - d, cv2.COLORMAP_JET)
    color[depth_mm == 0] = 0
    return color


def make_tile(color, boards, lines, tile_width, depth=None):
    """color: BGR full size, boards: [(board, Nx2 full-res corners)], lines: [(text, color)]"""
    s = tile_width / color.shape[1]
    tile = cv2.resize(
        color, (tile_width, round(color.shape[0] * s)), interpolation=cv2.INTER_AREA
    )
    for b, corners in boards:
        for u, v in corners * s:
            cv2.circle(
                tile,
                (int(u), int(v)),
                3,
                BOARD_COLORS[b % len(BOARD_COLORS)],
                -1,
                cv2.LINE_AA,
            )
    if depth is not None:
        inset_w = tile_width // 4
        inset = cv2.resize(
            depth_colormap(depth),
            (inset_w, round(depth.shape[0] * inset_w / depth.shape[1])),
            interpolation=cv2.INTER_NEAREST,
        )
        h, w = inset.shape[:2]
        tile[-h - 4 : -4, -w - 4 : -4] = inset
    draw_text_lines(tile, lines)
    return tile


def mosaic(tiles, footer=()):
    """grid of tiles (ceil(sqrt(n)) columns) + optional footer lines"""
    cols = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    out = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * th : r * th + t.shape[0], c * tw : c * tw + t.shape[1]] = t
    if footer:
        if len(tiles) < rows * cols:  # free cell for the global status
            r, c = divmod(len(tiles), cols)
            draw_text_lines(
                out[r * th : (r + 1) * th, c * tw : (c + 1) * tw], list(footer)
            )
        else:
            draw_text_lines(out, list(footer), y0=out.shape[0] - 20 * len(footer) - 10)
    return out


def blank_tile(width, height, text):
    tile = np.zeros((height, width, 3), np.uint8)
    draw_text_lines(tile, [(text, RED)])
    return tile
