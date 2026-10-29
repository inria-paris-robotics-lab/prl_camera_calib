"""ChArUco corner counting with the MC-Calib board layout (one board per cube face, marker ids
offset per board as in McCalib::createCharucoBoards). No ROS dependency.

Works with OpenCV >= 4.7 (CharucoDetector) and with the older aruco API (Ubuntu 24.04 / Jazzy
ships OpenCV 4.6): there, the board ids cannot be set, so the markers of each board are
shifted back to 0.. before the interpolation.
"""

import cv2
import numpy as np


def read_board_config(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(path)

    def vec(key):
        node = fs.getNode(key)
        return (
            [int(node.at(i).real()) for i in range(node.size())]
            if not node.empty()
            else []
        )

    nb = int(fs.getNode("number_board").real())
    nx, ny = vec("number_x_square_per_board"), vec("number_y_square_per_board")
    if not nx:
        nx = [int(fs.getNode("number_x_square").real())] * nb
        ny = [int(fs.getNode("number_y_square").real())] * nb
    cfg = {
        "nb": nb,
        "nx": nx,
        "ny": ny,
        "length_square": fs.getNode("length_square").real(),
        "length_marker": fs.getNode("length_marker").real(),
    }
    fs.release()
    return cfg


class CharucoCounter:
    def __init__(self, cfg, dictionary=cv2.aruco.DICT_6X6_1000):
        self.dict = cv2.aruco.getPredefinedDictionary(dictionary)
        self.new_api = hasattr(cv2.aruco, "CharucoDetector")
        sq, mk = cfg["length_square"], cfg["length_marker"]
        self.boards, self.offsets = [], []
        offset = 0
        for nx, ny in zip(cfg["nx"], cfg["ny"]):
            n_markers = (nx * ny) // 2
            ids = np.arange(offset, offset + n_markers, dtype=np.int32)
            if self.new_api:
                board = cv2.aruco.CharucoBoard((nx, ny), sq, mk, self.dict, ids)
                self.boards.append(cv2.aruco.CharucoDetector(board))
            else:
                self.boards.append(
                    cv2.aruco.CharucoBoard_create(nx, ny, sq, mk, self.dict)
                )
            self.offsets.append((offset, n_markers))
            offset += n_markers
        if not self.new_api:
            self.params = cv2.aruco.DetectorParameters_create()

    def detect(self, img, scale=1.0):
        """ChArUco corners per detected board, [(board index, Nx2 corners in full-resolution pixels)]"""
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        found = []
        if self.new_api:
            for b, det in enumerate(self.boards):
                corners = det.detectBoard(gray)[0]
                if corners is not None and len(corners):
                    found.append((b, corners.reshape(-1, 2) / scale))
            return found
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dict, parameters=self.params
        )
        if ids is None:
            return found
        ids = ids.ravel()
        for b, (board, (offset, n)) in enumerate(zip(self.boards, self.offsets)):
            sel = (ids >= offset) & (ids < offset + n)
            if sel.sum() < 2:
                continue
            n_c, c_corners, _ = cv2.aruco.interpolateCornersCharuco(
                [corners[k] for k in np.flatnonzero(sel)],
                (ids[sel] - offset).reshape(-1, 1),
                gray,
                board,
            )
            if n_c:
                found.append((b, c_corners.reshape(-1, 2) / scale))
        return found

    def count(self, img, scale=1.0):
        """number of ChArUco corners detected over all boards"""
        return sum(len(c) for _, c in self.detect(img, scale))
