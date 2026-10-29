"""Calibration pose generation for the cube held by the robot.

Samples tool poses (base frame) whose cube center lies in a workspace box, with varied tilts and yaws (position,
tilt and yaw drawn together from a scrambled Sobol sequence), and keeps the poses where the cube is predicted visible
by enough cameras when approximate camera poses are known (previous calibration).

No ROS dependency. Transforms are 4x4 numpy arrays in meters.
"""

from typing import Optional

import numpy as np
import pinocchio as pin

try:
    from scipy.stats import qmc

    _HAS_QMC = True
except ImportError:  # pragma: no cover
    _HAS_QMC = False


def space_filling_samples(
    lower: np.ndarray, upper: np.ndarray, n: int, seed: Optional[int] = None
) -> np.ndarray:
    """n points in the box [lower, upper] from a scrambled Sobol sequence (uniform random without scipy).
    The first k points of a call are the same for any n >= k (same seed): a prefix is still space-filling."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    d = lower.shape[0]
    if n <= 0:
        return np.empty((0, d))
    if _HAS_QMC:
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        unit = sampler.random(n)
    else:  # pragma: no cover
        rng = np.random.default_rng(seed)
        unit = rng.uniform(size=(n, d))
    return lower + unit * (upper - lower)


class Camera:
    """Approximate camera model for the visibility prediction (optical frame: z forward)."""

    def __init__(self, name, T_base_cam, K, width, height):
        self.name, self.T_base_cam, self.K, self.size = (
            name,
            np.asarray(T_base_cam),
            np.asarray(K),
            (width, height),
        )
        self.T_cam_base = np.linalg.inv(self.T_base_cam)

    def sees(
        self,
        T_base_tool,
        cube_center,
        face_normals,
        margin_px=150,
        depth_range=(0.3, 2.5),
        max_view_angle_deg=65.0,
    ):
        """True if the cube center projects in the image and at least one face is turned to the camera."""
        T = self.T_cam_base @ T_base_tool
        c = T[:3, :3] @ cube_center + T[:3, 3]
        if not depth_range[0] < c[2] < depth_range[1]:
            return False
        u, v = (self.K @ (c / c[2]))[:2]
        w, h = self.size
        if not (margin_px < u < w - margin_px and margin_px < v < h - margin_px):
            return False
        to_cam = -c / np.linalg.norm(c)
        cos_max = np.cos(np.radians(max_view_angle_deg))
        return any((T[:3, :3] @ n) @ to_cam > cos_max for n in face_normals)


class PoseGenerator:
    def __init__(
        self,
        center,
        half_extent,
        base_orientation,
        max_tilt_deg=35.0,
        max_yaw_deg=90.0,
        cube_center_in_tool=(0.0, 0.0, 0.1),
        face_normals_in_tool=None,
        cameras=(),
        min_cameras=2,
        seed=0,
    ):
        """
        center, half_extent:  sampling box of the CUBE CENTER in the base frame (m)
        base_orientation:     3x3 tool orientation in the base frame around which tilts/yaws are sampled, or a
                              list of them in priority order (sample_candidates: same cube center and tilt/yaw
                              with each base orientation, e.g. tool up then tool down near the table)
        cube_center_in_tool:  approximate cube center in the tool frame (m)
        face_normals_in_tool: normals of the 5 visible faces (default: cube on the flange axis -> +z, +-x, +-y)
        cameras:              [Camera]; empty -> no visibility filtering (first calibration)
        """
        self.center, self.half = (
            np.asarray(center, float),
            np.asarray(half_extent, float),
        )
        R = np.asarray(base_orientation, float)
        self.R0s = [R] if R.ndim == 2 else list(R)
        self.R0 = self.R0s[0]
        self.max_tilt, self.max_yaw = np.radians(max_tilt_deg), np.radians(max_yaw_deg)
        self.cube_center = np.asarray(cube_center_in_tool, float)
        self.normals = [
            np.asarray(n, float)
            for n in (
                face_normals_in_tool
                or [(0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
            )
        ]
        self.cameras, self.min_cameras = list(cameras), min_cameras
        self.seed = seed
        self._unit, self._next = (
            np.empty((0, 6)),
            0,
        )  # Sobol points in [0, 1]^6, consumed in order

    def _unit_point(self):
        """next point of the 6-D sequence (x, y, z, tilt direction, tilt, yaw); the block is regenerated twice
        as long when exhausted (same seed -> same prefix, the sequence just continues)"""
        if self._next == len(self._unit):
            self._unit = space_filling_samples(
                np.zeros(6), np.ones(6), max(1024, 2 * len(self._unit)), self.seed
            )
        self._next += 1
        return self._unit[self._next - 1]

    def sample_candidates(self):
        """[T_base_tool] for one cube center (Sobol in the workspace box) and one tilt/yaw, one pose per base
        orientation in priority order: the cube stays at the same place, the flange goes around it"""
        u = self._unit_point()
        p = self.center + (2 * u[:3] - 1) * self.half
        tilt_dir = 2 * np.pi * u[3]
        tilt = self.max_tilt * np.sqrt(u[4])  # uniform over the tilt disc
        R_tilt = pin.exp3(tilt * np.array([np.cos(tilt_dir), np.sin(tilt_dir), 0.0]))
        R_yaw = pin.exp3(np.array([0.0, 0.0, (2 * u[5] - 1) * self.max_yaw]))
        poses = []
        for R0 in self.R0s:
            T = np.eye(4)
            T[:3, :3] = R0 @ R_tilt @ R_yaw  # tilt/yaw expressed in the tool frame
            T[:3, 3] = p - T[:3, :3] @ self.cube_center
            poses.append(T)
        return poses

    def sample(self):
        """tool pose whose cube center (cube_center_in_tool) fills the workspace box, first base orientation"""
        return self.sample_candidates()[0]

    def visible_cameras(self, T_base_tool):
        return [
            c.name
            for c in self.cameras
            if c.sees(T_base_tool, self.cube_center, self.normals)
        ]

    def generate(self, n, max_tries=100000):
        """n tool poses (4x4, base frame), ordered to keep the robot motions short."""
        poses = []
        for _ in range(max_tries):
            if len(poses) == n:
                break
            for T in self.sample_candidates():  # first visible base orientation
                if not self.cameras or len(self.visible_cameras(T)) >= self.min_cameras:
                    poses.append(T)
                    break
        if len(poses) < n:
            raise RuntimeError(
                f"only {len(poses)}/{n} poses with the cube visible by >= {self.min_cameras} "
                "cameras: widen the sampling box or lower min_cameras"
            )
        return order_poses(poses)


def pose_distance(Ta, Tb, rot_weight=0.3):
    """position distance (m) + rot_weight * rotation angle (rad)"""
    c = np.clip((np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1) / 2, -1, 1)
    return np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) + rot_weight * np.arccos(c)


def order_poses(poses, start=None, return_indices=False):
    """greedy nearest-neighbour ordering"""
    left = list(range(len(poses)))
    order = []
    cur = start if start is not None else poses[0]
    while left:
        k = min(left, key=lambda i: pose_distance(cur, poses[i]))
        order.append(k)
        left.remove(k)
        cur = poses[k]
    return order if return_indices else [poses[k] for k in order]
