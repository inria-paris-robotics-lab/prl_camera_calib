"""Robot-world / hand-eye calibration from an MC-Calib result and the recorded robot poses.

The cube is held by the robot, the cameras are fixed:
    C_i = X . B_i . Z     C_i = T_ref_cube(i), B_i = T_base_tool(i)
with ref = the MC-Calib reference camera (Cam_001). Unknowns: X = T_ref_base, Z = T_tool_cube.

Initialization: pairs of poses give C_i C_j^-1 . X = X . B_i B_j^-1 (AX = XB, Park's method),
then Z is averaged from X and the poses, then both are refined jointly (Levenberg-Marquardt).

Known-tool variant (robot_world_known_tool): Z is given (URDF tool -> cube center, combined with the
cube-center frame of the MC-Calib object, cube_center_frame) and only X is estimated.
Implemented with numpy (cv2.calibrateRobotWorldHandEye is not in every OpenCV python build).

No ROS dependency: usable offline on a session folder.
All transforms are 4x4 numpy arrays in meters (MC-Calib writes millimeters).
"""

from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin

MM = 1e-3


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return to_T(R.T, -R.T @ t)


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _seq(node):
    return np.array([node.at(j).real() for j in range(node.size())])


# --------------------------------------------------------------------------- MC-Calib I/O
def load_mccalib(calib_dir):
    """Cameras, cube 3D points and detected keypoints of an MC-Calib save_path.

    returns dict:
      cameras:   [{K, dist, T_ref_cam (m), group, width, height}] in Cam_001.. order
      object:    {(board, corner): xyz (m)} of object_0 (the cube)
      keypoints: {frame: [(camera, board, corner_ids, pts_2d Nx2)]}
    """
    d = Path(calib_dir)
    fc = cv2.FileStorage(str(d / "calibrated_cameras_data.yml"), cv2.FILE_STORAGE_READ)
    cams = []
    for i in range(int(fc.getNode("nb_camera").real())):
        n = fc.getNode(f"camera_{i}")
        P = n.getNode("camera_pose_matrix").mat()  # camera -> reference, mm
        cams.append(
            {
                "K": n.getNode("camera_matrix").mat(),
                "dist": n.getNode("distortion_vector").mat().ravel(),
                "T_ref_cam": to_T(P[:3, :3], P[:3, 3] * MM),
                "group": int(n.getNode("camera_group").real()),
                "width": int(n.getNode("img_width").real()),
                "height": int(n.getNode("img_height").real()),
            }
        )

    fo = cv2.FileStorage(str(d / "calibrated_objects_data.yml"), cv2.FILE_STORAGE_READ)
    pts = fo.getNode("object_0").getNode("points").mat()  # rows: x y z board corner
    obj = {
        (int(pts[3, k]), int(pts[4, k])): pts[:3, k].astype(np.float64) * MM
        for k in range(pts.shape[1])
    }

    fk = cv2.FileStorage(str(d / "detected_keypoints_data.yml"), cv2.FILE_STORAGE_READ)
    kps = {}
    for i in range(int(fk.getNode("nb_camera").real())):
        n = fk.getNode(f"camera_{i}")
        frames, boards = (
            _seq(n.getNode("frame_idxs")).astype(int),
            _seq(n.getNode("board_idxs")).astype(int),
        )
        p2d, ids = n.getNode("pts_2d"), n.getNode("charuco_idxs")
        for k, (f, b) in enumerate(zip(frames, boards)):
            kps.setdefault(int(f), []).append(
                (i, int(b), _seq(ids.at(k)).astype(int), _seq(p2d.at(k)).reshape(-1, 2))
            )
    return {"cameras": cams, "object": obj, "keypoints": kps}


# --------------------------------------------------------------------------- cube pose per frame
def _residuals(x, obs):
    T_ref_cube = to_T(pin.exp3(x[:3]), x[3:])
    res = []
    for T_cam_ref, K, dist, P3, P2 in obs:
        T = T_cam_ref @ T_ref_cube
        proj, _ = cv2.projectPoints(P3, pin.log3(T[:3, :3]), T[:3, 3], K, dist)
        res.append((proj.reshape(-1, 2) - P2).ravel())
    return np.concatenate(res)


def _refine(x, obs, iters=15):
    """Levenberg-Marquardt on the multi-camera reprojection error (6 parameters, numeric Jacobian)."""
    lam = 1e-3
    r = _residuals(x, obs)
    for _ in range(iters):
        J = np.empty((r.size, 6))
        for j in range(6):
            h = 1e-6 if j < 3 else 1e-7
            dx = np.zeros(6)
            dx[j] = h
            J[:, j] = (_residuals(x + dx, obs) - r) / h
        A, g = J.T @ J, J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        r_new = _residuals(x + step, obs)
        if r_new @ r_new < r @ r:
            x, r, lam = x + step, r_new, lam * 0.3
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            lam *= 10
    return x, r


def cube_poses(calib, min_points=12):
    """{frame: (T_ref_cube (m), rms reprojection px, n points, n cameras)}, jointly over all cameras."""
    cams, obj = calib["cameras"], calib["object"]
    out = {}
    for frame, views in sorted(calib["keypoints"].items()):
        per_cam = {}
        for cam, board, ids, p2 in views:
            keep = [k for k, c in enumerate(ids) if (board, int(c)) in obj]
            if keep:
                P3 = np.array([obj[(board, int(ids[k]))] for k in keep])
                o = per_cam.setdefault(cam, [[], []])
                o[0].append(P3)
                o[1].append(p2[keep])
        obs = []
        for cam, (P3, P2) in per_cam.items():
            c = cams[cam]
            obs.append(
                (inv_T(c["T_ref_cam"]), c["K"], c["dist"], np.vstack(P3), np.vstack(P2))
            )
        n = sum(len(o[3]) for o in obs)
        if n < min_points:
            continue
        # initialization: PnP in the camera that sees the most points
        T_cam_ref, K, dist, P3, P2 = max(obs, key=lambda o: len(o[3]))
        ok, rv, tv = cv2.solvePnP(P3, P2, K, dist)
        if not ok:
            continue
        T0 = inv_T(T_cam_ref) @ to_T(pin.exp3(rv.ravel()), tv)
        x, r = _refine(np.concatenate([pin.log3(T0[:3, :3]), T0[:3, 3]]), obs)
        out[frame] = (
            to_T(pin.exp3(x[:3]), x[3:]),
            float(np.sqrt(np.mean(r**2) * 2)),
            n,
            len(obs),
        )
    return out


# --------------------------------------------------------------------------- robot-world / hand-eye
def closest_rotation(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def _ax_xb(C, B, min_angle_deg=5.0):
    """X from AX = XB on all pose pairs with enough relative rotation (Park & Martin)."""
    alphas, betas, lhs = [], [], []
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            A, Bij = C[i] @ inv_T(C[j]), B[i] @ inv_T(B[j])
            a, b = pin.log3(A[:3, :3]), pin.log3(Bij[:3, :3])
            if np.degrees(np.linalg.norm(b)) < min_angle_deg:
                continue
            alphas.append(a)
            betas.append(b)
            lhs.append((A, Bij))
    if len(alphas) < 2:
        raise ValueError(
            "not enough rotation between the robot poses (tilt the cube more)"
        )
    R = closest_rotation(
        np.einsum("ki,kj->ij", np.array(alphas), np.array(betas))
    )  # alpha = R beta
    M = np.vstack([A[:3, :3] - np.eye(3) for A, _ in lhs])
    v = np.concatenate([R @ Bij[:3, 3] - A[:3, 3] for A, Bij in lhs])
    t = np.linalg.lstsq(M, v, rcond=None)[0]
    return to_T(R, t)


def _pose_residuals(p, C, B, rot_scale):
    X = to_T(pin.exp3(p[:3]), p[3:6])
    Z = to_T(pin.exp3(p[6:9]), p[9:12])
    res = []
    for Ci, Bi in zip(C, B):
        Tp = X @ Bi @ Z
        res.append(Tp[:3, 3] - Ci[:3, 3])
        res.append(pin.log3(Tp[:3, :3].T @ Ci[:3, :3]) * rot_scale)
    return np.concatenate(res)


def robot_world_hand_eye(T_ref_cube, T_base_tool, rot_scale=0.1, iters=30):
    """T_ref_cube, T_base_tool: matched lists of 4x4 (m). Returns (T_ref_base, T_tool_cube, residuals).

    rot_scale (m/rad) weights the rotation residuals against the translation ones
    (0.1: 1 mrad of rotation counts as 0.1 mm).
    """
    if len(T_ref_cube) < 3:
        raise ValueError(
            "robot-world/hand-eye needs at least 3 poses (10+ with varied rotations recommended)"
        )
    C, B = list(T_ref_cube), list(T_base_tool)
    X = _ax_xb(C, B)
    Zs = [inv_T(Bi) @ inv_T(X) @ Ci for Ci, Bi in zip(C, B)]
    Z = to_T(
        closest_rotation(sum(T[:3, :3] for T in Zs)),
        np.mean([T[:3, 3] for T in Zs], axis=0),
    )

    p = np.concatenate([pin.log3(X[:3, :3]), X[:3, 3], pin.log3(Z[:3, :3]), Z[:3, 3]])
    p = _levenberg_marquardt(lambda q: _pose_residuals(q, C, B, rot_scale), p, iters)
    T_ref_base = to_T(pin.exp3(p[:3]), p[3:6])
    T_tool_cube = to_T(pin.exp3(p[6:9]), p[9:12])
    return T_ref_base, T_tool_cube, residuals(T_ref_base, T_tool_cube, C, B)


def _levenberg_marquardt(fun, p, iters=30, h=1e-7):
    r, lam = fun(p), 1e-3
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for j in range(p.size):
            dp = np.zeros(p.size)
            dp[j] = h
            J[:, j] = (fun(p + dp) - r) / h
        A, g = J.T @ J, J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        r_new = fun(p + step)
        if r_new @ r_new < r @ r:
            p, r, lam = p + step, r_new, lam * 0.3
            if np.linalg.norm(step) < 1e-12:
                break
        else:
            lam *= 10
    return p


def robot_world_known_tool(
    T_ref_cube, T_base_tool, T_tool_cube, rot_scale=0.1, iters=30
):
    """X = T_ref_base with Z = T_tool_cube given. Returns (T_ref_base, residuals).
    Each pose gives X_i = C_i Z^-1 B_i^-1: averaged, then refined on all poses with Z fixed."""
    if len(T_ref_cube) < 1:
        raise ValueError("no pose")
    C, B, Z = list(T_ref_cube), list(T_base_tool), np.asarray(T_tool_cube)
    Xs = [Ci @ inv_T(Z) @ inv_T(Bi) for Ci, Bi in zip(C, B)]
    X = to_T(
        closest_rotation(sum(T[:3, :3] for T in Xs)),
        np.mean([T[:3, 3] for T in Xs], axis=0),
    )
    z = np.concatenate([pin.log3(Z[:3, :3]), Z[:3, 3]])
    p = _levenberg_marquardt(
        lambda q: _pose_residuals(np.concatenate([q, z]), C, B, rot_scale),
        np.concatenate([pin.log3(X[:3, :3]), X[:3, 3]]),
        iters,
    )
    X = to_T(pin.exp3(p[:3]), p[3:6])
    return X, residuals(X, Z, C, B)


def outlier_poses(res_mm, floor_mm=2.0):
    """mask of the poses whose residual is > max(3 x median, floor_mm): bad detection, cube slipped, robot moving"""
    return res_mm > max(3 * np.median(res_mm), floor_mm)


def fit_robust(solve, C, B, min_poses=10):
    """solve(C, B) -> (X, Z, residuals), refitted once without the outlier poses (if >= min_poses remain).
    Returns (X, Z, residuals of the kept poses, kept mask)"""
    X, Z, res = solve(C, B)
    bad = outlier_poses(res[:, 0])
    if bad.any() and (~bad).sum() >= min_poses:
        X, Z, res = solve(
            [c for c, b in zip(C, bad) if not b], [t for t, b in zip(B, bad) if not b]
        )
    else:
        bad[:] = False
    return X, Z, res, ~bad


# --------------------------------------------------------------------------- cube center of the MC-Calib object
AXES = {
    "+x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "+y": (0, 1, 0),
    "-y": (0, -1, 0),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
}


def parse_board_axes(text):
    """'0:+z,1:+x,2:+y,3:-x,4:-y' -> {0: '+z', ...}"""
    out = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        board, axis = item.split(":")
        if axis.strip() not in AXES:
            raise ValueError(f"cube_board_axes: {axis!r} not in {list(AXES)}")
        out[int(board)] = axis.strip()
    return out


def cube_center_frame(obj, board_axes):
    """T_center_object: the MC-Calib object frame expressed in the cube-center frame (URDF box axes).

    obj: {(board, corner): xyz (m)} (load_mccalib), board_axes: {board: '+x'...} outward direction of
    each board in the cube-center frame. The center is the point at the same distance (half side) from
    every board plane, the rotation maps the board normals onto the given axes.
    Returns (T_center_object, side length (m), max angle between a board normal and its axis (deg)).
    """
    boards = sorted({b for b, _ in obj})
    missing = [b for b in boards if b not in board_axes]
    if missing:
        raise ValueError(f"cube_board_axes has no axis for boards {missing}")
    pts = {
        b: np.array([xyz for (bb, _), xyz in obj.items() if bb == b]) for b in boards
    }
    c0 = np.mean(np.vstack(list(pts.values())), axis=0)
    normals, offsets = [], []
    for b in boards:
        m = pts[b].mean(axis=0)
        n = np.linalg.svd(pts[b] - m)[2][2]
        n = n if n @ (m - c0) > 0 else -n  # outward
        normals.append(n)
        offsets.append(n @ m)
    # n_b . c + half = n_b . p_b  for every board
    sol = np.linalg.lstsq(
        np.hstack([np.array(normals), np.ones((len(boards), 1))]),
        np.array(offsets),
        rcond=None,
    )[0]
    center, half = sol[:3], sol[3]
    axes = np.array([AXES[board_axes[b]] for b in boards], float)
    R = closest_rotation(axes.T @ np.array(normals))  # axis_b = R n_b
    err = max(
        float(np.degrees(np.arccos(np.clip(a @ (R @ n), -1, 1))))
        for a, n in zip(axes, normals)
    )
    return to_T(R, -R @ center), float(2 * half), err


def residuals(T_ref_base, T_tool_cube, T_ref_cube, T_base_tool):
    """per pose (translation error mm, rotation error deg) between measured and predicted cube pose"""
    out = []
    for Tm, Tb in zip(T_ref_cube, T_base_tool):
        Tp = T_ref_base @ Tb @ T_tool_cube
        out.append(
            (
                float(np.linalg.norm(Tp[:3, 3] - Tm[:3, 3]) / MM),
                rot_angle_deg(Tp[:3, :3].T @ Tm[:3, :3]),
            )
        )
    return np.array(out)


def base_to_cameras(T_ref_base, calib):
    """[T_base_cam] for every camera (m)"""
    T_base_ref = inv_T(T_ref_base)
    return [T_base_ref @ c["T_ref_cam"] for c in calib["cameras"]]
