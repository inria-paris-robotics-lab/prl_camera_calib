"""Quality report of a calibration session (MC-Calib + robot-world/hand-eye), offline.

    ros2 run calib_orchestrator evaluate_session <session_dir> [--compare <other_session>] [--json out.json]

Checks, each rated OK / WARN / FAIL:
  reprojection   MC-Calib reprojection error per camera, frames used per camera, single camera group
  poses          number of poses, tool up / down balance, tilt spread
  hand_eye       robot-world/hand-eye residuals (cube predicted by the robot vs cube seen by the cameras), outlier
                 poses dropped (> 3 x median, as the orchestrator does; WARN when more than 10 %)
  scale          hand-eye residual when the MC-Calib scene is rescaled: a clear minimum away from 1 means a wrong
                 square_size (the cameras only see the scene up to a scale, the robot measures in meters)
  cube_height    observability of the cube position along the tool axis (residual increase when forced 5 mm off;
                 traded with a vertical camera shift when the tool axis is always vertical), URDF agreement
  stability      camera positions over random subsets of the poses (bootstrap)
  compare        camera displacement with another session of the same rig (repeatability)

No ROS dependency. Transforms are 4x4 numpy arrays in meters.
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
import yaml

from calib_orchestrator import hand_eye as he

OK, WARN, FAIL = "OK", "WARN", "FAIL"
DEFAULT_BOARD_AXES = "0:-y,1:+y,2:+z,3:+x,4:-x"


def rate(value, warn, fail):
    """OK below warn, WARN below fail, FAIL above"""
    return OK if value < warn else WARN if value < fail else FAIL


def scaled(C, s):
    return [he.to_T(T[:3, :3], T[:3, 3] * s) for T in C]


def camera_names(d, n):
    """short names (golf...) from cams_extrinsics.yml when present, else Cam_001..."""
    p = Path(d) / "cams_extrinsics.yml"
    if p.exists():
        frames = [
            t["child_frame"]
            for t in yaml.safe_load(p.read_text()).get("transforms", [])
        ]
        if len(frames) == n:
            return [f.split("_camera")[0] for f in frames]
    return [f"Cam_{i + 1:03d}" for i in range(n)]


def load_session(d):
    """MC-Calib result, matched cube / robot poses and the square size of the session"""
    d = Path(d)
    calib = he.load_mccalib(d)
    cubes = he.cube_poses(calib)
    frames = yaml.safe_load((d / "robot_poses.yml").read_text())["frames"]
    robot = {int(f["frame"]): np.array(f["T_base_tool"]) for f in frames}
    common = sorted(set(cubes) & set(robot))
    m = re.search(r"^square_size:\s*([0-9.]+)", (d / "mc_calib.yml").read_text(), re.M)
    return {
        "calib": calib,
        "frames": common,
        "C": [cubes[f][0] for f in common],
        "B": [robot[f] for f in common],
        "n_robot": len(robot),
        "square_mm": float(m.group(1)) if m else float("nan"),
    }


def reprojection_errors(d):
    """{camera index: [errors px]} from MC-Calib reprojection_error_data.yml, number of camera groups"""
    fs = cv2.FileStorage(
        str(Path(d) / "reprojection_error_data.yml"), cv2.FILE_STORAGE_READ
    )
    groups = int(fs.getNode("nb_camera_group").real())
    out = {}
    for gi in range(groups):
        g = fs.getNode(f"camera_group_{gi}")
        for f in g.keys():
            fn = g.getNode(f)
            if not fn.isMap():
                continue
            for c in fn.keys():
                cn = fn.getNode(c)
                if cn.isMap() and not cn.getNode("error_list").empty():
                    out.setdefault(int(c.split("_")[1]), []).append(
                        cn.getNode("error_list").mat().ravel()
                    )
    return {c: np.concatenate(v) for c, v in out.items()}, groups


def best_scale(C, B):
    """(scale of the camera translations minimizing the hand-eye residual, that residual mm): scan 0.97-1.03,
    parabola through the minimum. Away from 1: the MC-Calib square_size is wrong by that factor"""
    scales = np.arange(0.970, 1.0301, 0.002)
    curve = np.array(
        [he.robot_world_hand_eye(scaled(C, k), B)[2][:, 0].mean() for k in scales]
    )
    i = int(curve.argmin())
    lo, hi = max(i - 4, 0), min(i + 5, len(scales))
    p = np.polyfit(scales[lo:hi], curve[lo:hi], 2)
    return (float(-p[1] / (2 * p[0])) if p[0] > 0 else float(scales[i])), float(
        curve[i]
    )


def fit_forced_height(C, B, T_center_obj, X0, Z0, height):
    """hand-eye with the cube center forced at `height` along the tool z axis, everything else free"""
    Zc0 = Z0 @ he.inv_T(T_center_obj)

    def unpack(q):
        return he.to_T(pin.exp3(q[:3]), q[3:6]), he.to_T(
            pin.exp3(q[6:9]), [q[9], q[10], height]
        ) @ T_center_obj

    def fun(q):
        X, Z = unpack(q)
        return he._pose_residuals(
            np.concatenate(
                [pin.log3(X[:3, :3]), X[:3, 3], pin.log3(Z[:3, :3]), Z[:3, 3]]
            ),
            C,
            B,
            0.1,
        )

    q = he._levenberg_marquardt(
        fun,
        np.concatenate(
            [pin.log3(X0[:3, :3]), X0[:3, 3], pin.log3(Zc0[:3, :3]), Zc0[:3, 3][:2]]
        ),
        40,
    )
    X, Z = unpack(q)
    return X, Z, he.residuals(X, Z, C, B)


def evaluate(
    session_dir,
    compare=None,
    board_axes=DEFAULT_BOARD_AXES,
    urdf_cube_height=None,
    bootstrap=30,
    seed=0,
    log=print,
):
    s = load_session(session_dir)
    calib, C, B = s["calib"], s["C"], s["B"]
    names = camera_names(session_dir, len(calib["cameras"]))
    report = {"session": str(session_dir), "checks": {}}

    def check(key, status, summary, **data):
        report["checks"][key] = {"status": status, "summary": summary, **data}
        log(f"[{status:4s}] {key:12s} {summary}")

    # ------------------------------------------------------------------ MC-Calib reprojection
    errs, groups = reprojection_errors(session_dir)
    per_cam = {
        names[c]: {
            "frames_points": int(e.size),
            "mean_px": float(e.mean()),
            "p95_px": float(np.percentile(e, 95)),
        }
        for c, e in sorted(errs.items())
    }
    worst = max(v["mean_px"] for v in per_cam.values())
    status = FAIL if groups != 1 else rate(worst, 0.5, 1.0)
    check(
        "reprojection",
        status,
        f"{groups} camera group(s), mean "
        + ", ".join(
            f"{n} {v['mean_px']:.2f} px (p95 {v['p95_px']:.2f})"
            for n, v in per_cam.items()
        ),
        groups=groups,
        cameras=per_cam,
    )

    # ------------------------------------------------------------------ pose set
    tool_z = np.array([b[:3, 2] for b in B])
    up = int((tool_z[:, 2] > 0).sum())
    tilt = np.degrees(np.arccos(np.clip(np.abs(tool_z[:, 2]), -1, 1)))
    frac_up = up / max(len(B), 1)
    status = (
        FAIL
        if len(B) < 15
        else WARN
        if len(B) < 30 or not 0.25 <= frac_up <= 0.75
        else OK
    )
    check(
        "poses",
        status,
        f"{len(B)} poses used / {s['n_robot']} captured, tool up {up} / down {len(B) - up}, "
        f"tilt from vertical median {np.median(tilt):.0f} deg (max {tilt.max():.0f})",
        used=len(B),
        captured=s["n_robot"],
        up=up,
        down=len(B) - up,
    )

    # ------------------------------------------------------------------ hand-eye
    # outlier poses dropped as in the orchestrator; the next checks use the kept poses
    X, Z, res, kept = he.fit_robust(he.robot_world_hand_eye, C, B)
    outliers = [f for f, k in zip(s["frames"], kept) if not k]
    C, B = [c for c, k in zip(C, kept) if k], [b for b, k in zip(B, kept) if k]
    status = rate(res[:, 0].mean(), 1.5, 3.0)
    if len(outliers) > 0.1 * len(kept):
        status = max(status, WARN, key=[OK, WARN, FAIL].index)
    check(
        "hand_eye",
        status,
        f"residual {res[:, 0].mean():.2f} mm (max {res[:, 0].max():.2f}), {res[:, 1].mean():.3f} deg"
        + (f", outlier frames dropped {outliers}" if outliers else ""),
        mean_mm=float(res[:, 0].mean()),
        max_mm=float(res[:, 0].max()),
        mean_deg=float(res[:, 1].mean()),
        outliers=outliers,
    )

    # ------------------------------------------------------------------ scale
    best, residual_at_best = best_scale(C, B)
    gain = 1 - residual_at_best / res[:, 0].mean()
    status = (
        FAIL
        if abs(best - 1) > 0.003 and gain > 0.4
        else WARN
        if abs(best - 1) > 0.0015 and gain > 0.2
        else OK
    )
    check(
        "scale",
        status,
        f"best scale {best:.4f} (residual {residual_at_best:.2f} mm, -{gain * 100:.0f} %)"
        + (
            f" -> square_size {s['square_mm']:.2f} mm is probably {s['square_mm'] * best:.2f} mm, cameras "
            f"off by ~{abs(1 - best) * 1000:.0f} mm per meter"
            if status != OK
            else ""
        ),
        best_scale=best,
        residual_at_best_mm=float(residual_at_best),
        square_mm=s["square_mm"],
        suggested_square_mm=s["square_mm"] * best,
    )

    # ------------------------------------------------------------------ cube height along the tool axis
    T_center_obj, side, err = he.cube_center_frame(
        calib["object"], he.parse_board_axes(board_axes)
    )
    h = float((Z @ he.inv_T(T_center_obj))[2, 3])
    cams = he.base_to_cameras(X, calib)
    sens, shift = [], []
    for dh in (-0.005, 0.005):
        Xf, _, rf = fit_forced_height(C, B, T_center_obj, X, Z, h + dh)
        sens.append(rf[:, 0].mean() - res[:, 0].mean())
        shift.append(
            max(
                np.linalg.norm(a[:3, 3] - b[:3, 3])
                for a, b in zip(cams, he.base_to_cameras(Xf, calib))
            )
        )
    # observed when a wrong height clearly raises the residual
    status = FAIL if min(sens) < 0.2 else WARN if min(sens) < 0.5 else OK
    summary = (
        f"cube center {h * 1e3:.1f} mm along the tool axis; forcing it 5 mm off raises the residual by "
        f"{min(sens):.2f} mm and moves the cameras up to {max(shift) * 1e3:.1f} mm"
    )
    extra = {}
    if urdf_cube_height is not None:
        Xu, _, ru = fit_forced_height(C, B, T_center_obj, X, Z, urdf_cube_height)
        du = max(
            np.linalg.norm(a[:3, 3] - b[:3, 3])
            for a, b in zip(cams, he.base_to_cameras(Xu, calib))
        )
        summary += (
            f"; URDF {urdf_cube_height * 1e3:.1f} mm: {(h - urdf_cube_height) * 1e3:+.1f} mm, "
            f"cameras move {du * 1e3:.1f} mm"
        )
        extra = {"urdf_height_m": urdf_cube_height, "urdf_camera_shift_mm": du * 1e3}
        status = max(
            status,
            rate(abs(h - urdf_cube_height) * 1e3, 2.0, 4.0),
            key=[OK, WARN, FAIL].index,
        )
    if min(sens) < 0.5:
        summary += " (poorly observed: add tool-up poses / larger tilts)"
    check(
        "cube_height",
        status,
        summary,
        height_m=h,
        board_spacing_mm=side * 1e3,
        camera_shift_5mm=max(shift) * 1e3,
        **extra,
    )

    # ------------------------------------------------------------------ bootstrap stability
    rng = np.random.default_rng(seed)
    k = max(int(0.7 * len(C)), 3)
    pos = []
    for _ in range(bootstrap):
        idx = rng.choice(len(C), k, replace=False)
        Xb, _, _ = he.robot_world_hand_eye([C[j] for j in idx], [B[j] for j in idx])
        pos.append([T[:3, 3] for T in he.base_to_cameras(Xb, calib)])
    std = np.linalg.norm(np.array(pos).std(axis=0), axis=1) * 1e3  # per camera, mm
    check(
        "stability",
        rate(std.max(), 1.0, 3.0),
        f"camera position spread over {bootstrap} subsets of {k} poses: "
        + ", ".join(f"{n} {v:.2f} mm" for n, v in zip(names, std)),
        spread_mm=dict(zip(names, std.tolist())),
    )

    # ------------------------------------------------------------------ comparison with another session
    if compare:
        try:
            o = load_session(compare)
            r = (
                s["square_mm"] / o["square_mm"]
            )  # other session brought to this square_size
            cams_o = [
                dict(
                    c,
                    T_ref_cam=he.to_T(
                        c["T_ref_cam"][:3, :3], c["T_ref_cam"][:3, 3] * r
                    ),
                )
                for c in o["calib"]["cameras"]
            ]
            cal_o = dict(o["calib"], cameras=cams_o)
            Xo, _, _, _ = he.fit_robust(
                he.robot_world_hand_eye, scaled(o["C"], r), o["B"]
            )
            other = he.base_to_cameras(Xo, cal_o)
            d = [
                (
                    np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1e3,
                    he.rot_angle_deg((he.inv_T(a) @ b)[:3, :3]),
                )
                for a, b in zip(cams, other)
            ]
            check(
                "compare",
                rate(max(m for m, _ in d), 2.0, 5.0),
                f"vs {Path(compare).name}: "
                + ", ".join(
                    f"{n} {m:.1f} mm {a:.2f} deg" for n, (m, a) in zip(names, d)
                )
                + (
                    f" (its square_size {o['square_mm']} mm rescaled to {s['square_mm']} mm)"
                    if r != 1
                    else ""
                ),
                displacement=dict(zip(names, d)),
            )
        except Exception as e:  # noqa: BLE001 - a missing / incomplete session only skips this check
            check("compare", WARN, f"cannot compare with {compare}: {e}")

    worst = max(
        (c["status"] for c in report["checks"].values()), key=[OK, WARN, FAIL].index
    )
    report["status"] = worst
    log(f"overall: {worst}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "session",
        help="calibration session folder (MC-Calib outputs + robot_poses.yml)",
    )
    ap.add_argument(
        "--compare", help="another session of the same camera rig (repeatability)"
    )
    ap.add_argument(
        "--board-axes",
        default=DEFAULT_BOARD_AXES,
        help="board outward axes in the cube frame",
    )
    ap.add_argument(
        "--urdf-cube-height",
        type=float,
        help="URDF cube center along the tool axis (m), e.g. 0.127",
    )
    ap.add_argument("--bootstrap", type=int, default=30)
    ap.add_argument("--json", help="also write the report there")
    a = ap.parse_args(argv)
    report = evaluate(
        a.session, a.compare, a.board_axes, a.urdf_cube_height, a.bootstrap
    )
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2, default=float))
    return 0 if report["status"] != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
