import numpy as np

import pinocchio as pin

from calib_orchestrator.hand_eye import inv_T, robot_world_hand_eye, to_T


def rot_axis(axis, angle):
    return pin.exp3(np.asarray(axis, float) / np.linalg.norm(axis) * angle)


def random_T(rng, t_scale):
    return to_T(
        rot_axis(rng.normal(size=3), rng.uniform(-np.pi, np.pi)),
        rng.normal(size=3) * t_scale,
    )


def test_robot_world_hand_eye_synthetic():
    rng = np.random.default_rng(1)
    T_ref_base = random_T(rng, 1.0)
    T_tool_cube = random_T(rng, 0.1)
    T_base_tool = [
        to_T(
            rot_axis(rng.normal(size=3), rng.uniform(-0.8, 0.8)),
            [0.4, 0.1, 0.3] + rng.normal(size=3) * 0.1,
        )
        for _ in range(25)
    ]
    noise = [
        to_T(rot_axis(rng.normal(size=3), np.radians(0.05)), rng.normal(size=3) * 2e-4)
        for _ in T_base_tool
    ]
    T_ref_cube = [
        T_ref_base @ Tb @ T_tool_cube @ n for Tb, n in zip(T_base_tool, noise)
    ]

    T_rb, T_tc, res = robot_world_hand_eye(T_ref_cube, T_base_tool)
    assert np.linalg.norm(T_rb[:3, 3] - T_ref_base[:3, 3]) < 2e-3
    assert np.linalg.norm(T_tc[:3, 3] - T_tool_cube[:3, 3]) < 2e-3
    assert res[:, 0].mean() < 2.0  # mm


def test_inv_T():
    T = random_T(np.random.default_rng(3), 1.0)
    assert np.allclose(T @ inv_T(T), np.eye(4))


def _random_T(rng, t_scale):
    return to_T(pin.exp3(rng.normal(size=3)), rng.normal(size=3) * t_scale)


def test_known_tool_recovers_X():
    from calib_orchestrator.hand_eye import robot_world_known_tool

    rng = np.random.default_rng(1)
    X, Z = _random_T(rng, 1.0), _random_T(rng, 0.1)
    B = [_random_T(rng, 0.5) for _ in range(20)]
    C = []
    for Bi in B:
        noise = to_T(pin.exp3(rng.normal(size=3) * 1e-3), rng.normal(size=3) * 3e-4)
        C.append(X @ Bi @ Z @ noise)
    X_est, res = robot_world_known_tool(C, B, Z)
    assert np.linalg.norm(X_est[:3, 3] - X[:3, 3]) < 1e-3
    assert res[:, 0].mean() < 2.0


def test_cube_center_frame_synthetic():
    from calib_orchestrator.hand_eye import (
        AXES,
        cube_center_frame,
        inv_T,
        parse_board_axes,
    )

    rng = np.random.default_rng(2)
    axes = parse_board_axes("0:-x,1:+x,2:+z,3:-y,4:+y")
    half = 0.07
    obj = {}
    for b, a in axes.items():  # 4x4 corners on each face, in the cube-center frame
        n = np.array(AXES[a], float)
        u = np.roll(n, 1) if abs(n[0]) < 1 else np.array([0.0, 1.0, 0.0])
        v = np.cross(n, u)
        for k, (i, j) in enumerate([(i, j) for i in range(4) for j in range(4)]):
            obj[(b, k)] = n * half + u * (i - 1.5) * 0.028 + v * (j - 1.5) * 0.028
    T_center_obj = _random_T(
        rng, 0.05
    )  # object frame = arbitrary frame (MC-Calib reference board)
    obj = {k: (inv_T(T_center_obj) @ np.append(p, 1))[:3] for k, p in obj.items()}
    T, side, err = cube_center_frame(obj, axes)
    assert (
        abs(side - 0.14) < 1e-9 and err < 1e-4
    )  # deg, arccos near 1 is not more precise
    assert np.allclose(T, T_center_obj, atol=1e-9)
