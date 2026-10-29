import numpy as np

import pinocchio as pin

from calib_orchestrator.evaluate import best_scale, fit_forced_height, rate
from calib_orchestrator.hand_eye import fit_robust, robot_world_hand_eye, to_T


def synthetic(scale=1.0, n=30, seed=0):
    """C_i = X B_i Z with cube poses up / down (tool z varied) and translations seen `scale` times too large"""
    rng = np.random.default_rng(seed)
    X = to_T(pin.exp3(np.array([2.0, 0.3, -0.5])), [0.3, -0.8, 1.2])
    Z = to_T(np.eye(3), [0.0, 0.0, 0.127])
    B = []
    for k in range(n):
        flip = np.diag([1.0, -1.0, -1.0]) if k % 2 else np.eye(3)
        R = pin.exp3(rng.normal(size=3) * 0.35) @ flip
        B.append(to_T(R, [-0.2, 0.0, 0.3] + rng.uniform(-0.2, 0.2, 3)))
    noise = [
        to_T(pin.exp3(rng.normal(size=3) * 2e-4), rng.normal(size=3) * 2e-4) for _ in B
    ]
    C = [X @ b @ Z @ e for b, e in zip(B, noise)]
    return [to_T(c[:3, :3], c[:3, 3] * scale) for c in C], B, X, Z


def test_best_scale_finds_a_wrong_square_size():
    C, B, _, _ = synthetic(scale=1.0)
    assert abs(best_scale(C, B)[0] - 1.0) < 0.001
    C, B, _, _ = synthetic(
        scale=1.01
    )  # square_size 1 % too large, as 28 mm instead of 27.72 mm
    best, residual = best_scale(C, B)
    assert abs(best - 1 / 1.01) < 0.001
    assert residual < robot_world_hand_eye(C, B)[2][:, 0].mean()


def test_forced_height_raises_the_residual_when_observed():
    C, B, _, Z = synthetic()
    X, Ze, res = robot_world_hand_eye(C, B)
    _, _, forced = fit_forced_height(C, B, np.eye(4), X, Ze, float(Ze[2, 3]) + 0.005)
    assert abs(Ze[2, 3] - 0.127) < 1e-3
    assert (
        forced[:, 0].mean() - res[:, 0].mean() > 0.5
    )  # mm: the cube_height check rates this OK


def test_rate():
    assert [rate(v, 1.0, 2.0) for v in (0.5, 1.5, 2.5)] == ["OK", "WARN", "FAIL"]


def test_fit_robust_drops_outlier_poses():
    C, B, X, _ = synthetic(n=30)
    C[5] = C[5] @ to_T(np.eye(3), [0.004, 0.0, 0.0])  # bad detection: cube 4 mm off
    C[9] = C[9] @ to_T(np.eye(3), [0.0, 0.0, -0.010])  # 10 mm off
    Xr, _, res, kept = fit_robust(robot_world_hand_eye, C, B)
    assert not kept[5] and not kept[9] and kept.sum() == 28
    assert res[:, 0].max() < 1.0  # mm, residuals of the kept poses
    assert np.linalg.norm(Xr[:3, 3] - X[:3, 3]) < 1e-3
