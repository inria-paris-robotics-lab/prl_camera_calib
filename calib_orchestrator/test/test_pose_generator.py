import numpy as np

from calib_orchestrator.pose_generator import Camera, PoseGenerator, pose_distance


def look_at(eye, target):
    """camera optical frame (z forward, y down) at eye looking at target, in the base frame"""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    T[:3, 3] = eye
    return T


def test_generate_visible_and_ordered():
    K = np.array([[1700, 0, 1920], [0, 1700, 1080], [0, 0, 1.0]])
    center = np.array([0.45, 0.0, 0.35])
    cams = [
        Camera(
            f"cam_{k}",
            look_at(center + 1.2 * np.array([np.cos(a), np.sin(a), 0.6]), center),
            K,
            3840,
            2160,
        )
        for k, a in enumerate(np.radians([0, 120, 240]))
    ]
    R_down = np.diag([1.0, -1.0, -1.0])  # tool z pointing down
    gen = PoseGenerator(
        center,
        (0.12, 0.12, 0.08),
        R_down,
        cube_center_in_tool=(0, 0, 0.1),
        cameras=cams,
        min_cameras=2,
        seed=4,
    )
    poses = gen.generate(30)
    assert len(poses) == 30
    for T in poses:
        assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-9)
        assert len(gen.visible_cameras(T)) >= 2
    # greedy ordering: consecutive moves shorter than a random order on average
    ordered = np.mean([pose_distance(a, b) for a, b in zip(poses, poses[1:])])
    shuffled = [poses[k] for k in np.random.default_rng(0).permutation(len(poses))]
    assert ordered < np.mean(
        [pose_distance(a, b) for a, b in zip(shuffled, shuffled[1:])]
    )


def test_cube_center_stays_in_workspace():
    center, half, cube = (
        np.array([-0.23, 0.0, 0.25]),
        np.array([0.12, 0.12, 0.08]),
        np.array([0, 0, 0.122]),
    )
    gen = PoseGenerator(
        center,
        half,
        np.diag([1.0, -1.0, -1.0]),
        max_tilt_deg=35,
        cube_center_in_tool=cube,
        seed=1,
    )
    for T in gen.generate(200):
        c = T[:3, :3] @ cube + T[:3, 3]
        assert np.all(np.abs(c - center) <= half + 1e-12)


def test_candidates_same_cube_center_priority_order():
    import pinocchio as pin

    cube = np.array([0, 0, 0.122])
    up, down = pin.rpy.rpyToMatrix(0, 0, 0), pin.rpy.rpyToMatrix(np.pi, 0, 0)
    gen = PoseGenerator(
        [0, 0, 0.2],
        [0.1, 0.1, 0.1],
        [up, down],
        max_tilt_deg=30,
        cube_center_in_tool=cube,
        seed=2,
    )
    for _ in range(50):
        T_up, T_down = gen.sample_candidates()
        assert np.allclose(
            T_up[:3, :3] @ cube + T_up[:3, 3], T_down[:3, :3] @ cube + T_down[:3, 3]
        )
        assert (
            T_up[2, 2] > 0.8 and T_down[2, 2] < -0.8
        )  # tool z up / down (tilt <= 30 deg)
        assert T_down[2, 3] > T_up[2, 3]  # flange above the cube when tool down
