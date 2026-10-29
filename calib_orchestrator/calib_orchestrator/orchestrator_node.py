"""Calibration orchestrator: action server `~/calibrate` (inria_calib_interfaces/action/Calibrate).

For each target pose: MoveIt plan + execute -> capture service (camera_rig) -> record the tool
pose (TF) and joints. Then MC-Calib on the session, cube pose per frame, robot-world / hand-eye
calibration, cams_extrinsics.yml (base_frame -> camera optical frames) and fixed_cameras.yaml
(same poses in the prl_ur5_robot_configuration fixed_cameras format, loaded by the URDF).

Threading: the action execute callback is a coroutine awaiting the MoveIt action and the
capture service, everything in a ReentrantCallbackGroup spun by a MultiThreadedExecutor
(a plain single-threaded spin deadlocks when a callback waits for another callback).
"""

import copy
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from ur_msgs.srv import SetPayload
from visualization_msgs.msg import Marker, MarkerArray

from inria_calib_interfaces.action import Calibrate
from inria_calib_interfaces.srv import Capture

from . import hand_eye, mc_calib
from .evaluate import DEFAULT_BOARD_AXES, evaluate
from .motion import MoveGroupClient
from .pose_generator import Camera, PoseGenerator, order_poses
from .session import (
    Session,
    load_joint_poses,
    read_extrinsics,
    write_extrinsics,
    write_fixed_cameras,
)


class CalibOrchestrator(Node):
    def __init__(self):
        super().__init__("calib_orchestrator")
        p = self.declare_parameter
        # robot / MoveIt
        p("planning_group", "ur_manipulator")
        p("base_frame", "base_link")
        p("tool_frame", "tool0")
        p(
            "joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )
        p("move_action", "move_action")
        p("pipeline_id", "ompl")
        p("planner_id", "")
        p("execution_retries", 3)  # after CONTROL_FAILED / TIMED_OUT
        p("execution_retry_delay_s", 2.0)
        p("velocity_scaling", 0.1)
        p("acceleration_scaling", 0.1)
        # payload set on the UR before each calibration (it is lost when the robot program restarts), <= 0: not set
        p("payload_service", "/left_io_and_status_controller/set_payload")
        p("payload_mass", 0.0)  # kg
        p("payload_cog", [0.0, 0.0, 0.12])  # m, flange frame
        # capture
        p("capture_service", "/orbbec_rig/capture")
        p("settle_s", 1.0)
        p("motion_only", False)  # reach the poses without capturing
        p("depth_median_frames", 0)
        p("min_corners", 6)
        p("min_cameras", 2)
        # pose generation
        p("workspace_center", [0.45, 0.0, 0.35])
        p("workspace_half_extent", [0.12, 0.12, 0.08])
        p(
            "tool_rpy", [0.0, 0.0, 0.0, 3.14159, 0.0, 0.0]
        )  # roll pitch yaw triplets, in priority order
        p(
            "balance_orientations", True
        )  # split the poses evenly between the tool_rpy orientations
        p("max_tilt_deg", 35.0)
        p("max_yaw_deg", 90.0)
        p(
            "cube_center_in_tool", [0.0, 0.0, 0.1]
        )  # used when TF tool_frame -> cube_frame is not available
        p("cube_size", 0.150)  # RViz markers (m)
        p(
            "fov_intrinsics", ""
        )  # FOV markers intrinsics, "" = factory 4K of this package
        p("fov_depth", 1.5)  # FOV pyramid length (m), 0 = hidden
        p(
            "previous_extrinsics", ""
        )  # cams_extrinsics.yml of a previous run: visibility filtering
        p("seed", 0)
        p(
            "check_ik", True
        )  # keep only poses with a collision-free IK (MoveIt /compute_ik)
        p("ik_service", "/compute_ik")
        p("ik_timeout_s", 0.05)
        p("max_samples_per_pose", 50)
        # MC-Calib / outputs
        p("data_root", os.path.expanduser("~/calib_sessions"))
        p("mc_calib_binary", "")
        p("mc_calib_config", "")  # template, root / save paths rewritten per session
        p("intrinsics_init", "")  # "" = <session>/factory_intrinsics.yml if present
        p(
            "camera_frames",
            [
                "cam_11_color_optical_frame",
                "cam_12_color_optical_frame",
                "cam_10_color_optical_frame",
            ],
        )  # MC-Calib order Cam_001, Cam_002, ...
        p("extrinsics_out", "")  # also copy cams_extrinsics.yml there
        p(
            "hand_eye_mode", "estimate"
        )  # estimate | urdf_cube (tool -> cube from the URDF)
        p("cube_frame", "")  # TF frame of the cube center
        p(
            "cube_board_axes", ""
        )  # outward axis of each board in cube_frame, e.g. "0:-x,1:+x,2:+z,3:-y,4:+y"
        # fixed_cameras yaml of the URDF, written with the poses of <name_prefix>_camera_calibration_pose
        p("fixed_cameras_template", "")  # "" = not written
        p("fixed_cameras_out", "")  # also copy fixed_cameras.yaml there
        p("evaluate", True)  # quality report <session>/evaluation.txt|json
        p("evaluate_compare", "")  # previous session of the same rig
        p("evaluate_bootstrap", 30)

        self.cb = ReentrantCallbackGroup()
        self.capture_cli = self.create_client(
            Capture, self.param("capture_service"), callback_group=self.cb
        )
        self.ik_cli = self.create_client(
            GetPositionIK, self.param("ik_service"), callback_group=self.cb
        )
        self.payload_cli = self.create_client(
            SetPayload, self.param("payload_service"), callback_group=self.cb
        )
        self.motion = MoveGroupClient(
            self,
            self.param("planning_group"),
            self.param("base_frame"),
            self.param("tool_frame"),
            action_name=self.param("move_action"),
            pipeline_id=self.param("pipeline_id"),
            planner_id=self.param("planner_id"),
            velocity_scaling=self.param("velocity_scaling"),
            acceleration_scaling=self.param("acceleration_scaling"),
            callback_group=self.cb,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.joint_state = None
        self.create_subscription(
            JointState, "/joint_states", self.on_joint_state, 10, callback_group=self.cb
        )
        self.busy = threading.Lock()
        # latched: RViz added later still gets the poses of the last goal
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "~/calib_markers",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.cube_states = {}  # target index -> (T_base_tool, status)
        self.marker_lock = (
            threading.Lock()
        )  # publish_cubes runs from the action and the parameter callbacks
        self.fov_models = self.load_fov_intrinsics()
        self.publish_cubes()  # workspace box and FOVs visible before the first goal
        self.add_post_set_parameters_callback(self.on_params_set)
        self.server = ActionServer(
            self,
            Calibrate,
            "~/calibrate",
            execute_callback=self.execute,
            goal_callback=self.on_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.cb,
        )
        self.get_logger().info("ready: action ~/calibrate")

    def param(self, name):
        return self.get_parameter(name).value

    def on_joint_state(self, msg):
        self.joint_state = msg

    def on_goal(self, _goal):
        # released at the end of execute()
        if not self.busy.acquire(blocking=False):
            self.get_logger().warn("calibration already running, goal rejected")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    # ------------------------------------------------------------------ helpers
    def current_joints(self):
        names = self.param("joint_names")
        js = self.joint_state
        if js is None:
            return names, None
        pos = dict(zip(js.name, js.position))
        return names, [pos[n] for n in names] if all(n in pos for n in names) else None

    def lookup_T(self, parent, child, stamp=None, timeout_s=0.5):
        """4x4 T_parent_child from TF at stamp (None: latest)"""
        tf = self.tf_buffer.lookup_transform(
            parent,
            child,
            Time.from_msg(stamp) if stamp else Time(),
            timeout=Duration(seconds=timeout_s),
        )
        t, q = tf.transform.translation, tf.transform.rotation
        T = np.eye(4)
        T[:3, :3] = pin.Quaternion(q.w, q.x, q.y, q.z).normalized().matrix()
        T[:3, 3] = (t.x, t.y, t.z)
        return T

    def tool_pose(self, stamp):
        """T_base_tool at the capture stamp (falls back to the latest TF: the robot is static)"""
        base, tool = self.param("base_frame"), self.param("tool_frame")
        try:
            return self.lookup_T(base, tool, stamp)
        except TransformException:
            return self.lookup_T(base, tool)

    # status -> RGBA: planned target, plan ok (dry run), reached / captured, rejected (cube not seen), failed
    MARKER_COLORS = {
        "target": (0.6, 0.6, 0.6, 0.15),
        "planned": (0.2, 0.4, 1.0, 0.35),
        "reached": (0.1, 0.8, 0.2, 0.5),
        "captured": (0.1, 0.8, 0.2, 0.5),
        "rejected": (1.0, 0.6, 0.0, 0.5),
        "failed": (1.0, 0.1, 0.1, 0.5),
    }

    def set_cube(self, i, status, T_base_tool=None):
        """Update the marker of target i (T_base_tool: None = keep the known one) and republish all cubes."""
        if T_base_tool is None:
            T_base_tool = self.cube_states.get(i, (None, None))[0]
        if T_base_tool is None:  # joint target never reached: position unknown
            return
        self.cube_states[i] = (np.asarray(T_base_tool), status)
        self.publish_cubes()

    def cube_center_in_tool(self):
        """cube center in the tool frame from TF (tool_frame -> cube_frame), else the cube_center_in_tool parameter"""
        if self.param("cube_frame"):
            try:
                return self.lookup_T(
                    self.param("tool_frame"), self.param("cube_frame"), timeout_s=0.2
                )[:3, 3]
            except TransformException:
                pass
        return np.asarray(self.param("cube_center_in_tool"), float)

    def current_tool_pose(self):
        try:
            return self.tool_pose(self.get_clock().now().to_msg())
        except TransformException:
            return None

    def on_params_set(self, params):
        if any(
            p.name
            in (
                "workspace_center",
                "workspace_half_extent",
                "cube_size",
                "cube_center_in_tool",
                "base_frame",
                "fov_depth",
                "camera_frames",
            )
            for p in params
        ):
            self.publish_cubes()

    def load_fov_intrinsics(self):
        """[(K, width, height)] in MC-Calib order, [] if the file is missing"""
        path = self.param("fov_intrinsics")
        if not path:
            from ament_index_python.packages import get_package_share_directory

            path = os.path.join(
                get_package_share_directory("calib_orchestrator"),
                "config",
                "femto_mega",
                "factory_intrinsics_4k.yml",
            )
        fs = cv2.FileStorage(os.path.expanduser(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            self.get_logger().warn(f"no camera FOV markers: cannot read {path}")
            return []
        out = []
        for i in range(int(fs.getNode("nb_camera").real())):
            n = fs.getNode(f"camera_{i}")
            out.append(
                (
                    n.getNode("camera_matrix").mat(),
                    int(n.getNode("img_width").real()),
                    int(n.getNode("img_height").real()),
                )
            )
        return out

    # one color per camera (Cam_001, Cam_002, Cam_003, ...)
    FOV_COLORS = [(0.2, 0.8, 1.0), (1.0, 0.4, 0.8), (0.6, 1.0, 0.3), (1.0, 0.7, 0.2)]

    def fov_markers(self, stamp):
        """field of view pyramid of each camera, in its optical frame"""
        depth, markers = float(self.param("fov_depth")), []
        if depth <= 0:
            return markers
        for i, (frame, (K, w, h)) in enumerate(
            zip(self.param("camera_frames"), self.fov_models)
        ):
            # image corners back-projected at fov_depth, without distortion
            corners = [
                Point(
                    x=float((u - K[0, 2]) / K[0, 0] * depth),
                    y=float((v - K[1, 2]) / K[1, 1] * depth),
                    z=depth,
                )
                for u, v in ((0, 0), (w, 0), (w, h), (0, h))
            ]
            apex = Point()
            r, g, b = self.FOV_COLORS[i % len(self.FOV_COLORS)]
            edges = Marker(ns="fov", id=3 * i, type=Marker.LINE_LIST, action=Marker.ADD)
            sides = Marker(
                ns="fov", id=3 * i + 1, type=Marker.TRIANGLE_LIST, action=Marker.ADD
            )
            label = Marker(
                ns="fov", id=3 * i + 2, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD
            )
            for k in range(4):
                c, c_next = corners[k], corners[(k + 1) % 4]
                edges.points += [apex, c, c, c_next]
                sides.points += [apex, c, c_next]
            label.pose.position.z = depth
            label.text = frame.removesuffix("_color_optical_frame").removesuffix(
                "_camera"
            )
            for m, alpha in ((edges, 0.9), (sides, 0.06), (label, 1.0)):
                m.header.frame_id, m.header.stamp = frame, stamp
                m.pose.orientation.w = 1.0
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, alpha
            edges.scale.x = 0.004
            sides.scale.x = sides.scale.y = sides.scale.z = 1.0
            label.scale.z = 0.05
            markers += [edges, sides, label]
        return markers

    def workspace_markers(self, stamp):
        """transparent box + edges: sampling box of the cube center (workspace_center +- workspace_half_extent)"""
        center, half = (
            self.param("workspace_center"),
            np.asarray(self.param("workspace_half_extent"), float),
        )
        box = Marker(ns="workspace", id=0, type=Marker.CUBE, action=Marker.ADD)
        edges = Marker(ns="workspace", id=1, type=Marker.LINE_LIST, action=Marker.ADD)
        label = Marker(
            ns="workspace", id=2, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD
        )
        for m in (box, edges, label):
            m.header.frame_id, m.header.stamp = self.param("base_frame"), stamp
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, center)
            m.pose.orientation.w = 1.0
        box.scale.x, box.scale.y, box.scale.z = map(float, 2 * half)
        box.color.r, box.color.g, box.color.b, box.color.a = 0.9, 0.9, 0.2, 0.08
        corners = [
            np.array([sx, sy, sz]) * half
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
        for a in range(8):
            for b in range(a + 1, 8):
                if np.count_nonzero(corners[a] != corners[b]) == 1:  # the 12 edges
                    for c in (corners[a], corners[b]):
                        edges.points.append(
                            Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
                        )
        edges.scale.x = 0.003
        edges.color.r, edges.color.g, edges.color.b, edges.color.a = 0.9, 0.9, 0.2, 0.8
        label.pose.position.z += float(half[2]) + 0.03
        label.scale.z = 0.03
        label.text = "workspace"
        label.color.r, label.color.g, label.color.b, label.color.a = 0.9, 0.9, 0.2, 1.0
        return [box, edges, label]

    def publish_cubes(self):
        with self.marker_lock:
            self._publish_cubes()

    def _publish_cubes(self):
        """~/calib_markers: one cube per target colored by status, the workspace box and the camera FOVs"""
        stamp = self.get_clock().now().to_msg()
        clear = Marker(action=Marker.DELETEALL)
        clear.header.frame_id = self.param("base_frame")
        # FOVs stamped 0: RViz uses the latest (static) camera TF
        arr = MarkerArray(
            markers=[clear]
            + self.workspace_markers(stamp)
            + self.fov_markers(TimeMsg())
        )
        size, center = (
            self.param("cube_size"),
            np.append(self.cube_center_in_tool(), 1.0),
        )
        for i, (T, status) in sorted(self.cube_states.items()):
            pos = (T @ center)[:3]
            q = pin.Quaternion(T[:3, :3]).coeffs()
            for ns, kind in (("cube", Marker.CUBE), ("index", Marker.TEXT_VIEW_FACING)):
                m = Marker(ns=ns, id=i, type=kind, action=Marker.ADD)
                m.header.frame_id, m.header.stamp = self.param("base_frame"), stamp
                m.pose.position.x, m.pose.position.y, m.pose.position.z = map(
                    float, pos
                )
                if kind == Marker.CUBE:
                    (
                        m.pose.orientation.x,
                        m.pose.orientation.y,
                        m.pose.orientation.z,
                        m.pose.orientation.w,
                    ) = map(float, q)
                    m.scale.x = m.scale.y = m.scale.z = float(size)
                    m.color.r, m.color.g, m.color.b, m.color.a = self.MARKER_COLORS[
                        status
                    ]
                else:
                    m.pose.orientation.w = 1.0
                    m.pose.position.z += 0.6 * size
                    m.scale.z = 0.04
                    m.text = str(i + 1)
                    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                arr.markers.append(m)
        self.marker_pub.publish(arr)

    @staticmethod
    def corners_by_camera(resp):
        """{'golf': 42, 'foxtrot': 0, ...} from a Capture response"""
        return dict(zip(resp.camera_names, resp.charuco_corners))

    async def solve_ik(self, T_base_tool, seed=None):
        """collision-free IK joint solution (joint_names order), or (None, error).
        seed: joint values (joint_names order) the solver starts from, default the current state"""
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = (
            self.param("planning_group"),
            self.param("tool_frame"),
            True,
        )
        timeout = float(self.param("ik_timeout_s"))
        r.timeout = DurationMsg(sec=int(timeout), nanosec=int((timeout % 1) * 1e9))
        if self.joint_state is not None:
            r.robot_state.joint_state = copy.deepcopy(self.joint_state)
        if seed is not None:
            js = r.robot_state.joint_state
            values = dict(zip(js.name, js.position))
            values.update(zip(self.param("joint_names"), map(float, seed)))
            js.name, js.position = list(values), list(values.values())
            js.velocity, js.effort = [], []
        r.pose_stamped.header.frame_id = self.param("base_frame")
        pose = r.pose_stamped.pose
        pose.position.x, pose.position.y, pose.position.z = map(
            float, T_base_tool[:3, 3]
        )
        q = pin.Quaternion(T_base_tool[:3, :3]).coeffs()
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = map(float, q)
        resp = await self.ik_cli.call_async(req)
        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, resp.error_code.val
        pos = dict(
            zip(resp.solution.joint_state.name, resp.solution.joint_state.position)
        )
        return [pos[n] for n in self.param("joint_names")], MoveItErrorCodes.SUCCESS

    async def make_targets(self, goal):
        """[(kind, target, T_base_tool or None)]: ('joints', [..], T) | ('pose', T_base_tool, T)"""
        if goal.poses_file:
            names, poses = load_joint_poses(goal.poses_file)
            if names and names != self.param("joint_names"):
                raise ValueError(
                    f"{goal.poses_file}: joint names {names} != parameter joint_names"
                )
            return [("joints", q, None) for q in poses]
        cameras = []
        if self.param("previous_extrinsics"):
            _, entries = read_extrinsics(self.param("previous_extrinsics"))
            cameras = [
                Camera(name, T, K, *size) for name, T, K, size in entries if size[0] > 0
            ]
        rpys = list(self.param("tool_rpy"))
        if not rpys or len(rpys) % 3:
            raise ValueError(f"tool_rpy must be roll, pitch, yaw triplets, got {rpys}")
        rpys = [rpys[k : k + 3] for k in range(0, len(rpys), 3)]
        gen = PoseGenerator(
            self.param("workspace_center"),
            self.param("workspace_half_extent"),
            [pin.rpy.rpyToMatrix(*rpy) for rpy in rpys],
            max_tilt_deg=self.param("max_tilt_deg"),
            max_yaw_deg=self.param("max_yaw_deg"),
            cube_center_in_tool=self.cube_center_in_tool(),
            cameras=cameras,
            min_cameras=self.param("min_cameras"),
            seed=self.param("seed"),
        )
        n = goal.num_poses or 40
        if not self.param("check_ik"):
            return [("pose", T, T) for T in gen.generate(n)]
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"IK service {self.param('ik_service')} not available (move_group running?)"
            )
        balanced = self.param("balance_orientations")
        quota = (
            [n // len(rpys) + (k < n % len(rpys)) for k in range(len(rpys))]
            if balanced
            else [n] * len(rpys)
        )
        found, failures, used, tries = [], {}, [0] * len(rpys), 0
        while len(found) < n and tries < n * self.param("max_samples_per_pose"):
            tries += 1
            reasons = []
            for k, T in enumerate(
                gen.sample_candidates()
            ):  # same cube center, orientations by priority
                if used[k] >= quota[k]:
                    continue
                if gen.cameras and len(gen.visible_cameras(T)) < gen.min_cameras:
                    reasons.append("not visible")
                    continue
                joints, code = await self.solve_ik(T)
                if joints is None:  # unreachable or in collision (avoid_collisions)
                    reasons.append(
                        "no collision-free IK"
                        if code == MoveItErrorCodes.NO_IK_SOLUTION
                        else f"IK error {code}"
                    )
                    continue
                found.append((T, joints, k))
                used[k] += 1
                break
            else:
                key = "/".join(sorted(set(reasons)))  # every orientation failed
                failures[key] = failures.get(key, 0) + 1
        per_rpy = {
            f"rpy {rpy}": f"{c}/{q}" if balanced else c
            for rpy, c, q in zip(rpys, used, quota)
        }
        self.get_logger().info(
            f"{len(found)}/{n} collision-free poses from {tries} cube centers, "
            f"orientations used {per_rpy}, rejected {failures}"
        )
        if not found:
            raise RuntimeError(
                f"no reachable collision-free pose in the workspace ({failures})"
            )
        if balanced and any(u < q for u, q in zip(used, quota)):
            self.get_logger().warn(
                f"orientation quotas not reached {per_rpy}: move/enlarge the workspace box or "
                "raise max_samples_per_pose"
            )
        # orientation by orientation (greedy order inside each group): a single wrist flip
        order, cur = [], None
        for k in range(len(rpys)) if balanced else [None]:
            group = [i for i, f in enumerate(found) if k is None or f[2] == k]
            if group:
                sub = order_poses(
                    [found[i][0] for i in group], start=cur, return_indices=True
                )
                order += [group[j] for j in sub]
                cur = found[order[-1]][0]
        # IK chained along the sequence: consecutive targets stay in the same configuration (no elbow / wrist flip)
        targets, prev = [], self.current_joints()[1]
        for k in order:
            T, joints, _ = found[k]
            if prev is not None:
                chained, _ = await self.solve_ik(T, seed=prev)
                if chained is not None:
                    joints = chained
            targets.append(("joints", joints, T))
            prev = joints
        jumps = [
            float(np.degrees(np.max(np.abs(np.subtract(b[1], a[1])))))
            for a, b in zip(targets, targets[1:])
        ]
        if jumps:
            self.get_logger().info(
                f"largest joint move between consecutive targets: median {np.median(jumps):.0f} "
                f"deg, max {max(jumps):.0f} deg"
            )
        return targets

    # ------------------------------------------------------------------ action
    async def execute(self, gh):
        goal, result = gh.request, Calibrate.Result()
        result.mean_reprojection_px = result.hand_eye_residual_mm = -1.0
        try:
            return await self.run(gh, goal, result)
        except Exception as e:  # noqa: BLE001 - report any failure through the action result
            self.get_logger().error(f"calibration failed: {e}")
            result.success, result.message = False, str(e)
            gh.abort()
            return result
        finally:
            self.busy.release()  # taken in on_goal

    async def run(self, gh, goal, result):
        log = self.get_logger()
        session = Session(
            os.path.expanduser(self.param("data_root")),
            os.path.expanduser(goal.session_dir),
        )
        result.session_dir = str(session.dir)
        targets = await self.make_targets(goal)
        self.cube_states = {}
        for i, (_, _, T) in enumerate(targets):
            if T is not None:
                self.cube_states[i] = (np.asarray(T), "target")
        self.publish_cubes()
        motion_only = self.param("motion_only") and not goal.dry_run
        if (
            not goal.dry_run
            and not motion_only
            and not self.capture_cli.wait_for_service(timeout_sec=5.0)
        ):
            raise RuntimeError(
                f"capture service {self.param('capture_service')} not available"
            )
        if not self.motion.wait():
            raise RuntimeError("move_group action server not available")
        if not goal.dry_run:
            await self.set_payload()

        fb = Calibrate.Feedback()
        fb.num_poses = len(targets)

        def feedback(stage, i):
            fb.stage, fb.pose_index, fb.poses_captured = stage, i, len(session.frames)
            gh.publish_feedback(fb)

        joint_names = self.param("joint_names")
        planned = []
        for i, (kind, target, _) in enumerate(targets):
            if gh.is_cancel_requested:
                session.save()
                gh.canceled()
                result.message = "canceled"
                result.poses_captured = len(session.frames)
                return result
            feedback("planning" if goal.dry_run else "moving", i)
            constraints = (
                self.motion.joint_constraints(joint_names, target)
                if kind == "joints"
                else self.motion.pose_constraints(target)
            )
            ok, msg = await self.motion.move(constraints, plan_only=goal.dry_run)
            for retry in range(self.param("execution_retries")):
                if ok or self.motion.last_error_code not in (
                    MoveItErrorCodes.CONTROL_FAILED,
                    MoveItErrorCodes.TIMED_OUT,
                ):
                    break
                log.warn(
                    f"pose {i + 1}/{len(targets)}: execution failed ({msg}), "
                    f"retry {retry + 1}/{self.param('execution_retries')}"
                )
                time.sleep(float(self.param("execution_retry_delay_s")))
                ok, msg = await self.motion.move(constraints, plan_only=goal.dry_run)
            if not ok:
                current = self.current_joints()[1]
                log.warn(
                    f"pose {i + 1}/{len(targets)} skipped: {msg}\n"
                    f"  from (deg) {np.degrees(current).round(1).tolist() if current is not None else '?'}\n"
                    f"  to   (deg) {np.degrees(target).round(1).tolist() if kind == 'joints' else 'pose goal'}"
                )
                self.set_cube(i, "failed")
                continue
            if goal.dry_run:
                self.set_cube(i, "planned")
                planned.append(i + 1)
                log.info(f"pose {i + 1}/{len(targets)}: plan ok")
                continue

            if motion_only:
                time.sleep(float(self.param("settle_s")))
                _, joints = self.current_joints()
                now = self.get_clock().now()
                T_tool = self.tool_pose(now.to_msg())
                session.add_frame(
                    now.nanoseconds * 1e-9,
                    joint_names,
                    joints if joints is not None else [],
                    T_tool,
                    [],
                )
                self.set_cube(i, "reached", T_tool)
                log.info(f"pose {i + 1}/{len(targets)}: reached (motion only)")
                feedback("reached", i)
                continue

            feedback("capturing", i)
            req = Capture.Request(
                session_dir=str(session.dir),
                frame_name=session.frame_name(),
                depth_median_frames=self.param("depth_median_frames"),
                settle_s=float(self.param("settle_s")),
            )
            resp = await self.capture_cli.call_async(req)
            fb.charuco_corners = list(resp.charuco_corners)
            n_seen = sum(c >= self.param("min_corners") for c in resp.charuco_corners)
            if not resp.success or n_seen < self.param("min_cameras"):
                # not kept: delete the files, the next capture reuses the same frame name
                for path in resp.image_paths:
                    Path(path).unlink(missing_ok=True)
                log.warn(
                    f"pose {i + 1}/{len(targets)} rejected: {resp.message or ''} "
                    f"(cube seen by {n_seen} cameras, corners {self.corners_by_camera(resp)})"
                )
                self.set_cube(i, "rejected", self.current_tool_pose())
                continue
            _, joints = self.current_joints()
            T_tool = self.tool_pose(resp.stamp)
            session.add_frame(
                Time.from_msg(resp.stamp).nanoseconds * 1e-9,
                joint_names,
                joints if joints is not None else [],
                T_tool,
                resp.charuco_corners,
            )
            self.set_cube(i, "captured", T_tool)
            log.info(
                f"pose {i + 1}/{len(targets)}: frame {len(session.frames) - 1} captured, "
                f"corners {self.corners_by_camera(resp)}"
            )
            feedback("captured", i)

        result.poses_captured = len(session.frames)
        if goal.dry_run or motion_only or not goal.run_calibration:
            result.success = True
            result.message = (
                f"dry run: {len(planned)}/{len(targets)} poses planned (from the current state), "
                f"failed: {sorted(set(range(1, len(targets) + 1)) - set(planned))}"
                if goal.dry_run
                else f"motion only: {len(session.frames)}/{len(targets)} poses reached"
                if motion_only
                else "capture done (calibration not requested)"
            )
            gh.succeed()
            return result

        feedback("calibrating", len(targets))
        log.info(
            f"MC-Calib running on {len(session.frames)} poses (~1 min, log: {session.dir / 'mc_calib.log'})"
        )
        self.calibrate(session, result)
        feedback("hand_eye", len(targets))
        self.hand_eye(session, result)
        evaluation = ""
        if self.param("evaluate"):
            feedback("evaluating", len(targets))
            log.info("evaluation running (~1 min)")
            evaluation = self.evaluate_session(session)
        result.success = True
        result.message = (
            f"{result.poses_captured} poses, MC-Calib {result.mean_reprojection_px:.2f} px, "
            f"hand-eye {result.hand_eye_residual_mm:.2f} mm{evaluation}"
        )
        log.info(result.message)
        gh.succeed()
        return result

    async def set_payload(self):
        """payload_mass / payload_cog on the UR controller; raises if it is not applied"""
        mass, cog = (
            float(self.param("payload_mass")),
            [float(v) for v in self.param("payload_cog")],
        )
        if mass <= 0:
            self.get_logger().warn(
                "payload_mass not set: the UR keeps its current payload (check it on the pendant)"
            )
            return
        if not self.payload_cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"payload service {self.param('payload_service')} not available"
            )
        req = SetPayload.Request(mass=mass)
        req.center_of_gravity.x, req.center_of_gravity.y, req.center_of_gravity.z = cog
        if not (await self.payload_cli.call_async(req)).success:
            raise RuntimeError(
                f"set_payload {mass} kg {cog} refused (External Control program running?)"
            )
        self.get_logger().info(f"payload set: {mass} kg, center of gravity {cog} m")

    def evaluate_session(self, session):
        """quality report of the calibrated session -> <session>/evaluation.txt|json, summary for the result"""
        lines, log = [], self.get_logger()
        try:
            report = evaluate(
                session.dir,
                compare=os.path.expanduser(self.param("evaluate_compare")) or None,
                board_axes=self.param("cube_board_axes") or DEFAULT_BOARD_AXES,
                urdf_cube_height=float(self.cube_center_in_tool()[2]),
                bootstrap=self.param("evaluate_bootstrap"),
                log=lines.append,
            )
        except Exception as e:  # noqa: BLE001 - the calibration itself is kept, only its report is missing
            log.warn(f"evaluation failed: {e}")
            return ", evaluation failed"
        (session.dir / "evaluation.txt").write_text("\n".join(lines) + "\n")
        (session.dir / "evaluation.json").write_text(
            json.dumps(report, indent=2, default=float)
        )
        for (
            line
        ) in lines:  # one call per severity: rclpy forbids changing it on the same line
            if line.startswith("[FAIL]"):
                log.error(line)
            elif line.startswith("[WARN]"):
                log.warn(line)
            else:
                log.info(line)
        bad = [k for k, c in report["checks"].items() if c["status"] != "OK"]
        return f", evaluation {report['status']}" + (
            f" ({', '.join(bad)})" if bad else ""
        )

    def calibrate(self, session, result):
        """MC-Calib on the session (blocking: runs in this executor thread)."""
        binary, template = (
            os.path.expanduser(self.param(n))
            for n in ("mc_calib_binary", "mc_calib_config")
        )
        if not binary or not template:
            raise RuntimeError(
                "parameters mc_calib_binary and mc_calib_config are required"
            )
        init = self.param("intrinsics_init")
        if not init:
            factory = session.dir / "factory_intrinsics.yml"
            init = str(factory) if factory.exists() else "None"
        config = mc_calib.write_config(
            template, session.dir / "mc_calib.yml", str(session.dir), init
        )
        mc_calib.run(binary, config, session.dir / "mc_calib.log")
        result.mean_reprojection_px = mc_calib.mean_reprojection_px(session.dir)

    def hand_eye(self, session, result):
        calib = hand_eye.load_mccalib(session.dir)
        if len({c["group"] for c in calib["cameras"]}) != 1:
            raise RuntimeError(
                "MC-Calib ended with several camera groups: not enough shared views"
            )
        cubes, robot = hand_eye.cube_poses(calib), session.robot_poses()
        frames = sorted(set(cubes) & set(robot))
        C, B = [cubes[f][0] for f in frames], [robot[f] for f in frames]
        mode, log = self.param("hand_eye_mode"), self.get_logger()
        if mode == "urdf_cube":
            T_known = self.urdf_tool_to_cube(calib)

            def solve(C, B):
                X, res = hand_eye.robot_world_known_tool(C, B, T_known)
                return X, T_known, res
        elif mode == "estimate":
            solve = hand_eye.robot_world_hand_eye
        else:
            raise RuntimeError(
                f"hand_eye_mode {mode!r}: expected 'estimate' or 'urdf_cube'"
            )
        T_ref_base, T_tool_cube, res, kept = hand_eye.fit_robust(solve, C, B)
        if not kept.all():
            log.warn(
                f"hand-eye: outlier frames {[f for f, k in zip(frames, kept) if not k]} dropped "
                "(residual > 3 x median: bad detection or cube moved)"
            )
            frames = [f for f, k in zip(frames, kept) if k]
        if (
            mode == "estimate"
            and self.param("cube_board_axes")
            and self.param("cube_frame")
        ):
            try:  # cross-check with the URDF
                d = hand_eye.inv_T(self.urdf_tool_to_cube(calib)) @ T_tool_cube
                log.info(
                    f"tool -> cube estimated vs URDF: {np.linalg.norm(d[:3, 3]) / hand_eye.MM:.1f} mm, "
                    f"{hand_eye.rot_angle_deg(d[:3, :3]):.2f} deg"
                )
            except (TransformException, ValueError) as e:
                log.warn(f"no URDF cross-check of tool -> cube: {e}")
        result.hand_eye_residual_mm = float(res[:, 0].mean())
        log.info(
            f"hand-eye ({mode}) on {len(frames)} frames: {res[:, 0].mean():.2f} mm "
            f"(max {res[:, 0].max():.2f}), {res[:, 1].mean():.3f} deg"
        )
        cams = calib["cameras"]
        frames_names = self.param("camera_frames")
        if len(frames_names) != len(cams):
            raise RuntimeError(
                f"camera_frames has {len(frames_names)} names for {len(cams)} cameras"
            )
        out = session.dir / "cams_extrinsics.yml"
        T_base_cams = hand_eye.base_to_cameras(T_ref_base, calib)
        write_extrinsics(
            out,
            self.param("base_frame"),
            frames_names,
            T_base_cams,
            cams,
            T_tool_cube,
            res,
        )
        if self.param("extrinsics_out"):
            Path(os.path.expanduser(self.param("extrinsics_out"))).write_text(
                out.read_text()
            )
        result.extrinsics_file = str(out)
        if self.param("fixed_cameras_template"):
            self.write_fixed_cameras(session, frames_names, T_base_cams)

    def urdf_tool_to_cube(self, calib):
        """T_tool_object: URDF tool -> cube center (TF) composed with the cube-center frame of the MC-Calib object"""
        if not self.param("cube_frame") or not self.param("cube_board_axes"):
            raise ValueError("parameters cube_frame and cube_board_axes are required")
        T_center_obj, side, err = hand_eye.cube_center_frame(
            calib["object"], hand_eye.parse_board_axes(self.param("cube_board_axes"))
        )
        self.get_logger().info(
            f"MC-Calib cube: side {side / hand_eye.MM:.1f} mm between opposite boards, "
            f"board normals within {err:.2f} deg of cube_board_axes"
        )
        if err > 5.0:
            raise ValueError(
                f"cube_board_axes does not match the MC-Calib cube (normal {err:.0f} deg off its axis)"
            )
        return (
            self.lookup_T(
                self.param("tool_frame"), self.param("cube_frame"), timeout_s=1.0
            )
            @ T_center_obj
        )

    def write_fixed_cameras(self, session, frames_names, T_base_cams):
        """fixed_cameras.yaml: pose of <name_prefix>_camera_calibration_pose such that the URDF chain down to
        the color optical frame lands on the calibrated optical frame (chain read from TF, ~identity for the
        femto-mega). The poses are expressed in base_frame: it must be the parent of the fixed cameras."""
        suffix = "_camera_color_optical_frame"
        poses = {}
        for frame, T_base_opt in zip(frames_names, T_base_cams):
            if not frame.endswith(suffix):
                raise RuntimeError(f"camera frame {frame} is not <name_prefix>{suffix}")
            prefix = frame[: -len(suffix)]
            mount = f"{prefix}_camera_calibration_pose"
            try:
                T_mount_opt = self.lookup_T(mount, frame, timeout_s=1.0)
            except TransformException as e:
                self.get_logger().warn(
                    f"no TF {mount} -> {frame} ({e}): optical frame used as mount pose"
                )
                T_mount_opt = np.eye(4)
            poses[prefix] = T_base_opt @ hand_eye.inv_T(T_mount_opt)
        out = session.dir / "fixed_cameras.yaml"
        write_fixed_cameras(
            os.path.expanduser(self.param("fixed_cameras_template")), out, poses
        )
        if self.param("fixed_cameras_out"):
            Path(os.path.expanduser(self.param("fixed_cameras_out"))).write_text(
                out.read_text()
            )
        self.get_logger().info(
            f"fixed cameras poses ({self.param('base_frame')}) written to {out}"
        )


def main():
    rclpy.init()
    node = CalibOrchestrator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
