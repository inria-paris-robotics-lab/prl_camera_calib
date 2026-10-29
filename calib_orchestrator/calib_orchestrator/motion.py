"""Thin async client of MoveIt's move_group action (moveit_msgs/action/MoveGroup).

Joint goals are used to replay validated poses, pose goals (tool frame) for generated poses.
Prefer the Pilz PTP planner (pipeline 'pilz_industrial_motion_planner', planner 'PTP') for
predictable point-to-point motions next to the cameras.
"""

import numpy as np
import pinocchio as pin
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive


class MoveGroupClient:
    def __init__(
        self,
        node,
        group,
        base_frame,
        tool_frame,
        action_name="move_action",
        pipeline_id="ompl",
        planner_id="",
        velocity_scaling=0.1,
        acceleration_scaling=0.1,
        planning_time=5.0,
        callback_group=None,
    ):
        self.node, self.group, self.base_frame, self.tool_frame = (
            node,
            group,
            base_frame,
            tool_frame,
        )
        self.pipeline_id, self.planner_id = pipeline_id, planner_id
        self.vel, self.acc, self.planning_time = (
            velocity_scaling,
            acceleration_scaling,
            planning_time,
        )
        self.client = ActionClient(
            node, MoveGroup, action_name, callback_group=callback_group
        )
        self.last_error_code = None  # MoveItErrorCodes value of the last move()

    def wait(self, timeout_s=10.0):
        return self.client.wait_for_server(timeout_sec=timeout_s)

    def _request(self, constraints):
        req = MotionPlanRequest()
        req.group_name = self.group
        req.pipeline_id, req.planner_id = self.pipeline_id, self.planner_id
        req.num_planning_attempts = 5
        req.allowed_planning_time = self.planning_time
        req.max_velocity_scaling_factor = self.vel
        req.max_acceleration_scaling_factor = self.acc
        req.start_state.is_diff = True
        req.goal_constraints = [constraints]
        return req

    @staticmethod
    def joint_constraints(names, positions, tol=1e-3):
        c = Constraints()
        for n, p in zip(names, positions):
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=n,
                    position=float(p),
                    tolerance_above=tol,
                    tolerance_below=tol,
                    weight=1.0,
                )
            )
        return c

    def pose_constraints(self, T_base_tool, pos_tol=1e-3, rot_tol=1e-2):
        c = Constraints()
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(
            float, T_base_tool[:3, 3]
        )
        pc = PositionConstraint()
        pc.header.frame_id, pc.link_name, pc.weight = (
            self.base_frame,
            self.tool_frame,
            1.0,
        )
        pc.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[pos_tol])
        )
        pc.constraint_region.primitive_poses.append(pose)
        oc = OrientationConstraint()
        oc.header.frame_id, oc.link_name, oc.weight = (
            self.base_frame,
            self.tool_frame,
            1.0,
        )
        q = pin.Quaternion(np.asarray(T_base_tool, float)[:3, :3]).coeffs()
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = map(
            float, q
        )
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = (
            oc.absolute_z_axis_tolerance
        ) = rot_tol
        c.position_constraints.append(pc)
        c.orientation_constraints.append(oc)
        return c

    async def move(self, constraints, plan_only=False):
        """plan (+ execute). Returns (success, message)."""
        goal = MoveGroup.Goal()
        goal.request = self._request(constraints)
        goal.planning_options.plan_only = plan_only
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self.last_error_code = None
        handle = await self.client.send_goal_async(goal)
        if not handle.accepted:
            return False, "move_group rejected the goal"
        res = await handle.get_result_async()
        code = self.last_error_code = res.result.error_code.val
        if (
            res.status != GoalStatus.STATUS_SUCCEEDED
            or code != MoveItErrorCodes.SUCCESS
        ):
            return (
                False,
                f"move_group failed (status {res.status}, MoveIt error code {code})",
            )
        return True, "ok"
