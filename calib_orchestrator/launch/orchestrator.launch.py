"""Calibration orchestrator.

  ros2 launch calib_orchestrator orchestrator.launch.py [config:=/path/orchestrator.yaml]

MoveIt (ur_moveit_config), the UR driver and camera_rig are launched separately. The calibrated cameras are
published by the robot URDF (fixed_cameras.yaml), not by this package.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("calib_orchestrator"),
                        "config",
                        "orchestrator.yaml",
                    ]
                ),
            ),
            Node(
                package="calib_orchestrator",
                executable="orchestrator",
                name="calib_orchestrator",
                parameters=[LaunchConfiguration("config")],
                output="screen",
            ),
        ]
    )
