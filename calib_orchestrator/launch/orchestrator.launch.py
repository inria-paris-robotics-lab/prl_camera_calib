"""Calibration orchestrator.

  ros2 launch calib_orchestrator orchestrator.launch.py [config:=/path/orchestrator.yaml]
      [mc_calib_config:=/path/calib_param.yml] [fixed_cameras_template:=/path/fixed_cameras.yaml]

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
            DeclareLaunchArgument(
                "mc_calib_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("calib_orchestrator"),
                        "config",
                        "MC_Calib",
                        "calib_param_cube_femto.yml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "fixed_cameras_template",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("prl_ur5_robot_configuration"),
                        "config",
                        "fixed_cameras",
                        "dataset_collection.yaml",
                    ]
                ),
            ),
            Node(
                package="calib_orchestrator",
                executable="orchestrator",
                name="calib_orchestrator",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "mc_calib_config": LaunchConfiguration("mc_calib_config"),
                        "fixed_cameras_template": LaunchConfiguration(
                            "fixed_cameras_template"
                        ),
                    },
                ],
            ),
        ]
    )
