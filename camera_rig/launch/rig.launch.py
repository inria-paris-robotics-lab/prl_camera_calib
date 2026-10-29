"""Camera rig of the calibration: capture service ~/capture, preview ~/preview.

ros2 launch camera_rig rig.launch.py                     # orbbec_rig: Femto Mega opened with pyorbbecsdk
ros2 launch camera_rig rig.launch.py rig:=realsense      # realsense_rig: topics of realsense2_camera
ros2 launch camera_rig rig.launch.py config:=/path/orbbec_rig.yaml board_config:=/path/calib_param.yml
ros2 run rqt_image_view rqt_image_view /orbbec_rig/preview
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rig = LaunchConfiguration("rig")
    node = PythonExpression(["'", rig, "' + '_rig'"])
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rig", default_value="orbbec", choices=["orbbec", "realsense"]
            ),
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("camera_rig"),
                        "config",
                        PythonExpression(["'", rig, "_rig.yaml'"]),
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "board_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("calib_orchestrator"),
                        "config",
                        "MC_Calib",
                        "calib_param_cube_femto.yml",
                    ]
                ),
            ),
            Node(
                package="camera_rig",
                executable=node,
                name=node,
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {"board_config": LaunchConfiguration("board_config")},
                ],
            ),
        ]
    )
