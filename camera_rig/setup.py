from glob import glob

from setuptools import find_packages, setup

package_name = "camera_rig"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Timothée Carecchio",
    maintainer_email="timothee.carecchio@inria.fr",
    description="Camera rigs for the calibration: Femto Mega (pyorbbecsdk) or RealSense (ROS topics) capture service, preview, RViz clouds",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "orbbec_rig = camera_rig.rig_node:main_orbbec",
            "realsense_rig = camera_rig.rig_node:main_realsense",
            "cloud_viz = camera_rig.cloud_viz:main",
        ],
    },
)
