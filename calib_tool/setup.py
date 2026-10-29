from setuptools import find_packages, setup
from glob import glob

package_name = "calib_tool"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            glob("config/*.yaml") + glob("config/*.pdf"),
        ),
        (
            "share/" + package_name + "/config/charuco_boards",
            glob("config/charuco_boards/*.bmp"),
        ),
        ("share/" + package_name + "/urdf", glob("urdf/*.xacro") + glob("urdf/*.urdf")),
        ("share/" + package_name + "/meshes", glob("meshes/*.stl")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Timothée Carecchio",
    maintainer_email="timothee.carecchio@inria.fr",
    description="Calibration tool on the UR flange: 5-face ChArUco cube (URDF/SRDF, mesh, printable boards)",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "make_charuco_pdf = calib_tool.make_charuco_pdf:main",
        ],
    },
)
