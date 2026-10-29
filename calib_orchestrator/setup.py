from glob import glob

from setuptools import find_packages, setup

package_name = "calib_orchestrator"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config/MC_Calib", glob("config/MC_Calib/*.yml")),
        (
            "share/" + package_name + "/config/femto_mega",
            glob("config/femto_mega/*.yml"),
        ),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Timothée Carecchio",
    maintainer_email="timothee.carecchio@inria.fr",
    description="Robot-driven multi-camera calibration and robot-world/hand-eye",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "orchestrator = calib_orchestrator.orchestrator_node:main",
            "evaluate_session = calib_orchestrator.evaluate:main",
        ],
    },
)
