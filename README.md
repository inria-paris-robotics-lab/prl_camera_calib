# prl_camera_calib

Robot-driven calibration of the fixed cameras of the mantis setup (3 Orbbec Femto Mega `golf`, `foxtrot`, `echo`, or
RealSense). The left arm holds a 5-face ChArUco cube and moves it in front of the cameras; at each pose a synchronized
image set is captured. Then:

1. MC-Calib calibrates the cameras together (intrinsics and camera-to-camera extrinsics);
2. a robot-world / hand-eye calibration places the camera rig in the robot frame (`prl_ur5_base`).

The result is written in the `prl_ur5_robot_configuration` format (`fixed_cameras.yaml`), so the URDF places the
cameras at their calibrated pose.

## Contents
1. [Packages](#packages)
2. [Principle](#principle)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Session outputs](#session-outputs)
7. [RViz visualization](#rviz-visualization)
8. [Troubleshooting](#troubleshooting)
9. [Known limits](#known-limits)

## Packages

| Package | Role |
|---|---|
| `inria_calib_interfaces` | `Capture.srv` service, `Calibrate.action` action |
| `camera_rig` | Capture nodes: `orbbec_rig` opens the Femto Mega with pyorbbecsdk (4K), `realsense_rig` reads the RealSense from the `realsense2_camera` topics. `cloud_viz`: light RViz point clouds from the Orbbec driver streams |
| `calib_orchestrator` | Generates the poses, drives MoveIt, triggers the captures, runs MC-Calib, the hand-eye and the evaluation, publishes the RViz markers |
| `calib_tool` | URDF/SRDF of the adapter and cube (`calib-cube` gripper type), mesh, printable ChArUco boards |

```
                    ┌──────────────────────────┐   action ~/calibrate
          user ───> │    calib_orchestrator    │
                    │ poses -> MoveIt -> capture│
                    │ -> MC-Calib -> hand-eye   │──> RViz ~/calib_markers
                    └──────┬────────────┬──────┘
          move_action /    │            │  service /orbbec_rig/capture
          compute_ik       ▼            ▼
                    ┌───────────┐  ┌──────────────┐  pyorbbecsdk   ┌──────────────┐
                    │ move_group│  │  orbbec_rig  │ <────────────> │ 3 Femto Mega │
                    └───────────┘  └──────────────┘   (Ethernet)   └──────────────┘
```

`orbbec_rig` opens the cameras itself instead of using the official driver: a decoded 4K image weighs ~25 MB, which
DDS cannot carry for 3 cameras. The node keeps the MJPG frames and decodes them in 4K only when capturing. The
official Orbbec driver must not run at the same time. Calibrating at 4K rather than with the driver's 1280x720 streams
gives about 3x better repeatability between sessions.

## Principle

### 1. Pose generation
For each pose, the orchestrator draws from a scrambled Sobol sequence:
- the position of the cube center in a workspace box (`workspace_center ± workspace_half_extent`);
- a tilt (≤ `max_tilt_deg`) and a rotation about the tool axis (≤ `max_yaw_deg`).

The tool orientations of `tool_rpy` are tried in priority order: tool up (top board facing the cameras), then tool
down (flange above the cube, to reach the cube centers close to the table). With `balance_orientations` each
orientation gets the same share of the poses: the hand-eye needs varied tool axes, otherwise the cube height along the
tool axis cannot be told apart from a vertical shift of the cameras. Each candidate is checked with MoveIt
`/compute_ik` (`avoid_collisions`). The poses are then ordered for short motions, and the IK is chained from one pose
to the next to avoid elbow or wrist flips.

### 2. Acquisition
For each pose: MoveIt plans and executes, the orchestrator waits `settle_s`, then `camera_rig` captures. Only frames
received after the request + `settle_s` and synchronized across cameras (spread < `sync_slop_s`) are used. A pose is
kept when at least `min_cameras` cameras see at least `min_corners` ChArUco corners. The flange pose is read from TF
at the image timestamp.

### 3. MC-Calib
MC-Calib reconstructs the cube (3D position of every corner of the 5 boards) and calibrates the cameras together:
intrinsics and pose of each camera relative to the reference camera (Cam_001 = golf).

### 4. Robot-world / hand-eye
For each pose *i*: **Bᵢ** = flange in the base (robot), **Cᵢ** = cube in the reference camera (PnP then multi-camera
refinement). With **X** = robot base seen from the cameras and **Z** = cube on the flange:
```
Cᵢ = X · Bᵢ · Z
```
- Initialization: `Cᵢ·Cⱼ⁻¹ · X = X · Bᵢ·Bⱼ⁻¹` for each pair of poses (Z cancels out), then Z by averaging.
- Refinement: X and Z jointly (Levenberg-Marquardt) over all poses.
- Outliers: poses with a residual > 3 × median are dropped and the fit is redone once.
- Result: `T_base->camera = X⁻¹ · T_ref->camera` for each camera.

A residual around 0.5 mm is typical; above 2 mm, check the cube mounting and that the robot is still at capture time.
With `hand_eye_mode: urdf_cube`, Z is taken from the URDF and only X is estimated.

## Installation

ROS 2 Jazzy, `prl_docker` mantis image (`robots/mantis/Dockerfile.mantis`): it installs the Python dependencies,
pyorbbecsdk2 and MC-Calib (`mc_calib_calibrate` on the PATH). The container runs with `--network host` (cameras
192.168.1.10/11/12 and robots).

Outside this image:
```bash
sudo apt install -y python3-numpy python3-scipy python3-yaml python3-opencv python3-pil \
    build-essential cmake libopencv-dev libopencv-contrib-dev libceres-dev libgoogle-glog-dev libeigen3-dev \
    libboost-log-dev libboost-system-dev libboost-test-dev
rosdep install --from-paths src/prl_camera_calib --ignore-src -y
# --no-deps: pip numpy / opencv would shadow the ones ROS is built against
pip install --no-deps --break-system-packages pyorbbecsdk2==2.1.2
git clone https://github.com/rameau-fr/MC-Calib.git && cd MC-Calib && git checkout 77cb043
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target calibrate -j$(nproc)
sudo ln -s $PWD/build/apps/calibrate/calibrate /usr/local/bin/mc_calib_calibrate
```

Workspace:
```bash
cd ~/share/mantis_ws
colcon build --symlink-install --packages-select inria_calib_interfaces calib_tool camera_rig calib_orchestrator \
    prl_ur5_description prl_ur5_moveit
source install/setup.bash
```

Tool: in `prl_ur5_robot_configuration/config/standard_setup.yaml`, left arm `gripper: "calib-cube"`. Boards to print
at 100 %: `calib_tool/config/charuco_boards_cube_14cm.pdf` (regenerate with
`ros2 run calib_tool make_charuco_pdf calib_tool/config/charuco_boards out.pdf`).

## Usage

In every terminal: `cd ~/share/mantis_ws && source install/setup.bash`.

### Simulation
```bash
ros2 launch prl_ur5_run sim.launch.py
ros2 launch calib_orchestrator orchestrator.launch.py
ros2 param set /calib_orchestrator motion_only true
ros2 action send_goal --feedback /calib_orchestrator/calibrate inria_calib_interfaces/action/Calibrate "{num_poses: 40}"
```
`dry_run: true` only plans. With `motion_only` the arm reaches the poses; the resulting `validated_poses.yml` can be
replayed on the real robot.

### Real robot
Set `payload_mass` (cube + adapter) in `orchestrator.yaml` before the first run.
```bash
ros2 launch prl_ur5_run real.launch.py activate_cameras:=false
ros2 launch camera_rig rig.launch.py                          # rig:=realsense for RealSense cameras
ros2 run rqt_image_view rqt_image_view /orbbec_rig/preview      # optional
ros2 launch calib_orchestrator orchestrator.launch.py

# short test: is the cube seen? (log "corners {'golf': .., 'foxtrot': .., 'echo': ..}")
ros2 action send_goal --feedback /calib_orchestrator/calibrate inria_calib_interfaces/action/Calibrate \
  "{num_poses: 5, run_calibration: false}"
# full calibration (~15 min)
ros2 action send_goal --feedback /calib_orchestrator/calibrate inria_calib_interfaces/action/Calibrate \
  "{num_poses: 100, run_calibration: true}"
```

### Other commands
```bash
# manual capture
ros2 service call /orbbec_rig/capture inria_calib_interfaces/srv/Capture \
  "{session_dir: /home/ros/share/calib_sessions/test, frame_name: '000000', depth_median_frames: 0, settle_s: 0.5}"

# replay the poses of a session
ros2 action send_goal --feedback /calib_orchestrator/calibrate inria_calib_interfaces/action/Calibrate \
  "{poses_file: /home/ros/share/calib_sessions/<session>/validated_poses.yml, run_calibration: true}"

# tune the workspace box live (visible in RViz)
ros2 param set /calib_orchestrator workspace_center "[0.0, 0.0, 0.20]"
ros2 param set /calib_orchestrator workspace_half_extent "[0.50, 0.50, 0.11]"
```

### `Calibrate` action
| Goal field | |
|---|---|
| `num_poses` | number of poses to generate (default 40) |
| `poses_file` | `validated_poses.yml` to replay instead of generating |
| `dry_run` | plan only |
| `run_calibration` | run MC-Calib, the hand-eye and the evaluation at the end |
| `session_dir` | session folder (empty = `<data_root>/session_<date>`) |

Result: `success`, `message`, `session_dir`, `poses_captured`, `mean_reprojection_px`, `hand_eye_residual_mm`,
`extrinsics_file`.

### Applying the result
Copy `<session>/fixed_cameras.yaml` to `prl_ur5_robot_configuration/config/fixed_cameras/dataset_collection.yaml`
(or set `fixed_cameras_out` to that path), then restart the robot. Only the `pose` entries are changed; the comments
and YAML anchors of the original file are not kept.

### Evaluating a session
After the hand-eye, the orchestrator writes a quality report to `<session>/evaluation.txt` and `evaluation.json`,
logs it and appends the overall status to the action result. A FAIL does not discard the calibration. Set
`evaluate_compare` to a previous session of the same rig to get the camera displacement.

Offline, on any session:
```bash
ros2 run calib_orchestrator evaluate_session ~/share/calib_sessions/<session> \
    --urdf-cube-height 0.127 --compare ~/share/calib_sessions/<previous_session>
```
Exit code 1 on FAIL, `--json report.json` to save the report.

| Check | What it detects |
|---|---|
| `reprojection` | MC-Calib error per camera (OK < 0.5 px), several camera groups |
| `poses` | too few poses, unbalanced tool up / down |
| `hand_eye` | robot / cameras disagreement (OK < 1.5 mm), outlier poses (WARN above 10 %) |
| `scale` | wrong `square_size`: the hand-eye is redone with the MC-Calib scene rescaled; a clear minimum away from 1 gives the real square size. A wrong size is invisible in the reprojection error |
| `cube_height` | cube position along the tool axis poorly observed (forcing it 5 mm off must raise the residual by ≥ 0.5 mm), difference with the URDF (OK < 2 mm) |
| `stability` | camera position spread over random subsets of the poses (OK < 1 mm) |
| `compare` | camera displacement with another session (OK < 2 mm) |

## Configuration

### `calib_orchestrator/config/orchestrator.yaml`
| Parameter | Role |
|---|---|
| `planning_group`, `base_frame`, `tool_frame` | `left_arm`, `prl_ur5_base` (table frame), `left_tool0` |
| `workspace_center`, `workspace_half_extent` | box of the cube center (m, `base_frame`); keep its bottom ~9 cm above the table |
| `tool_rpy` | tool orientations in priority order (roll pitch yaw triplets), default up then down |
| `balance_orientations` | split the poses evenly between the `tool_rpy` orientations |
| `max_tilt_deg`, `max_yaw_deg` | orientation variety |
| `cube_center_in_tool` | used only when TF `tool_frame` -> `cube_frame` is not available |
| `check_ik` | keep only poses with a collision-free IK |
| `settle_s` | wait after a motion before capturing |
| `min_corners`, `min_cameras` | criterion to keep a pose |
| `velocity_scaling`, `acceleration_scaling` | MoveIt scaling |
| `payload_mass`, `payload_cog` | payload of the arm carrying the cube (kg, m in the flange frame), set on the UR before each calibration; 0 = not set |
| `execution_retries` | retries after `CONTROL_FAILED` |
| `mc_calib_binary` | MC-Calib binary (`mc_calib_calibrate` on the PATH) |
| `camera_frames` | optical frames in MC-Calib order |
| `hand_eye_mode` | `estimate` or `urdf_cube` |
| `cube_board_axes` | outward axis of each board in the cube frame: URDF cross-check, required by `urdf_cube` |
| `fixed_cameras_out` | also copy the result (`fixed_cameras` format) there |
| `previous_extrinsics` | `cams_extrinsics.yml` of a previous calibration: keep only poses predicted visible by ≥ `min_cameras` |
| `evaluate`, `evaluate_compare` | quality report, optional session to compare with |

### Launch arguments
Files resolved in the installed packages, whatever the workspace location:

| Launch | Argument | Default |
|---|---|---|
| `orchestrator.launch.py` | `mc_calib_config` | `calib_orchestrator/config/MC_Calib/calib_param_cube_femto.yml` |
| `orchestrator.launch.py` | `fixed_cameras_template` | `prl_ur5_robot_configuration/config/fixed_cameras/dataset_collection.yaml` |
| `rig.launch.py` | `board_config` (ChArUco corner count) | same MC-Calib config |

### `camera_rig/config/orbbec_rig.yaml`
| Parameter | Role |
|---|---|
| `cameras_config` | cameras file of `prl_ur5_robot_configuration` (IP, port, `sync_mode`, fps): file name in `config/fixed_cameras` (`""` = `cameras_config.yaml`) or a path |
| `names` | MC-Calib order: `[golf_camera, foxtrot_camera, echo_camera]` -> Cam_001, Cam_002, Cam_003 |
| `color_width`, `color_height` | 3840 × 2160 |
| `fps` | 5 (0 = `color_fps` of `cameras_config`): one image per still pose is enough, and 4K at 15 fps loads the network shared with the arms |
| `sync_slop_s` | max timestamp spread between cameras (0.02 s) |

The primary camera (echo) is always started last. The sync mode stored in the cameras is only compared with the file.
Never open a secondary camera alone.

### `camera_rig/config/realsense_rig.yaml`
`realsense_rig` does not open the cameras: the robot bringup starts `realsense2_camera` for the cameras of
`cameras_config.yaml` with `type: realsense` and `activate: true`. The node reads
`<topic_prefix>/color/image_raw/compressed` and `color/camera_info` of each `names` entry, the factory intrinsics from
`camera_info` and the serial from `cameras_config.yaml`. Color only, no hardware sync (`sync_slop_s` 0.1 s).
`color_transport: raw` reads `color/image_raw`, whose large frames are often lost in best effort.

```bash
ros2 launch camera_rig rig.launch.py rig:=realsense
```
In `orchestrator.yaml`, set `capture_service: /realsense_rig/capture`, `camera_frames:
[alpha_camera_color_optical_frame, ...]` in the order of `names`, and a `fixed_cameras_template` listing these
`name_prefix` (`type: realsense-d435i`).

### MC-Calib (`calib_orchestrator/config/MC_Calib/calib_param_cube_femto.yml`)
`square_size` must match the printed boards: 27.72 mm for the current cube (nominal 28). Measure across 3 squares with
a caliper; the `scale` check of the evaluation detects a mismatch. `root_path`, `save_path` and `cam_params_path` are
rewritten for each session.

## Session outputs
`~/share/calib_sessions/session_<date>/`:

| File | Content | Written |
|---|---|---|
| `Cam_001/` … `Cam_003/` | images numbered for MC-Calib (001 = golf, 002 = foxtrot, 003 = echo) | each capture |
| `robot_poses.yml` | per pose: timestamp, joints, `T_base_tool`, corners per camera | each accepted pose |
| `validated_poses.yml` | joint configurations reached (replayable) | each accepted pose |
| `factory_intrinsics.yml`, `cameras.txt` | factory intrinsics (MC-Calib initialization), Cam ↔ device | first capture |
| `mc_calib.yml`, `mc_calib.log`, `calibrated_*.yml` | MC-Calib config and results | calibration |
| `cams_extrinsics.yml` | cameras in `base_frame`, `tool_to_cube`, residuals | hand-eye |
| `fixed_cameras.yaml` | same result in the `prl_ur5_robot_configuration` format | hand-eye |
| `evaluation.txt`, `evaluation.json` | quality report | evaluation |

## RViz visualization
Add the MarkerArray `/calib_orchestrator/calib_markers`, fixed frame `prl_ur5_base`:

| Namespace | |
|---|---|
| `workspace` | box of the cube centers (updated live with `ros2 param set`) |
| `fov` | field of view pyramid of each camera (`fov_depth`) |
| `cube` / `index` | one cube per pose: grey = target, blue = planned (dry run), green = captured, orange = rejected (cube not seen enough), red = failed |

Place the box where the pyramids of the 3 cameras intersect.

### Point clouds (`cloud_viz`)
The Orbbec driver's own clouds are too heavy for DDS / RViz with 3 cameras and are disabled in `cameras_config.yaml`.
`cloud_viz` rebuilds light colored clouds from the driver streams `depth/image_raw/compressedDepth` (lossless, registered
to color), `color/image_raw/compressed` and `color/camera_info`:

```bash
ros2 run camera_rig cloud_viz                              # ~/golf/points, ~/foxtrot/points, ~/echo/points
ros2 run camera_rig cloud_viz --ros-args -p fuse:=true     # ~/fused/points in prl_ur5_base
```

| Parameter | Default | |
|---|---|---|
| `stride`, `rate_hz` | 2, 2.0 | one pixel out of `stride` in each direction, publication rate |
| `max_depth` | 1.3 | m along the optical axis |
| `flying_rel`, `flying_win` | 0.02, 5 | flying pixels at depth edges (0 = off) |
| `grazing_deg`, `grazing_step` | 75, 2 | surfaces almost parallel to the view ray (0 = off) |
| `support_win`, `support_min` | 5, 12 | isolated points (0 = off) |
| `crop_min`, `crop_max` | table box | fused cloud kept inside this box of `base_frame` |
| `voxel` | 0.005 | fused cloud: points merged per voxel (0 = off) |

Display only; the recorded streams are untouched.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Black RViz window | `export QT_X11_NO_MITSHM=1` |
| `numpy/ndarrayobject.h: No such file` at build time | stale CMake cache: `rm -rf build/inria_calib_interfaces install/inria_calib_interfaces` and rebuild |
| `A module compiled using NumPy 1.x cannot run in NumPy 2` | pip numpy in `~/.local`: `pip uninstall numpy`, reinstall pyorbbecsdk2 with `--no-deps` |
| `package 'xxx' not found` at launch | leftover of a removed package: `rm -rf build/xxx install/xxx`, new terminal |
| Protective stop "position deviates from path", no collision | wrong payload on the UR: set `payload_mass` |
| `Connection to reverse interface dropped`, arm stops without protective stop | the UR did not receive the commands in time: network shared with the cameras, CPU throttling, no real-time priority. `sudo ethtool -C eth0 rx-usecs 0`, `--ulimit rtprio=99 --ulimit memlock=-1` on the container, ideally a dedicated network interface for the arms. Restart the program: `ros2 service call /left_io_and_status_controller/resend_robot_program std_srvs/srv/Trigger` |
| `CONTROL_FAILED` | execution refused (controller not ready): retried automatically |
| `Unable to solve the planning problem` | target hard to reach; joint limits should be ±180° |
| Few poses / many `no collision-free IK` | move the box closer to the arm, raise its bottom |
| Rejected poses (orange) | cube not seen by enough cameras: move the box toward the intersection of the fields of view |
| `realsense_rig`: `no .../color/camera_info` | RealSense driver not running: `activate_cameras:=true`, camera `activate: true` |
| Sync mode warning | set `sync_mode` in the cameras (OrbbecViewer) as in `cameras_config.yaml` |
| `no color set within sync_slop_s` | Femto not synced (primary not started last) or RealSense: raise `sync_slop_s` |
| Hand-eye residual > 2 mm | cube loose on the flange, `settle_s` too short, or a camera moved during the acquisition |

## Known limits
- The Femto depth (ToF) is a few mm off the table plane and slightly tilted per camera; the color calibration is not
  affected. The dark UR links are badly measured in depth.
- The RealSense path has no hardware synchronization and captures color only.
