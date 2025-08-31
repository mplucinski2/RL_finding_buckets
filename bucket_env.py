import math
import random
from typing import Dict, List, Optional, Tuple, Set
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import os
import airsim
import time

cv2.setUseOptimized(True)
cv2.setNumThreads(1)


def _is_valid_pose(pose: airsim.Pose) -> bool:
    x = pose.position.x_val
    y = pose.position.y_val
    z = pose.position.z_val
    return not (math.isnan(x) or math.isnan(y) or math.isnan(z))

class BucketSearchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        #input image size
        image_size_hw: Tuple[int, int] = (84, 84),
        camera_name: str = "0",
        #pitch angle of the camera in the body frame, pitch down was tested initially for constant-altitude flight but in the new setup forward camera is enough due to usually lower flight altitude
        camera_pitch_deg: float = 0.0,
        #arbitrary realistic field of view for a drone
        camera_fov_deg: float = 85.0,
        #camera offset in the body x axis, so that the drone frame is not in view actually
        camera_offset_x: float = 0.3,
        #initial altitude of the vehicle after takeoff
        cruise_altitude_z_m: float = -7.0,
        #maximum requested speed of the vehicle in the body xy frame
        speed_mps: float = 8.0,
        #duration of a single action in seconds
        action_duration_s: float = 0.25,
        #maximum numer of steps before termination
        max_steps: int = 2000,
        #penalty for each step to encourage faster task completion
        step_penalty: float = 0.01,
        #penalty for colliding with obstacles or ground 
        collision_penalty: float = 200.0,
        #reward for finding all five buckets in an episode
        success_reward: float = 1000.0,
        #vehicle name to use, if not specified default used
        vehicle_name: Optional[str] = None,
        #after how many steps should the segmentation frame be fetched and processed for reward, can potentially increase fps 
        seg_reward_every_n: int = 1,
        #printing collisions detected via airsim api to check
        print_collisions: bool = False,
        #interpolation method for resizing the image
        resize_interpolation: int = cv2.INTER_AREA,
        #reward for finding a bucket
        found_bucket_bonus: float = 500.0,
        first_seen_bonus: float = 0.0, #for now
        unfound_in_view_reward_scale: float = 0.0, #for now
        #segmentation mask for each bucket, those colors corresponds to the 200, 201, 202, 203, 204 ids set
        seg_mask_lower_bgr: Tuple[int, int, int] = (190, 68, 107),
        seg_mask_upper_bgr: Tuple[int, int, int] = (190, 68, 107),
        seg_mask_lower_bgr_2: Tuple[int, int, int] = (139, 199, 23),
        seg_mask_upper_bgr_2: Tuple[int, int, int] = (139, 199, 23),
        seg_mask_lower_bgr_3: Tuple[int, int, int] = (168, 88, 171),
        seg_mask_upper_bgr_3: Tuple[int, int, int] = (168, 88, 171),
        seg_mask_lower_bgr_4: Tuple[int, int, int] = (58, 202, 136),
        seg_mask_upper_bgr_4: Tuple[int, int, int] = (58, 202, 136),
        seg_mask_lower_bgr_5: Tuple[int, int, int] = (86, 46, 6),
        seg_mask_upper_bgr_5: Tuple[int, int, int] = (86, 46, 6),
        #size of square cell in meters
        explore_cell_size_m: float = 2.0,
        #reward for visiting a previous not visited cell in an episodeencouraging exploration of the unvisited places in the environment
        explore_reward: float = 0.2,
        coverage_found_threshold_frac: float = 0.02,
        #reward for seeing stand(s) from a closer distance than before during the episode
        coverage_delta_reward_scale: float = 4000.0,
        #legacy param
        motion_mode: str = "velocity",
        continuous_vy_vz: bool = True,
        max_vertical_mps: float = 1.0,
        #maximum flight altitude specified as a height of most obstacles in the environment
        max_altitude_m: float = 15.0,
        #penalty from flying above the maximum specified altitude
        above_altitude_penalty: float = 200.0,
        #do not apply the altitude restriction at the beginning of sim as during take-off the vehicle first goes higher than desired altitude and terminates the episode
        altitude_grace_steps: int = 20,
        # require altitude to exceed the limit for this many consecutive steps (after grace)
        altitude_violation_patience_steps: int = 5,
        #optional debug for seeing what the network actually receives as image input
        save_rgb_debug: bool = False,
        #directory for debug rgb images
        rgb_debug_dir: Optional[str] = None,
        #how often rgb debug images should be saved, high by default as it is just a check and no need for higher frequency
        rgb_debug_every: int = 20000,
        #deprecated naming, actually specifies if and how the state vector should be included, either not at all, as additional image channels(not really used anymore) or as a dictionary (used)
        altitude_obs: str = "none",
        #firstly last action was not included as technically it is related to the current body-frame velocity, but that is not actually the case for forwardonly drivetrain
        include_last_action_in_state: bool = True,
        # toggle randomization at reset (start position/yaw/altitude)
        randomize_reset: bool = True,
    ) -> None:
        super().__init__()
        self.image_h, self.image_w = image_size_hw
        self.camera_name = camera_name
        self.camera_pitch_deg = camera_pitch_deg
        self.camera_fov_deg = camera_fov_deg
        self.camera_offset_x = float(camera_offset_x)
        self.cruise_altitude_z_m = cruise_altitude_z_m
        self.action_duration_s = action_duration_s
        self.speed_mps = speed_mps
        self.motion_mode = str(motion_mode).lower().strip()
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.success_reward = success_reward
        self.vehicle_name = vehicle_name
        self.seg_reward_every_n = max(1, int(seg_reward_every_n))
        self.coverage_found_threshold_frac = coverage_found_threshold_frac
        self.coverage_delta_reward_scale = coverage_delta_reward_scale
        self._seg_mask_lower_bgr = np.array(seg_mask_lower_bgr, dtype=np.uint8)
        self._seg_mask_upper_bgr = np.array(seg_mask_upper_bgr, dtype=np.uint8)
        self._seg_mask_lower_bgr_2 = np.array(seg_mask_lower_bgr_2, dtype=np.uint8) if seg_mask_lower_bgr_2 is not None else None
        self._seg_mask_upper_bgr_2 = np.array(seg_mask_upper_bgr_2, dtype=np.uint8) if seg_mask_upper_bgr_2 is not None else None
        self._seg_mask_lower_bgr_3 = np.array(seg_mask_lower_bgr_3, dtype=np.uint8) if seg_mask_lower_bgr_3 is not None else None
        self._seg_mask_upper_bgr_3 = np.array(seg_mask_upper_bgr_3, dtype=np.uint8) if seg_mask_upper_bgr_3 is not None else None
        self._seg_mask_lower_bgr_4 = np.array(seg_mask_lower_bgr_4, dtype=np.uint8) if seg_mask_lower_bgr_4 is not None else None
        self._seg_mask_upper_bgr_4 = np.array(seg_mask_upper_bgr_4, dtype=np.uint8) if seg_mask_upper_bgr_4 is not None else None
        self._seg_mask_lower_bgr_5 = np.array(seg_mask_lower_bgr_5, dtype=np.uint8) if seg_mask_lower_bgr_5 is not None else None
        self._seg_mask_upper_bgr_5 = np.array(seg_mask_upper_bgr_5, dtype=np.uint8) if seg_mask_upper_bgr_5 is not None else None
        self.explore_cell_size_m = float(explore_cell_size_m)
        self.explore_reward = float(explore_reward)
        self.continuous_vy_vz = bool(continuous_vy_vz)
        self.max_altitude_m = float(max_altitude_m)
        self.above_altitude_penalty = float(above_altitude_penalty)
        self.max_vertical_mps = float(max_vertical_mps)
        self.altitude_grace_steps = int(altitude_grace_steps)
        self.altitude_violation_patience_steps = int(altitude_violation_patience_steps)

        self.client: airsim.MultirotorClient = airsim.MultirotorClient()
        self.client.confirmConnection()
        alt_mode = str(altitude_obs).lower().strip()
        self.altitude_obs_mode = alt_mode if alt_mode in ("none", "image", "dict") else "none"
        self._include_last_action: bool = bool(include_last_action_in_state)
        self.randomize_reset: bool = bool(randomize_reset)

        if self.altitude_obs_mode == "image":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.image_h, self.image_w, 4), dtype=np.uint8)
        elif self.altitude_obs_mode == "dict":
            #image in uint8, additional inputs as float32 values
            img_space = spaces.Box(low=0, high=255, shape=(3, self.image_h, self.image_w), dtype=np.uint8)
            alt_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            #base state vector consisting of two heading c
            #optionally appending the two last action values to shis additional input
            state_dim = 10 if self._include_last_action else 8
            state_space = spaces.Box(low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32)
            self.observation_space = spaces.Dict({"img": img_space, "alt": alt_space, "state": state_space})
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.image_h, self.image_w, 3), dtype=np.uint8)

        # Action space: continuous [a_y, a_z] ∈ [-1,1]^2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        #runtime state
        self.current_step: int = 0
        self._last_raw_image_w: Optional[int] = None
        self._last_raw_image_h: Optional[int] = None
        self._prev_segmask_frac: float = 0.0
        self._no_improve_steps: int = 0
        #tracking if each bucke was found or not during the episode
        self._color_found: Dict[int, bool] = {0: False, 1: False, 2: False, 3: False, 4: False}
        #tracking if a given color was seen before, used for first seen bonus if enabled
        self._ever_seen_by_color: Dict[int, bool] = {0: False, 1: False, 2: False, 3: False, 4: False}
        #grid for optional bonus for exploration
        self._visited_cells: Set[Tuple[int, int]] = set()
        self._omega_max_rad_s: float = 4.0
        self._last_action_ay: float = 0.0
        self._last_action_az: float = 0.0
        #caching image requests
        #uncompressed for faster rendering
        self._req_scene = airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)
        self._req_seg = airsim.ImageRequest(self.camera_name, airsim.ImageType.Segmentation, False, False)
        #last good frame used during simulation hiccups
        self._last_scene_bgr: Optional[np.ndarray] = None
        # Per-step image caches to coalesce RPCs
        self._cache_step_idx: int = -1
        self._cache_scene_bgr: Optional[np.ndarray] = None
        self._cache_seg_bgr: Optional[np.ndarray] = None
        # Segmentation reward caches and defaults
        self._last_seg_reward: float = 0.0
        self._last_segfrac: float = 0.0
        self._last_found_count: int = 0
        self._total_targets: int = 5
        # reward tuning
        self.first_seen_bonus: float = float(first_seen_bonus)
        self.unfound_in_view_reward_scale: float = float(unfound_in_view_reward_scale)
        self.print_collisions: bool = bool(print_collisions)
        self._resize_interp: int = int(resize_interpolation)
        self.found_bucket_bonus: float = float(found_bucket_bonus)
        #success in the episode flag for printing in the terminal
        self._success_printed: bool = False
        # success awarded flag to avoid double counting
        self._success_awarded: bool = False
        #tracki
        self._max_cov_by_color: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        # rgb/gray debug saving
        self.save_rgb_debug: bool = bool(locals().get("save_rgb_debug", False))
        self.rgb_debug_dir: Optional[str] = locals().get("rgb_debug_dir", None)
        self.rgb_debug_every: int = int(locals().get("rgb_debug_every", 200))
        if self.save_rgb_debug and self.rgb_debug_dir:
            os.makedirs(self.rgb_debug_dir, exist_ok=True)


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if self.vehicle_name is None:
            self.vehicle_name = "SimpleFlight"
        #resettngta
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        #taking-off 5 meters to prevent overshoot over maximum altitude during take-off or crash with the ground for lower starting altitudes
        try:
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        try:
            self.client.moveToZAsync(-5.0, self.speed_mps, vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        try:
            base_pose = self.client.simGetVehiclePose(self.vehicle_name)
            if self.randomize_reset:
                dx = random.uniform(-7.0, 5.0)
                dy = random.uniform(-75.0, 100.0)
                yaw_deg = random.uniform(0.0, 360.0)
                start_altitude_m = random.uniform(3.0, 12.0)
            else:
                dx = 0.0
                dy = 0.0
                yaw_deg = 0.0
                start_altitude_m = 7.0
            self._current_takeoff_altitude_m = float(start_altitude_m)
            tx = float(base_pose.position.x_val) + dx
            ty = float(base_pose.position.y_val) + dy
            tz = -float(start_altitude_m)
            self.client.moveToPositionAsync(
                tx, ty, tz,
                self.speed_mps,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=float(yaw_deg)),
                vehicle_name=self.vehicle_name,
            ).join()
            #checking stability before rl start
            try:
                self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
            except Exception:
                pass
            time.sleep(4.0)
        except Exception:
            pass

        #setting the pitch angle and fov of the camera
        down_quat = airsim.to_quaternion(math.radians(float(self.camera_pitch_deg)), 0.0, 0.0)
        self.client.simSetCameraPose(
            self.camera_name,
            airsim.Pose(
                airsim.Vector3r(float(self.camera_offset_x), 0.0, 0.0),
                down_quat
            ),
            vehicle_name=self.vehicle_name,
        )
        self.client.simSetCameraFov(self.camera_name, float(self.camera_fov_deg), vehicle_name=self.vehicle_name)

        self._stand_positions_cache = None
        self.current_step = 0
        #the current cell is visited
        self._visited_cells.clear()
        x0, y0 = self._get_position()
        c0 = self._cell_of(x0, y0)
        self._visited_cells.add(c0)
        self._prev_segmask_frac = 0.0
        self._no_improve_steps = 0
        self._alt_violation_steps: int = 0
        self._color_found = {0: False, 1: False, 2: False, 3: False, 4: False}
        self._max_cov_by_color = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self._ever_seen_by_color = {0: False, 1: False, 2: False, 3: False, 4: False}
        self._success_printed = False
        self._success_awarded = False
        #deprecated code for measuring the step length
        try:
            self._last_pose = self.client.simGetVehiclePose(self.vehicle_name)
        except Exception:
            self._last_pose = None
        obs = self._get_observation()
        info: Dict[str, float] = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        #all actions are in the xy plane
        if self.continuous_vy_vz:
            #v_y, v_z are two continuous actions that can be taken, v_x is just computed to keep the total desired velocity in the xy body plane constant
            try:
                if isinstance(action, (list, tuple, np.ndarray)):
                    a_y = float(action[0])
                    a_z = float(action[1]) if len(action) > 1 else 0.0
                else:
                    a_y = float(action)
                    a_z = 0.0
            except Exception:
                a_y, a_z = 0.0, 0.0
            vy = float(np.clip(a_y, -1.0, 1.0)) * float(self.speed_mps)
            vz = float(np.clip(a_z, -1.0, 1.0)) * float(self.max_vertical_mps)
            #remember last action inputs
            try:
                self._last_action_ay = float(np.clip(a_y, -1.0, 1.0))
                self._last_action_az = float(np.clip(a_z, -1.0, 1.0))
            except Exception:
                self._last_action_ay = 0.0
                self._last_action_az = 0.0
            #vx computed from the total desired velocity
            vxy2 = max(0.0, float(self.speed_mps) ** 2 - vy * vy)
            vx = float(vxy2 ** 0.5)
            try:
                self.client.moveByVelocityBodyFrameAsync(
                    vx=vx,
                    vy=vy,
                    vz=0.4+vz/2,
                duration=self.action_duration_s,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
                vehicle_name=self.vehicle_name,
            ).join()
            except Exception:
                pass
        else:
            try:
                if isinstance(action, (list, tuple, np.ndarray)):
                    a_y = float(action[0])
                    a_z = float(action[1]) if len(action) > 1 else 0.0
                else:
                    a_y = float(action)
                    a_z = 0.0
            except Exception:
                a_y, a_z = 0.0, 0.0
            vy = float(np.clip(a_y, -1.0, 1.0)) * float(self.speed_mps)
            vz = float(np.clip(a_z, -1.0, 1.0)) * float(self.max_vertical_mps)
            vxy2 = max(0.0, float(self.speed_mps) ** 2 - vy * vy)
            vx = float(vxy2 ** 0.5)
            try:
                self.client.moveByVelocityBodyFrameAsync(
                    vx=vx,
                    vy=vy,
                    vz=0.4 + vz/2,
                    duration=self.action_duration_s,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
                    vehicle_name=self.vehicle_name,
                ).join()
            except Exception:
                pass

        #state fetched once per step
        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        except Exception:
            state = None
        obs = self._get_observation()
        reward, terminated, info = self._compute_reward_and_terminated_info(state=state)
        truncated = False
        self.current_step += 1
        #current position for logging
        try:
            if state is not None:
                p = state.kinematics_estimated.position
                x_now, y_now = float(p.x_val), float(p.y_val)
            else:
                x_now, y_now = self._get_position()
        except Exception:
            x_now, y_now = self._get_position()
        info["pos_x"] = x_now
        info["pos_y"] = y_now
        #updating visited cells and exploration reward if new cell visited
        try:
            cell = self._cell_of(x_now, y_now)
            if cell not in self._visited_cells:
                self._visited_cells.add(cell)
                try:
                    reward += float(self.explore_reward)
                except Exception:
                    pass
        except Exception:
            pass
        if self.current_step >= self.max_steps:
            truncated = True
        #computing the total sum of max semgentation coverages at the end of each episode
        if terminated or truncated:
            try:
                info["final_sum_max_seg_cov"] = float(sum(self._max_cov_by_color.values()))
            except Exception:
                pass
            try:
                info["visited_cells"] = float(len(self._visited_cells))
            except Exception:
                pass
        return obs, reward, terminated, truncated, info


    def _get_position(self) -> Tuple[float, float]:
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        return state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val

    def _get_altitude(self) -> float:
        #airsim has z axis in the world frame pointing down
        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            z = float(state.kinematics_estimated.position.z_val)
            return float(-z)
        except Exception:
            return 0.0


    def _interpret_planar_body_delta(self, action: int) -> Tuple[float, float]:
        # returning dx dy in meters for for motion in the camera direction (in the horizontal plane)
        if action == 0: 
            return 1.0, 0.0
        return 0.0, 0.0

    def _cell_of(self, x: float, y: float) -> Tuple[int, int]:
        cs = max(1e-6, self.explore_cell_size_m)
        return int(math.floor(x / cs)), int(math.floor(y / cs))
    def _make_state_vector(self) -> np.ndarray:
        try:
            st = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            q = st.kinematics_estimated.orientation
            #rotation matrix from world to body frame to compute the velocity in the body frame
            qw, qx, qy, qz = float(q.w_val), float(q.x_val), float(q.y_val), float(q.z_val)
            R11 = 1 - 2*(qy*qy + qz*qz)
            R12 = 2*(qx*qy - qz*qw)
            R13 = 2*(qx*qz + qy*qw)
            R21 = 2*(qx*qy + qz*qw)
            R22 = 1 - 2*(qx*qx + qz*qz)
            R23 = 2*(qy*qz - qx*qw)
            R31 = 2*(qx*qz - qy*qw)
            R32 = 2*(qy*qz + qx*qw)
            R33 = 1 - 2*(qx*qx + qy*qy)
            #encoding heading 
            _, _, yaw = airsim.to_eularian_angles(q)
            ys = math.sin(yaw)
            yc = math.cos(yaw)
            #velocity in the inertial (ned) frame
            vn = float(st.kinematics_estimated.linear_velocity.x_val)
            ve = float(st.kinematics_estimated.linear_velocity.y_val)
            vd = float(st.kinematics_estimated.linear_velocity.z_val)
            vbx = R11*vn + R12*ve + R13*vd
            vby = R21*vn + R22*ve + R23*vd
            vbz = R31*vn + R32*ve + R33*vd
            #normalizing velocities by max desired velocity
            s = max(1e-6, float(self.speed_mps))
            sz = max(1e-6, float(self.max_vertical_mps))
            vbx_n = float(np.clip(vbx / s, -1.0, 1.0))
            vby_n = float(np.clip(vby / s, -1.0, 1.0))
            vbz_n = float(np.clip(vbz / sz, -1.0, 1.0))
            #angular rates in the body frame
            p = float(st.kinematics_estimated.angular_velocity.x_val)
            qv = float(st.kinematics_estimated.angular_velocity.y_val)
            r = float(st.kinematics_estimated.angular_velocity.z_val)
            wmax = max(1e-6, float(self._omega_max_rad_s))
            p_n = float(np.clip(p / wmax, -1.0, 1.0))
            q_n = float(np.clip(qv / wmax, -1.0, 1.0))
            r_n = float(np.clip(r / wmax, -1.0, 1.0))
        except Exception:
            ys, yc = 0.0, 1.0
            vbx_n = vby_n = vbz_n = 0.0
            p_n = q_n = r_n = 0.0
        base = [ys, yc, vbx_n, vby_n, vbz_n, p_n, q_n, r_n]
        if self._include_last_action:
            base.extend([float(self._last_action_ay), float(self._last_action_az)])
        return np.array(base, dtype=np.float32)


    def _get_observation(self) -> np.ndarray:
        #only using rgb (scene) when it comes to visual input
        need_seg_this_step = ((self.current_step % self.seg_reward_every_n) == 0)
        #requesting both scene and segmentation images at once
        self._fetch_images(want_scene=True, want_seg=need_seg_this_step)
        if self._cache_scene_bgr is None:
            if self.altitude_obs_mode == "dict":
                return {"img": np.zeros((3, self.image_h, self.image_w), dtype=np.uint8), "alt": np.zeros((1,), dtype=np.float32)}
            ch = 4 if self.altitude_obs_mode == "image" else 3
            return np.zeros((self.image_h, self.image_w, ch), dtype=np.uint8)
        bgr = self._cache_scene_bgr
        if self.altitude_obs_mode == "image":
            # append altitude as extra uint8 channel (normalized 0..255)
            try:
                alt = float(self._get_altitude())
            except Exception:
                alt = 0.0
            alt_norm = float(np.clip(alt / max(1e-6, float(self.max_altitude_m)), 0.0, 1.0))
            alt_plane = np.full((self.image_h, self.image_w, 1), int(alt_norm * 255.0 + 0.5), dtype=np.uint8)
            bgr = np.concatenate([bgr, alt_plane], axis=2)
        elif self.altitude_obs_mode == "dict":
            try:
                alt = float(self._get_altitude())
            except Exception:
                alt = 0.0
            alt_norm = float(np.clip(alt / max(1e-6, float(self.max_altitude_m)), 0.0, 1.0))
            chw = np.transpose(bgr, (2, 0, 1))
            state_vec = self._make_state_vector()
            return {"img": chw.astype(np.uint8, copy=False), "alt": np.array([alt_norm], dtype=np.float32), "state": state_vec}
        return bgr

    def _fetch_images(self, want_scene: bool = False, want_seg: bool = False) -> None:
        #cache reset for the new step
        if self._cache_step_idx != self.current_step:
            self._cache_step_idx = self.current_step
            self._cache_scene_bgr = None
            self._cache_seg_bgr = None
        reqs: List[airsim.ImageRequest] = []
        tags: List[str] = []
        if want_scene:
            reqs.append(self._req_scene)
            tags.append("scene")
        if want_seg:
            reqs.append(self._req_seg)
            tags.append("seg")
        if not reqs:
            return
        try:
            resps = self.client.simGetImages(reqs, vehicle_name=self.vehicle_name)
        except Exception:
            return
        for tag, resp in zip(tags, resps):
            if tag == "scene":
                img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
                # prefer uncompressed buffer first
                h = int(getattr(resp, "height", 0))
                w = int(getattr(resp, "width", 0))
                self._cache_scene_bgr = None
                if h > 0 and w > 0:
                    if img1d.size == h * w * 3:
                        self._cache_scene_bgr = img1d.reshape((h, w, 3))
                    elif img1d.size == h * w * 4:
                        # BGRA -> drop alpha
                        self._cache_scene_bgr = img1d.reshape((h, w, 4))[:, :, :3]
                    else:
                        self._cache_scene_bgr = None
                # carry-forward last good frame
                if self._cache_scene_bgr is None:
                    self._cache_scene_bgr = self._last_scene_bgr
                else:
                    self._last_scene_bgr = self._cache_scene_bgr
                try:
                    if self._cache_scene_bgr is not None and not np.any(self._cache_scene_bgr):
                        print("warning: rgb frame fully black")
                except Exception:
                    pass
            elif tag == "seg":
                img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
                if getattr(resp, "height", 0) > 0 and getattr(resp, "width", 0) > 0 and img1d.size in (resp.height * resp.width * 3, resp.height * resp.width * 4):
                    h, w = int(resp.height), int(resp.width)
                    buf = img1d.reshape((h, w, -1))
                    self._cache_seg_bgr = buf[:, :, :3]
                else:
                    self._cache_seg_bgr = None

    def _compute_segmask_reward(self) -> Tuple[float, float, int, int]:
        # binary masks for segmented bucket stands
        seg_bgr = None
        if self._cache_step_idx == self.current_step and self._cache_seg_bgr is not None:
            seg_bgr = self._cache_seg_bgr
        else:
            self._fetch_images(want_seg=True)
            seg_bgr = self._cache_seg_bgr
        if seg_bgr is None:
            return 0.0, 0.0, 0, 5

        #downscaling to observation resolution to accelerate masking, not really needed if settings.json is implemented correctly
        if seg_bgr.shape[1] != self.image_w or seg_bgr.shape[0] != self.image_h:
            seg_bgr = cv2.resize(seg_bgr, (self.image_w, self.image_h), interpolation=cv2.INTER_NEAREST)

        h, w = seg_bgr.shape[:2]
        total_px = float(h * w)

        #color masking done with 24bit packed bgr
        b = seg_bgr[:, :, 0].astype(np.uint32)
        g = seg_bgr[:, :, 1].astype(np.uint32)
        r = seg_bgr[:, :, 2].astype(np.uint32)
        packed = (b<<16) | (g<<8) | r
        def _pack_bgr(arr: Optional[np.ndarray]) -> Optional[int]:
            if arr is None:
                return None
            try:
                bb, gg, rr = int(arr[0]), int(arr[1]), int(arr[2])
                return (bb << 16) | (gg << 8) | rr
            except Exception:
                return None

        c1 = _pack_bgr(self._seg_mask_lower_bgr)
        c2 = _pack_bgr(self._seg_mask_lower_bgr_2)
        c3 = _pack_bgr(self._seg_mask_lower_bgr_3)
        c4 = _pack_bgr(self._seg_mask_lower_bgr_4)
        c5 = _pack_bgr(self._seg_mask_lower_bgr_5)
        m1 = (packed == c1) if c1 is not None else np.zeros((h, w), dtype=bool)
        m2 = (packed == c2) if c2 is not None else np.zeros((h, w), dtype=bool)
        m3 = (packed == c3) if c3 is not None else np.zeros((h, w), dtype=bool)
        m4 = (packed == c4) if c4 is not None else np.zeros((h, w), dtype=bool)
        m5 = (packed == c5) if c5 is not None else np.zeros((h, w), dtype=bool)
        masks = [m1, m2, m3, m4, m5]
        #faster union and counting due to uin8 conversion
        m1u = m1.astype(np.uint8)
        m2u = m2.astype(np.uint8)
        m3u = m3.astype(np.uint8)
        m4u = m4.astype(np.uint8)
        m5u = m5.astype(np.uint8)
        combined_u = m1u
        combined_u = cv2.bitwise_or(combined_u, m2u)
        combined_u = cv2.bitwise_or(combined_u, m3u)
        combined_u = cv2.bitwise_or(combined_u, m4u)
        combined_u = cv2.bitwise_or(combined_u, m5u)
        frac = float(cv2.countNonZero(combined_u)) / max(1.0, total_px)
        reward_components: float = 0.0
        found_count = 0
        coverage_deltas: List[float] = []
        cov_fracs: List[float] = []
        for idx, mi_u in enumerate([m1u, m2u, m3u, m4u, m5u]):
            cov_frac = float(cv2.countNonZero(mi_u)) / max(1.0, total_px)
            cov_fracs.append(cov_frac)
            #optional reward for seeing a given bucket stand for the first time in a given episode
            if self.first_seen_bonus > 0.0:
                if cov_frac > 0.0 and not self._ever_seen_by_color.get(idx,False):  
                    reward_components += self.first_seen_bonus
                    self._ever_seen_by_color[idx] = True
            #additional reward if a given bucket stand is seen from closer distance (is bigger in the field of view)
            if not self._color_found.get(idx, False) and cov_frac < self.coverage_found_threshold_frac:
                delta = max(0.0, cov_frac - float(self._max_cov_by_color.get(idx, 0.0)))
                if delta > 0.0:
                    coverage_deltas.append(delta)
                    self._max_cov_by_color[idx] = cov_frac
            prev_found = bool(self._color_found.get(idx, False))
            if cov_frac >= self.coverage_found_threshold_frac:
                self._color_found[idx] = True
                if (not prev_found) and (self.found_bucket_bonus != 0.0):
                    reward_components += self.found_bucket_bonus
            if self._color_found.get(idx, False):
                found_count += 1

        # Small per-step reward: keep unfound colors (below threshold) in view
        unfound_masks_u = [mi_u for idx, mi_u in enumerate([m1u, m2u, m3u, m4u, m5u]) if not self._color_found.get(idx, False)]
        if unfound_masks_u and (self.unfound_in_view_reward_scale != 0.0):
            combined_unfound = unfound_masks_u[0]
            for mi_u in unfound_masks_u[1:]:
                combined_unfound = cv2.bitwise_or(combined_unfound, mi_u)
            in_view_frac = float(cv2.countNonZero(combined_unfound)) / max(1.0, total_px)
            reward_components += self.unfound_in_view_reward_scale * in_view_frac
        else:
            in_view_frac = 0.0

        #multiplyng the increase in coverage fractions by a high constant to scale it properly
        if coverage_deltas:
            reward_components += float(self.coverage_delta_reward_scale) * float(sum(coverage_deltas))
        #deprecated code for stagnation penalty
        self._prev_segmask_frac = frac
        return reward_components, frac, found_count, 5

    def _get_scene_bgr(self) -> Optional[np.ndarray]:
        try:
            resp = self.client.simGetImages([self._req_scene], vehicle_name=self.vehicle_name)[0]
            img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            if img1d.size == 0:
                return None
            h = int(getattr(resp, "height", 0))
            w = int(getattr(resp, "width", 0))
            if h > 0 and w > 0:
                if img1d.size == h * w * 3:
                    bgr = img1d.reshape((h, w, 3))
                elif img1d.size == h * w * 4:
                    bgr = img1d.reshape((h, w, 4))[:, :, :3]
                else:
                    return None
                self._last_raw_image_h, self._last_raw_image_w = bgr.shape[:2]
                try:
                    if not np.any(bgr):
                        print("warning: rgb frame fully black")
                except Exception:
                    pass
            return bgr
            return None
        except Exception:
            return None


    def get_scene_bgr(self) -> Optional[np.ndarray]:
        return self._get_scene_bgr()

    def get_segmentation_bgr(self) -> Optional[np.ndarray]:
        try:
            resp = self.client.simGetImages([self._req_seg], vehicle_name=self.vehicle_name)[0]
            img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            if img1d.size == 0:
                return None
            #either taking compressed png or raw buffer, mostly reducing to raw buffers now as they are faster to fetch
            if getattr(resp, "height", 0) > 0 and getattr(resp, "width", 0) > 0:
                h, w = int(resp.height), int(resp.width)
                if img1d.size == h * w * 3:
                    return img1d.reshape((h, w, 3))
                if img1d.size == h * w * 4:
                    return img1d.reshape((h, w, 4))[:, :, :3]
            return None
        except Exception:
            return None




    def _compute_reward_and_terminated_info(self, state: Optional[airsim.MultirotorState] = None) -> Tuple[float, bool, Dict[str, float]]:
        #penalizing each step taken to encourage faster task completion
        reward = - self.step_penalty
        terminated = False
        info: Dict[str, float] = {
            "step": float(self.current_step),
        }
        #during take-off in airsim, the agent first goes higher than desired altitude, may cross max distance and cause early termination, hence a grace period at the start of each episode
        within_grace = (int(self.current_step) < int(self.altitude_grace_steps))
        #penalty for colliding with obstacles and ground
        try:
            collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
            has_collided = bool(collision.has_collided)
        except Exception:
            has_collided = False
        if has_collided:
            info["collision"] = 1.0
            if within_grace:
                #collisions ignored during grace period to avoid collision with the ground for a small take off velocity (the drone overshoots down a bit after first taking off)
                info["collision_grace_ignored"] = 1.0
            else:
                reward -= self.collision_penalty
                terminated = True
            if self.print_collisions:
                try:
                    name = getattr(collision, "object_name", "?")
                    depth = getattr(collision, "penetration_depth", None)
                    p = getattr(collision, "position", None)
                    if p is not None:
                        px = getattr(p, "x_val", 0.0)
                        py = getattr(p, "y_val", 0.0)
                        pz = getattr(p, "z_val", 0.0)
                        pos_str = f"({px:.2f},{py:.2f},{pz:.2f})"
                    else:
                        pos_str = "(-,-,-)"
                    ts = getattr(collision, "time_stamp", 0)
                    if depth is None:
                        print(f"[COLLISION] name={name} pos={pos_str} ts={ts} step={self.current_step}")
                    else:
                        print(f"[COLLISION] name={name} depth={depth:.3f} pos={pos_str} ts={ts} step={self.current_step}")
                except Exception:
                    pass
        else:
            info["collision"] = 0.0
        #terminating the episode and penalizing if set max altitude is crossed, to prohibit the agent of going over the obstacles
        try:
            if state is None:
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            z = float(state.kinematics_estimated.position.z_val)
            alt = float(-z)
            info["altitude_m"] = float(alt)
            if alt > float(self.max_altitude_m):
                if within_grace:
                    info["above_alt_penalized"] = 0.0
                    info["above_alt_grace_ignored"] = 1.0
                    self._alt_violation_steps = 0
                else:
                    self._alt_violation_steps += 1
                    info["alt_violation_steps"] = float(self._alt_violation_steps)
                    if self._alt_violation_steps >= max(1, int(self.altitude_violation_patience_steps)):
                        reward -= float(self.above_altitude_penalty)
                        terminated = True
                        info["above_alt_penalized"] = 1.0
                    else:
                        info["above_alt_penalized"] = 0.0
            else:
                info["above_alt_penalized"] = 0.0
                self._alt_violation_steps = 0
        except Exception:
            pass

        #rewards for bucket finding
        try:
            if (self.current_step % max(1, int(self.seg_reward_every_n))) == 0:
                seg_r, seg_frac, found_count, total_targets = self._compute_segmask_reward()
                reward += float(seg_r)
                info["segmask_frac"] = float(seg_frac)
                info["buckets_found"] = float(found_count)
                info["total_targets"] = float(total_targets)
                if (not self._success_printed) and int(found_count) >= int(total_targets):
                    print(f"[SUCCESS] All {int(total_targets)} buckets found at step {int(self.current_step)}.")
                    self._success_printed = True
                #if successfull (finding all buckets), terminating the episode and giving reward
                if (not self._success_awarded) and int(found_count) >= int(total_targets):
                    if self.success_reward != 0.0:
                        reward += float(self.success_reward)
                    terminated = True
                    info["success"] = 1.0
                    self._success_awarded = True
        except Exception:
            pass
        return reward, terminated, info


__all__ = ["BucketSearchEnv"]


