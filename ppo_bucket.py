import argparse
import os
from typing import Optional, Callable
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import torch
from torch.utils.tensorboard import SummaryWriter
from nets import ImpalaCNNExtractor, ResNet18LikeExtractor #deprecated, used for previous attempts
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import airsim
import math
import cv2
from sb3_contrib import RecurrentPPO
from bucket_env import BucketSearchEnv

def _print_bucket_and_stand_seg_ids() -> None:
    """checking if all stands properly detected"""
    client = airsim.MultirotorClient()
    client.confirmConnection()
    pattern = ".*([Bb]ucket|[Ss]tand).*" 
    names = client.simListSceneObjects(pattern)
    n = len(names)
    print(f"detected bucket objects:{n}")
    for name in names:
        obj_id = client.simGetSegmentationObjectID(name)
        print(f"[SEG] {name} -> id={obj_id}")
    


def _assign_unique_stand_ids(base_id: int = 200) -> None:
    """
    assigning unique segmentation id for each bucket stand, each id corresponds to a different color
    """
    client = airsim.MultirotorClient()
    client.confirmConnection()
    stand_names = client.simListSceneObjects(".*([Ss]tand).*")
    stand_names = sorted(set(stand_names))
    next_id = int(base_id) % 256
    for name in stand_names:
        if next_id <= 0:
            next_id = 1
        ok = client.simSetSegmentationObjectID(name, int(next_id), False)
        print(f"[SEG] set id {next_id:3d} for {name} -> {'ok' if ok else 'fail'}")
        next_id = (next_id + 1) % 256


def _print_stand_distances(vehicle: Optional[str] = None) -> None:
    """checking distance to buckets for choosing the sensible step size"""
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        veh_name = vehicle
        try:
            if not veh_name:
                vlist = client.listVehicles()
                veh_name = vlist[0] if vlist else "SimpleFlight"
        except Exception:
            veh_name = vehicle if vehicle else "SimpleFlight"
        #getting vehicle posiion
        try:
            state = client.getMultirotorState(vehicle_name=veh_name)
            vp = state.kinematics_estimated.position
            vx, vy, vz = float(vp.x_val), float(vp.y_val), float(vp.z_val)
        except Exception:
            pose = client.simGetVehiclePose(veh_name)
            vp = pose.position
            vx, vy, vz = float(vp.x_val), float(vp.y_val), float(vp.z_val)
        #listing bucket stands and computing euclidean distances to them
        names = client.simListSceneObjects(".*([Ss]tand).*")
        dists = []
        for name in names:
            try:
                pose = client.simGetObjectPose(name)
                px, py, pz = float(pose.position.x_val), float(pose.position.y_val), float(pose.position.z_val)
                if any(math.isnan(v) for v in (px, py, pz)):
                    continue
                d = math.sqrt((px - vx) ** 2 + (py - vy) ** 2 + (pz - vz) ** 2)
                dists.append((d, name))
            except Exception:
                pass
        dists.sort(key=lambda t: t[0])
        print(f"[dist] Vehicle='{veh_name}' → {len(dists)} stand distances (m):")
        for d, name in dists:
            print(f"[dist] {name}: {d:.2f} m")
    except Exception as e:
        print(f"[dist] failed to compute stand distances: {e}")

def make_env(
    image_size: int = 84,
    vehicle: Optional[str] = None,
    seg_reward_every_n: int = 1,
    action_dur: float = 0.5,
    camera_pitch_deg: float = 0.0,
    camera_name: str = "0",
    camera_fov_deg: float = 85.0,
    camera_offset_x: float = 0.3,
    speed_mps: float = 8.0,
    unfound_in_view_scale: float = 0.0,
    found_bucket_bonus: float = 500.0,
    first_seen_bonus: float = 25.0,
    explore_reward: float = 0.1,
    explore_cell_size: float = 4.0,
    coverage_delta_reward_scale: float = 4000.0,
    coverage_found_threshold_frac: float = 0.02,
    success_reward: float = 1000.0,
    step_penalty: float = 0.01,
    collision_penalty: float = 200.0,
    motion_mode: str = "velocity",
    save_rgb_debug: bool = False,
    rgb_debug_dir: Optional[str] = None,
    rgb_debug_every: int = 200000,
    max_altitude_m: float = 20.0,
    above_altitude_penalty: float = 200.0,
    altitude_grace_steps: int = 20,
    continuous_vy_vz: bool = True,
    max_vertical_mps: float = 1.0,
    altitude_obs: str = "none",
    include_last_action_in_state: bool = True,
    randomize_reset: bool = True,
):
    env = BucketSearchEnv(
        image_size_hw=(image_size, image_size),
        vehicle_name=vehicle,
        camera_pitch_deg=camera_pitch_deg,
        camera_name=camera_name,
        camera_fov_deg=camera_fov_deg,
        camera_offset_x=camera_offset_x,
        seg_reward_every_n=seg_reward_every_n,
        action_duration_s=action_dur,
        speed_mps=speed_mps,
        unfound_in_view_reward_scale=unfound_in_view_scale,
        found_bucket_bonus=found_bucket_bonus,
        first_seen_bonus=first_seen_bonus,
        explore_reward=explore_reward,
        explore_cell_size_m=explore_cell_size,
        coverage_delta_reward_scale=coverage_delta_reward_scale,
        coverage_found_threshold_frac=coverage_found_threshold_frac,
        success_reward=success_reward,
        step_penalty=step_penalty,
        collision_penalty=collision_penalty,
        motion_mode=motion_mode,
        save_rgb_debug=save_rgb_debug,
        rgb_debug_dir=rgb_debug_dir,
        rgb_debug_every=int(rgb_debug_every),
        max_altitude_m=max_altitude_m,
        above_altitude_penalty=above_altitude_penalty,
        altitude_grace_steps=int(altitude_grace_steps),
        continuous_vy_vz=bool(continuous_vy_vz),
        max_vertical_mps=max_vertical_mps,
        altitude_obs=altitude_obs,
        include_last_action_in_state=bool(include_last_action_in_state),
        randomize_reset=bool(randomize_reset),
    )
    return env


def _linear_schedule(initial_value: float) -> Callable[[float], float]:
    """LR scheduler"""
    init = float(initial_value)
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * init
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1) #not really used, can be useful on more powerful machines
    parser.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument("--speed-mps", type=float, default=25.0)
    parser.add_argument("--unfound-in-view-scale", type=float, default=2.0)
    parser.add_argument("--first-seen-bonus", type=float, default=25.0)
    parser.add_argument("--found-bucket-bonus", type=float, default=200.0)
    parser.add_argument("--explore-reward", type=float, default=0.1)
    parser.add_argument("--explore-cell-size", type=float, default=4.0)
    parser.add_argument("--coverage-delta-reward-scale", type=float, default=4000.0)
    parser.add_argument("--coverage-found-threshold-frac", type=float, default=0.02)
    parser.add_argument("--success-reward", type=float, default=1000.0)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--collision-penalty", type=float, default=100.0)
    parser.add_argument("--max-altitude-m", type=float, default=8.0, help="Altitude threshold in meters (above this penalize)")
    parser.add_argument("--above-altitude-penalty", type=float, default=200.0, help="Penalty applied when altitude exceeds threshold")
    parser.add_argument("--altitude-grace-steps", type=int, default=20, help="Do not terminate for altitude for the first N steps after reset")
    parser.add_argument("--save-rgb-debug", action="store_true", help="Save RGB frames periodically for debugging")
    parser.add_argument("--rgb-debug-dir", type=str, default=None, help="Directory for saved RGB debug frames (defaults under save-dir)")
    parser.add_argument("--rgb-debug-every", type=int, default=2000, help="Save RGB debug every N steps")
    parser.add_argument("--vz-correction-mps", type=float, default=1.0, help="Alias of max vertical speed for z correction (m/s)")
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0, help="Camera pitch angle in degrees (positive is down)")
    parser.add_argument("--camera-fov-deg", type=float, default=85.0)
    parser.add_argument("--camera-name", type=str, default="0")
    parser.add_argument("--camera-offset-x", type=float, default=0.3, help="Camera forward offset in body frame (meters)")
    parser.add_argument("--altitude-obs", type=str, default="none", choices=["none", "image", "dict"], help="Include altitude as extra image channel or dict scalar")
    parser.add_argument("--dict-state-legacy-8d", action="store_true", help="When using --altitude-obs dict, use 8-D state (no last action) to match older checkpoints")
    parser.add_argument("--motion-mode", type=str, default="velocity", choices=["velocity", "position"])
    parser.add_argument("--vehicle", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=os.path.join("tb_logs", "PPO_Bucket"))
    parser.add_argument("--randomize-reset", action="store_true", help="Randomize initial position/yaw/altitude at reset")
    parser.add_argument("--seg-reward-every-n", type=int, default=1, help="Compute segmentation reward every N steps (>=1)")
    parser.add_argument("--action-dur", type=float, default=0.25, help="Env action duration seconds")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout length per update")
    parser.add_argument("--n-epochs", type=int, default=3, help="Number of epochs per update")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Base learning rate")
    parser.add_argument("--lr-linear", action="store_true", help="Use linear LR schedule from initial LR to 0 over training")
    parser.add_argument("--resize-interp", type=str, default="area", choices=["area", "nearest"], help="cv2 resize interpolation for obs")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint-save-norm", action="store_true", help="Also save VecNormalize stats at checkpoint")
    parser.add_argument("--log-interval", type=int, default=1, help="SB3 log interval (print training stats every N updates)")
    parser.add_argument("--resume", type=str, default=None, help="Path to PPO checkpoint .zip to resume from")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--backbone", type=str, default="impala", choices=["impala", "resnet18"])
    parser.add_argument("--features-dim", type=int, default=512)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--clip-range-vf", type=float, default=0.2, help="Clip range for value function updates (<=0 to disable)")
    parser.add_argument("--target-kl", type=float, default=0.02, help="Early stop updates when approx_kl exceeds this")
    parser.add_argument("--adam-eps", type=float, default=1e-5, help="Adam optimizer epsilon")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (autocast) in CNN feature extractor")

    args = parser.parse_args()

    def _make_thunk(rank: int):
        def _env_fn():
            env = make_env(
                image_size=args.image_size,
                vehicle=args.vehicle,
                # RGB-only
                seg_reward_every_n=max(1, int(args.seg_reward_every_n)),
                action_dur=float(args.action_dur),
                camera_pitch_deg=float(args.camera_pitch_deg),
                camera_name=str(args.camera_name),
                camera_fov_deg=float(args.camera_fov_deg),
                camera_offset_x=float(args.camera_offset_x),
                # removed yaw controls
                speed_mps=float(args.speed_mps),
                unfound_in_view_scale=float(args.unfound_in_view_scale),
                first_seen_bonus=float(args.first_seen_bonus),
                found_bucket_bonus=float(args.found_bucket_bonus),
                explore_reward=float(args.explore_reward),
                explore_cell_size=float(args.explore_cell_size),
                coverage_delta_reward_scale=float(args.coverage_delta_reward_scale),
                coverage_found_threshold_frac=float(args.coverage_found_threshold_frac),
                success_reward=float(args.success_reward),
                step_penalty=float(args.step_penalty),
                collision_penalty=float(args.collision_penalty),
                motion_mode=str(args.motion_mode),
                max_altitude_m=float(args.max_altitude_m),
                above_altitude_penalty=float(args.above_altitude_penalty),
                altitude_grace_steps=int(args.altitude_grace_steps),
                save_rgb_debug=bool(args.save_rgb_debug),
                rgb_debug_dir=(args.rgb_debug_dir if args.rgb_debug_dir else os.path.join(args.save_dir, "rgb_debug")),
                rgb_debug_every=int(args.rgb_debug_every),
                continuous_vy_vz=True,
                max_vertical_mps=float(args.vz_correction_mps),
                altitude_obs=str(args.altitude_obs),
                include_last_action_in_state=(not bool(args.dict_state_legacy_8d)),
                randomize_reset=bool(args.randomize_reset),
            )
            base_env = getattr(env, "unwrapped", env)
            if hasattr(base_env, "_resize_interp"):
                base_env._resize_interp = cv2.INTER_AREA if args.resize_interp == "area" else cv2.INTER_NEAREST
            env = Monitor(env)
            env.reset(seed=(int(args.seed) + int(rank)))
            return env
        return _env_fn

    thunks = [_make_thunk(i) for i in range(max(1, int(args.n_envs)))]
    if int(args.n_envs) > 1 and args.vec_env == "subproc":
        vec_env = SubprocVecEnv(thunks)
    else:
        vec_env = DummyVecEnv(thunks)

    #seeding for reproducibility
    set_random_seed(int(args.seed))
    #restoring vecnormalize stats for resumed training from a checkpoint
    stats_path = None
    if args.resume is not None:
        try:
            resume_dir = os.path.dirname(args.resume)
            cands = [f for f in os.listdir(resume_dir) if f.startswith("vecnormalize_") and f.endswith(".pkl")]
            if cands:
                cands.sort()
                stats_path = os.path.join(resume_dir, cands[-1])
        except Exception:
            stats_path = None
    if stats_path is not None:
        try:
            vec_env = VecNormalize.load(stats_path, vec_env)
        except Exception:
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    #using reward normalization and vectorized env
    vec_env.training = True
    vec_env.norm_obs = False
    vec_env.norm_reward = True
    #transposing only for image observations
    if str(args.altitude_obs).lower().strip() != "dict":
        vec_env = VecTransposeImage(vec_env)
    class ProgressCallback(BaseCallback):
        def __init__(self, log_dir: str, verbose: int = 0):
            super().__init__(verbose)
            self.writer = SummaryWriter(log_dir=log_dir)
            #episode-specific accumulators
            self._current_ep_cov_sum: float = 0.0
            self._current_ep_cov_len: int = 0
            #stat accumulators per iteration for tracking performance
            self._iter_sum_buckets: int = 0
            self._iter_num_eps: int = 0
            self._iter_sum_cov: float = 0.0
            self._iter_num_cov: int = 0
            self._iter_sum_visited: float = 0.0
            self._iter_num_visited: int = 0
            self._iter_sum_final_cov: float = 0.0
        def _on_step(self) -> bool:
            infos = self.locals.get("infos")
            if infos and len(infos) > 0:
                info0 = infos[0]
                if isinstance(info0, dict) and ("segmask_frac" in info0):
                    try:
                        self._current_ep_cov_sum += float(info0.get("segmask_frac", 0.0))
                        self._current_ep_cov_len += 1
                    except Exception:
                        pass
                try:
                    for info_i in infos:
                        if isinstance(info_i, dict) and ("episode" in info_i):
                            bf_final = int(float(info_i.get("buckets_found", 0)))
                            self._iter_sum_buckets += bf_final
                            self._iter_num_eps += 1
                            if self._current_ep_cov_len > 0:
                                ep_avg_cov = self._current_ep_cov_sum / float(self._current_ep_cov_len)
                                self._iter_sum_cov += ep_avg_cov
                                self._iter_num_cov += 1
                            try:
                                vcells = float(info_i.get("visited_cells", 0.0))
                                self._iter_sum_visited += vcells
                                self._iter_num_visited += 1
                            except Exception:
                                pass
                            #the sum of total max segment coverages
                            try:
                                final_sum_cov = float(info_i.get("final_sum_max_seg_cov", 0.0))
                                self._iter_sum_final_cov += final_sum_cov
                            except Exception:
                                pass
                            self._current_ep_cov_sum = 0.0
                            self._current_ep_cov_len = 0
                except Exception:
                    pass
                try:
                    n_steps_val = int(getattr(self.model, "n_steps", 0))
                except Exception:
                    n_steps_val = 0
                if n_steps_val > 0 and (self.num_timesteps % n_steps_val == 0):
                    #adding buckets found per iteration to get the per episode average
                    if self._iter_num_eps > 0:
                        iter_avg_buckets = float(self._iter_sum_buckets) / float(self._iter_num_eps)
                        try:
                            self.writer.add_scalar("metrics/avg_buckets_found", iter_avg_buckets, self.num_timesteps)
                        except Exception:
                            pass
                        try:
                            self.logger.record("rollout/avg_buckets_found", iter_avg_buckets)
                        except Exception:
                            pass
                    # final total segment coverage per episode
                    if self._iter_num_eps > 0:
                        iter_avg_final_cov = float(self._iter_sum_final_cov) / float(self._iter_num_eps)
                        try:
                            self.writer.add_scalar("metrics/avg_final_total_seg_coverage", iter_avg_final_cov, self.num_timesteps)
                        except Exception:
                            pass
                        try:
                            self.logger.record("rollout/avg_final_total_seg_coverage", iter_avg_final_cov)
                        except Exception:
                            pass
                    #average number of visited cells per episode
                    if self._iter_num_visited > 0:
                        iter_avg_visited = float(self._iter_sum_visited) / float(self._iter_num_visited)
                        try:
                            self.writer.add_scalar("metrics/avg_visited_cells", iter_avg_visited, self.num_timesteps)
                        except Exception:
                            pass
                        try:
                            self.logger.record("rollout/avg_visited_cells", iter_avg_visited)
                        except Exception:
                            pass
                    try:
                        self.writer.flush()
                    except Exception:
                        pass
                    self._iter_sum_buckets = 0
                    self._iter_num_eps = 0
                    self._iter_sum_cov = 0.0
                    self._iter_num_cov = 0
                    self._iter_sum_visited = 0.0
                    self._iter_num_visited = 0
                    self._iter_sum_final_cov = 0.0

            return True

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using {device_str}")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    #pre-resetting the vector environment for initializing
    _ = vec_env.reset()
    #printing ids for debugging purposes
    _print_bucket_and_stand_seg_ids()
    #assigning different ids to stands to differentiate buckets found
    _assign_unique_stand_ids(base_id=200)
    _print_bucket_and_stand_seg_ids()
    _print_stand_distances(vehicle=args.vehicle)
    is_dict_obs = (str(args.altitude_obs).lower().strip()=="dict")
    if is_dict_obs:
        policy_name = "MultiInputLstmPolicy"
        #use sb3's default combined extractor for dict obs 
        policy_kwargs = dict(
            lstm_hidden_size=int(args.lstm_hidden_size),
        )
    else:
        policy_name = "CnnLstmPolicy"
        feat_cls = ImpalaCNNExtractor if args.backbone == "impala" else ResNet18LikeExtractor
        policy_kwargs = dict(
            features_extractor_class=feat_cls,
            features_extractor_kwargs=dict(features_dim=int(args.features_dim), use_coord=True, use_amp=bool(args.amp)),
            lstm_hidden_size=int(args.lstm_hidden_size),
        )

    #introducing linear lr annealing
    if args.lr_linear:
        lr_param = _linear_schedule(float(args.learning_rate))
    else:
        lr_param = float(args.learning_rate)
    if args.resume:
        model = RecurrentPPO.load(args.resume, env=vec_env, device=device_str)
        #enforing setting lr even when resuming from checkpint
        try:
            if args.lr_linear:
                model.lr_schedule = _linear_schedule(float(args.learning_rate))
                new_lr = float(args.learning_rate)
            else:
                const_lr = float(args.learning_rate)
                model.lr_schedule = (lambda _progress_remaining: const_lr)
                new_lr = const_lr
            if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
                for group in model.policy.optimizer.param_groups:
                    group["lr"] = new_lr
        except Exception:
            pass
    else:
        model = RecurrentPPO(
            policy=policy_name,
            env=vec_env,
            verbose=1,
            tensorboard_log=args.save_dir,
            device=device_str,
            policy_kwargs=policy_kwargs,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=lr_param,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            clip_range_vf=(None if float(args.clip_range_vf) <= 0 else float(args.clip_range_vf)),
            target_kl=float(args.target_kl),
            ent_coef=args.ent_coef,
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
        )
    #trying to set epsilon for adam
    try:
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for group in model.policy.optimizer.param_groups:
                group["eps"] = float(args.adam_eps)
    except Exception:
        pass
    os.makedirs(args.save_dir, exist_ok=True)
    run_log_dir = args.save_dir
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, int(args.checkpoint_every)),
        save_path=run_log_dir,
        name_prefix="ppo_bucket",
        save_vecnormalize=bool(args.checkpoint_save_norm),
    )
    progress_cb = ProgressCallback(log_dir=run_log_dir)
    model.learn(total_timesteps=args.total_steps, callback=[checkpoint_cb, progress_cb], log_interval=max(1, int(args.log_interval)))
    model.save(os.path.join(args.save_dir, "ppo_bucket_final"))

if __name__ == "__main__":
    main()


