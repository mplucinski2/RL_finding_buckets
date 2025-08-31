import argparse
import os
from typing import Any, Dict, Optional
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from sb3_contrib import RecurrentPPO
import csv
from ppo_bucket import make_env


def _build_eval_env(args: argparse.Namespace, seed_value: int):
    def _fn():
        env = make_env(
            image_size=args.image_size,
            vehicle=args.vehicle,
            seg_reward_every_n=max(1, int(args.seg_reward_every_n)),
            action_dur=float(args.action_dur),
            camera_pitch_deg=float(args.camera_pitch_deg),
            camera_name=str(args.camera_name),
            camera_fov_deg=float(args.camera_fov_deg),
            camera_offset_x=float(args.camera_offset_x),
            speed_mps=float(args.speed_mps),
            unfound_in_view_scale=float(args.unfound_in_view_scale),
            found_bucket_bonus=float(args.found_bucket_bonus),
            first_seen_bonus=float(args.first_seen_bonus),
            explore_reward=float(args.explore_reward),
            explore_cell_size=float(args.explore_cell_size),
            coverage_delta_reward_scale=float(args.coverage_delta_reward_scale),
            coverage_found_threshold_frac=float(args.coverage_found_threshold_frac),
            success_reward=float(args.success_reward),
            motion_mode=str(args.motion_mode),
            max_altitude_m=float(args.max_altitude_m),
            above_altitude_penalty=float(args.above_altitude_penalty),
            altitude_grace_steps=int(args.altitude_grace_steps),
            save_rgb_debug=False,
            rgb_debug_dir=None,
            rgb_debug_every=int(args.rgb_debug_every),
            continuous_vy_vz=True,
            max_vertical_mps=float(args.vz_correction_mps),
            altitude_obs=str(args.altitude_obs),
            include_last_action_in_state=(not bool(args.dict_state_legacy_8d)),
            # base penalties
            step_penalty=float(args.step_penalty),
            collision_penalty=float(args.collision_penalty),
        )
        env = Monitor(env)
        env.reset(seed=int(seed_value))
        return env
    vec = DummyVecEnv([_fn])
    if str(args.altitude_obs).lower().strip() != "dict":
        vec = VecTransposeImage(vec)
    return vec

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/home/mayloneus/Desktop/reinforcement_learning/logs/PPO_correct_reward/ppo_bucket_280000_steps.zip")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs='*', default=list(range(50)), help="list of seeds, by default [0,49]")
    parser.add_argument("--save-positions", type=str, default="positions.npy")
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--vehicle", type=str, default=None)
    parser.add_argument("--seg-reward-every-n", type=int, default=1)
    parser.add_argument("--action-dur", type=float, default=0.25)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--camera-name", type=str, default="0")
    parser.add_argument("--camera-fov-deg", type=float, default=85.0)
    parser.add_argument("--camera-offset-x", type=float, default=0.3)
    parser.add_argument("--motion-mode", type=str, default="velocity", choices=["velocity", "position"])
    parser.add_argument("--speed-mps", type=float, default=8.0)
    parser.add_argument("--vz-correction-mps", type=float, default=1.0)
    parser.add_argument("--altitude-obs", type=str, default="dict", choices=["none", "image", "dict"])
    parser.add_argument("--dict-state-legacy-8d", action="store_true")
    #rewards and penalties used during training
    parser.add_argument("--unfound-in-view-scale", type=float, default=0.0)
    parser.add_argument("--found-bucket-bonus", type=float, default=500.0)
    parser.add_argument("--first-seen-bonus", type=float, default=0.0)
    parser.add_argument("--explore-reward", type=float, default=0.2)
    parser.add_argument("--explore-cell-size", type=float, default=2.0)
    parser.add_argument("--coverage-delta-reward-scale", type=float, default=4000.0)
    parser.add_argument("--coverage-found-threshold-frac", type=float, default=0.02)
    parser.add_argument("--success-reward", type=float, default=1000.0)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--collision-penalty", type=float, default=200.0)
    parser.add_argument("--max-altitude-m", type=float, default=15.0)
    parser.add_argument("--above-altitude-penalty", type=float, default=200.0)
    parser.add_argument("--altitude-grace-steps", type=int, default=30)
    parser.add_argument("--rgb-debug-every", type=int, default=200000)

    args = parser.parse_args()
    device = "cuda"
    model = RecurrentPPO.load(args.checkpoint, env=None, device=device)
    seeds = args.seeds if (args.seeds and len(args.seeds) > 0) else [int(args.seed)]
    temp_env = _build_eval_env(args, seeds[0])
    try:
        base_env = temp_env.envs[0].unwrapped  
        max_len = int(getattr(base_env, "max_steps", 2000))
    except Exception:
        max_len = 2000
    finally:
        try:
            temp_env.close()
        except Exception:
            pass

    n_ep = int(args.n_episodes)
    positions_all = np.full((len(seeds), n_ep, max_len, 3), np.nan, dtype=np.float32)

    #aggregating stats per seed
    per_seed_metrics = []
    #per episode statistics agggregated for all seeds
    all_ep_buckets: list[float] = []
    all_ep_success: list[float] = []
    all_ep_final_cov: list[float] = []
    all_ep_visited: list[float] = []
    all_ep_len: list[int] = []
    all_ep_reward: list[float] = []
    all_ep_success_step: list[float] = [] 
    #rows for csv saving for later analysis
    episode_rows = []
    #collecting the values of all actions taken
    all_action_ay: list[float] = []
    all_action_az: list[float] = []
    #
    all_altitude_m: list[float] = []
    for si, seed_val in enumerate(seeds):
        env = _build_eval_env(args, seed_val)
        model.set_env(env)
        ep_buckets = []
        ep_success = []
        ep_final_cov = []
        ep_visited = []
        ep_len = []
        ep_ret = []

        for epi in range(n_ep):
            #changing seeds to vary the behavior of the agent
            try:
                obs = env.reset(seed=int(seed_val) + int(epi))
            except TypeError:
                obs = env.reset()
            state = None
            episode_start = np.ones((env.num_envs,), dtype=bool)
            done = False
            t = 0
            ep_reward_sum = 0.0
            last_bf = 0.0
            last_cov = 0.0
            last_visited = 0.0
            success_flag = 0.0
            success_step = float("nan")
            while not done and t < max_len:
                #stochastic actions to actually capture the variance in the model policy
                action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=False)
                #collecting action values
                try:
                    arr = np.asarray(action, dtype=float)
                    if arr.ndim == 2 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim == 1 and arr.size >= 2:
                        all_action_ay.append(float(arr[0]))
                        all_action_az.append(float(arr[1]))
                except Exception:
                    pass
                obs, rewards, dones, infos = env.step(action)
                ep_reward_sum += float(rewards[0])
                try:
                    info0: Dict[str, Any] = infos[0]
                    if "pos_x" in info0 and "pos_y" in info0:
                        x = float(info0.get("pos_x", 0.0))
                        y = float(info0.get("pos_y", 0.0))
                    else:
                        x = y = 0.0
                    z = float(info0.get("altitude_m", 0.0))
                    if t < max_len:
                        positions_all[si, epi, t, 0] = x
                        positions_all[si, epi, t, 1] = y
                        positions_all[si, epi, t, 2] = z
                    try:
                        if np.isfinite(z):
                            all_altitude_m.append(float(z))
                    except Exception:
                        pass
                    if "buckets_found" in info0:
                        last_bf = float(info0.get("buckets_found", last_bf))
                    if "final_sum_max_seg_cov" in info0:
                        last_cov = float(info0.get("final_sum_max_seg_cov", last_cov))
                    if "visited_cells" in info0:
                        last_visited = float(info0.get("visited_cells", last_visited))
                    if "success" in info0:
                        s_val = float(info0.get("success", success_flag))
                        if s_val >= 1.0 and not np.isfinite(success_step):
                            success_step = float(t)
                        success_flag = s_val
                except Exception:
                    pass
                done = bool(dones[0])
                episode_start = dones
                t += 1
            ep_buckets.append(last_bf)
            ep_success.append(success_flag)
            ep_final_cov.append(last_cov)
            ep_visited.append(last_visited)
            ep_len.append(t)
            ep_ret.append(ep_reward_sum)
            #total aggregatation of the values
            all_ep_buckets.append(float(last_bf))
            all_ep_success.append(float(success_flag))
            all_ep_final_cov.append(float(last_cov))
            all_ep_visited.append(float(last_visited))
            all_ep_len.append(int(t))
            all_ep_reward.append(float(ep_reward_sum))
            all_ep_success_step.append(float(success_step))
            #csv rows
            episode_rows.append({
                "seed": int(seed_val),
                "episode": int(epi),
                "buckets_found": float(last_bf),
                "success": float(success_flag),
                "final_total_seg_coverage": float(last_cov),
                "visited_cells": float(last_visited),
                "episode_length": int(t),
                "episode_return": float(ep_reward_sum),
                "success_step": (float(success_step) if np.isfinite(success_step) else ""),
            })

        #avereages per-seed
        seed_metrics = dict(
            seed=int(seed_val),
            avg_buckets=float(np.mean(ep_buckets)) if ep_buckets else 0.0,
            avg_successes=float(np.mean(ep_success)) if ep_success else 0.0,
            avg_final_cov=float(np.mean(ep_final_cov)) if ep_final_cov else 0.0,
            avg_visited=float(np.mean(ep_visited)) if ep_visited else 0.0,
            avg_len=float(np.mean(ep_len)) if ep_len else 0.0,
            avg_reward=float(np.mean(ep_ret)) if ep_ret else 0.0,
        )
        per_seed_metrics.append(seed_metrics)

        try:
            env.close()
        except Exception:
            pass
    def _agg(key: str) -> float:
        vals = [m[key] for m in per_seed_metrics]
        return float(np.mean(vals)) if vals else 0.0

    def _std(key: str) -> float:
        vals = [m[key] for m in per_seed_metrics]
        return float(np.std(vals)) if vals else 0.0

    #saving positions (x,y,z) with seed metadata
    np.savez(args.save_positions if args.save_positions.endswith('.npz') else args.save_positions + '.npz',
             positions=positions_all, seeds=np.array(seeds, dtype=np.int32))

    print("per-seed averages:")
    for m in per_seed_metrics:
        print(f"  seed={m['seed']} buckets={m['avg_buckets']:.3f} success={m['avg_successes']:.3f} cov={m['avg_final_cov']:.3f} visited={m['avg_visited']:.3f} len={m['avg_len']:.1f} reward={m['avg_reward']:.3f}")
    print("overall mean + std across seesd")
    print(f"buckets:{_agg('avg_buckets'):.3f} ± {_std('avg_buckets'):.3f}")
    print(f"success:{_agg('avg_successes'):.3f} ± {_std('avg_successes'):.3f}")
    print(f"coverage:{_agg('avg_final_cov'):.3f} ± {_std('avg_final_cov'):.3f}")
    print(f"visited:{_agg('avg_visited'):.3f} ± {_std('avg_visited'):.3f}")
    print(f"length:{_agg('avg_len'):.1f} ± {_std('avg_len'):.1f}")
    print(f"reward:{_agg('avg_reward'):.3f} ± {_std('avg_reward'):.3f}")
    print(f"saved positions to:{os.path.abspath(args.save_positions if args.save_positions.endswith('.npz') else args.save_positions + '.npz')} with shape {positions_all.shape}")

    # Save CSVs (per-episode and per-seed summary)
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(results_dir, exist_ok=True)
    try:
        ep_csv_path = os.path.join(results_dir, "eval_episodes.csv")
        if episode_rows:
            fieldnames = [
                "seed", "episode", "buckets_found", "success", "final_total_seg_coverage",
                "visited_cells", "episode_length", "episode_return", "success_step",
            ]
            with open(ep_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(episode_rows)
        #per-seed csv summary
        seed_csv_path = os.path.join(results_dir, "eval_per_seed_summary.csv")
        if per_seed_metrics:
            fieldnames_seed = [
                "seed", "avg_buckets", "avg_successes", "avg_final_cov",
                "avg_visited", "avg_len", "avg_reward",
            ]
            with open(seed_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames_seed)
                writer.writeheader()
                writer.writerows(per_seed_metrics)
        overall_csv_path = os.path.join(results_dir, "eval_overall_summary.csv")
        with open(overall_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "std"])
            writer.writerow(["buckets", _agg('avg_buckets'), _std('avg_buckets')])
            writer.writerow(["success", _agg('avg_successes'), _std('avg_successes')])
            writer.writerow(["coverage", _agg('avg_final_cov'), _std('avg_final_cov')])
            writer.writerow(["visited", _agg('avg_visited'), _std('avg_visited')])
            writer.writerow(["length", _agg('avg_len'), _std('avg_len')])
            writer.writerow(["reward", _agg('avg_reward'), _std('avg_reward')])
        print(f"wrote summary to": {overall_csv_path}")
    except Exception as e:
        print(f"failed during writing csvs: {e}")
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skipping histograms")
    else:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        def _hist(data, xlabel: str, fname: str, bins: int | list | np.ndarray = 30, range_: tuple | None = None) -> None:
            arr = np.asarray(data, dtype=float)
            if arr.size == 0:
                return
            fig, ax = plt.subplots(figsize=(8, 5))
            if range_ is not None:
                ax.hist(arr[np.isfinite(arr)], bins=bins, range=range_, color="#3366cc", alpha=0.9)
            else:
                ax.hist(arr[np.isfinite(arr)], bins=bins, color="#3366cc", alpha=0.9)
            ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("count")
            fig.tight_layout()
            out_path = os.path.join(plots_dir, fname)
            fig.savefig(out_path, format="svg", bbox_inches="tight")
            plt.close(fig)
        _hist(all_ep_buckets, "buckets_found (per episode)", "hist_buckets_found.svg", bins=np.arange(-0.5, 6.5, 1.0), range_=(-0.5, 5.5))
        _hist(all_ep_success, "success (0/1, per episode)", "hist_success.svg", bins=np.arange(-0.5, 2.0, 1.0), range_=(-0.5, 1.5))
        _hist(all_ep_final_cov, "final_total_seg_coverage (per episode)", "hist_final_total_seg_coverage.svg", bins=30)
        _hist(all_ep_visited, "visited_cells (per episode)", "hist_visited_cells.svg", bins=30)
        _hist(all_ep_len, "episode_length (steps)", "hist_episode_length.svg", bins=30)
        _hist(all_ep_reward, "episode_return", "hist_episode_return.svg", bins=30)
        #only plotting finite vals of successful episodes
        success_steps = [v for v in all_ep_success_step if np.isfinite(v)]
        _hist(success_steps, "success_step (only successful episodes)", "hist_success_step.svg", bins=30)
        #actions vals distributions
        if all_action_ay:
            _hist(all_action_ay,"action a_y", "hist_action_ay.svg", bins=np.linspace(-1.0, 1.0, 41), range_=(-1.0, 1.0))
        if all_action_az:
            _hist(all_action_az,"action az", "hist_action_az.svg", bins=np.linspace(-1.0, 1.0, 41), range_=(-1.0, 1.0))
        #altitude distribution (1D)
        if all_altitude_m:
            _hist(all_altitude_m, "altitude [m]", "hist_altitude_m.svg", bins=50)

def _has_cuda() -> bool:
    try:
        import torch  

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()


