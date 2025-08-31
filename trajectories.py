import argparse
import os
from typing import List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

#hardcoded bucket stand positions
HARDCODED_STANDS = [
    (-81.89, 39.98),
    (27.95, -29.64),
    (33.71, 110.49),
    (39.42, 58.40),
    (75.58, 8.31),
]
def load_positions(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"]) 
        seeds = np.asarray(data.get("seeds", np.arange(positions.shape[0], dtype=np.int32)))
    return positions, seeds

def plot_trajectories(
    positions: np.ndarray,
    seeds: Sequence[int],
    episode_index: int,
    stands_xy: Sequence[Tuple[float, float]],
    out_svg_path: str,
) -> None:
    num_seeds = positions.shape[0]
    if episode_index < 0 or episode_index >= positions.shape[1]:
        raise ValueError(f"episode_index {episode_index} out of range [0, {positions.shape[1]-1}]")

    os.makedirs(os.path.dirname(out_svg_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    for si in range(num_seeds):
        seed_label = f"seed {int(seeds[si])}" if len(seeds) == num_seeds else f"seed_idx {si}"
        xy = positions[si, episode_index, :, :2]  # (T, 2)
        x = xy[:, 0]
        y = xy[:, 1]
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        line = ax.plot(x[mask], y[mask], linewidth=1.8, alpha=0.95, label=seed_label)[0]
        c = line.get_color()
        ax.scatter([x[mask][0]], [y[mask][0]], s=28, color=c, marker="o", zorder=3)
        ax.scatter([x[mask][-1]], [y[mask][-1]], s=36, color=c, marker="x", zorder=3)

    #plotting bucket stand postions
    if stands_xy:
        sx = [p[0] for p in stands_xy]
        sy = [p[1] for p in stands_xy]
        ax.scatter(sx, sy, marker="P", s=72, c="#444444", label="bucket stands", zorder=4)
        #labelling bucket stands for differentiation
        for idx, (bx, by) in enumerate(stands_xy):
            try:
                label = f"[{idx:02d}]"
            except Exception:
                label = f"[{idx}]"
            ax.annotate(label, (bx, by), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color="#222222", weight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.minorticks_on()
    ax.grid(which='major', linestyle=":", linewidth=0.7, alpha=0.7)
    ax.grid(which='minor', linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="/home/mayloneus/Desktop/reinforcement_learning/positions_eval.npz")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--plots-dir", type=str, default="/home/mayloneus/Desktop/reinforcement_learning/plots")
    args = parser.parse_args()
    positions, seeds = load_positions(args.npz)
    stands_xy: List[Tuple[float, float]] = HARDCODED_STANDS
    print("Using hardcoded bucket stand positions (x, y):")
    for idx, (sx, sy) in enumerate(stands_xy):
        try:
            print(f"  [{idx:02d}] x={sx:.2f}, y={sy:.2f}")
        except Exception:
            print(f"  [{idx:02d}] x={sx}, y={sy}")

    ckpt_hint = os.path.splitext(os.path.basename(args.npz))[0]
    out_svg = os.path.join(args.plots_dir, f"trajectories_{ckpt_hint}_ep{int(args.episode_index)}.svg")
    plot_trajectories(positions, seeds, int(args.episode_index), stands_xy, out_svg)

if __name__ == "__main__":
    main()


