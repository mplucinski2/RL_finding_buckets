import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

HARDCODED_STANDS: List[Tuple[float, float]] = [
    (-81.89, 39.98),
    (27.95, -29.64),
    (33.71, 110.49),
    (39.42, 58.40),
    (75.58, 8.31),
]


def load_positions(npz_path: str) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"])  # (S, E, T, 3)
    return positions


def aggregate_xy(positions: np.ndarray, episode_index: int) -> Tuple[np.ndarray, np.ndarray]:
    if episode_index < 0 or episode_index >= positions.shape[1]:
        raise ValueError(f"episode_index {episode_index} out of range [0, {positions.shape[1]-1}]")
    xy = positions[:, episode_index, :, :2]  # (S, T, 2)
    x = xy[..., 0].ravel()
    y = xy[..., 1].ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def plot_heatmap(x: np.ndarray, y: np.ndarray, stands: List[Tuple[float, float]], out_svg: str, bins: int = 200) -> None:
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    #computing area bounds of the maps
    x_candidates = []
    y_candidates = []
    if x.size > 0 and y.size > 0:
        x_candidates.extend([float(np.min(x)), float(np.max(x))])
        y_candidates.extend([float(np.min(y)), float(np.max(y))])
    if stands:
        sx = [p[0] for p in stands]
        sy = [p[1] for p in stands]
        x_candidates.extend([min(sx), max(sx)])
        y_candidates.extend([min(sy), max(sy)])

    if not x_candidates or not y_candidates:
        xmin = ymin = -10.0
        xmax = ymax = 10.0
    else:
        xmin, xmax = min(x_candidates), max(x_candidates)
        ymin, ymax = min(y_candidates), max(y_candidates)
        pad_x = 0.03 * max(1.0, (xmax - xmin))
        pad_y = 0.03 * max(1.0, (ymax - ymin))
        xmin, xmax = xmin - pad_x, xmax + pad_x
        ymin, ymax = ymin - pad_y, ymax + pad_y
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    im = ax.imshow(hist.T, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="magma", aspect="equal", vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("visits")
    #overlaying bucket stands
    if stands:
        sx = [p[0] for p in stands]
        sy = [p[1] for p in stands]
        ax.scatter(sx, sy, marker="P", s=72, c="#00ff88", edgecolors="#004422", linewidths=0.8, label="bucket stands", zorder=3)
        for idx, (bx, by) in enumerate(stands):
            label = f"[{idx:02d}]"
            ax.annotate(label, (bx, by), xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color="#e6ffe6", weight="bold")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8, frameon=False)
    #ensuring no titles in plots
    try:
        ax.set_title("")
    except Exception:
        pass
    try:
        fig.suptitle("")
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="/home/mayloneus/Desktop/reinforcement_learning/positions_eval.npz")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--plots-dir", type=str, default="/home/mayloneus/Desktop/reinforcement_learning/plots")
    args = parser.parse_args()
    positions = load_positions(args.npz)
    x, y = aggregate_xy(positions, int(args.episode_index))
    ckpt_hint = os.path.splitext(os.path.basename(args.npz))[0]
    out_svg = os.path.join(args.plots_dir, f"heatmap_{ckpt_hint}_ep{int(args.episode_index)}.svg")
    plot_heatmap(x, y, HARDCODED_STANDS, out_svg, bins=int(args.bins))

if __name__ == "__main__":
    main()
