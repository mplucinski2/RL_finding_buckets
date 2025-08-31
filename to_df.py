import argparse
from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
"""
this script is used to convert the tensorboard data from multiple runs into multiple csv
files per each metric and then to compare runs on a svg graphic
"""

def _is_event_file(path: str) -> bool:
    try:
        name = os.path.basename(path)
        return os.path.isfile(path) and name.startswith("events.out.tfevents.")
    except Exception:
        return False


def _list_event_files_in_dir(dir_path: str) -> list[str]:
    try:
        return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if _is_event_file(os.path.join(dir_path, f))]
    except Exception:
        return []


def _collect_inputs(input_path: str) -> list[tuple[str, str]]:
    """
    collecting tensorboard event files in the given path
    """
    inputs: list[tuple[str, str]] = []
    if os.path.isfile(input_path):
        #if a single tf file inside the dir
        run_id = os.path.basename(os.path.dirname(input_path)) or os.path.basename(input_path)
        inputs.append((input_path, run_id))
        return inputs
    subdirs = [entry.path for entry in os.scandir(input_path) if entry.is_dir()]
    subdirs.sort()
    if subdirs:
        for run_path in subdirs:
            inputs.append((run_path, os.path.basename(run_path)))
        return inputs
    #looking for tf files directly in the dir
    event_files = _list_event_files_in_dir(input_path)
    event_files.sort()
    for ef in event_files:
        inputs.append((ef, os.path.basename(ef)))
    return inputs

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TensorBoard scalars to CSV and SVG plots")
    parser.add_argument(
        "--path",
        type=str,
        default="/home/mayloneus/Desktop/reinforcement_learning/logs/PPO_correct_reward",
        help=(
            "parent directory of many or single run"
        ),
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=3000,
        help="minimum step to include in plots",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=350000,
        help="maximum step to include in plots",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="disable legend on plots",
    )
    parser.add_argument(
        "--no-trendline",
        action="store_true",
        help="disable trendline on plots",
    )
    return parser.parse_args()

args = _parse_args()
inputs = _collect_inputs(args.path)
if not inputs:
    raise SystemExit(f"No runs or event files found under: {args.path}")
#folder name is used to label the curves 
frames: list[pd.DataFrame] = []
for reader_input, run_id in inputs:
    try:
        r = SummaryReader(reader_input)
        df_run = r.scalars.copy()
        if df_run is None or df_run.empty:
            continue
        df_run["run_id"] = str(run_id)
        frames.append(df_run[["tag", "value", "step", "run_id"]])
        print(f"loaded{len(df_run)} rows from run '{run_id}'")
    except Exception as exc:
        print(f"failed to read{reader_input}:{exc}")


scalars_df = pd.concat(frames, ignore_index=True)
scalars_df["run_id"] = scalars_df["run_id"].astype(str)
#sorting by nr of steps to make sure that the order is clear
sorted_df = scalars_df.sort_values(["run_id", "tag", "step"], kind="mergesort")

#first converting to csvs per metricand then plotting, simpler to implement than directly plotting
output_csv = "tensorboard_data_all_runs.csv"
sorted_df.to_csv(output_csv, index=False)

#output dirs creation (project root assumed as this script's directory)
project_root_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(project_root_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

def sanitize_for_filename(value: str) -> str:
    """
    making sure the string can be used as a filename
    """
    if not isinstance(value, str):
        value = str(value)
    value = value.replace(os.sep, "_").replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "untitled"

def save_plots_by_tag(dataframe: pd.DataFrame, output_dir: str) -> None:
    unique_tags = dataframe["tag"].dropna().unique().tolist()
    for tag_value, tag_df in dataframe.groupby("tag"):
        #filtering steps in requested window
        tag_df = tag_df[(tag_df["step"] >= int(args.min_step)) & (tag_df["step"] <= int(args.max_step))]
        if tag_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for run_id, run_df in tag_df.groupby("run_id"):
            run_df = run_df.sort_values("step", kind="mergesort")
            line = ax.plot(run_df["step"], run_df["value"], label=str(run_id), linewidth=1.5, alpha=0.9)[0]
            color = line.get_color()
            #linear trendline to better represent the actual performance of a given run
            if not args.no_trendline:
                x = run_df["step"].to_numpy(dtype=float)
                y = run_df["value"].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 2:
                    try:
                        slope, intercept = np.polyfit(x[mask], y[mask], 1)
                        x_min = max(float(args.min_step), float(np.min(x[mask])))
                        x_max = min(float(args.max_step), float(np.max(x[mask])))
                        if x_max > x_min:
                            x_line = np.array([x_min, x_max], dtype=float)
                            y_line = slope * x_line + intercept
                            ax.plot(x_line, y_line, linestyle="--", color=color, linewidth=1.2, alpha=0.9)
                    except Exception:
                        pass
        ax.set_xlim(int(args.min_step), int(args.max_step))
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        # main grid
        ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.7)
        # minor ticks + minigrid
        try:
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.4)
        except Exception:
            pass
        if not args.no_legend:
            ax.legend(loc="best", fontsize=8, frameon=False)
        fig.tight_layout()
        safe_tag = sanitize_for_filename(str(tag_value))
        out_path = os.path.join(output_dir, f"{safe_tag}.svg")
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)

save_plots_by_tag(sorted_df, plots_dir)
#2d csv tables per metric (run-steps axes)
csv_by_tag_dir = os.path.join(project_root_dir, "csv_by_tag")
os.makedirs(csv_by_tag_dir, exist_ok=True)

def export_pivot_csvs_by_tag(dataframe: pd.DataFrame, output_dir: str) -> None:
    unique_tags = dataframe["tag"].dropna().unique().tolist()
    for tag_value, tag_df in dataframe.groupby("tag"):
        #not every run ends at the same step (e.g. baseline), clip to requested max
        tag_df = tag_df[(tag_df["step"] >= 0) & (tag_df["step"] <= int(args.max_step))]
        if tag_df.empty:
            continue
        #runs taken into acocunt for preview
        run_ids = sorted(tag_df["run_id"].astype(str).unique().tolist())
        print(f"Tag: {tag_value} → runs detected: {len(run_ids)}\n  {run_ids[:10]}{' ...' if len(run_ids) > 10 else ''}")
        print("Sample rows before pivot:")
        print(tag_df[["run_id", "step", "value"]].head())
        pivot_df = tag_df.pivot_table(index="step", columns="run_id", values="value", aggfunc="mean")
        pivot_df = pivot_df.sort_index(kind="mergesort")
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
        safe_tag = sanitize_for_filename(str(tag_value))
        out_csv = os.path.join(output_dir, f"{safe_tag}.csv")
        pivot_df.to_csv(out_csv, index=True)
export_pivot_csvs_by_tag(sorted_df, csv_by_tag_dir)