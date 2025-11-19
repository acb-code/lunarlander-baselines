#!/usr/bin/env python3
"""
Orchestrate Nautilus vs. CleanRL PPO training runs plus plotting/CSV export.

Example:
python scripts/run_lunarlander_compare.py \
    --nautilus-config externals/nautilus/nautilus/configs/lunarlander_ppo.yaml \
    --cleanrl-script externals/cleanrl/cleanrl/ppo.py \
    --env-id LunarLander-v2 \
    --seed 1 \
    --track-wandb
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import yaml


NAUTILUS_TAG = "charts/episodic_return"
CLEANRL_TAG = "rollout/ep_rew_mean"
CLEANRL_FALLBACK_TAGS = ["charts/episodic_return"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train + compare Nautilus PPO vs. CleanRL PPO.")
    parser.add_argument(
        "--nautilus-config",
        type=Path,
        default=Path("externals/nautilus/nautilus/configs/lunarlander_ppo.yaml"),
        help="Path to the Nautilus YAML config file.",
    )
    parser.add_argument(
        "--cleanrl-script",
        type=Path,
        default=Path("externals/cleanrl/cleanrl/ppo.py"),
        help="Path to CleanRL's PPO entrypoint.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="LunarLander-v2",
        help="Environment ID to run in both frameworks (Gymnasium naming, e.g. LunarLander-v2).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Shared RNG seed.")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip launching training jobs and only parse run dirs supplied via flags.",
    )
    parser.add_argument(
        "--track-wandb",
        action="store_true",
        help="Forward --track/Weights & Biases flags to both frameworks.",
    )
    parser.add_argument(
        "--nautilus-extra",
        action="append",
        default=None,
        help="Extra CLI overrides for Nautilus (pass multiple times for more args).",
    )
    parser.add_argument(
        "--cleanrl-extra",
        action="append",
        default=None,
        help="Extra CLI overrides for CleanRL (pass multiple times for more args).",
    )
    parser.add_argument(
        "--nautilus-run-dir",
        type=Path,
        default=None,
        help="Existing Nautilus run directory to reuse (implies --skip-training for Nautilus).",
    )
    parser.add_argument(
        "--cleanrl-run-dir",
        type=Path,
        default=None,
        help="Existing CleanRL run directory to reuse (implies --skip-training for CleanRL).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for plots/CSVs.",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="lunarlander_cleanrl_vs_nautilus.png",
        help="Filename for the comparison plot (stored inside --artifacts-dir).",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="lunarlander_cleanrl_vs_nautilus.csv",
        help="Filename for the CSV export (stored inside --artifacts-dir).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable writing a concatenated CSV of the aligned curves.",
    )
    parser.add_argument(
        "--num-plot-points",
        type=int,
        default=200,
        help="Number of points to use on the shared timestep axis when aligning curves.",
    )
    parser.add_argument(
        "--nautilus-tag",
        type=str,
        default=NAUTILUS_TAG,
        help="TensorBoard scalar tag to read from Nautilus runs.",
    )
    parser.add_argument(
        "--cleanrl-tag",
        type=str,
        default=CLEANRL_TAG,
        help="TensorBoard scalar tag to read from CleanRL runs (falls back to charts/episodic_return if missing).",
    )
    parser.add_argument(
        "--figure-title",
        type=str,
        default=None,
        help="Custom title for the Matplotlib figure (defaults to '<env>: CleanRL vs. Nautilus (PPO)').",
    )
    return parser.parse_args()


def parse_extra_args(extra: Optional[Sequence[str]]) -> List[str]:
    parsed: List[str] = []
    if not extra:
        return parsed
    for chunk in extra:
        parsed.extend(shlex.split(chunk))
    return parsed


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    printable = " ".join(cmd)
    print(f"[cmd] ({cwd}) {printable}")
    subprocess.run(cmd, cwd=cwd, check=True)


def list_run_dirs(runs_root: Path) -> set[Path]:
    if not runs_root.exists():
        return set()
    return {p.resolve() for p in runs_root.iterdir() if p.is_dir()}


def detect_new_run_dir(runs_root: Path, before: set[Path]) -> Path:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs directory {runs_root} not created by training command.")
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found under {runs_root}")
    new_dirs = [p for p in candidates if p.resolve() not in before]
    target_pool = new_dirs or candidates
    target_pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return target_pool[0]


def prepare_nautilus_config(config_path: Path, env_id: str) -> Path:
    """
    Create a temporary config with the requested env_id so Nautilus can target Gymnasium.
    """
    original_cfg = yaml.safe_load(config_path.read_text()) or {}
    original_cfg["env_id"] = env_id
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml")
    with tmp as fp:
        yaml.safe_dump(original_cfg, fp)
    return Path(tmp.name)


def launch_nautilus(
    config_path: Path,
    seed: int,
    track_wandb: bool,
    extra_args: Sequence[str],
    env_id: str,
) -> Path:
    config_path = config_path.resolve()
    repo_root = Path("externals/nautilus").resolve()
    runner = repo_root / "nautilus" / "runners" / "ppo_runner.py"
    if not runner.exists():
        raise FileNotFoundError(f"Could not find Nautilus runner at {runner}")
    runs_root = repo_root / "runs"
    before = list_run_dirs(runs_root)
    effective_config = prepare_nautilus_config(config_path, env_id)
    try:
        cmd = [sys.executable, str(runner), "--config", str(effective_config), "--seed", str(seed)]
        cmd.extend(["--env-id", env_id])
        if track_wandb:
            cmd.append("--track")
        cmd.extend(extra_args)
        run_command(cmd, cwd=repo_root)
        run_dir = detect_new_run_dir(runs_root, before)
        print(f"[nautilus] Latest run directory: {run_dir}")
        return run_dir
    finally:
        effective_config.unlink(missing_ok=True)


def launch_cleanrl(
    script_path: Path,
    seed: int,
    track_wandb: bool,
    extra_args: Sequence[str],
    env_id: str,
) -> Path:
    repo_root = Path("externals/cleanrl").resolve()
    script_path = script_path.resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find CleanRL script at {script_path}")
    runs_root = repo_root / "runs"
    before = list_run_dirs(runs_root)
    cmd = [
        sys.executable,
        str(script_path),
        "--env-id",
        env_id,
        "--seed",
        str(seed),
        "--num-envs",
        "16",
        "--total-timesteps",
        "3000000",
        "--num-steps",
        "1024",
        "--learning-rate",
        "0.001",
        "--target-kl",
        "0.01",
        "--ent-coef",
        "0.02",
        "--exp-name",
        f"ppo_{env_id}",
    ]
    if track_wandb:
        cmd.extend(["--track", "--wandb-project-name", "nautilus-ppo"])
    cmd.extend(extra_args)
    run_command(cmd, cwd=repo_root)
    run_dir = detect_new_run_dir(runs_root, before)
    print(f"[cleanrl] Latest run directory: {run_dir}")
    return run_dir


def newest_event_file(run_dir: Path) -> Path:
    event_files = sorted(
        (p for p in run_dir.rglob("events.out.tfevents.*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir}")
    return event_files[-1]


def collect_scalars(run_dir: Path, tag: str, fallback_tags: Optional[Sequence[str]] = None) -> pd.DataFrame:
    event_path = newest_event_file(run_dir)
    accumulator = event_accumulator.EventAccumulator(str(event_path))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    candidate_tags = [tag] + list(fallback_tags or [])
    selected_tag = next((candidate for candidate in candidate_tags if candidate in tags), None)
    if selected_tag is None:
        raise KeyError(
            f"Scalar tag '{tag}' not found in {event_path}. "
            f"Also tried {fallback_tags or []}. Available: {tags}"
        )
    scalars = accumulator.Scalars(selected_tag)
    if not scalars:
        raise ValueError(f"No scalar entries for tag '{selected_tag}' in {event_path}")
    data = {
        "timesteps": [s.step for s in scalars],
        "value": [s.value for s in scalars],
    }
    return pd.DataFrame(data).sort_values("timesteps")


def align_curves(curves: Iterable[Tuple[str, pd.DataFrame]], num_points: int) -> pd.DataFrame:
    curves = list(curves)
    if len(curves) < 2:
        raise ValueError("Need at least two curves to align for comparison.")
    max_shared = min(df["timesteps"].max() for _, df in curves)
    if max_shared <= 0:
        raise ValueError("Non-positive shared max timestep; make sure runs logged progress.")
    grid = np.linspace(0, max_shared, num_points)
    aligned_frames = []
    for label, df in curves:
        values = np.interp(grid, df["timesteps"].to_numpy(), df["value"].to_numpy())
        aligned_frames.append(
            pd.DataFrame(
                {
                    "timesteps": grid,
                    "value": values,
                    "source": label,
                }
            )
        )
    return pd.concat(aligned_frames, ignore_index=True)


def plot_curves(aligned_df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for source, sub_df in aligned_df.groupby("source"):
        ax.plot(sub_df["timesteps"], sub_df["value"], label=source)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[plot] Saved comparison figure to {output_path}")


def main() -> None:
    args = parse_args()
    figure_title = args.figure_title or f"{args.env_id}: CleanRL vs. Nautilus (PPO)"
    artifacts_dir = args.artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    naut_extra = parse_extra_args(args.nautilus_extra)
    clean_extra = parse_extra_args(args.cleanrl_extra)

    nautilus_run_dir = args.nautilus_run_dir.resolve() if args.nautilus_run_dir else None
    cleanrl_run_dir = args.cleanrl_run_dir.resolve() if args.cleanrl_run_dir else None

    if not args.skip_training:
        if not nautilus_run_dir:
            nautilus_run_dir = launch_nautilus(
                args.nautilus_config, args.seed, args.track_wandb, naut_extra, args.env_id
            )
        if not cleanrl_run_dir:
            cleanrl_run_dir = launch_cleanrl(
                args.cleanrl_script, args.seed, args.track_wandb, clean_extra, args.env_id
            )
    else:
        if not nautilus_run_dir or not cleanrl_run_dir:
            raise ValueError("--skip-training requires both --nautilus-run-dir and --cleanrl-run-dir.")

    assert nautilus_run_dir is not None
    assert cleanrl_run_dir is not None

    naut_df = collect_scalars(nautilus_run_dir, args.nautilus_tag)
    clean_df = collect_scalars(
        cleanrl_run_dir, args.cleanrl_tag, fallback_tags=CLEANRL_FALLBACK_TAGS
    )

    comparison_df = align_curves(
        [
            (f"Nautilus PPO ({args.env_id})", naut_df),
            (f"CleanRL PPO ({args.env_id})", clean_df),
        ],
        num_points=args.num_plot_points,
    )

    figure_path = artifacts_dir / args.figure_name
    plot_curves(comparison_df, figure_title, figure_path)

    if not args.no_csv:
        csv_path = artifacts_dir / args.csv_name
        comparison_df.to_csv(csv_path, index=False)
        print(f"[csv] Wrote aligned scalars to {csv_path}")

    print("[done] Comparison complete.")


if __name__ == "__main__":
    main()
