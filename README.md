# LunarLander Baselines Comparison

This repository ties together the Nautilus PPO implementation and CleanRL PPO to make like-for-like LunarLander experiments (defaulting to Gymnasium's `LunarLander-v2`) easy to launch, track, and visualize.

## Python Environment

The comparison script assumes a single Conda environment shared by both projects.

1. Create/activate the environment:
   ```bash
   conda create -n lunarlander python=3.10 -y
   conda activate lunarlander
   python -m pip install -U pip
   ```
2. Install both projects in editable mode with the extras required for LunarLander + developer tooling (including Nautilus's JAX components and Gymnasium's Box2D envs):
   ```bash
   pip install -r requirements-compare.txt
   ```
   *This installs Nautilus with `[dev,envs,jax]` (bringing in PyTorch, WandB, tensorboard, pandas, JAX, etc.), `gymnasium[box2d]>=0.29` so `LunarLander-v2` is available, and CleanRL with `[box2d]`. If your OS image lacks `swig`, install it through your system package manager before running the command so `box2d-py` can build.*

## Usage

The automation entrypoint lives in `scripts/run_lunarlander_compare.py`. Typical example:

```bash
python scripts/run_lunarlander_compare.py \
  --nautilus-config externals/nautilus/nautilus/configs/lunarlander_ppo.yaml \
  --cleanrl-script externals/cleanrl/cleanrl/ppo.py \
  --env-id LunarLander-v2 \
  --seed 1 \
  --track-wandb
```

This launches both training runs, extracts TensorBoard scalars (`charts/episodic_return` from Nautilus and `rollout/ep_rew_mean` from CleanRL), aligns the curves over a shared timestep axis, and writes the figure + CSV inside `artifacts/`.

Use `--skip-training --nautilus-run-dir ... --cleanrl-run-dir ...` to generate comparison plots from prior runs, and `--nautilus-extra/--cleanrl-extra` to forward CLI overrides for ablations. If you have a newer Gymnasium build that exposes `LunarLander-v3`, override the default with `--env-id LunarLander-v3`.
