#!/usr/bin/env python3
"""Generate a fixed, deterministic episode manifest for BottleCap Turning's
already-collected offline-RL data (collect_bottle_cap_tactile.py's output).

Much simpler than bidexhands' equivalent (bidexhands/tactile_collection/bc/
generate_episode_manifest.py): BottleCap's 1000 successful episodes all live
under ONE flat `<data_root>/successful_episodes/episode_NNNNNN/` directory
(no per-seed shard dirs, so no shard_dir/episode pairing is needed -- a bare
episode-dir path identifies a sample on its own).

Usage:
    python generate_bottlecap_manifest.py \
        --data_root <cluster>/collected_bottle_cap_1k \
        --max_episodes 1000 --val_frac 0.1 --seed 0 \
        --out bottle_cap_manifest.json

Every episode_* dir missing trajectory_env0.npz, pressure_grids.npz, or
rgb_frames/ is skipped rather than silently included half-complete (mirrors
collect_bottle_cap_tactile.py's save_episode(), which always writes all three
together -- an incomplete dir means a partially-written or interrupted save).
"""
import os
import json
import argparse

import numpy as np


def discover_episodes(data_root):
    """Returns a sorted list of complete episode_* directory paths under
    <data_root>/successful_episodes/."""
    success_dir = os.path.join(data_root, "successful_episodes")
    if not os.path.isdir(success_dir):
        raise FileNotFoundError(f"no successful_episodes/ dir under {data_root}")

    episodes = []
    for ep_dirname in sorted(os.listdir(success_dir)):
        ep_path = os.path.join(success_dir, ep_dirname)
        if not ep_dirname.startswith("episode_") or not os.path.isdir(ep_path):
            continue
        has_traj = os.path.isfile(os.path.join(ep_path, "trajectory_env0.npz"))
        has_press = os.path.isfile(os.path.join(ep_path, "pressure_grids.npz"))
        has_rgb = os.path.isdir(os.path.join(ep_path, "rgb_frames"))
        if has_traj and has_press and has_rgb:
            episodes.append(ep_path)
        else:
            print(f"[skip] incomplete episode: {ep_path} "
                  f"(traj={has_traj} press={has_press} rgb={has_rgb})", flush=True)
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                     help="dir containing successful_episodes/ "
                          "(e.g. .../collected_bottle_cap_1k)")
    ap.add_argument("--max_episodes", type=int, default=1000)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episodes = discover_episodes(args.data_root)
    print(f"[discover] {len(episodes)} complete successful episodes found "
          f"under {args.data_root}", flush=True)
    if len(episodes) < args.max_episodes:
        print(f"[warn] only {len(episodes)} episodes available, "
              f"fewer than --max_episodes {args.max_episodes}", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    idx = idx[:args.max_episodes]
    n_val = max(1, int(round(len(idx) * args.val_frac)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    manifest = {
        "task": "bottle_cap",
        "data_root": args.data_root,
        "seed": args.seed,
        "max_episodes": args.max_episodes,
        "val_frac": args.val_frac,
        "train": [episodes[i] for i in train_idx],
        "val": [episodes[i] for i in val_idx],
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] {len(manifest['train'])} train / {len(manifest['val'])} val "
          f"episodes written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
