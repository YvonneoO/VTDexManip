#!/usr/bin/env python3
"""Compute BottleCap's GT tactile task_vmax (99th-percentile denormalization
scalar), for infer_pred_tactile_offline.py's required --task_vmax argument.

Same formula as bidexhands' compute_task_vmax.py / train_bc_student.py's
compute_tac_vmax() (percentile over pressure values across a manifest's
episodes, NaN treated as 0) -- but BottleCap's pressure_grids.npz has a
SINGLE "pressure_grid" key (single-handed collection), not bidexhands'
left_pressure_grid/right_pressure_grid split, so this is a separate small
script rather than a shared import across the two repos.

Usage:
    python compute_task_vmax.py --manifest bottlecap_manifest_1000.json \
        --percentile 99 --n_episodes 200
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--n_episodes", type=int, default=200)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    entries = manifest["train"][:args.n_episodes]

    values = []
    for ep_dir in entries:
        with np.load(os.path.join(ep_dir, "pressure_grids.npz")) as press:
            grid = np.nan_to_num(np.asarray(press["pressure_grid"], dtype=np.float32))
            values.append(grid.reshape(-1))
    all_values = np.concatenate(values)
    vmax = max(float(np.percentile(all_values, args.percentile)), 1e-6)
    print(f"[vmax] task=bottle_cap n_episodes={len(entries)} "
          f"percentile={args.percentile} -> task_vmax={vmax}", flush=True)


if __name__ == "__main__":
    main()
