#!/usr/bin/env python3
"""Same purpose as inspect_episode.py but for collect_handover_tactile.py's
bimanual pressure_grids.npz (left_pressure_grid/right_pressure_grid keys) --
pure numpy, no IsaacGym/GPU needed.

Usage:
    python tactile_collection/inspect_bimanual_episode.py <episode_dir>
"""
import sys
import os

import numpy as np


def main():
    ep_dir = sys.argv[1]
    press = np.load(os.path.join(ep_dir, "pressure_grids.npz"))
    print(f"[pressure] keys: {list(press.files)}")
    for side in ("left", "right"):
        grid = press[f"{side}_pressure_grid"]
        nonzero = grid[grid > 0]
        print(f"[{side}] shape={grid.shape} nonzero_fraction={float((grid > 0).mean()):.4%} "
              f"nonzero_count={nonzero.size}")
        if nonzero.size:
            print(f"[{side}] nonzero: min={nonzero.min():.4f} max={nonzero.max():.4f} mean={nonzero.mean():.4f}")
        contact_count = press[f"{side}_contact_count"]
        print(f"[{side}] contact_count: min={contact_count.min()} max={contact_count.max()} "
              f"mean={contact_count.mean():.2f} nonzero_steps={(contact_count > 0).sum()}/{len(contact_count)}")

    traj = np.load(os.path.join(ep_dir, "trajectory_env0.npz"))
    print(f"\n[trajectory] keys: {list(traj.files)}")
    print(f"[trajectory] T={len(traj['frame_index'])}")
    print(f"[trajectory] obj_state[0]={traj['obj_state'][0]}")
    print(f"[trajectory] native_success[-3:]={traj['native_success'][-3:]}")


if __name__ == "__main__":
    main()
