#!/usr/bin/env python3
"""Print keys/shapes/dtypes/sample-values for one collected episode's
trajectory_env0.npz and pressure_grids.npz -- a quick sanity/example dump,
no IsaacGym or GPU needed (pure numpy).

Usage:
    python tactile_collection/inspect_episode.py <episode_dir>
"""
import sys
import os

import numpy as np


def dump(npz_path, label):
    if not os.path.isfile(npz_path):
        print(f"[{label}] MISSING: {npz_path}")
        return
    data = np.load(npz_path, allow_pickle=True)
    print(f"[{label}] {npz_path} -- {len(data.files)} keys")
    for k in data.files:
        arr = np.asarray(data[k])
        if arr.ndim == 0:
            print(f"  {k}: scalar value={arr}")
        else:
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")


def main():
    if len(sys.argv) != 2:
        print("Usage: inspect_episode.py <episode_dir>")
        sys.exit(1)
    ep_dir = sys.argv[1]

    traj_path = os.path.join(ep_dir, "trajectory_env0.npz")
    press_path = os.path.join(ep_dir, "pressure_grids.npz")
    frames_dir = os.path.join(ep_dir, "rgb_frames")

    dump(traj_path, "trajectory")
    dump(press_path, "pressure")

    if os.path.isdir(frames_dir):
        frames = sorted(os.listdir(frames_dir))
        print(f"[rgb_frames] {frames_dir} -- {len(frames)} files, "
              f"first={frames[0] if frames else None} last={frames[-1] if frames else None}")

    # Concrete numeric examples, not just shapes.
    traj = np.load(traj_path)
    press = np.load(press_path)
    T = traj["frame_index"].shape[0]
    print(f"\n[example] episode length T={T}")
    print(f"[example] reward[:5]={traj['reward'][:5]}")
    print(f"[example] reward[-5:]={traj['reward'][-5:]}")
    print(f"[example] done[-3:]={traj['done'][-3:]}")
    print(f"[example] native_success[-3:]={traj['native_success'][-3:]}")
    print(f"[example] actions[0][:6]={traj['actions'][0][:6]}")
    print(f"[example] dof_pos[0][:6]={traj['dof_pos'][0][:6]}")
    print(f"[example] object_pose[0]={traj['object_pose'][0]}")
    print(f"[example] camera_eye[0]={traj['camera_eye'][0]}  camera_lookat[0]={traj['camera_lookat'][0]}")

    pg = press["pressure_grid"]
    nonzero = pg[pg > 0]
    print(f"\n[pressure] pressure_grid shape={pg.shape} (T, 21, 21)")
    print(f"[pressure] nonzero fraction={float((pg > 0).mean()):.4%}")
    if nonzero.size:
        print(f"[pressure] nonzero: min={nonzero.min():.4f} max={nonzero.max():.4f} "
              f"mean={nonzero.mean():.4f} (Pa)")
    print(f"[pressure] valid_mask sum={int(np.asarray(press['valid_mask']).sum())} / "
          f"{np.asarray(press['valid_mask']).size} (should be 217)")
    print(f"[pressure] mapped_force_fraction[:5]={press['mapped_force_fraction'][:5]}")
    print(f"[pressure] contact_count[:10]={press['contact_count'][:10]}")


if __name__ == "__main__":
    main()
