#!/usr/bin/env python3
"""Build a d3rlpy MDPDataset (v2.x API) from BottleCap Turning's already-collected
successful-episode trajectories, for offline IQL training (train_iql.py).

Two arms (--arm):
  p_only    : observation = flatten(dof_pos) ++ flatten(object_pose)
  p_gt_tac  : same, ++ flatten(pressure_grid)  (217-taxel/21x21 EgoTouch grid,
              ground-truth from the sim's own contact solver -- NOT a predicted
              tactile channel; that would be a third arm, not built here)

Per-step schema (confirmed 2026-09-07 by reading
tactile_collection/collect_bottle_cap_tactile.py's save_episode(), lines ~63-99
-- read directly rather than trusting any summary if this drifts):
  trajectory_env0.npz : frame_index, dof_pos, object_pose, actions, reward,
                          done, native_success, camera_eye, camera_lookat
  pressure_grids.npz  : pressure_grid (T,21,21) float32 Pa -- single-handed,
                          NOT split into left/right (unlike bidexhands) --
                          plus force_grid_n, valid_mask, taxel_area_m2, and
                          diagnostic fields (source_force_n, contact_count,
                          mapped_force_fraction) not used here.

terminal = `done` AS-IS. This conflates true task termination with timeout
truncation (both just set done=True in the collector) -- a known simplification
per the task spec, not fixed here. Since every episode in the manifest is a
*successful* episode, d3rlpy's own EpisodeGenerator splits transitions into
per-episode boundaries wherever a concatenated `terminals` array has a 1, which
is exactly where each episode's own `done` array ends -- so concatenating all
episodes' arrays end-to-end and passing the concatenated `done` as `terminals`
is sufficient without any extra bookkeeping.

d3rlpy v2.x API notes (verified 2026-09-07 against the `master` branch source,
since this environment can't `pip install` and check locally):
  - `d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals)` still
    exists in v2.x as a thin backward-compat wrapper (d3rlpy/dataset/compat.py)
    that builds an EpisodeGenerator + InfiniteBuffer and passes them to the
    generic ReplayBuffer it subclasses -- this is the intended v2.x construction
    path for "I have flat arrays" data, not a deprecated v1 relic.
  - Saving is `dataset.dump(f)` where `f` is an open BINARY file object (not a
    bare path string) -- confirmed via ReplayBuffer.dump(self, f: BinaryIO).
    train_iql.py's loader must open the same way (`ReplayBuffer.load(f, ...)`).

Usage:
    python build_mdp_dataset.py --manifest bottle_cap_manifest.json \
        --arm p_gt_tac --out bottle_cap_p_gt_tac.h5
"""
import os
import json
import argparse

import numpy as np


REQUIRED_TRAJ_KEYS = ["dof_pos", "object_pose", "actions", "reward", "done"]


def _flatten_step(arr):
    """(T, ...) -> (T, prod(...)), float32."""
    arr = np.asarray(arr, dtype=np.float32)
    T = arr.shape[0]
    return arr.reshape(T, -1)


def load_episode_obs_action_reward_terminal(ep_dir, arm):
    traj = np.load(os.path.join(ep_dir, "trajectory_env0.npz"))
    missing = [k for k in REQUIRED_TRAJ_KEYS if k not in traj.files]
    if missing:
        raise KeyError(f"{ep_dir}: trajectory_env0.npz missing keys {missing} "
                        f"(has {traj.files}) -- schema drift from collect_bottle_cap_tactile.py?")

    dof_pos = _flatten_step(traj["dof_pos"])
    object_pose = _flatten_step(traj["object_pose"])
    obs_parts = [dof_pos, object_pose]
    T = dof_pos.shape[0]
    assert object_pose.shape[0] == T, \
        f"{ep_dir}: dof_pos T={T} != object_pose T={object_pose.shape[0]}"

    if arm == "p_gt_tac":
        press = np.load(os.path.join(ep_dir, "pressure_grids.npz"))
        if "pressure_grid" not in press.files:
            raise KeyError(f"{ep_dir}: pressure_grids.npz missing 'pressure_grid' "
                            f"(has {press.files})")
        pressure_grid = np.nan_to_num(np.asarray(press["pressure_grid"], dtype=np.float32), nan=0.0)
        pressure_flat = pressure_grid.reshape(pressure_grid.shape[0], -1)
        assert pressure_flat.shape[0] == T, \
            f"{ep_dir}: trajectory T={T} != pressure_grid T={pressure_flat.shape[0]}"
        obs_parts.append(pressure_flat)

    observations = np.concatenate(obs_parts, axis=-1)
    actions = np.asarray(traj["actions"], dtype=np.float32)
    rewards = np.asarray(traj["reward"], dtype=np.float32).reshape(-1)
    terminals = np.asarray(traj["done"], dtype=np.float32).reshape(-1)
    assert actions.shape[0] == T and rewards.shape[0] == T and terminals.shape[0] == T, \
        f"{ep_dir}: per-step array length mismatch (T={T}, actions={actions.shape[0]}, " \
        f"rewards={rewards.shape[0]}, terminals={terminals.shape[0]})"
    return observations, actions, rewards, terminals


def sanity_check_one_episode(ep_dir, arm):
    """Defensive first step: load one real episode and print keys/shapes before
    committing to processing all of them -- catches schema drift cheaply."""
    traj = np.load(os.path.join(ep_dir, "trajectory_env0.npz"))
    print(f"[sanity] {ep_dir}/trajectory_env0.npz keys: {traj.files}", flush=True)
    for k in traj.files:
        print(f"[sanity]   {k}: shape={np.asarray(traj[k]).shape} dtype={np.asarray(traj[k]).dtype}", flush=True)
    press = np.load(os.path.join(ep_dir, "pressure_grids.npz"))
    print(f"[sanity] {ep_dir}/pressure_grids.npz keys: {press.files}", flush=True)
    for k in press.files:
        v = np.asarray(press[k])
        if v.ndim > 0:
            print(f"[sanity]   {k}: shape={v.shape} dtype={v.dtype}", flush=True)
        else:
            print(f"[sanity]   {k}: value={v}", flush=True)

    obs, act, rew, term = load_episode_obs_action_reward_terminal(ep_dir, arm)
    print(f"[sanity] built obs={obs.shape} action={act.shape} reward={rew.shape} "
          f"terminal={term.shape} (arm={arm})", flush=True)
    if not term[-1]:
        print(f"[sanity][warn] {ep_dir}: last-step done=False -- episode boundary "
              f"won't be marked in the concatenated terminals array", flush=True)


def build_dataset(episode_dirs, arm):
    obs_list, act_list, rew_list, term_list = [], [], [], []
    n_ok = 0
    for ep_dir in episode_dirs:
        try:
            obs, act, rew, term = load_episode_obs_action_reward_terminal(ep_dir, arm)
        except (KeyError, AssertionError, FileNotFoundError) as e:
            print(f"[skip] {ep_dir}: {e}", flush=True)
            continue
        obs_list.append(obs)
        act_list.append(act)
        rew_list.append(rew)
        term_list.append(term)
        n_ok += 1

    if n_ok == 0:
        raise RuntimeError("no episodes could be loaded -- check --manifest / --arm")

    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    rewards = np.concatenate(rew_list, axis=0)
    terminals = np.concatenate(term_list, axis=0)
    episode_returns = [float(r.sum()) for r in rew_list]
    return observations, actions, rewards, terminals, n_ok, episode_returns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                     help="manifest JSON from generate_bottlecap_manifest.py")
    ap.add_argument("--arm", choices=["p_only", "p_gt_tac"], required=True)
    ap.add_argument("--out", required=True, help="output dataset path (.h5)")
    ap.add_argument("--split", choices=["train", "val"], default="train",
                     help="which manifest split to build (default: train, per spec -- "
                          "run again with --split val --out <val_path> for a held-out set)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    episode_dirs = manifest[args.split]
    if not episode_dirs:
        raise RuntimeError(f"manifest[{args.split!r}] is empty: {args.manifest}")

    sanity_check_one_episode(episode_dirs[0], args.arm)

    observations, actions, rewards, terminals, n_ok, episode_returns = build_dataset(episode_dirs, args.arm)

    import d3rlpy  # deferred: only needed once we're actually building the dataset object
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w+b") as f:
        dataset.dump(f)

    print(f"[done] arm={args.arm} split={args.split}", flush=True)
    print(f"[done] n_episodes={n_ok} n_transitions={observations.shape[0]} "
          f"obs_dim={observations.shape[1]}", flush=True)
    print(f"[done] mean_episode_return={np.mean(episode_returns):.4f} "
          f"(min={np.min(episode_returns):.4f} max={np.max(episode_returns):.4f})", flush=True)
    print(f"[done] action range: min={actions.min(axis=0)} max={actions.max(axis=0)}", flush=True)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
