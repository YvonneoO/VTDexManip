#!/usr/bin/env python3
"""Bimanual GT-pose crop smoke test for VTDexManip's HandOver task, mirroring
the DexterousHands repo's gt_pose_crop_smoke_multienv.py -- verifies the
crop-box geometry (both hands, static camera, env_origin handling) visually
BEFORE trusting it inside a real "predtac" training run.

Boots the real "handover-predtac" task/config (so numEnvs/obs_dim match a
real launch exactly), sets a DUMMY PREDTAC_RUN_ID (no live predtac_server.py
needed -- env.step() will call compute_predtac_obs() each tick, which
submits/polls the IPC files harmlessly with nothing listening; tactile
values just stay zero, irrelevant to this geometry-only check), steps with
random actions, then re-derives the SAME crop boxes compute_predtac_obs()
just computed (via the same predtac_utils.crop_boxes_for_env call, fed the
task's own live state) purely to draw + save them for visual inspection --
does not modify handover.py further for this.

Usage (run as a PLAIN SCRIPT from the repo root, NOT `python -m ...` -- see
below for why): PREDTAC_RUN_ID=smoke_test python tv_tasks/tasks/base1/predtac_smoke.py \\
    --task handover-predtac --rl_device cuda:0 --seed 3204 --headless
"""
import os
import sys

os.environ.setdefault("PREDTAC_RUN_ID", "smoke_test")

# `python -m tv_tasks.tasks.base1.predtac_smoke` (the original invocation)
# CANNOT work here, no matter this file's own import order: `-m` resolves the
# dotted path by importing tv_tasks -> tv_tasks.tasks -> tv_tasks.tasks.base1
# BEFORE this file's body ever runs, and tv_tasks/tasks/__init__.py
# unconditionally does `from .bottle_cap import BottleCap` -> ... ->
# `from isaacgym.torch_utils import *`. If anything already imported torch by
# then, isaacgym's own gymdeps guard aborts -- and there's no way for this
# file's own imports (which haven't executed yet at that point) to prevent
# it. Bit us 2026-09-10: reordering imports here alone did NOT fix the
# original crash; running as a plain script instead does, matching how the
# working train_agent.py entry point runs (plain script, not -m) -- so we
# manually put the repo root on sys.path the way `-m` would have, since a
# plain `python path/to/file.py` puts the FILE's own directory there instead.
sys.path.insert(0, os.getcwd())

from utils.hydra_utils import get_args, parse_sim_params, parse_task, set_np_formatting, set_seed
from tv_tasks.tasks.base1.predtac_utils import crop_boxes_for_env

import cv2
import numpy as np
import torch


def draw_boxes(frame_bgr, sides):
    out = frame_bgr.copy()
    colors = {"left": (255, 80, 80), "right": (80, 220, 80)}
    for side, info in sides.items():
        x1, y1, x2, y2 = [int(round(c)) for c in info["box"]]
        color = colors.get(side, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, side, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def main():
    set_np_formatting()
    args = get_args()
    sim_params = parse_sim_params(args)
    set_seed(args.models["seed"], args.models["torch_deterministic"])
    env = parse_task(args, sim_params)
    task = env.task

    n_steps = int(os.environ.get("PREDTAC_SMOKE_STEPS", "10"))
    out_dir = os.path.abspath(os.environ.get("PREDTAC_SMOKE_OUT", "predtac_smoke_out"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[setup] task={args.task} num_envs={env.num_envs} cam={task.cam_w}x{task.cam_h}", flush=True)

    obs = env.reset()
    for step in range(n_steps):
        with torch.no_grad():
            actions = 0.3 * (2.0 * torch.rand(env.num_envs, env.num_actions, device=env.rl_device) - 1.0)
            obs, rew, done, info = env.step(actions)

    # Re-derive the same boxes compute_predtac_obs() just used internally,
    # purely for visualization -- see module docstring.
    env_origin_np = task.env_origin.detach().cpu().numpy()
    right_pts_np = task.fingertip_pos.detach().cpu().numpy()
    left_pts_np = task.a_fingertip_pos.detach().cpu().numpy()

    results = []
    for i in range(env.num_envs):
        frame_rgb = task.camera_rgb_tensor_list[i][0][:, :, :3].detach().cpu().numpy().astype(np.uint8)
        frame_bgr = frame_rgb[:, :, ::-1].copy()
        view_matrix = np.asarray(task.gym.get_camera_view_matrix(task.sim, task.envs[i], task.camera_handles[i][0]), dtype=np.float64)
        proj_matrix = np.asarray(task.gym.get_camera_proj_matrix(task.sim, task.envs[i], task.camera_handles[i][0]), dtype=np.float64)
        sides = crop_boxes_for_env(right_pts_np[i], left_pts_np[i], env_origin_np[i],
                                    view_matrix, proj_matrix, task.cam_w, task.cam_h)

        raw_path = os.path.join(out_dir, f"env{i:02d}_raw.png")
        overlay_path = os.path.join(out_dir, f"env{i:02d}_gtbox.png")
        cv2.imwrite(raw_path, frame_bgr)
        cv2.imwrite(overlay_path, draw_boxes(frame_bgr, sides))

        results.append({"env": i, "sides": {k: v["box"] for k, v in sides.items()}})
        box_str = " ".join(f"{k}_box={[round(c, 1) for c in v['box']]}" for k, v in sides.items())
        print(f"[env {i}] sides={list(sides.keys())} {box_str}", flush=True)

    n_with_any = sum(1 for r in results if len(r["sides"]) >= 1)
    n_with_both = sum(1 for r in results if len(r["sides"]) == 2)
    print(f"\n[summary] {env.num_envs} envs, {n_with_any} with >=1 hand box, "
          f"{n_with_both} with both hands boxed. Wrote images to {out_dir}", flush=True)
    print("PREDTAC_SMOKE_DONE", flush=True)
    if n_with_any == 0:
        raise SystemExit("no hand ever projected into frame -- camera/crop geometry is broken")


if __name__ == "__main__":
    main()
