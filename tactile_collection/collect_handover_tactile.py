"""Collect full-frame (RGB + both-hand DOF + object/goal state + EgoTouch tactile)
successful Bimanual Hand-over trajectories using a trained handover-t_scr_gt_priv
checkpoint.

Bimanual adaptation of collect_bottle_cap_tactile.py -- Hand-over has TWO hand
actors ("hand" = primary/right, "another_hand" = receiving/left, confirmed via
base1/shadow_hand.py:595 and handover.py:315), so this saves BOTH hands' DOF and
BOTH hands' pressure grids (left_pressure_grid/right_pressure_grid keys, matching
bidexhands' own two-grid schema exactly -- unlike BottleCap/Lever Sliding's single
"pressure_grid" key, since those tasks are genuinely single-handed). Note:
handover.py's own compute_sensor_obs() (the RL policy's actual tactile input) only
reads "another_hand" contact -- the primary "hand"'s raw contact force is still
directly available via task.contact_force[task.hand_contact_idx] (never wired into
the policy's own observation, see handover.py's commented-out bimanual sensor code)
and is saved here anyway for a complete/faithful RGB+tactile record of both hands.

Also saves task.compute_object_state() directly (raw 24-dim object+goal state
tensor, set_goal=True -- the exact "task-relevant state" tensor the as-shipped
base/t_scr_gt_priv checkpoints consume) as `obj_state` per frame, so downstream
IQL feature-building can use the identical task-relevant-state input the trained
policy saw, not a re-derived approximation.

Hides the visual goal-marker object (a second copy of the object rendered at
the target hand-off location, see handover.py's _load_goal/reset_target_pose)
from the RGB camera every step by teleporting it far outside any camera's
frustum -- confirmed safe because reward/success (compute_object_state's own
self.goal_pose) reads self.goal_states, a separate cached tensor never tied to
this actor's live position; the marker's root_state_tensor entry is used ONLY
for rendering. Without this, the RGB video shows an unexplained second ball
next to one hand -- a privileged goal cue no real camera would ever see.

Usage:
    python tactile_collection/collect_handover_tactile.py --task handover-t_scr_gt_priv \
        --rl_device cuda:0 --resume_model <ckpt> --test --seed 111 --headless \
        --target_successes 2 --out_dir <dir>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args
from model.process_sarl import process_sarl

import torch  # must come after the isaacgym-importing modules above
from isaacgym import gymapi, gymtorch

from egotouch_taxels import EgoTouchTaxelMapper  # noqa: E402

HAND_COLOR = (0.42, 0.52, 0.56)
MAPPING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
GOAL_MARKER_HIDE_POS = (0.0, 0.0, -10.0)  # well below the floor, outside every camera's frustum


def hide_goal_marker(task):
    """Teleport the visual goal-marker object out of camera view every step (see
    module docstring for why this is safe -- purely cosmetic, decoupled from
    reward/success).

    Needs an explicit step_graphics() after the tensor-API teleport: base1/
    shadow_hand.py's compute_pixel_obs() calls gym.render_all_camera_sensors()
    directly with no graphics sync of its own (confirmed by reading it) --
    without this, the render still reflects whatever pose env.step()'s own
    internal step_graphics() last synced (i.e. the PRE-teleport position),
    reproducing the marker for exactly one frame every time (confirmed via a
    real smoke-test screenshot showing the marker on frame 0 only)."""
    idx = task.goal_object_indices
    task.root_state_tensor[idx, 0] = GOAL_MARKER_HIDE_POS[0]
    task.root_state_tensor[idx, 1] = GOAL_MARKER_HIDE_POS[1]
    task.root_state_tensor[idx, 2] = GOAL_MARKER_HIDE_POS[2]
    task.root_state_tensor[idx, 7:13] = 0.0
    idx_int32 = idx.to(torch.int32)
    task.gym.set_actor_root_state_tensor_indexed(
        task.sim, gymtorch.unwrap_tensor(task.root_state_tensor),
        gymtorch.unwrap_tensor(idx_int32), len(idx_int32),
    )
    task.gym.step_graphics(task.sim)


def env0(tensor_or_array, idx):
    if tensor_or_array is None:
        return None
    arr = tensor_or_array
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    if arr.ndim == 0 or arr.shape[0] <= idx:
        return None
    return arr[idx].copy()


def save_episode(out_dir, episode_id, buf):
    ep_dir = os.path.join(out_dir, "successful_episodes", "episode_{:06d}".format(episode_id))
    frames_dir = os.path.join(ep_dir, "rgb_frames")
    os.makedirs(frames_dir, exist_ok=True)

    import imageio.v2 as imageio
    for i, frame in enumerate(buf["rgb_frames"]):
        imageio.imwrite(os.path.join(frames_dir, "frame_{:06d}.png".format(i)), frame.astype(np.uint8))

    pressure = {
        "left_pressure_grid": np.asarray(buf["left_pressure_grid"], dtype=np.float32),
        "right_pressure_grid": np.asarray(buf["right_pressure_grid"], dtype=np.float32),
        "left_force_grid_n": np.asarray(buf["left_force_grid_n"], dtype=np.float32),
        "right_force_grid_n": np.asarray(buf["right_force_grid_n"], dtype=np.float32),
        "left_contact_count": np.asarray(buf["left_contact_count"], dtype=np.int32),
        "right_contact_count": np.asarray(buf["right_contact_count"], dtype=np.int32),
        "valid_mask": buf["valid_mask"],
        "taxel_area_m2": buf["taxel_area_m2"].astype(np.float32),
        "pressure_unit": "Pa", "force_unit": "N", "area_unit": "m^2",
        "layout": "EgoTouch-21x21-217-taxels-bimanual",
        "sensed_hand_note": "right (\"hand\" actor) is NOT part of the trained policy's own "
                             "observation -- only left (\"another_hand\") feeds compute_sensor_obs(); "
                             "right is recorded anyway for a complete record",
        "num_frames": np.asarray(len(buf["rgb_frames"]), dtype=np.int32),
    }
    np.savez_compressed(os.path.join(ep_dir, "pressure_grids.npz"), **pressure)

    trajectory = {
        "frame_index": np.arange(len(buf["rgb_frames"]), dtype=np.int32),
        "dof_pos": np.asarray(buf["dof_pos"], dtype=np.float32),
        "another_dof_pos": np.asarray(buf["another_dof_pos"], dtype=np.float32),
        "object_pose": np.asarray(buf["object_pose"], dtype=np.float32),
        "obj_state": np.asarray(buf["obj_state"], dtype=np.float32),
        "actions": np.asarray(buf["actions"], dtype=np.float32),
        "reward": np.asarray(buf["reward"], dtype=np.float32),
        "done": np.asarray(buf["done"], dtype=bool),
        "native_success": np.asarray(buf["native_success"], dtype=np.float32),
        "camera_eye": np.asarray(buf["camera_eye"], dtype=np.float32),
        "camera_lookat": np.asarray(buf["camera_lookat"], dtype=np.float32),
    }
    np.savez_compressed(os.path.join(ep_dir, "trajectory_env0.npz"), **trajectory)
    return ep_dir


def new_buf():
    return {
        "rgb_frames": [], "left_pressure_grid": [], "right_pressure_grid": [],
        "left_force_grid_n": [], "right_force_grid_n": [],
        "left_contact_count": [], "right_contact_count": [],
        "dof_pos": [], "another_dof_pos": [], "object_pose": [], "obj_state": [],
        "actions": [], "reward": [], "done": [],
        "native_success": [], "camera_eye": None, "camera_lookat": None,
        "valid_mask": None, "taxel_area_m2": None,
    }


def main():
    set_np_formatting()
    args = get_args()
    target_successes = int(os.environ.get("TARGET_SUCCESSES", "2"))
    max_steps = int(os.environ.get("MAX_STEPS", "2000000"))
    out_dir = os.environ.get("OUT_DIR", "collected_handover_tactile")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.models['seed'], args.models['torch_deterministic'])
    sim_params = parse_sim_params(args)
    env = parse_task(args, sim_params)
    task = env.task
    num_envs = task.num_envs

    if not hasattr(task, "img_buf"):
        task.img_buf = torch.zeros((num_envs, 224, 224, 3), device=task.device, dtype=torch.float)
        print("[collect] manually allocated task.img_buf for RGB capture", flush=True)

    for i, env_ptr in enumerate(task.envs):
        for actor_name in ("hand", "another_hand"):
            hand_actor = task.gym.find_actor_handle(env_ptr, actor_name)
            n_bodies = task.gym.get_actor_rigid_body_count(env_ptr, hand_actor)
            for b in range(n_bodies):
                task.gym.set_rigid_body_color(env_ptr, hand_actor, b, gymapi.MESH_VISUAL, gymapi.Vec3(*HAND_COLOR))

    right_mapping = os.path.join(MAPPING_DIR, "pressure_position_mapping_right.json")
    left_mapping = os.path.join(MAPPING_DIR, "pressure_position_mapping_left.json")
    right_mappers = [
        EgoTouchTaxelMapper(task.gym, env_ptr, "hand", "right", right_mapping)
        for env_ptr in task.envs
    ]
    left_mappers = [
        EgoTouchTaxelMapper(task.gym, env_ptr, "another_hand", "left", left_mapping)
        for env_ptr in task.envs
    ]

    sarl = process_sarl(args, env, args.models, args.logger_dir)
    print("Loading model from {}".format(args.resume_model), flush=True)
    sarl.test(args.resume_model)

    obs = env.reset()
    bufs = [new_buf() for _ in range(num_envs)]
    total_successes = 0
    episode_id = 0
    step = 0

    while total_successes < target_successes and step < max_steps:
        with torch.no_grad():
            actions = sarl.actor_critic.act_inference(obs)
            next_obs, rews, dones, infos = env.step(actions)
            obs.copy_(next_obs)

        hide_goal_marker(task)
        task.compute_pixel_obs()
        obj_state = task.compute_object_state(set_goal=True)
        rews_np = rews.detach().cpu().numpy()
        dones_np = dones.detach().cpu().numpy().astype(bool)
        successes_np = infos["successes"].detach().cpu().numpy() if isinstance(infos, dict) and "successes" in infos else None
        actions_np = actions.detach().cpu().numpy()

        for i, env_ptr in enumerate(task.envs):
            contacts = task.gym.get_env_rigid_contacts(env_ptr)
            right_pa, right_force_n, right_diag = right_mappers[i].project(contacts)
            left_pa, left_force_n, left_diag = left_mappers[i].project(contacts)
            buf = bufs[i]
            buf["rgb_frames"].append(task.img_buf[i].detach().cpu().numpy().clip(0, 255))
            buf["right_pressure_grid"].append(right_pa)
            buf["left_pressure_grid"].append(left_pa)
            buf["right_force_grid_n"].append(right_force_n)
            buf["left_force_grid_n"].append(left_force_n)
            buf["right_contact_count"].append(right_diag["contact_count"])
            buf["left_contact_count"].append(left_diag["contact_count"])
            buf["valid_mask"] = left_mappers[i].valid_mask
            buf["taxel_area_m2"] = left_mappers[i].taxel_area_m2
            object_row = int(task.object_indices[i].item())
            buf["dof_pos"].append(env0(task.shadow_hand_dof_pos, i))
            buf["another_dof_pos"].append(env0(task.shadow_hand_another_dof_pos, i))
            buf["object_pose"].append(env0(task.root_state_tensor, object_row))
            buf["obj_state"].append(env0(obj_state, i))
            buf["actions"].append(actions_np[i].copy())
            buf["reward"].append(float(rews_np[i]))
            buf["done"].append(bool(dones_np[i]))
            native_success = float(successes_np[i]) if successes_np is not None else float("nan")
            buf["native_success"].append(native_success)

        if step == 0:
            for i, env_ptr in enumerate(task.envs):
                bufs[i]["_camera_eye"] = task.camera_eye_list[i % len(task.camera_eye_list)]
                bufs[i]["_camera_lookat"] = task.camera_lookat_list[i % len(task.camera_lookat_list)]

        for i in range(num_envs):
            buf = bufs[i]
            buf["camera_eye"] = buf.get("_camera_eye", gymapi.Vec3(0, 0, 0))
            eye = buf["camera_eye"]
            buf["camera_eye"] = [eye.x, eye.y, eye.z] if hasattr(eye, "x") else eye
            lookat = buf.get("_camera_lookat", gymapi.Vec3(0, 0, 0))
            buf["camera_lookat"] = [lookat.x, lookat.y, lookat.z] if hasattr(lookat, "x") else lookat
            if dones_np[i]:
                succeeded = (successes_np is not None and successes_np[i] > 0)
                if succeeded and total_successes < target_successes:
                    save_episode(out_dir, episode_id, buf)
                    total_successes += 1
                    episode_id += 1
                    print("SAVED episode={} total_successes={}/{} step={}".format(
                        episode_id - 1, total_successes, target_successes, step), flush=True)
                bufs[i] = new_buf()

        step += 1
        if step % 500 == 0:
            print("PROGRESS step={} total_successes={}/{}".format(step, total_successes, target_successes), flush=True)

    print("DONE total_successes={} steps={}".format(total_successes, step), flush=True)


if __name__ == "__main__":
    main()
