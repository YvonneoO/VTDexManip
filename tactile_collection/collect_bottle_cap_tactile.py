"""Collect full-frame (RGB + hand DOF + object pose + EgoTouch tactile) successful
BottleCap Turning trajectories using a trained bottle_cap-vt_all_cls checkpoint.

Mirrors bidexhands' rollout_tactile_rgb_chest.py output convention so the data is
directly comparable: one directory per successful episode under
<out_dir>/successful_episodes/episode_NNNNNN/ containing rgb_frames/frame_%06d.png,
pressure_grids.npz (EgoTouch 217-taxel/hand pressure, via the SAME
EgoTouchTaxelMapper used for bidexhands -- BottleCap's ShadowHand uses the identical
"robot0:ffdistal"-style rigid body naming and "hand"/"object" actor names, confirmed
2026-08-30), and trajectory_env0.npz (dof_pos, object pose, actions, reward, done,
native_success -- every simulation step, no frame skip).

BottleCap is single-handed (unlike bidexhands' bimanual tasks), and --test mode
hardcodes numEnvs to len(env_dict) (10, one per seen object) -- this script uses
all 10 in parallel (one EgoTouchTaxelMapper + one episode buffer per env) rather
than fighting that override down to num_envs=1, since each env's contacts/bodies
are already independently indexed (DOMAIN_ENV), so this is safe and ~10x faster
than a single-env collector.

Usage:
    python tactile_collection/collect_bottle_cap_tactile.py --task bottle_cap-vt_all_cls \
        --rl_device cuda:0 --resume_model <ckpt> --test --seed 111 --headless \
        --target_successes 1000 --out_dir <dir>
"""
import argparse
import os
import sys

import numpy as np

# This script lives in tactile_collection/, but utils.hydra_utils/model.process_sarl
# are resolved relative to the repo root (same as eval_agent.py/train_agent.py,
# which live AT the root) -- add cwd (the sbatch launcher cd's into VTDEX_ROOT
# before invoking python) so those imports resolve the same way.
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args
from model.process_sarl import process_sarl

import torch  # must come after the isaacgym-importing modules above
from isaacgym import gymapi

from egotouch_taxels import EgoTouchTaxelMapper  # noqa: E402

HAND_COLOR = (0.42, 0.52, 0.56)  # bidexhands data-collection convention (BIDEX_HAND_COLOR_RGB default)
MAPPING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


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
        "pressure_grid": np.asarray(buf["pressure_grid"], dtype=np.float32),
        "force_grid_n": np.asarray(buf["force_grid_n"], dtype=np.float32),
        "source_force_n": np.asarray(buf["source_force_n"], dtype=np.float32),
        "reconstructed_force_n": np.asarray(buf["reconstructed_force_n"], dtype=np.float32),
        "contact_count": np.asarray(buf["contact_count"], dtype=np.int32),
        "mapped_force_fraction": np.asarray(buf["mapped_force_fraction"], dtype=np.float32),
        "valid_mask": buf["valid_mask"],
        "taxel_area_m2": buf["taxel_area_m2"].astype(np.float32),
        "pressure_unit": "Pa", "force_unit": "N", "area_unit": "m^2",
        "layout": "EgoTouch-21x21-217-taxels-single-hand",
        "num_frames": np.asarray(len(buf["rgb_frames"]), dtype=np.int32),
    }
    np.savez_compressed(os.path.join(ep_dir, "pressure_grids.npz"), **pressure)

    trajectory = {
        "frame_index": np.arange(len(buf["rgb_frames"]), dtype=np.int32),
        "dof_pos": np.asarray(buf["dof_pos"], dtype=np.float32),
        "object_pose": np.asarray(buf["object_pose"], dtype=np.float32),
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
        "rgb_frames": [], "pressure_grid": [], "force_grid_n": [], "source_force_n": [],
        "reconstructed_force_n": [], "contact_count": [], "mapped_force_fraction": [],
        "dof_pos": [], "object_pose": [], "actions": [], "reward": [], "done": [],
        "native_success": [], "camera_eye": None, "camera_lookat": None,
        "valid_mask": None, "taxel_area_m2": None,
    }


def main():
    set_np_formatting()
    args = get_args()
    target_successes = int(os.environ.get("TARGET_SUCCESSES", "1000"))
    max_steps = int(os.environ.get("MAX_STEPS", "2000000"))
    out_dir = os.environ.get("OUT_DIR", "collected_bottle_cap_tactile")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.models['seed'], args.models['torch_deterministic'])
    sim_params = parse_sim_params(args)
    env = parse_task(args, sim_params)
    task = env.task
    num_envs = task.num_envs

    if not hasattr(task, "img_buf"):
        task.img_buf = torch.zeros((num_envs, 224, 224, 3), device=task.device, dtype=torch.float)
        print("[collect] manually allocated task.img_buf for RGB capture", flush=True)

    # Hand-color override (data-collection convention only, matches bidexhands)
    for i, env_ptr in enumerate(task.envs):
        hand_actor = task.gym.find_actor_handle(env_ptr, "hand")
        n_bodies = task.gym.get_actor_rigid_body_count(env_ptr, hand_actor)
        for b in range(n_bodies):
            task.gym.set_rigid_body_color(env_ptr, hand_actor, b, gymapi.MESH_VISUAL, gymapi.Vec3(*HAND_COLOR))

    mapping_path = os.path.join(MAPPING_DIR, "pressure_position_mapping_right.json")
    mappers = [
        EgoTouchTaxelMapper(task.gym, env_ptr, "hand", "right", mapping_path)
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

        task.compute_pixel_obs()
        rews_np = rews.detach().cpu().numpy()
        dones_np = dones.detach().cpu().numpy().astype(bool)
        # VTDexManip's env.step() infos key is "successes" (see model/ppo/ppo.py's own
        # eval() method: infos["successes"][new_ids]) -- NOT "consecutive_successes",
        # which is the bidexhands convention.
        successes_np = infos["successes"].detach().cpu().numpy() if isinstance(infos, dict) and "successes" in infos else None
        actions_np = actions.detach().cpu().numpy()

        for i, env_ptr in enumerate(task.envs):
            contacts = task.gym.get_env_rigid_contacts(env_ptr)
            pressure_pa, force_grid_n, diag = mappers[i].project(contacts)
            buf = bufs[i]
            # task.img_buf is already in [0,255] raw pixel range -- compute_pixel_obs()'s
            # own "/ 255." normalization is commented out in shadow_hand.py. Multiplying
            # by 255 again here (an earlier version of this line did) blew every value
            # past 255 and clipped to solid white -- caught 2026-08-30 via the uploaded
            # smoke-test check videos (RGB panel was blank white, tactile panel was fine).
            buf["rgb_frames"].append(task.img_buf[i].detach().cpu().numpy().clip(0, 255))
            buf["pressure_grid"].append(pressure_pa)
            buf["force_grid_n"].append(force_grid_n)
            buf["source_force_n"].append(diag["source_force_n"])
            buf["reconstructed_force_n"].append(diag["reconstructed_force_n"])
            buf["contact_count"].append(diag["contact_count"])
            buf["mapped_force_fraction"].append(diag["mapped_force_fraction"])
            buf["valid_mask"] = mappers[i].valid_mask
            buf["taxel_area_m2"] = mappers[i].taxel_area_m2
            object_row = int(task.object_indices[i].item())
            buf["dof_pos"].append(env0(task.dof_pos, i))
            buf["object_pose"].append(env0(task.root_state_tensor, object_row))
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
