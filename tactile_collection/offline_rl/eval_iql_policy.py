#!/usr/bin/env python3
"""Roll out a trained d3rlpy IQL policy in the real BottleCap Turning sim and
measure success rate -- the missing piece train_iql.py doesn't provide (it
only trains and saves checkpoints, never evaluates in-sim).

Runs in the `vtdexmanip` conda env (has IsaacGym). Never imports d3rlpy itself
-- d3rlpy 2.x needs torch>=2.5, vtdexmanip's IsaacGym build is pinned to
torch==2.0.1+cu118, and the two cannot coexist in one process. Instead this
process spawns iql_action_server.py (this same directory) as a subprocess
running under the `d3rlpy_offline` env's python, and exchanges one batched
JSON request per env step over the subprocess's stdin/stdout.

Builds observations with the EXACT SAME schema as build_mdp_dataset.py, so a
policy trained on that dataset gets the input distribution it expects:
  p_only   : flatten(dof_pos) ++ flatten(object_pose)
  p_gt_tac : same, ++ flatten(pressure_grid)   (live per-step EgoTouch grid,
             computed the same way collect_bottle_cap_tactile.py did during
             data collection -- same EgoTouchTaxelMapper, same mapping json)

Does NOT load any VTDexManip PPO checkpoint or call process_sarl/sarl.act* --
actions come entirely from the external IQL policy. `--task bottle_cap-base`
is used only to stand up the env cheaply (obs_type is irrelevant since
task.obs_states_buf is never read here).

Usage:
    python eval_iql_policy.py \
        --task bottle_cap-base --seed 111 --headless \
        --arm p_gt_tac --checkpoint runs/iql_p_gt_tac/final_full.d3 \
        --d3rlpy_python /scratch/project/prj-02-phai-lab/yqq/envs/d3rlpy_offline/bin/python \
        --num_trajs 100 --device cuda:0
"""
import os
import sys
import json
import subprocess
import argparse

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args  # noqa: E402

import torch  # noqa: E402 -- must come after the isaacgym-importing modules above
from isaacgym import gymapi  # noqa: E402

from egotouch_taxels import EgoTouchTaxelMapper  # noqa: E402

MAPPING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")


def env0(tensor_or_array, idx):
    if tensor_or_array is None:
        return None
    arr = tensor_or_array
    arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
    if arr.ndim == 0 or arr.shape[0] <= idx:
        return None
    return arr[idx].copy()


class IQLActionClient:
    """Owns the iql_action_server.py subprocess for the lifetime of the eval run."""

    def __init__(self, d3rlpy_python, checkpoint, device):
        server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iql_action_server.py")
        self.proc = subprocess.Popen(
            [d3rlpy_python, server_path, "--checkpoint", checkpoint, "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, bufsize=1,
        )
        ready = self.proc.stdout.readline().strip()
        if ready != "READY":
            raise RuntimeError(f"iql_action_server.py did not report READY (got: {ready!r}); "
                                f"check its stderr above for the real error")

    def act(self, obs_batch):
        """obs_batch: (num_envs, obs_dim) float array -> (num_envs, action_dim) float array."""
        self.proc.stdin.write(json.dumps({"obs": obs_batch.tolist()}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"iql_action_server.py step error: {resp['error']}")
        return np.asarray(resp["action"], dtype=np.float32)

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        self.proc.wait(timeout=10)


def build_batched_obs(task, mappers, arm):
    """One row per env, matching build_mdp_dataset.py's per-step flatten order exactly."""
    num_envs = task.num_envs
    dof_pos = task.dof_pos.detach().cpu().numpy()  # (num_envs, n_dof)
    rows = []
    for i in range(num_envs):
        object_row = int(task.object_indices[i].item())
        object_pose = env0(task.root_state_tensor, object_row)  # (13,)
        parts = [dof_pos[i].reshape(-1), object_pose.reshape(-1)]
        if arm == "p_gt_tac":
            contacts = task.gym.get_env_rigid_contacts(task.envs[i])
            pressure_pa, _force_grid_n, _diag = mappers[i].project(contacts)
            parts.append(np.asarray(pressure_pa, dtype=np.float32).reshape(-1))
        rows.append(np.concatenate(parts).astype(np.float32))
    return np.stack(rows, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["p_only", "p_gt_tac"], required=True)
    ap.add_argument("--checkpoint", required=True, help="final_full.d3 from train_iql.py")
    ap.add_argument("--d3rlpy_python", required=True,
                     help="path to the d3rlpy_offline env's python (NOT this process's own)")
    ap.add_argument("--device", default="cuda:0", help="device the IQL policy itself runs on")
    ap.add_argument("--num_trajs", type=int, default=100,
                     help="stop once this many episodes have completed (matches eval_agent.py's "
                          "own paper-protocol convention of counting completed episodes)")
    args, remaining = ap.parse_known_args()

    # get_args() (hydra_utils) owns --task/--seed/--headless/--resume_model/etc and reads
    # sys.argv directly -- hand it only the args it defines, not ours.
    sys.argv = [sys.argv[0]] + remaining
    set_np_formatting()
    vtdex_args = get_args()
    set_seed(vtdex_args.models['seed'], vtdex_args.models['torch_deterministic'])

    sim_params = parse_sim_params(vtdex_args)
    env = parse_task(vtdex_args, sim_params)
    task = env.task
    num_envs = task.num_envs

    mapping_path = os.path.join(MAPPING_DIR, "pressure_position_mapping_right.json")
    mappers = None
    if args.arm == "p_gt_tac":
        mappers = [
            EgoTouchTaxelMapper(task.gym, env_ptr, "hand", "right", mapping_path)
            for env_ptr in task.envs
        ]

    client = IQLActionClient(args.d3rlpy_python, args.checkpoint, args.device)

    obs = env.reset()
    completed = 0
    successes = 0
    step = 0
    try:
        while completed < args.num_trajs:
            batched_obs = build_batched_obs(task, mappers, args.arm)
            actions_np = client.act(batched_obs)
            actions = torch.as_tensor(actions_np, dtype=torch.float32, device=task.device)

            with torch.no_grad():
                next_obs, rews, dones, infos = env.step(actions)
                obs.copy_(next_obs)

            dones_np = dones.detach().cpu().numpy().astype(bool)
            successes_np = (infos["successes"].detach().cpu().numpy()
                             if isinstance(infos, dict) and "successes" in infos else None)
            for i in range(num_envs):
                if dones_np[i]:
                    completed += 1
                    if successes_np is not None and successes_np[i] > 0:
                        successes += 1

            step += 1
            if step % 200 == 0:
                sr = successes / completed if completed else float("nan")
                print(f"PROGRESS step={step} completed={completed}/{args.num_trajs} "
                      f"successes={successes} running_sr={sr:.4f}", flush=True)
    finally:
        client.close()

    success_rate = successes / completed if completed else float("nan")
    result = {
        "arm": args.arm, "checkpoint": os.path.abspath(args.checkpoint),
        "num_trajs_target": args.num_trajs, "episodes_completed": completed,
        "successes": successes, "success_rate": success_rate, "steps": step,
    }
    print("IQL_EVAL_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
