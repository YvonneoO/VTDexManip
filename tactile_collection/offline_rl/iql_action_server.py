#!/usr/bin/env python3
"""Persistent stdin/stdout action server for a trained d3rlpy IQL policy.

Runs in the `d3rlpy_offline` conda env (has d3rlpy, no IsaacGym). A separate
process in the `vtdexmanip` env (has IsaacGym, no d3rlpy) launches this as a
subprocess and exchanges one JSON line per env step -- this two-process split
is required, not a style choice: d3rlpy 2.x needs torch>=2.5 (see
setup_d3rlpy_env.sh), but vtdexmanip's IsaacGym build is pinned to
torch==2.0.1+cu118 (confirmed in vtdex_train_511098.out) -- the two torch
builds cannot coexist in one Python process, so policy inference (this
process) and simulation stepping (eval_iql_policy.py, in vtdexmanip) are two
processes bridged over a pipe instead.

Protocol (line-delimited JSON over stdin/stdout, unbuffered). One request per
ENV STEP, batched across all parallel envs (not one round-trip per env -- with
bottle_cap's num_envs=10 that would be 10x the pipe overhead for no reason):
  in:  {"obs": [[f, f, ...], [f, f, ...], ...]}   -- shape (num_envs, obs_dim)
  out: {"action": [[f, f, ...], [f, f, ...], ...]} -- shape (num_envs, action_dim)
  in:  "QUIT"                  -- clean shutdown, no response
Also prints a single line "READY" to stdout once the policy is loaded, so the
parent process knows not to write requests before the checkpoint is on GPU.

Usage (normally launched as a subprocess by eval_iql_policy.py, not by hand):
    python iql_action_server.py --checkpoint runs/iql_p_gt_tac/final_full.d3 --device cuda:0
"""
import sys
import json
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                     help="final_full.d3 written by train_iql.py's IQL.save() "
                          "(self-describing -- config + weights, no separate "
                          "IQLConfig needed to reconstruct the algo)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import d3rlpy
    policy = d3rlpy.load_learnable(args.checkpoint, device=args.device)
    print("READY", flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line or line == "QUIT":
            break
        try:
            req = json.loads(line)
            obs = np.asarray(req["obs"], dtype=np.float32)  # (num_envs, obs_dim)
            actions = policy.predict(obs)  # (num_envs, action_dim)
            print(json.dumps({"action": np.asarray(actions, dtype=np.float32).tolist()}), flush=True)
        except Exception as e:  # noqa: BLE001 -- report and keep serving rather than dying on one bad step
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
