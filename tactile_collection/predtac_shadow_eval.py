"""Shadow evaluation of the online tactile predictor: a trained GT-tactile policy (e.g. slide-t_scr_gt) drives the env as usual
(TacGT observations), while the Pred-Tac pipeline (render -> crops -> predtac_server -> response) runs alongside on the SAME frames and
is only MONITORED (PREDTAC_SHADOW=1 makes the task call compute_predtac_obs() and discard the result; PREDTAC_QUALITY_LOG=1 prints
the hand-level quality lines). This measures online predictor quality on frames from a competent policy -- the distribution the
fine-tuning data came from -- instead of the near-contactless frames of a freshly initialised PPO policy.

    PREDTAC_SHADOW=1 PREDTAC_QUALITY_LOG=1 PREDTAC_RUN_ID=... PREDTAC_NUM_ENVS=<len(env_dict)> ... \
    python tactile_collection/predtac_shadow_eval.py --task slide-t_scr_gt --rl_device cuda:0 --resume_model <ckpt> --test \
        --seed 1 --headless      # SHADOW_STEPS (default 800) env steps
"""
import os
import sys

sys.path.insert(0, os.getcwd())
try:  # online Pred-Tac runs: turn a hung IsaacGym/gymtorch startup into a quick failure (see startup_watchdog.py)
    import startup_watchdog as _sw
    _sw.start()
except Exception as _e:  # never let the watchdog break a run
    _sw = None
    print("[watchdog] not active:", _e, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args
from model.process_sarl import process_sarl

import torch  # must come after the isaacgym-importing modules above


def main():
    set_np_formatting()
    args = get_args()
    steps = int(os.environ.get("SHADOW_STEPS", "800"))
    set_seed(args.models['seed'], args.models['torch_deterministic'])
    sim_params = parse_sim_params(args)
    env = parse_task(args, sim_params)
    sarl = process_sarl(args, env, args.models, args.logger_dir)
    print("Loading model from {}".format(args.resume_model), flush=True)
    sarl.test(args.resume_model)
    obs = env.reset()
    n_done, n_succ = 0, 0
    rec = {"actions": [], "dones": [], "progress": []}      # PREDTAC_DUMP_DIR set: the expert's actions, so the episode can be re-simulated
    for t in range(steps):
        with torch.no_grad():
            actions = sarl.actor_critic.act_inference(obs)
            next_obs, rews, dones, infos = env.step(actions)
            obs.copy_(next_obs)
        if _sw is not None:
            _sw.progress()
        d = dones.detach().cpu().numpy().astype(bool)
        if os.environ.get("PREDTAC_DUMP_DIR"):
            rec["actions"].append(actions.detach().cpu().numpy().astype("float32"))
            rec["dones"].append(d)
            rec["progress"].append(env.progress_buf.detach().cpu().numpy() if hasattr(env, "progress_buf") else d.astype("int64") * 0)
        if d.any():
            n_done += int(d.sum())
            if isinstance(infos, dict) and "successes" in infos:
                n_succ += int((infos["successes"].detach().cpu().numpy()[d] > 0).sum())
        if (t + 1) % 100 == 0:
            print(f"[shadow] step={t + 1}/{steps} episodes_done={n_done} successes={n_succ}", flush=True)
    if os.environ.get("PREDTAC_DUMP_DIR") and rec["actions"]:
        import numpy as np
        np.savez(os.path.join(os.environ["PREDTAC_DUMP_DIR"], "shadow_actions.npz"), **{k: np.stack(v) for k, v in rec.items()})
    print(f"[shadow] DONE steps={steps} episodes_done={n_done} successes={n_succ}", flush=True)


if __name__ == "__main__":
    main()
