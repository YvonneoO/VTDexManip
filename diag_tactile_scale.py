import numpy as np

from utils.logger import DataLog
import os
from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args
from model.process_sarl import process_sarl
import torch


def diag_tactile_scale(max_trajs=30):
    """One-off diagnostic (not part of the ablation pipeline): hooks
    task.compute_sensor_obs(gt_continuous=True) -- the exact call TacGT/t_scr_gt
    uses to build its raw tactile channel -- during a real held-out eval rollout
    on an already-trained checkpoint, and reports the observed magnitude
    distribution. Checks whether the raw per-link contact-force L2-norm
    actually saturates the pipeline's clip_observations=5.0 clamp during
    real contact, since no scale factor is applied to this channel anywhere
    in vtdexmanip_fork (unlike robot_qpos/robot_qvel/robot_dof_force/
    fingertip_force, which all get an explicit scale before concatenation)."""
    set_np_formatting()
    args = get_args()
    set_seed(args.models['seed'], args.models['torch_deterministic'])

    sim_params = parse_sim_params(args)
    env = parse_task(args, sim_params)
    task = env.task

    samples = []
    orig_fn = task.compute_sensor_obs

    def hooked(gt_continuous=False):
        out = orig_fn(gt_continuous=gt_continuous)
        if gt_continuous:
            samples.append(out.detach().float().reshape(-1).cpu().numpy())
        return out

    task.compute_sensor_obs = hooked

    logger = DataLog()
    assert os.path.isdir(args.logger_dir)
    logger.log_kv('model', f'{os.path.basename(args.resume_model)[:-3]}')
    sarl = process_sarl(args, env, args.models, args.logger_dir)
    sarl.eval(logger, max_trajs=max_trajs, record_video=False)

    if not samples:
        print("[tactile-diag] NO gt_continuous samples were captured -- obs_type on this "
              "checkpoint's task is not TacGT, or compute_sensor_obs was never called "
              "with gt_continuous=True. Nothing to report.")
        return

    all_vals = np.concatenate(samples)
    nonzero = all_vals[all_vals > 0]
    print(f"[tactile-diag] task={args.task} n_samples={all_vals.size} "
          f"n_steps={len(samples)} n_nonzero={nonzero.size} ({100*nonzero.size/all_vals.size:.2f}% active)")
    print(f"[tactile-diag] ALL   min={all_vals.min():.6f} max={all_vals.max():.6f} mean={all_vals.mean():.6f}")
    if nonzero.size:
        pcts = [50, 75, 90, 95, 99, 99.9, 100]
        vals = np.percentile(nonzero, pcts)
        pct_str = " ".join(f"p{p}={v:.4f}" for p, v in zip(pcts, vals))
        print(f"[tactile-diag] NONZERO-ONLY (excludes no-contact steps): {pct_str}")
    for thresh in (0.01, 0.1, 1.0, 5.0, 10.0):
        frac = (all_vals > thresh).mean()
        print(f"[tactile-diag] frac(raw > {thresh}) = {frac:.4%}")
    print(f"[tactile-diag] clip_observations for this task = "
          f"{env.task.cfg['env'].get('clip_observations', 'unknown')}")


if __name__ == '__main__':
    diag_tactile_scale()
