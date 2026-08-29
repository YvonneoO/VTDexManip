import os

from utils.logger import DataLog
from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args
from model.process_sarl import process_sarl

import torch  # must come after the isaacgym-importing modules above


def rollout_record():
    """Same as eval_agent.py but with record_video=True -- eval_agent.eval_policy()
    hardcodes record_video=False, so this is a thin fork of it, not an override flag
    (the upstream function doesn't expose one)."""
    set_np_formatting()
    args = get_args()
    set_seed(args.models['seed'], args.models['torch_deterministic'])

    sim_params = parse_sim_params(args)
    env = parse_task(args, sim_params)
    logger = DataLog()
    assert os.path.isdir(args.logger_dir)

    # base_task.py only allocates task.img_buf when obs_type is VisOnly/VisTac (a
    # memory-saving skip for non-visual policies) -- but the camera sensors
    # themselves are always created (enable_camera_sensors=True is hardcoded at
    # task construction, independent of obs_type), so compute_pixel_obs() works
    # fine for ANY model once img_buf exists. Recording a debug rollout video is
    # a sim-visualization concern, not a policy-observation one, so allocate it
    # here regardless of what the loaded policy actually consumes.
    task = env.task
    if not hasattr(task, "img_buf"):
        task.img_buf = torch.zeros(
            (task.num_envs, 224, 224, 3), device=task.device, dtype=torch.float)
        print(f"[rollout_record] task.img_buf was not allocated for this obs_type "
              f"-- created it manually ({tuple(task.img_buf.shape)}) for video recording only.")

    logger.log_kv('model', f'{os.path.basename(args.resume_model)[:-3]}')
    sarl = process_sarl(args, env, args.models, args.logger_dir)
    # max_trajs=1 -> per-env break threshold is low so the job exits promptly once
    # the one video segment (max_episode_length frames) has been written.
    sarl.eval(logger, max_trajs=1, record_video=True)

    print(args.logger_dir)


if __name__ == '__main__':
    rollout_record()
