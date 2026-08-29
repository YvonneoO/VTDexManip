import os

from utils.logger import DataLog
from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args
from model.process_sarl import process_sarl


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

    logger.log_kv('model', f'{os.path.basename(args.resume_model)[:-3]}')
    sarl = process_sarl(args, env, args.models, args.logger_dir)
    # max_trajs=1 -> per-env break threshold is low so the job exits promptly once
    # the one video segment (max_episode_length frames) has been written.
    sarl.eval(logger, max_trajs=1, record_video=True)

    print(args.logger_dir)


if __name__ == '__main__':
    rollout_record()
