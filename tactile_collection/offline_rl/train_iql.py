#!/usr/bin/env python3
"""Train d3rlpy's own IQL (Implicit Q-Learning) implementation on a BottleCap
Turning offline dataset built by build_mdp_dataset.py. Does NOT hand-implement
IQL -- uses d3rlpy.algos.IQLConfig directly (v2.x API, verified 2026-09-07
against the `master` branch source since this environment can't pip-install
d3rlpy to check locally):

  - `d3rlpy.dataset.ReplayBuffer.load(f, buffer)` is the v2.x load path for a
    file written by MDPDataset.dump() -- `f` an open BINARY file object, and
    `buffer` a fresh `InfiniteBuffer()` (load reconstructs episodes from the
    dumped HDF5 and refills whatever buffer instance you hand it).
  - `IQLConfig(...).create(device=...)` returns an `IQL` algo instance;
    `.fit(dataset, n_steps=..., n_steps_per_epoch=..., logger_adapter=...)`
    trains it and is responsible for calling create_impl() itself the first
    time (observation/action shapes come from the dataset) -- no separate
    build_with_dataset() call is required before fit().
  - Logging: `d3rlpy.logging.FileAdapterFactory` (CSV) and
    `TensorboardAdapterFactory` are always available. `d3rlpy.logging.
    WanDBAdapterFactory` DOES exist in v2.x (d3rlpy/logging/wandb_adapter.py)
    but its own __init__ raises ImportError if the `wandb` package isn't
    installed -- so it's only wired in here behind `--wandb` AND a local
    `import wandb` probe, never unconditionally (per the task spec: don't
    guess at an API that may not exist -- this one does exist, but its
    dependency might not).  Multiple adapters are combined with
    `d3rlpy.logging.CombineAdapterFactory([...])` (d3rlpy/logging/utils.py).
  - Final checkpoint: `IQL.save_model(path)` (network weights) is used for the
    "final policy checkpoint" the task asks for, since it's readonly-loadable
    without needing the original IQLConfig to reconstruct the algo wrapper for
    a bare inference/rollout use case; `IQL.save(path)` (full config+weights,
    resumable) is ALSO written since it costs nothing extra and is the more
    robust artifact if someone wants to resume training later.

Usage:
    python train_iql.py --dataset bottle_cap_p_gt_tac.h5 --out runs/iql_p_gt_tac \
        --n_steps 100000 --device cuda:0
"""
import os
import json
import argparse


def build_logger_adapter(out_dir, experiment_name, use_wandb, wandb_project):
    import d3rlpy

    adapters = [
        d3rlpy.logging.FileAdapterFactory(root_dir=os.path.join(out_dir, "logs")),
        d3rlpy.logging.TensorboardAdapterFactory(root_dir=os.path.join(out_dir, "tensorboard")),
    ]
    if use_wandb:
        try:
            import wandb  # noqa: F401 -- probe only; WanDBAdapterFactory itself re-imports it
            adapters.append(d3rlpy.logging.WanDBAdapterFactory(project=wandb_project))
            print("[logging] wandb available -- adding WanDBAdapterFactory", flush=True)
        except ImportError:
            print("[logging][warn] --wandb was set but `wandb` is not installed "
                  "in this env -- skipping wandb logging, CSV+Tensorboard only", flush=True)
    return d3rlpy.logging.CombineAdapterFactory(adapters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help=".h5 file from build_mdp_dataset.py")
    ap.add_argument("--out", required=True, help="output dir for checkpoints + logs")
    ap.add_argument("--n_steps", type=int, default=100_000)
    ap.add_argument("--n_steps_per_epoch", type=int, default=1_000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--actor_learning_rate", type=float, default=3e-4)
    ap.add_argument("--critic_learning_rate", type=float, default=3e-4)
    ap.add_argument("--expectile", type=float, default=0.7,
                     help="IQL expectile regression tau (d3rlpy default 0.7)")
    ap.add_argument("--weight_temp", type=float, default=3.0,
                     help="IQL advantage-weighting inverse temperature (d3rlpy default 3.0)")
    ap.add_argument("--device", default="cuda:0", help="e.g. cuda:0 or cpu")
    ap.add_argument("--save_interval", type=int, default=10,
                     help="save an epoch checkpoint every N epochs (d3rlpy fit() default is 1; "
                          "10 keeps disk usage down over a 100k-step run)")
    ap.add_argument("--experiment_name", default=None,
                     help="defaults to the dataset file's basename")
    ap.add_argument("--wandb", action="store_true",
                     help="also log to Weights & Biases if the `wandb` package is installed "
                          "(WanDBAdapterFactory exists in d3rlpy v2.x but needs wandb itself)")
    ap.add_argument("--wandb_project", default="vtdexmanip-offline-iql")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    experiment_name = args.experiment_name or os.path.splitext(os.path.basename(args.dataset))[0]

    import d3rlpy  # deferred so --help works without the dependency installed

    with open(args.dataset, "rb") as f:
        dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())
    print(f"[data] loaded {args.dataset}: {dataset.size()} episodes, "
          f"{dataset.transition_count} transitions", flush=True)

    iql_config = d3rlpy.algos.IQLConfig(
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        batch_size=args.batch_size,
        expectile=args.expectile,
        weight_temp=args.weight_temp,
    )
    iql = iql_config.create(device=args.device)

    logger_adapter = build_logger_adapter(args.out, experiment_name, args.wandb, args.wandb_project)

    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    iql.fit(
        dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        experiment_name=experiment_name,
        logger_adapter=logger_adapter,
        save_interval=args.save_interval,
        show_progress=True,
    )

    # Final policy checkpoint. save_model() = weights only (portable, cheap to
    # load for eval/rollout); save() = weights + config (resumable training).
    iql.save_model(os.path.join(args.out, "final_model.pt"))
    iql.save(os.path.join(args.out, "final_full.d3"))
    print(f"[done] wrote final checkpoints to {args.out}", flush=True)


if __name__ == "__main__":
    main()
