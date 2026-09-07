"""Standalone HF upload for a VTDexManip checkpoint + its rollout-check videos.

Mirrors vision_eval.sbatch's checkpoint-upload convention (same repo_id, same
tactile_sr_ablation/vtdexmanip/checkpoints/<task>/ naming, one HF folder per task
so multi-arm tasks don't dump everything flat) so checkpoints land in the same
place whether uploaded via a completed eval job or standalone here. Adds a
tactile_sr_ablation/vtdexmanip/rollout_check/<task>-<model>_seed<seed>/ path for
videos, extending the earlier vtdexmanip/base_rollout_check/ convention to cover
every arm, not just base.

Usage: python upload_ckpt_and_rollout.py --task screw_faucet --model t_scr_gt
       python upload_ckpt_and_rollout.py --task screw_faucet --model t_scr_gt --delete_others
"""
import argparse
import glob
import os

from huggingface_hub import HfApi

REPO_ID = "qqyang/hora-v4-shadow-tennis"
VTDEX_ROOT_DEFAULT = "/scratch/project/prj-02-phai-lab/yqq/VTDexManip"


def latest_checkpoint(vtdex_root, task, model, seed):
    ckpt_glob = os.path.join(vtdex_root, "runs", "*", task, f"{task}-{model}", f"seed{seed}", "checkpoint", "model_*.pt")
    ckpts = glob.glob(ckpt_glob)
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(os.path.basename(p).rsplit("_", 1)[-1].split(".")[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", default="111")
    ap.add_argument("--vtdex_root", default=VTDEX_ROOT_DEFAULT)
    ap.add_argument("--skip_ckpt", action="store_true")
    ap.add_argument("--skip_video", action="store_true")
    ap.add_argument("--delete_others", action="store_true",
                     help="after a successful checkpoint upload, delete every other "
                          "model_*.pt in the same checkpoint dir (VISION scratch is "
                          "near quota) -- the just-uploaded latest one is kept, and "
                          "it's already backed up on HF at this point")
    args = ap.parse_args()

    task, model, seed = args.task, args.model, args.seed
    ckpt = latest_checkpoint(args.vtdex_root, task, model, seed)
    if ckpt is None:
        raise SystemExit(f"no checkpoint found for {task}-{model} seed{seed}")

    api = HfApi(token=os.environ["HF_TOKEN"])

    if not args.skip_ckpt:
        hf_ckpt_name = f"{task}-{model}_seed{seed}_{os.path.basename(ckpt)}"
        api.upload_file(
            path_or_fileobj=ckpt,
            path_in_repo=f"tactile_sr_ablation/vtdexmanip/checkpoints/{task}/{hf_ckpt_name}",
            repo_id=REPO_ID, repo_type="model",
        )
        print(f"UPLOADED checkpoint: {task}/{hf_ckpt_name}")

        if args.delete_others:
            ckpt_dir = os.path.dirname(ckpt)
            siblings = [p for p in glob.glob(os.path.join(ckpt_dir, "model_*.pt")) if p != ckpt]
            for p in siblings:
                os.remove(p)
            if siblings:
                print(f"DELETED {len(siblings)} older checkpoint(s) in {ckpt_dir}, kept {os.path.basename(ckpt)}")
            else:
                print(f"no older checkpoints to delete in {ckpt_dir}")
    else:
        print(f"skip_ckpt: {os.path.basename(ckpt)}")

    if not args.skip_video:
        run_dir = os.path.dirname(os.path.dirname(ckpt))  # .../seed<seed>
        video_dir = os.path.join(run_dir, "videos")
        vids = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        if not vids:
            print(f"WARNING: no videos found at {video_dir} -- run vision_rollout.sbatch first")
        for v in vids:
            path_in_repo = f"tactile_sr_ablation/vtdexmanip/rollout_check/{task}-{model}_seed{seed}/{os.path.basename(v)}"
            api.upload_file(path_or_fileobj=v, path_in_repo=path_in_repo, repo_id=REPO_ID, repo_type="model")
            print(f"UPLOADED video: {path_in_repo}")


if __name__ == "__main__":
    main()
