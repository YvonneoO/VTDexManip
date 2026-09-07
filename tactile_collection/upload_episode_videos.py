#!/usr/bin/env python3
"""Upload one or more episode side-by-side check videos (from
make_episode_videos.py) to HF, under the SAME repo/account convention as
upload_ckpt_and_rollout.py (qqyang/hora-v4-shadow-tennis, HF_TOKEN from
yqq/env.sh) -- a new path prefix
tactile_sr_ablation/vtdexmanip/data_collection_check/<label>/ since these are
a spot-check of collect_*_tactile.py's OWN output (RGB viewpoint + tactile
sanity), not a PPO policy's own rollout_check (that's the existing
tactile_sr_ablation/vtdexmanip/rollout_check/ prefix, unrelated purpose).

Usage:
    python tactile_collection/upload_episode_videos.py \
        --label slide-t_scr_gt_smoke5 \
        collected_slide_smoke5/successful_episodes/episode_000000/rgb_tactile_side_by_side.mp4 \
        collected_slide_smoke5/successful_episodes/episode_000001/rgb_tactile_side_by_side.mp4 \
        ...
"""
import argparse
import os

from huggingface_hub import HfApi

REPO_ID = "qqyang/hora-v4-shadow-tennis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--label", required=True, help="e.g. slide-t_scr_gt_smoke5")
    args = ap.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    for i, v in enumerate(args.videos):
        ep_name = os.path.basename(os.path.dirname(v))  # e.g. episode_000000
        filename = f"{ep_name}.mp4" if ep_name.startswith("episode_") else f"video_{i:02d}.mp4"
        path_in_repo = f"tactile_sr_ablation/vtdexmanip/data_collection_check/{args.label}/{filename}"
        api.upload_file(path_or_fileobj=v, path_in_repo=path_in_repo, repo_id=REPO_ID, repo_type="model")
        print(f"UPLOADED: {path_in_repo}")


if __name__ == "__main__":
    main()
