#!/usr/bin/env python3
"""Zero-shot Pred-Tac quality check for BottleCap: compare v2-dit's predicted
tactile (pred_pressure_grids.npz, from Ego2Contact's infer_pred_tactile_offline.py,
using the checkpoint that was ONLY ever sim-finetuned on bidexhands data --
BottleCap/VTDexManip was never part of that finetune) against BottleCap's own
ground-truth tactile (pressure_grids.npz's "pressure_grid" key).

This is a genuine zero-shot cross-embodiment/cross-simulator test, not a
finetune-leakage question -- the point is to find out whether the existing
1000 collected episodes are good enough to use for Pred-Tac + offline IQL
as-is, or whether a BottleCap-specific v2-dit finetune (on a SEPARATE held-out
batch of newly-collected episodes) is needed first.

Metrics (per episode, then averaged):
  - pearson_r: correlation between predicted and GT pressure values across all
    frames+cells -- a coarse "is there any real relationship at all" signal.
  - cIoU@0.05: contact IoU at a 5%-of-task_vmax threshold (this project's
    established eval convention elsewhere, e.g. Ego2Contact's TA baseline
    protocol) -- binarize both grids at 0.05*task_vmax, IoU of the two
    contact masks per frame, averaged over frames then episodes.

infer_pred_tactile_offline.py always predicts BOTH left_pressure_grid and
right_pressure_grid (bimanual model architecture) even for a single-hand
episode -- BottleCap is single-handed and always the RIGHT hand (see
collect_bottle_cap_tactile.py's use of pressure_position_mapping_right.json),
so this script compares against right_pressure_grid only and ignores the
left slot (expected to be near-zero/meaningless, no real left hand exists in
this task).

Usage:
    python check_bottlecap_predtac_quality.py --manifest bottlecap_manifest_smoke12.json \
        --task_vmax <value from compute_task_vmax.py>
"""
import argparse
import json
import os

import numpy as np


def contact_iou(pred, gt, thresh):
    pred_mask = pred > thresh
    gt_mask = gt > thresh
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return None  # no contact predicted OR present this frame -- not informative, skip
    inter = np.logical_and(pred_mask, gt_mask).sum()
    return float(inter) / float(union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task_vmax", type=float, required=True)
    ap.add_argument("--ciou_frac", type=float, default=0.05,
                     help="contact threshold as a fraction of task_vmax (0.05 = this "
                          "project's established cIoU@.05 convention)")
    ap.add_argument("--pred_file", default="pred_pressure_grids.npz")
    ap.add_argument("--split", default="train", choices=["train", "val", "all"])
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    entries = []
    if args.split in ("train", "all"):
        entries += manifest["train"]
    if args.split in ("val", "all"):
        entries += manifest["val"]

    thresh = args.ciou_frac * args.task_vmax
    per_episode = []
    skipped = []

    for ep_dir in entries:
        pred_path = os.path.join(ep_dir, args.pred_file)
        gt_path = os.path.join(ep_dir, "pressure_grids.npz")
        if not os.path.isfile(pred_path):
            skipped.append((ep_dir, "no pred_pressure_grids.npz"))
            continue
        with np.load(pred_path) as pred_npz, np.load(gt_path) as gt_npz:
            pred = np.nan_to_num(np.asarray(pred_npz["right_pressure_grid"], dtype=np.float32))
            gt = np.nan_to_num(np.asarray(gt_npz["pressure_grid"], dtype=np.float32))
        if pred.shape != gt.shape:
            skipped.append((ep_dir, f"shape mismatch pred={pred.shape} gt={gt.shape}"))
            continue

        r = np.corrcoef(pred.reshape(-1), gt.reshape(-1))[0, 1]
        ious = [contact_iou(pred[t], gt[t], thresh) for t in range(pred.shape[0])]
        ious = [x for x in ious if x is not None]
        ciou = float(np.mean(ious)) if ious else None

        per_episode.append({
            "episode": ep_dir, "pearson_r": float(r) if np.isfinite(r) else None,
            "ciou_at_frac": ciou, "n_frames_with_contact": len(ious), "n_frames_total": pred.shape[0],
        })
        print(f"[{os.path.basename(ep_dir)}] pearson_r={r:.4f} "
              f"cIoU@{args.ciou_frac}={ciou if ciou is None else f'{ciou:.4f}'} "
              f"({len(ious)}/{pred.shape[0]} frames had any contact)", flush=True)

    if skipped:
        print(f"[skip] {len(skipped)} episodes skipped:", flush=True)
        for ep_dir, reason in skipped:
            print(f"    {ep_dir}: {reason}", flush=True)

    valid_r = [e["pearson_r"] for e in per_episode if e["pearson_r"] is not None]
    valid_ciou = [e["ciou_at_frac"] for e in per_episode if e["ciou_at_frac"] is not None]
    print(f"\n[summary] n_episodes={len(per_episode)}  n_skipped={len(skipped)}", flush=True)
    if valid_r:
        print(f"[summary] mean pearson_r = {np.mean(valid_r):.4f} "
              f"(min={np.min(valid_r):.4f}, max={np.max(valid_r):.4f})", flush=True)
    if valid_ciou:
        print(f"[summary] mean cIoU@{args.ciou_frac} = {np.mean(valid_ciou):.4f} "
              f"(min={np.min(valid_ciou):.4f}, max={np.max(valid_ciou):.4f}) "
              f"over {len(valid_ciou)} episodes with at least one contact frame", flush=True)
    else:
        print("[summary] no episode had any GT+pred contact overlap to compute cIoU from", flush=True)


if __name__ == "__main__":
    main()
