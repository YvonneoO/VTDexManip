#!/usr/bin/env python3
"""Re-evaluate every periodic checkpoint from a completed VTDexManip run with the
paper's own held-out protocol (eval_agent.eval_policy -> PPO.eval, deterministic
act_inference, 100 episodes/object), to reconstruct a success-rate-vs-iteration
curve that isn't the in-training rolling success buffer. Mirrors hora_fork's
sweep_checkpoint_eval.py (same problem: in-training numbers aren't comparable to
a held-out eval, so build the held-out curve once instead of trusting the log).

Each checkpoint gets its own subprocess (fresh env/model construction, same as a
normal eval_agent.py --test invocation) since VTDexManip's env/model config is
tied to --task at process start.

Usage: python sweep_checkpoint_eval.py <ckpt_dir> <task_model> <out_csv> [--seed N] [--max_trajs N]
  ckpt_dir: e.g. runs/BottleCap/bottle_cap/bottle_cap-base/seed111/checkpoint
  task_model: e.g. bottle_cap-base
"""
import argparse
import re
import csv
import subprocess
import sys
from pathlib import Path

CKPT_RE = re.compile(r"model_(\d+)\.pt")
SUCCESS_RE = re.compile(r"Mean success:\s*([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("task_model")
    ap.add_argument("out_csv")
    ap.add_argument("--seed", default="111")
    ap.add_argument("--max_trajs", type=int, default=100)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    vtdex_root = Path(__file__).resolve().parent
    python_bin = sys.executable

    ckpts = []
    for f in sorted(ckpt_dir.glob("model_*.pt")):
        m = CKPT_RE.match(f.name)
        if m:
            ckpts.append((int(m.group(1)), f))
    ckpts.sort(key=lambda t: t[0])

    print(f"Found {len(ckpts)} periodic checkpoints in {ckpt_dir}", flush=True)
    rows = []
    for it, ckpt_path in ckpts:
        cmd = [
            python_bin, str(vtdex_root / "eval_agent.py"),
            "--task", args.task_model,
            "--rl_device", "cuda:0",
            "--resume_model", str(ckpt_path),
            "--test",
            "--seed", args.seed,
            "--headless",
        ]
        print(f"--- iter {it} ({ckpt_path.name}) ---", flush=True)
        proc = subprocess.run(cmd, cwd=str(vtdex_root), capture_output=True, text=True, timeout=1200)
        out = proc.stdout + proc.stderr
        matches = SUCCESS_RE.findall(out)
        if not matches:
            print(f"  NO SUCCESS LINE FOUND (returncode={proc.returncode}); tail:")
            print("  " + "\n  ".join(out.splitlines()[-20:]))
            success_rate = None
        else:
            success_rate = float(matches[-1]) * 100.0
        print(f"  success_rate={success_rate}", flush=True)
        rows.append({"iteration": it, "held_out_success_pct": success_rate})
        with open(args.out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["iteration", "held_out_success_pct"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"DONE. Wrote {len(rows)} rows to {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
