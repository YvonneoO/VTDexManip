#!/usr/bin/env python3
"""Check whether a PPO training log's mid-training success-rate peak is a genuine sustained
plateau or a single noisy spike -- pairs each "Learning iteration N/TOTAL" line with the
"Mean success rate: X" line that follows it, then reports count/mean/min/max within a window of
iterations centered on a given target, plus the raw sequence so a real plateau (values clustered
near the mean) can be told apart from a spike (one high value surrounded by much lower ones).

Usage: python analyze_success_band.py <logfile> <target_iter> [<half_width>]
  (half_width default 100 -> window is [target-100, target+100])
"""
import re
import sys


def main():
    logfile = sys.argv[1]
    target = int(sys.argv[2])
    half_width = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    lo, hi = target - half_width, target + half_width

    pairs = []
    cur_iter = None
    with open(logfile) as f:
        for line in f:
            m = re.search(r"Learning iteration (\d+)/\d+", line)
            if m:
                cur_iter = int(m.group(1))
                continue
            m = re.search(r"Mean success rate: ([\d.]+)", line)
            if m and cur_iter is not None:
                pairs.append((cur_iter, float(m.group(1))))
                cur_iter = None

    window = [(it, sr) for it, sr in pairs if lo <= it <= hi]
    if not window:
        print(f"no data in window [{lo}, {hi}]")
        return

    vals = [sr for _, sr in window]
    print(f"window=[{lo},{hi}] n={len(vals)} mean={sum(vals)/len(vals):.2f} "
          f"min={min(vals):.2f} max={max(vals):.2f}")
    stride = max(1, len(window) // 30)
    print("sampled sequence (iter: success%):")
    for it, sr in window[::stride]:
        print(f"  {it}: {sr:.2f}")


if __name__ == "__main__":
    main()
