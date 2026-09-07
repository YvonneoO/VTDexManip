#!/usr/bin/env python3
"""Render a single-hand EgoTouch-layout pressure grid as a video.

Single-hand variant of bidexhands' render_tactile.py (dexteroushands_fork/
bidexhands/tactile_collection/render_tactile.py) -- that script expects
bimanual `left_pressure_grid`/`right_pressure_grid` keys; VTDexManip tasks
(BottleCap, Lever Sliding, ...) are single-handed and collect_*_tactile.py
saves a single `pressure_grid` key instead. Same rendering conventions
(turbo colormap, dark background, shared colorbar) otherwise.
"""

import argparse
import os
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pressure_npz")
    parser.add_argument("output_mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--title", default="Tactile rollout")
    args = parser.parse_args()

    data = np.load(args.pressure_npz)
    grid = data["pressure_grid"]  # (T, 21, 21) Pa
    finite = grid[np.isfinite(grid)]
    positive = finite[finite > 0]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    vmax = max(vmax, 1.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_mp4)) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tactile_frames_") as frame_dir:
        frame_index = 0
        for step in range(0, grid.shape[0], max(1, args.stride)):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor="#101321")
            ax.set_facecolor("#101321")
            image = ax.imshow(grid[step], cmap="turbo", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title("Right Hand Pressure", color="white")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Pressure (Pa)", color="white")
            cbar.ax.tick_params(colors="white")
            fig.suptitle("{} — step {}".format(args.title, step), color="white")
            fig.savefig(
                os.path.join(frame_dir, "frame_{:06d}.png".format(frame_index)),
                dpi=120,
                facecolor=fig.get_facecolor(),
            )
            plt.close(fig)
            frame_index += 1
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", os.path.join(frame_dir, "frame_%06d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "20", args.output_mp4,
        ])
    print("Wrote {} (vmax {:.6g} Pa)".format(args.output_mp4, vmax))


if __name__ == "__main__":
    main()
