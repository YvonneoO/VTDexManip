#!/usr/bin/env python3
"""Build rgb.mp4 + tactile.mp4 + rgb_tactile_side_by_side.mp4 for one or more
already-collected episode directories (collect_bottle_cap_tactile.py /
collect_slide_tactile.py output) -- these collectors save raw frame_%06d.png
+ pressure_grids.npz but never render a check video, unlike bidexhands'
rollout_tactile_rgb_chest.py which does this inline during collection.

Usage:
    python tactile_collection/make_episode_videos.py \
        collected_slide_smoke5/successful_episodes/episode_000000 \
        collected_slide_smoke5/successful_episodes/episode_000001 \
        ... \
        --title "Lever Sliding (t_scr_gt, tactile-scale-fixed)"
"""
import argparse
import os
import subprocess
import sys


def encode_rgb(frames_dir, output_mp4, fps):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", output_mp4,
    ])


def render_tactile(script_dir, pressure_path, output_mp4, fps, stride, title):
    subprocess.check_call([
        sys.executable, os.path.join(script_dir, "render_tactile_single.py"),
        pressure_path, output_mp4, "--fps", str(fps), "--stride", str(stride),
        "--title", title,
    ])


def compose_side_by_side(rgb_mp4, tactile_mp4, output_mp4):
    subprocess.check_call([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", rgb_mp4, "-i", tactile_mp4,
        "-filter_complex",
        "[0:v]scale=600:-2,pad=600:600:(ow-iw)/2:(oh-ih)/2,setsar=1[rgb];"
        "[1:v]scale=600:600,setsar=1[tactile];"
        "[rgb][tactile]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        output_mp4,
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dirs", nargs="+")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--title", default="Tactile rollout")
    ap.add_argument("--keep_component_videos", action="store_true",
                     help="keep rgb.mp4/tactile.mp4 alongside the side-by-side compose "
                          "(default: only the composed video is kept)")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for ep_dir in args.episode_dirs:
        frames_dir = os.path.join(ep_dir, "rgb_frames")
        pressure_path = os.path.join(ep_dir, "pressure_grids.npz")
        if not os.path.isdir(frames_dir) or not os.path.isfile(pressure_path):
            print(f"[skip] {ep_dir}: missing rgb_frames/ or pressure_grids.npz", flush=True)
            continue

        rgb_mp4 = os.path.join(ep_dir, "rgb.mp4")
        tactile_mp4 = os.path.join(ep_dir, "tactile.mp4")
        paired_mp4 = os.path.join(ep_dir, "rgb_tactile_side_by_side.mp4")

        print(f"[render] {ep_dir}", flush=True)
        encode_rgb(frames_dir, rgb_mp4, args.fps)
        render_tactile(script_dir, pressure_path, tactile_mp4, args.fps, args.stride, args.title)
        compose_side_by_side(rgb_mp4, tactile_mp4, paired_mp4)

        if not args.keep_component_videos:
            os.unlink(rgb_mp4)
            os.unlink(tactile_mp4)

        print(f"[done] {ep_dir} -> {paired_mp4}", flush=True)


if __name__ == "__main__":
    main()
