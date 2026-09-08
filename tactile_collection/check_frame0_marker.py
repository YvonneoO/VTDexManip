#!/usr/bin/env python3
"""One-off check: does frame_000000.png still contain the goal-marker's
lavender/purple color signature (high R+B, lower G) anywhere in the frame?
Compares frame 0 against a later frame as a control -- if frame 0 has a
purple-pixel count similar to (not much higher than) the control frame, the
marker is gone from frame 0 too, not just from later frames.
"""
import sys

import numpy as np
from PIL import Image


def purple_pixel_count(png_path):
    img = np.asarray(Image.open(png_path).convert("RGB")).astype(np.int32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    # lavender/purple: R and B both clearly above G
    mask = (r > g + 20) & (b > g + 20) & (r > 100) & (b > 100)
    return int(mask.sum())


def main():
    ep_dir = sys.argv[1]
    frame0 = f"{ep_dir}/rgb_frames/frame_000000.png"
    frame_later = f"{ep_dir}/rgb_frames/frame_000030.png"
    c0 = purple_pixel_count(frame0)
    c_later = purple_pixel_count(frame_later)
    print(f"frame_000000.png purple-pixel-count={c0}")
    print(f"frame_000030.png purple-pixel-count={c_later}")


if __name__ == "__main__":
    main()
