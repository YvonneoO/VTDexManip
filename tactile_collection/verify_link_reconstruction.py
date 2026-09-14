#!/usr/bin/env python3
"""One-off check: can raw_per_link_force_n be reconstructed from the 217-
taxel force_grid_n alone (for OLD collected episodes that predate the
raw_per_link_force_n field)? Sums force_grid_n over each anatomical link's
own cell set (egotouch_taxels.py's _layout()) and compares against the
actual raw_per_link_force_n saved for the same episode, since
EgoTouchTaxelMapper.project() allocates each contact's force via Gaussian
weights that sum to 1 across exactly that contact's own body's cell set --
so summing back over that same cell set should recover the exact source
value, MODULO one known caveat: the "palm" and "lfmetacarpal" anatomical
groups overlap in raw cell-space (see _layout()'s palm definition, which
doesn't exclude lfmetacarpal's columns), so summing "palm's own cells"
double-counts any simultaneous lfmetacarpal contact. Every other one of
the 17 groups is disjoint from every other, so reconstruction should be
exact for those.

Usage: python tactile_collection/verify_link_reconstruction.py <pressure_grids.npz>
"""
import sys

import numpy as np

from egotouch_taxels import _load_valid_cells, _layout

MAPPING_PATH = "tactile_collection/assets/pressure_position_mapping_right.json"


def main():
    path = sys.argv[1]
    d = np.load(path)
    valid = _load_valid_cells(MAPPING_PATH)
    groups = _layout("right", valid)

    raw = d["raw_per_link_force_n"]
    names = [str(n) for n in d["raw_link_names"]]
    fg = d["force_grid_n"]

    print(f"episode: {path}, frames: {raw.shape[0]}")
    max_diff_by_link = {name: 0.0 for name in names}
    for t in range(raw.shape[0]):
        for i, name in enumerate(names):
            cells = groups[name]
            recon = sum(fg[t, r, c] for r, c in cells)
            diff = abs(float(raw[t, i]) - float(recon))
            max_diff_by_link[name] = max(max_diff_by_link[name], diff)

    print("max |raw - reconstructed| over all frames, per link:")
    for name, diff in sorted(max_diff_by_link.items(), key=lambda kv: -kv[1]):
        print(f"  {name:16s} max_diff={diff:.6f} N")


if __name__ == "__main__":
    main()
