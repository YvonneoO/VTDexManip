#!/usr/bin/env python3
"""One-off check: confirm the real held object's trajectory position is NOT
sitting at the goal-marker's hide position (would indicate the wrong actor
got teleported)."""
import sys

import numpy as np

d = np.load(sys.argv[1])
pose = d["object_pose"]
print(f"object_pose shape={pose.shape}")
print(f"object_pose[0][:3]={pose[0][:3]}")
print(f"object_pose[-1][:3]={pose[-1][:3]}")
