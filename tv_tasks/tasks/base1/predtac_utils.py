#!/usr/bin/env python3
"""Online Pred-Tac support for VTDexManip's bimanual tasks (handover.py first).

project_world_to_pixel is VENDORED VERBATIM from the DexterousHands repo's
bidexhands/tactile_collection/gt_pose_crop.py (do not fix bugs here without
fixing there too, or vice versa) -- derived as the algebraic inverse of Isaac
Gym's own point-cloud deprojection formula (validated there via a 2.8e-16
round-trip error against bidexhands/tasks/shadow_hand_point_cloud.py's own
depth_image_to_point_cloud_GPU). It is pure numpy, camera-convention-only, so
it needs no change to work against a DIFFERENT Isaac Gym task/repo -- only
the caller (crop_boxes_for_env below) is new, adapted to THIS codebase's own
camera setup:

  - VTDexManip's own camera is STATIC per env (fixed eye/lookat offset from a
    fixed table-center point, set once in ShadowHandBase._load_cameras --
    never repositioned like bidexhands' dynamic chest camera), so its view/
    proj matrices can be queried once and cached, no per-tick re-positioning.
  - rigid_body_states-derived points (fingertip_pos etc.) are local-per-env,
    same as bidexhands -- confirmed by ShadowHandBase._load_cameras already
    storing self.env_origin[env_id] = gym.get_env_origin(env_ptr) explicitly,
    the same quantity bidexhands' own multi-env camera bug fix needed (see
    gt_pose_crop.py's history for the debugging story -- add env_origin to
    points before projecting with the camera's own (global-frame) matrices,
    do NOT subtract it and do NOT leave it out).
"""
import numpy as np


def project_world_to_pixel(points_world, view_matrix, proj_matrix, width, height):
    """(N,3) world points -> (u, v, in_front), each shape (N,). See module
    docstring -- vendored from gt_pose_crop.py, do not diverge silently."""
    points_world = np.asarray(points_world, dtype=np.float64)
    n = points_world.shape[0]
    homo = np.concatenate([points_world, np.ones((n, 1), dtype=np.float64)], axis=1)
    cam = homo @ np.asarray(view_matrix, dtype=np.float64)
    x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]

    fu = 2.0 / float(proj_matrix[0, 0])
    fv = 2.0 / float(proj_matrix[1, 1])
    center_u = width / 2.0
    center_v = height / 2.0

    in_front = z < 0.0
    safe_z = np.where(in_front, z, -1.0)
    u = center_u - x * width / (safe_z * fu)
    v = center_v + y * height / (safe_z * fv)
    return u, v, in_front


def _hand_box_from_points(points_world_local, env_origin, view_matrix, proj_matrix, width, height,
                           pad_frac=0.30, min_pad_px=16.0):
    """points_world_local: (K,3) numpy, env-local frame (e.g. fingertip_pos
    for one hand in one env). pad_frac is larger than gt_pose_crop.py's
    default (0.15): fingertip positions alone (no palm/wrist points, unlike
    bidexhands' broader per-link set) undercover the hand, so pad more to
    compensate. Returns None if no point lands in front of the camera or the
    resulting box is degenerate after clamping to the image."""
    pts_global = points_world_local + np.asarray(env_origin, dtype=np.float64)[None, :]
    u, v, in_front = project_world_to_pixel(pts_global, view_matrix, proj_matrix, width, height)
    if not np.any(in_front):
        return None
    u_k, v_k = u[in_front], v[in_front]
    x1, x2 = float(u_k.min()), float(u_k.max())
    y1, y2 = float(v_k.min()), float(v_k.max())
    pad_x = max((x2 - x1) * pad_frac, min_pad_px)
    pad_y = max((y2 - y1) * pad_frac, min_pad_px)
    x1 = float(np.clip(x1 - pad_x, 0.0, width))
    x2 = float(np.clip(x2 + pad_x, 0.0, width))
    y1 = float(np.clip(y1 - pad_y, 0.0, height))
    y2 = float(np.clip(y2 + pad_y, 0.0, height))
    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
        return None
    return [x1, y1, x2, y2]


def crop_boxes_for_env(right_fingertip_pts, left_fingertip_pts, env_origin,
                        view_matrix, proj_matrix, width, height):
    """Both hands for one env. right_fingertip_pts/left_fingertip_pts: (5,3)
    numpy each (env-local). Returns a dict shaped like bidexhands'
    gt_pose_crop.gt_hand_boxes: {"left": {"box":[x1,y1,x2,y2]}, "right": {...}}
    (only keys with a valid box), so predtac_client.py's submit() can consume
    it unmodified."""
    sides = {}
    right_box = _hand_box_from_points(right_fingertip_pts, env_origin, view_matrix, proj_matrix, width, height)
    if right_box is not None:
        sides["right"] = {"box": right_box}
    left_box = _hand_box_from_points(left_fingertip_pts, env_origin, view_matrix, proj_matrix, width, height)
    if left_box is not None:
        sides["left"] = {"box": left_box}
    return sides
