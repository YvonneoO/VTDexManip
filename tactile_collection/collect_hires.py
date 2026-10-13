"""High-resolution wrapper around collect_slide_tactile.py / collect_handover_tactile.py / collect_bottle_cap_tactile.py.

Renders the task camera at HIRES_W x HIRES_H (default 1280 x 1280, same field of view and pose as the 224 x 224 training camera) and
makes the collection script save those full-resolution frames instead of the 224 x 224 policy image. The GT-tactile policies
(t_scr_gt / t_scr_gt_priv) take no image input, so the policy's behaviour is unchanged. Nothing in the tracked task code is modified:
the camera size is set on the env cfg before the task is built and compute_pixel_obs is replaced on the base classes at run time.

    python tactile_collection/collect_hires.py slide --task slide-t_scr_gt --rl_device cuda:0 --resume_model <ckpt> --test --seed 111 \
        --headless --target_successes 2 --out_dir <dir>
(everything after the first word is passed to the collection script unchanged; first word = slide | handover | bottle_cap)
"""
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W = int(os.environ.get("HIRES_W", "1280"))
H = int(os.environ.get("HIRES_H", "1280"))
which = sys.argv.pop(1)
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"collect_{which}_tactile.py")
assert os.path.isfile(script), script

import utils.hydra_utils as hu  # noqa: E402

_orig_get_args = hu.get_args


def _get_args(*a, **k):
    args = _orig_get_args(*a, **k)
    args.task_envs["env"]["img_w"] = W
    args.task_envs["env"]["img_h"] = H
    print(f"[hires] task camera {W}x{H}", flush=True)
    return args


hu.get_args = _get_args


def _hires_compute_pixel_obs(self):
    import torch
    self.gym.render_all_camera_sensors(self.sim)
    self.gym.start_access_image_tensors(self.sim)
    if getattr(self, "_hires_buf", None) is None:
        self._hires_buf = torch.zeros((self.num_envs, self.cam_h, self.cam_w, 3), dtype=torch.float, device=self.device)
    for i in range(self.num_envs):
        self._hires_buf[i] = self.camera_rgb_tensor_list[i][0][:, :, :3].float()
    self.gym.end_access_image_tensors(self.sim)
    self.img_buf = self._hires_buf          # the collection scripts read task.img_buf[i]
    return torch.flatten(self._hires_buf, start_dim=1, end_dim=-1)


for mod, cls in (("tv_tasks.tasks.base.shadow_hand", "ShadowHandBase"), ("tv_tasks.tasks.base1.shadow_hand", "ShadowHandBase")):
    try:
        m = __import__(mod, fromlist=[cls])
        getattr(m, cls).compute_pixel_obs = _hires_compute_pixel_obs
        print(f"[hires] patched {mod}.{cls}.compute_pixel_obs", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hires] could not patch {mod}: {e}", flush=True)

sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
