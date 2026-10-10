"""Online Pred-Tac observation for VTDexManip's SINGLE-hand tasks (slide, bottle_cap): the right hand's tactile comes from the
tactile-prediction server (Ego2Contact predtac_server.py, same async file IPC as bidexhands' and handover's PredTac arms).

Differences to handover.compute_predtac_obs (two hands): one hand only (34 dims = 17 continuous x PREDTAC_CONT_SCALE +
17 binary x PREDTAC_BIN_SCALE of the predictor's RIGHT channel -- the single-hand tasks' pressure is always the right-hand label),
hand boxes keyed by IMAGE side like the training cache (predtac_utils.crop_boxes_for_env), render/submit only every
PREDTAC_CAPTURE_EVERY-th env step, and a hand-level quality log (PREDTAC_QUALITY_LOG=1): predicted any-contact vs the sim's
ground-truth any-contact of the same env, at lags 0..K env steps, every PREDTAC_QUALITY_EVERY (200) steps.
"""
import os

import numpy as np
import torch


class PredTacObsMixin:
    predtac_single_hand = True
    _PQ_K = 10

    def _predtac_init(self):
        from tv_tasks.tasks.base1.predtac_client import PredTacClient

        run_id = os.environ.get("PREDTAC_RUN_ID")
        if not run_id:
            raise RuntimeError("PREDTAC_RUN_ID must be set for obs_type=PredTac (shared with the predtac_server run_id)")
        n_cam = len(self.camera_handles)
        self._predtac_n_ready = min(self.num_envs, n_cam)
        self._predtac_view, self._predtac_proj = [], []
        for i in range(self._predtac_n_ready):
            ch = self.camera_handles[i][0]
            self._predtac_view.append(np.asarray(self.gym.get_camera_view_matrix(self.sim, self.envs[i], ch), np.float64))
            self._predtac_proj.append(np.asarray(self.gym.get_camera_proj_matrix(self.sim, self.envs[i], ch), np.float64))
        self._predtac_client = PredTacClient(run_id, self.num_envs)
        self._predtac_capture_every = int(os.environ.get("PREDTAC_CAPTURE_EVERY", "1"))
        self._predtac_step = -1
        self._predtac_cont_scale = float(os.environ.get("PREDTAC_CONT_SCALE", "2.0"))
        self._predtac_bin_scale = float(os.environ.get("PREDTAC_BIN_SCALE", "3.0"))
        self._pq_hist, self._pq_n = [], 0
        self._pq_reset()
        print(f"[PredTac] {self._predtac_n_ready}/{self.num_envs} static cameras {self.cam_w}x{self.cam_h}, run_id={run_id}, "
              f"capture_every={self._predtac_capture_every}, scales cont={self._predtac_cont_scale} bin={self._predtac_bin_scale}",
              flush=True)

    def compute_predtac_obs(self):
        from tv_tasks.tasks.base1.predtac_utils import crop_boxes_for_env

        if not hasattr(self, "_predtac_client"):
            self._predtac_init()
        self._predtac_step += 1
        if self._predtac_step % self._predtac_capture_every == 0:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            origin = self.env_origin.detach().cpu().numpy()
            # all rigid bodies of the (right) hand actor, env-local: the box then covers palm + wrist like a SAM3 hand mask
            # (fingertip points alone gave boxes shifted/offset by ~30% of the box size vs SAM3)
            nb = int(self.num_shadow_hand_bodies)
            pts = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 0:nb, 0:3].detach().cpu().numpy()
            frames = np.zeros((self.num_envs, self.cam_h, self.cam_w, 3), dtype=np.uint8)
            sides = []
            for i in range(self.num_envs):
                if i >= self._predtac_n_ready:
                    sides.append({})
                    continue
                frames[i] = self.camera_rgb_tensor_list[i][0][:, :, :3].detach().cpu().numpy().astype(np.uint8)
                sides.append(crop_boxes_for_env(pts[i], None, origin[i], self._predtac_view[i], self._predtac_proj[i],
                                                self.cam_w, self.cam_h))
            self.gym.end_access_image_tensors(self.sim)
            self._predtac_client.submit(frames, sides)
            if os.environ.get("PREDTAC_DUMP_DIR"):
                from tv_tasks.tasks.base1.predtac_utils import dump_frames
                dump_frames(frames, sides, self._predtac_step)
            if self._predtac_step % (50 * self._predtac_capture_every) == 0:
                n_box = sum(1 for s in sides if s)
                print(f"[predtac][box] step={self._predtac_step} envs_with_box={n_box}/{self.num_envs}", flush=True)
        cont, binary = self._predtac_client.poll()             # (N,2,17): slot 0 = left output channel, 1 = right
        stale = self._predtac_client.staleness_ticks()
        if self._predtac_client._tick % 25 == 0 and self._predtac_step % self._predtac_capture_every == 0:
            print(f"[predtac][staleness] client_tick={self._predtac_client._tick} stale_ticks={stale}", flush=True)
        if os.environ.get("PREDTAC_QUALITY_LOG", "0") == "1":
            self._pq_update(binary[:, 1, :], stale)
        c = torch.from_numpy(np.nan_to_num(cont[:, 1, :], nan=0.0)).to(self.device)
        b = torch.from_numpy(np.nan_to_num(binary[:, 1, :], nan=0.0)).to(self.device)
        return torch.cat([self._predtac_cont_scale * c, self._predtac_bin_scale * b], dim=-1)   # (N, 34)

    # ---- hand-level online quality monitor -------------------------------------------------------------------------
    def _pq_reset(self):
        K = self._PQ_K + 1
        self._pq = dict(tp=np.zeros(K), pos=np.zeros(K), fp=np.zeros(K), neg=np.zeros(K), gt=0.0, pr=0.0, n=0)

    def _pq_update(self, bin_right, stale):
        contact = self.contact_force[self.hand_contact_idx].view(self.num_envs, self.num_force_sensors, 3)
        gt = (torch.norm(contact, p=2, dim=2) > 0.01).any(dim=1).cpu().numpy()            # (N,) any contact on the right hand
        self._pq_hist.append(gt)
        self._pq_hist = self._pq_hist[-(self._PQ_K + 1):]
        pred = bin_right.max(axis=1) > 0.5                                                # (N,) predicted any-link contact
        if not pred.any() and self._predtac_client.last_tick_seen < 0:
            return                                                                        # no response yet
        q = self._pq
        q["n"] += 1
        q["gt"] += gt.mean(); q["pr"] += pred.mean()
        for k, g in enumerate(reversed(self._pq_hist)):
            q["pos"][k] += g.sum(); q["neg"][k] += (~g).sum()
            q["tp"][k] += (pred & g).sum(); q["fp"][k] += (pred & ~g).sum()
        if q["n"] % int(os.environ.get("PREDTAC_QUALITY_EVERY", "200")) == 0:
            tpr = q["tp"] / np.maximum(q["pos"], 1); fpr = q["fp"] / np.maximum(q["neg"], 1)
            # balanced agreement per lag: higher = prediction at t matches GT contact at t-k better
            bal = 0.5 * (tpr + (1 - fpr)); best = int(np.argmax(bal))
            print(f"[predtac][quality] step={self._predtac_step} stale_ticks={stale} gt_pos={q['gt'] / q['n']:.3f} "
                  f"pred_pos={q['pr'] / q['n']:.3f} | k=0 tpr={tpr[0]:.3f} fpr={fpr[0]:.3f} | best_lag={best} "
                  f"tpr={tpr[best]:.3f} fpr={fpr[best]:.3f} | bal@lag: " +
                  " ".join(f"{k}:{bal[k]:.2f}" for k in range(len(bal))), flush=True)
            self._pq_reset()
