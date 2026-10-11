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
        self._predtac_apply_hand_color()
        print(f"[PredTac] {self._predtac_n_ready}/{self.num_envs} static cameras {self.cam_w}x{self.cam_h}, run_id={run_id}, "
              f"capture_every={self._predtac_capture_every}, scales cont={self._predtac_cont_scale} bin={self._predtac_bin_scale}",
              flush=True)

    def _predtac_apply_hand_color(self):
        """The predictor was fine-tuned on frames from the data collectors (collect_{slide,bottle_cap,handover}_tactile.py), which
        repaint every hand body (0.42, 0.52, 0.56) before rendering; the task envs themselves render near-black hands (handover
        also red sensor blocks). Same repaint here, otherwise the online input is out of distribution (predictions saturate).
        PREDTAC_HAND_COLOR=r,g,b overrides, "none" disables."""
        spec = os.environ.get("PREDTAC_HAND_COLOR", "0.42,0.52,0.56").strip().lower()
        if spec in ("", "none", "off"):
            return
        from isaacgym import gymapi
        rgb = [float(x) for x in spec.split(",")]
        n_act = 0
        for env_ptr in self.envs:
            for name in ("hand", "another_hand"):
                actor = self.gym.find_actor_handle(env_ptr, name)
                if actor is None or actor < 0:
                    continue
                for b in range(self.gym.get_actor_rigid_body_count(env_ptr, actor)):
                    self.gym.set_rigid_body_color(env_ptr, actor, b, gymapi.MESH_VISUAL, gymapi.Vec3(*rgb))
                n_act += 1
        print(f"[PredTac] hand bodies repainted rgb={rgb} actors={n_act}", flush=True)

    def compute_predtac_obs(self):
        from tv_tasks.tasks.base1.predtac_utils import crop_boxes_for_env

        if not hasattr(self, "_predtac_client"):
            self._predtac_init()
        self._predtac_step += 1
        if self._predtac_step % self._predtac_capture_every == 0:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            origin = self.env_origin.detach().cpu().numpy()
            # fingertips + palm (env-local). All hand-actor bodies was tried and is WRONG: arm/base links project far outside
            # the visible hand (box 3x the SAM3 box, IoU 0.1 on slide/bottle_cap); fingertips alone sit at one end of the hand.
            tips = self.fingertip_state[:, :, 0:3]
            if len(getattr(self, "palm_body_idx", [])) == self.num_envs:
                palm = self.rigid_body_states[self.palm_body_idx][:, 0:3].view(self.num_envs, 1, 3)
                tips = torch.cat([tips, palm], dim=1)
            pts = tips.detach().cpu().numpy()
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
            self._pq_update(binary[:, 1, :], stale, cont_right=cont[:, 1, :])
        c = torch.from_numpy(np.nan_to_num(cont[:, 1, :], nan=0.0)).to(self.device)
        b = torch.from_numpy(np.nan_to_num(binary[:, 1, :], nan=0.0)).to(self.device)
        return torch.cat([self._predtac_cont_scale * c, self._predtac_bin_scale * b], dim=-1)   # (N, 34)

    # ---- hand-level online quality monitor -------------------------------------------------------------------------
    def _pq_reset(self):
        K = self._PQ_K + 1
        self._pq = dict(tp=np.zeros(K), pos=np.zeros(K), fp=np.zeros(K), neg=np.zeros(K), gt=0.0, pr=0.0, n=0, npl=0.0, ngl=0.0)

    def _pq_label_links(self):
        """The predictor's TRAINING-label definition, computed online: per env, EgoTouchTaxelMapper.project(get_env_rigid_contacts) as the
        data collectors did, then /vmax, clip, max over each of the 17 anatomical links -> (N,17) link values and (N,17) link force (N).
        PREDTAC_QUALITY_CONTACTS=1 (right-hand tasks only). Lets the monitor score the predictor against the SAME quantity it was trained on,
        instead of the net-contact-force tensor (which can read 0 while the hand rests on the object)."""
        if not hasattr(self, "_pq_mappers"):
            import sys
            d = os.path.join(os.getcwd(), "tactile_collection")
            if d not in sys.path:
                sys.path.insert(0, d)
            from egotouch_taxels import EgoTouchTaxelMapper, _layout, _load_valid_cells
            mp = os.path.join(d, "assets", "pressure_position_mapping_right.json")
            self._pq_mappers = [EgoTouchTaxelMapper(self.gym, env_ptr, "hand", "right", mp) for env_ptr in self.envs]
            groups = _layout("right", _load_valid_cells(mp))
            self._pq_lnames = sorted(groups)
            self._pq_lmasks = np.zeros((len(self._pq_lnames), 21, 21), bool)
            for l, n in enumerate(self._pq_lnames):
                for r, c in groups[n]:
                    self._pq_lmasks[l, r, c] = True
            task = os.environ.get("TASK", "")
            default_vmax = 100146.2 if task.startswith("slide") else (94107.4 if task.startswith("bottle") else 0.0)
            self._pq_lvmax = float(os.environ.get("PREDTAC_QUALITY_VMAX", default_vmax))
            assert self._pq_lvmax > 0, "set PREDTAC_QUALITY_VMAX for PREDTAC_QUALITY_CONTACTS"
            print(f"[predtac][quality] label monitor on: vmax={self._pq_lvmax} links={self._pq_lnames}", flush=True)
        lab = np.zeros((self.num_envs, len(self._pq_lnames)), np.float32)
        lforce = np.zeros_like(lab)
        for i, env_ptr in enumerate(self.envs):
            pa, fg, _ = self._pq_mappers[i].project(self.gym.get_env_rigid_contacts(env_ptr))
            pn = np.clip(np.nan_to_num(pa) / self._pq_lvmax, 0.0, 1.0)
            fg = np.nan_to_num(fg)
            for l in range(len(self._pq_lnames)):
                m = self._pq_lmasks[l]
                lab[i, l] = pn[m].max()
                lforce[i, l] = fg[m].sum()
        return lab, lforce

    def _pq_update(self, bin_right, stale, contact_idx=None, cont_right=None):
        if not hasattr(self, "_pq"):
            self._pq_hist = []
            self._pq_reset()
        idx = contact_idx if contact_idx is not None else self.hand_contact_idx
        contact = self.contact_force[idx].view(self.num_envs, self.num_force_sensors, 3)
        force_n = torch.norm(contact, p=2, dim=2)                                         # (N, S) per force-sensor body
        gt_links = (force_n > 0.01)
        npz = os.environ.get("PREDTAC_QUALITY_NPZ")
        if not npz and os.environ.get("PREDTAC_DUMP_DIR"):
            npz = os.path.join(os.environ["PREDTAC_DUMP_DIR"], "quality_raw.npz")
        lab = lforce = None
        if os.environ.get("PREDTAC_QUALITY_CONTACTS", "0") == "1" and contact_idx is None and cont_right is not None:
            lab, lforce = self._pq_label_links()
        if npz and cont_right is not None:
            # raw per-tick arrays for offline calibration (threshold / link-order checks): predicted continuous + binary per link
            # vs GT per-sensor force norm, same tick; flushed every 100 updates
            if not hasattr(self, "_pq_raw"):
                self._pq_raw = dict(cont=[], binary=[], force=[])
            self._pq_raw["cont"].append(np.asarray(cont_right, np.float32).copy())
            self._pq_raw["binary"].append(np.asarray(bin_right, np.float32).copy())
            self._pq_raw["force"].append(force_n.detach().cpu().numpy().astype(np.float32))
            if lab is not None:
                self._pq_raw.setdefault("label", []).append(lab.copy())
                self._pq_raw.setdefault("lforce", []).append(lforce.copy())
            if len(self._pq_raw["cont"]) % 100 == 0:
                os.makedirs(os.path.dirname(os.path.abspath(npz)), exist_ok=True)
                np.savez(npz, **{k: np.stack(v) for k, v in self._pq_raw.items()})   # (ticks, N, 17|S)
        gt = gt_links.any(dim=1).cpu().numpy()                                            # (N,) any contact on the hand
        n_gt_links = gt_links.sum(dim=1).float().mean().item()
        self._pq_hist.append(gt)
        self._pq_hist = self._pq_hist[-(self._PQ_K + 1):]
        pred = bin_right.max(axis=1) > 0.5                                                # (N,) predicted any-link contact
        if not pred.any() and self._predtac_client.last_tick_seen < 0:
            return                                                                        # no response yet
        q = self._pq
        q["n"] += 1
        q["gt"] += gt.mean(); q["pr"] += pred.mean()
        q["npl"] += float((bin_right > 0.5).sum(axis=1).mean()); q["ngl"] += n_gt_links
        for k, g in enumerate(reversed(self._pq_hist)):
            q["pos"][k] += g.sum(); q["neg"][k] += (~g).sum()
            q["tp"][k] += (pred & g).sum(); q["fp"][k] += (pred & ~g).sum()
        if lab is not None:
            L = self.__dict__.setdefault("_pq_lab", dict(n=0, pos=0.0, tp=0.0, fp=0.0, neg=0.0, ta=0.0, la=0.0, both=0.0, xs=[], ys=[]))
            pb = np.asarray(cont_right) > 0.1
            lb = lab > 0.1
            L["n"] += 1; L["pos"] += lb.sum(); L["neg"] += (~lb).sum(); L["tp"] += (pb & lb).sum(); L["fp"] += (pb & ~lb).sum()
            ta = gt; la = (lforce.max(axis=1) > 0.01)
            L["ta"] += ta.mean(); L["la"] += la.mean(); L["both"] += (ta & la).mean()
            L["xs"].append(np.asarray(cont_right).ravel()); L["ys"].append(lab.ravel())
        if q["n"] % int(os.environ.get("PREDTAC_QUALITY_EVERY", "200")) == 0:
            tpr = q["tp"] / np.maximum(q["pos"], 1); fpr = q["fp"] / np.maximum(q["neg"], 1)
            # balanced agreement per lag: higher = prediction at t matches GT contact at t-k better
            bal = 0.5 * (tpr + (1 - fpr)); best = int(np.argmax(bal))
            step = getattr(self, "_predtac_step", self._predtac_client._tick)   # handover keeps its own capture path
            print(f"[predtac][quality] step={step} stale_ticks={stale} gt_pos={q['gt'] / q['n']:.3f} "
                  f"pred_pos={q['pr'] / q['n']:.3f} pred_links={q['npl'] / q['n']:.1f} gt_links={q['ngl'] / q['n']:.1f} | k=0 tpr={tpr[0]:.3f} fpr={fpr[0]:.3f} | best_lag={best} "
                  f"tpr={tpr[best]:.3f} fpr={fpr[best]:.3f} | bal@lag: " +
                  " ".join(f"{k}:{bal[k]:.2f}" for k in range(len(bal))), flush=True)
            L = self.__dict__.get("_pq_lab")
            if L is not None and L["n"] > 0:
                x = np.concatenate(L["xs"]); y = np.concatenate(L["ys"])
                c = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
                print(f"[predtac][quality][label] n={L['n']} label_pos={L['pos'] / max(L['pos'] + L['neg'], 1):.3f} "
                      f"link-tpr={L['tp'] / max(L['pos'], 1):.3f} link-fpr={L['fp'] / max(L['neg'], 1):.3f} corr(pred,label)={c:.3f} | "
                      f"any-contact: force-tensor {L['ta'] / L['n']:.3f} vs contact-API {L['la'] / L['n']:.3f} (both {L['both'] / L['n']:.3f})",
                      flush=True)
                self._pq_lab = dict(n=0, pos=0.0, tp=0.0, fp=0.0, neg=0.0, ta=0.0, la=0.0, both=0.0, xs=[], ys=[])
            self._pq_reset()
