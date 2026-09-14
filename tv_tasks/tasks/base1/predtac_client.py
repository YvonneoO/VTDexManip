#!/usr/bin/env python3
"""VTDexManip-side client for the online Pred-Tac IPC bridge (predtac_ipc.py,
same directory) -- vendored/adapted from the DexterousHands repo's
bidexhands/tactile_collection/predtac_client.py (same asynchronous/stale-
tolerant design, see predtac_ipc.py's module docstring for why).

Hand-slot convention: index 0 = left, index 1 = right -- matches the
tactile-prediction model's own input/output convention, same as the
DexterousHands-side client. handover.py's own crop_boxes_for_env (in
predtac_utils.py) already returns sides dicts keyed "left"/"right", so this
client is the one place that commits to a fixed wire-format slot order.
"""
import time

import numpy as np

from tv_tasks.tasks.base1 import predtac_ipc

NUM_LINKS = 17  # egotouch_taxels._layout() anatomical groups per hand


class PredTacClient:
    def __init__(self, run_id, num_envs, num_links=NUM_LINKS):
        self.run_id = run_id
        self.num_envs = num_envs
        self.num_links = num_links
        self.last_tick_seen = -1
        self.continuous = np.zeros((num_envs, 2, num_links), dtype=np.float32)
        self.binary = np.zeros((num_envs, 2, num_links), dtype=np.float32)
        self._tick = 0
        # See the DexterousHands-side predtac_client.py's identical guard
        # for the full rationale (a stale leftover response from a reused
        # run_id can permanently freeze a fresh session on one old frame).
        predtac_ipc.reset_run(run_id)

    def submit(self, frames_uint8, sides_all_envs):
        """frames_uint8: (num_envs,H,W,3) uint8. sides_all_envs: list of
        length num_envs of {"left": {"box": [x1,y1,x2,y2]}, "right": {...}}
        (either key possibly absent)."""
        boxes = np.full((self.num_envs, 2, 4), np.nan, dtype=np.float32)
        has_hand = np.zeros((self.num_envs, 2), dtype=bool)
        for i, sides in enumerate(sides_all_envs):
            for h, key in enumerate(("left", "right")):
                info = sides.get(key)
                if info is not None:
                    boxes[i, h] = info["box"]
                    has_hand[i, h] = True
        predtac_ipc.write_request(self.run_id, self._tick, frames_uint8, boxes, has_hand)
        self._tick += 1

    def poll(self):
        """Non-blocking. Updates self.continuous/self.binary in place if a
        newer server response is available; always returns the current
        (possibly stale, possibly still all-zero before the first response)
        (continuous, binary) pair, each (num_envs, 2, num_links)."""
        resp = predtac_ipc.read_response(self.run_id)
        if resp is not None and resp["tick"] > self.last_tick_seen:
            self.last_tick_seen = resp["tick"]
            # Defensive: see the DexterousHands-side predtac_client.py's
            # identical guard for the full rationale -- found live on this
            # framework's own Handover run (job 539404, iteration 267/4400),
            # a NaN-in-action-distribution crash consistent with an
            # unsanitized NaN tactile reading (e.g. a degenerate no-hand
            # frame) leaking through into the policy.
            self.continuous = np.nan_to_num(resp["continuous"], nan=0.0)
            self.binary = np.nan_to_num(resp["binary"], nan=0.0)
        return self.continuous, self.binary

    def staleness_ticks(self):
        """How many client ticks old the most recent response is, right now
        -- see the DexterousHands-side predtac_client.py's identical method
        for the full rationale (this file is kept in sync with that one).
        -1 = no submit made yet; None = no response ever received yet
        (still serving all-zero fallback) -- neither means zero lag."""
        if self._tick == 0:
            return -1
        if self.last_tick_seen < 0:
            return None
        return (self._tick - 1) - self.last_tick_seen

    def poll_blocking(self, timeout_s=15.0, poll_interval_s=0.02):
        """Blocking variant of poll() -- see the DexterousHands-side
        predtac_client.py's identical method for the full rationale (this
        file is kept in sync with that one). Waits for ANY response newer
        than what was known at call time, NOT an exact match on the tick
        just submitted -- the server's own frame_interval decimation means
        an exact-tick response may never arrive, which the first version of
        this method got wrong (always burned the full timeout, falling back
        one tick short every single call). Falls back to whatever's
        freshest and prints a warning if timeout_s elapses first."""
        if self._tick == 0:
            return self.continuous, self.binary
        seen_before = self.last_tick_seen
        deadline = time.time() + timeout_s
        while self.last_tick_seen <= seen_before:
            if time.time() > deadline:
                print(f"[predtac][blocking] timeout after {timeout_s}s waiting for a response "
                      f"newer than tick {seen_before} (submitted tick={self._tick - 1}) -- "
                      f"falling back to stale value", flush=True)
                break
            time.sleep(poll_interval_s)
            self.poll()
        return self.continuous, self.binary
