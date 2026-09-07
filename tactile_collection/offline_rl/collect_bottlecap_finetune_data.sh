#!/usr/bin/env bash
# Collect an ADDITIONAL ~1000 BottleCap episodes, separate from the original
# collected_bottle_cap_1k, specifically to be held out for a v2-dit finetune
# (never mixed into the offline-RL/IQL manifest pool -- avoids repeating the
# exact leakage mistake found and fixed for Pen/Scissors earlier this session:
# see check_pen_finetune_overlap.py / the --exclude flag on
# generate_episode_manifest.py). Different --seed (999, vs. whatever the
# original collection used) so this is a genuinely different trajectory batch,
# not a re-roll of the same one.
#
# Written as a real script (not an inline sbatch --wrap string) -- nested
# --wrap quoting through this session's tmux send-keys -> sbatch --wrap ->
# bash -c chain has repeatedly mangled $VAR references at this depth.
set -euo pipefail

YQQ="/scratch/project/prj-02-phai-lab/yqq"
source "${YQQ}/env.sh"

ENV_ROOT="${YQQ}/envs/vtdexmanip"
export LD_LIBRARY_PATH="${ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${ENV_ROOT}/bin:${PATH}"

cd "${YQQ}/VTDexManip"

TARGET_SUCCESSES=1000 OUT_DIR=collected_bottle_cap_finetune_1k \
  "${ENV_ROOT}/bin/python" tactile_collection/collect_bottle_cap_tactile.py \
  --task bottle_cap-vt_all_cls --rl_device cuda:0 \
  --resume_model runs/BottleCap/bottle_cap/bottle_cap-vt_all_cls/seed111/checkpoint/model_800.pt \
  --test --seed 999 --headless

echo COLLECT_BCAP_FINETUNE_DATA_DONE
