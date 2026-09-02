#!/usr/bin/env bash
#SBATCH --job-name=vtdex_upload
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_upload_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_upload_%j.err
#
# Uploads checkpoints + rollout-check videos (upload_ckpt_and_rollout.py) for a
# list of task:model pairs to HF. CPU-only -- this is network I/O, not compute.
#
# Required (--export=ALL,...): PAIRS (space-separated "task:model" tokens, e.g.
#   "bottle_cap:base bottle_cap:t_scr_gt slide:base")
# Optional: SEED (default 111), SKIP_VIDEO=1, SKIP_CKPT=1

set -euo pipefail

if [[ -z "${PAIRS:-}" ]]; then
  echo "PAIRS must be set (space-separated task:model tokens)" >&2
  exit 2
fi

YQQ=/scratch/project/prj-02-phai-lab/yqq
VTDEX_ROOT="${YQQ}/VTDexManip"
cd "${YQQ}"
source "${YQQ}/env.sh"

VTDEX_ENV="${YQQ}/envs/vtdexmanip"
export PATH="${VTDEX_ENV}/bin:${PATH}"

SEED="${SEED:-111}"
EXTRA_FLAGS=""
[[ -n "${SKIP_VIDEO:-}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_video"
[[ -n "${SKIP_CKPT:-}" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip_ckpt"

cd "${VTDEX_ROOT}"

for pair in ${PAIRS}; do
  task="${pair%%:*}"
  model="${pair##*:}"
  echo "=== ${task}-${model} ==="
  "${VTDEX_ENV}/bin/python" upload_ckpt_and_rollout.py --task "${task}" --model "${model}" --seed "${SEED}" ${EXTRA_FLAGS} || echo "FAILED: ${task}-${model}"
done

echo "---UPLOAD_BATCH_DONE---"
