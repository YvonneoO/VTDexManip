#!/usr/bin/env bash
#SBATCH --job-name=tac_diag
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/tac_diag_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/tac_diag_%j.err
#
# One-off: run diag_tactile_scale.py against an already-trained TacGT/t_scr_gt
# checkpoint to measure the raw per-link contact-force magnitude actually seen
# during a real held-out rollout, vs. the pipeline's clip_observations=5.0 clamp.
#
# Required (--export=ALL,...): TASK_NAME, MODEL_NAME (e.g. screw_faucet, t_scr_gt)
# Optional: SEED (default 111), MAX_TRAJS (default 10 -- just need the magnitude
#   distribution, not a full paper-matching eval)

set -euo pipefail

if [[ -z "${TASK_NAME:-}" || -z "${MODEL_NAME:-}" ]]; then
  echo "TASK_NAME and MODEL_NAME must both be set" >&2
  exit 2
fi

YQQ=/scratch/project/prj-02-phai-lab/yqq
VTDEX_ROOT="${YQQ}/VTDexManip"
cd "${YQQ}"
source "${YQQ}/env.sh"
mkdir -p "${YQQ}/logs" "${YQQ}/tmp"

export TMPDIR="${YQQ}/tmp"
export HF_HOME="${YQQ}/.cache/huggingface"
export TORCH_HOME="${YQQ}/.cache/torch"
export PYTHONUNBUFFERED=1

VTDEX_ENV="${YQQ}/envs/vtdexmanip"
export PATH="${VTDEX_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${VTDEX_ENV}/lib:${LD_LIBRARY_PATH:-}"

SEED="${SEED:-111}"
MAX_TRAJS="${MAX_TRAJS:-10}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "TASK_NAME=${TASK_NAME}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SEED=${SEED}"
echo "MAX_TRAJS=${MAX_TRAJS}"

cd "${VTDEX_ROOT}"

CKPT_DIR_GLOB="runs/*/${TASK_NAME}/${TASK_NAME}-${MODEL_NAME}/seed${SEED}/checkpoint"
CKPT=$(ls -1 ${CKPT_DIR_GLOB}/model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}')
if [[ -z "${CKPT}" ]]; then
  echo "No checkpoint found matching ${CKPT_DIR_GLOB}/model_*.pt" >&2
  exit 3
fi
echo "CKPT=${CKPT}"

"${VTDEX_ENV}/bin/python" -c "
import diag_tactile_scale, sys
sys.argv = ['diag_tactile_scale.py', '--task', '${TASK_NAME}-${MODEL_NAME}', '--rl_device', 'cuda:0', '--resume_model', '${CKPT}', '--test', '--seed', '${SEED}', '--headless']
diag_tactile_scale.diag_tactile_scale(max_trajs=${MAX_TRAJS})
"

echo "---DIAG_DONE: ${TASK_NAME}-${MODEL_NAME}---"
