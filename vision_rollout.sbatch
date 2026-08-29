#!/usr/bin/env bash
#SBATCH --job-name=vtdex_rollout
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_rollout_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_rollout_%j.err
#
# Renders rollout videos of a trained checkpoint (rollout_record.py -- a thin fork
# of eval_agent.py with record_video=True, which eval_agent.py's own eval_policy()
# hardcodes off). One mp4 per env, named <object>_<env_idx>.mp4, written under the
# checkpoint's own run dir (see cfg_env['video_path']). IsaacGym off-screen camera
# capture works headless (uses img_buf, not the interactive viewer).
#
# Required (--export=ALL,...): TASK_NAME (e.g. bottle_cap), MODEL_NAME (e.g. base),
#   CHECKPOINT (full path to a model_<iter>.pt)
# Optional: SEED (default 111)

set -euo pipefail

if [[ -z "${TASK_NAME:-}" || -z "${MODEL_NAME:-}" || -z "${CHECKPOINT:-}" ]]; then
  echo "TASK_NAME, MODEL_NAME, and CHECKPOINT must all be set" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "CHECKPOINT not found: ${CHECKPOINT}" >&2
  exit 3
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

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "TASK_NAME=${TASK_NAME}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "SEED=${SEED}"

cd "${VTDEX_ROOT}"

"${VTDEX_ENV}/bin/python" rollout_record.py \
  --task "${TASK_NAME}-${MODEL_NAME}" \
  --rl_device cuda:0 \
  --resume_model "${CHECKPOINT}" \
  --test \
  --seed "${SEED}" \
  --headless

echo "---ROLLOUT_DONE: ${TASK_NAME}-${MODEL_NAME}---"
