#!/usr/bin/env bash
#SBATCH --job-name=vtdex_sweep
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_sweep_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_sweep_%j.err
#
# Held-out eval sweep across every periodic checkpoint of a completed VTDexManip
# run -- reconstructs a success-rate-vs-iteration curve using the paper's own
# protocol (eval_agent.py --test) at every point, not just the final checkpoint.
# Generous walltime: each checkpoint eval takes ~3-4 min, ~14 checkpoints/run.
#
# Required (--export=ALL,...): TASK_NAME (e.g. bottle_cap), MODEL_NAME (e.g. base)
# Optional: SEED (default 111)

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

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "TASK_NAME=${TASK_NAME}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SEED=${SEED}"

cd "${VTDEX_ROOT}"

CKPT_DIR=$(ls -d runs/*/${TASK_NAME}/${TASK_NAME}-${MODEL_NAME}/seed${SEED}/checkpoint 2>/dev/null | head -1)
if [[ -z "${CKPT_DIR}" ]]; then
  echo "No checkpoint dir found for ${TASK_NAME}-${MODEL_NAME} seed${SEED}" >&2
  exit 3
fi
echo "CKPT_DIR=${CKPT_DIR}"

OUT_CSV="${YQQ}/csv_export/${TASK_NAME}_${MODEL_NAME}_heldout_sweep.csv"
mkdir -p "$(dirname "${OUT_CSV}")"

"${VTDEX_ENV}/bin/python" sweep_checkpoint_eval.py "${CKPT_DIR}" "${TASK_NAME}-${MODEL_NAME}" "${OUT_CSV}" --seed "${SEED}"

echo "---SWEEP_DONE: ${TASK_NAME}-${MODEL_NAME} -> ${OUT_CSV}---"
