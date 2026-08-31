#!/usr/bin/env bash
#SBATCH --job-name=vtdex_collect
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_collect_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_collect_%j.err
#
# Full-frame (RGB + dof_pos + EgoTouch tactile) successful-trajectory collector
# for BottleCap Turning -- tactile_collection/collect_bottle_cap_tactile.py.
# BottleCap is single-handed and --test mode hardcodes numEnvs=10 (one per seen
# object); the collector runs all 10 in parallel independently.
#
# Required (--export=ALL,...): TASK_NAME (e.g. bottle_cap), MODEL_NAME (e.g. vt_all_cls),
#   CHECKPOINT (full path to model_<iter>.pt), TARGET_SUCCESSES, OUT_DIR
# Optional: SEED (default 111), MAX_STEPS (default 2000000)

set -euo pipefail

if [[ -z "${TASK_NAME:-}" || -z "${MODEL_NAME:-}" || -z "${CHECKPOINT:-}" || -z "${TARGET_SUCCESSES:-}" || -z "${OUT_DIR:-}" ]]; then
  echo "TASK_NAME, MODEL_NAME, CHECKPOINT, TARGET_SUCCESSES, and OUT_DIR must all be set" >&2
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
export TARGET_SUCCESSES="${TARGET_SUCCESSES}"
export MAX_STEPS="${MAX_STEPS:-2000000}"
export OUT_DIR="${OUT_DIR}"

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
echo "TARGET_SUCCESSES=${TARGET_SUCCESSES}"
echo "OUT_DIR=${OUT_DIR}"

cd "${VTDEX_ROOT}"
mkdir -p "${OUT_DIR}"

"${VTDEX_ENV}/bin/python" tactile_collection/collect_bottle_cap_tactile.py \
  --task "${TASK_NAME}-${MODEL_NAME}" \
  --rl_device cuda:0 \
  --resume_model "${CHECKPOINT}" \
  --test \
  --seed "${SEED}" \
  --headless

echo "---COLLECT_DONE: ${TASK_NAME}-${MODEL_NAME}---"
