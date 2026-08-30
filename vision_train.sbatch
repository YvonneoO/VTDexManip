#!/usr/bin/env bash
#SBATCH --job-name=vtdex_train
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_train_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_train_%j.err
#
# VTDexManip training launcher (tactile-SR ablation). Reuses the repo's own
# train_agent.py verbatim -- task/model composition is {TASK_NAME}-{MODEL_NAME}
# per the upstream README (e.g. bottle_cap-vt_all_cls, bottle_cap-base,
# bottle_cap-t_scr). Checkpoints land under runs/<TaskCapitalized>/<task>/
# <task>-<model>/seed<seed>/checkpoint/.
#
# Required (--export=ALL,...): TASK_NAME (e.g. bottle_cap), MODEL_NAME (e.g.
#   base, vt_all_cls, t_scr)
# Optional: SEED (default 111, matching the README's own examples), RESUME_MODEL
#   (full path to a model_<iter>.pt -- resumes iteration counting from the
#   filename, per PPO.load(); vt_all_cls runs are ~7x slower/step than base and
#   can outrun a single --time budget, this is the resume path for that)

set -euo pipefail

if [[ -z "${TASK_NAME:-}" || -z "${MODEL_NAME:-}" ]]; then
  echo "TASK_NAME and MODEL_NAME must both be set (--export=ALL,TASK_NAME=bottle_cap,MODEL_NAME=base)" >&2
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

EXTRA_ARGS=()
[[ -n "${RESUME_MODEL:-}" ]] && EXTRA_ARGS+=("--resume_model=${RESUME_MODEL}")
echo "RESUME_MODEL=${RESUME_MODEL:-<none, fresh start>}"

"${VTDEX_ENV}/bin/python" train_agent.py \
  --task "${TASK_NAME}-${MODEL_NAME}" \
  --rl_device cuda:0 \
  --seed "${SEED}" \
  --headless \
  "${EXTRA_ARGS[@]}"

echo "---TRAINING DONE: ${TASK_NAME}-${MODEL_NAME} seed${SEED}---"
