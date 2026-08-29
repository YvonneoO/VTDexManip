#!/usr/bin/env bash
#SBATCH --job-name=vtdex_eval
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_eval_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_eval_%j.err
#
# VTDexManip held-out test eval (matches the paper's own protocol: act_inference,
# deterministic, max_trajs episodes -- default 100, same as Table 2). Auto-locates
# the highest-iteration checkpoint under runs/**/<task>-<model>/seed<seed>/checkpoint/
# so it works for any task without hardcoding the runs/<TaskDisplayName>/ prefix.
#
# Required (--export=ALL,...): TASK_NAME (e.g. bottle_cap), MODEL_NAME (e.g. base, vt_all_cls)
# Optional: SEED (default 111), MAX_TRAJS (default 100)
# Typically submitted with --dependency=afterok:<train_jobid> so it runs once training finishes.

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
MAX_TRAJS="${MAX_TRAJS:-100}"

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

# eval_policy() hardcodes max_trajs=100 in eval_agent.py; override via a small
# wrapper only if MAX_TRAJS differs from the default, to keep the common case
# identical to the paper's own eval_agent.py invocation.
if [[ "${MAX_TRAJS}" == "100" ]]; then
  "${VTDEX_ENV}/bin/python" eval_agent.py \
    --task "${TASK_NAME}-${MODEL_NAME}" \
    --rl_device cuda:0 \
    --resume_model "${CKPT}" \
    --test \
    --seed "${SEED}" \
    --headless
else
  "${VTDEX_ENV}/bin/python" -c "
import eval_agent, sys
sys.argv = ['eval_agent.py', '--task', '${TASK_NAME}-${MODEL_NAME}', '--rl_device', 'cuda:0', '--resume_model', '${CKPT}', '--test', '--seed', '${SEED}', '--headless']
eval_agent.eval_policy(max_trajs=${MAX_TRAJS})
"
fi

echo "---EVAL_DONE: ${TASK_NAME}-${MODEL_NAME} seed${SEED} ckpt=$(basename ${CKPT})---"
