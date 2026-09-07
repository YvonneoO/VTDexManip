#!/usr/bin/env bash
#SBATCH --job-name=vtdex_iql_eval
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_iql_eval_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_iql_eval_%j.err
#
# Rolls out a trained d3rlpy IQL policy (from offline_rl_iql.sbatch) in the
# real BottleCap sim and reports success rate. Runs in the `vtdexmanip` env
# (has IsaacGym); internally spawns iql_action_server.py under the
# `d3rlpy_offline` env's python as a subprocess (the two envs' torch builds
# are incompatible in one process -- see eval_iql_policy.py's docstring).
#
# Required (--export=ALL,...): ARM (p_only or p_gt_tac), CHECKPOINT (path to
#   a final_full.d3 written by train_iql.py)
# Optional: NUM_TRAJS (default 100, matches eval_agent.py's paper-protocol default)

set -euo pipefail

if [[ -z "${ARM:-}" || -z "${CHECKPOINT:-}" ]]; then
  echo "ARM and CHECKPOINT must both be set (--export=ALL,ARM=p_gt_tac,CHECKPOINT=...)" >&2
  exit 2
fi
if [[ "${ARM}" != "p_only" && "${ARM}" != "p_gt_tac" ]]; then
  echo "ARM must be p_only or p_gt_tac, got: ${ARM}" >&2
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
D3RLPY_ENV="${YQQ}/envs/d3rlpy_offline"
export PATH="${VTDEX_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${VTDEX_ENV}/lib:${LD_LIBRARY_PATH:-}"

NUM_TRAJS="${NUM_TRAJS:-100}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "ARM=${ARM}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "NUM_TRAJS=${NUM_TRAJS}"

cd "${VTDEX_ROOT}/tactile_collection/offline_rl"

"${VTDEX_ENV}/bin/python" eval_iql_policy.py \
  --task bottle_cap-base \
  --seed 111 \
  --headless \
  --arm "${ARM}" \
  --checkpoint "${CHECKPOINT}" \
  --d3rlpy_python "${D3RLPY_ENV}/bin/python" \
  --device cuda:0 \
  --num_trajs "${NUM_TRAJS}"

echo "---IQL_EVAL_DONE: arm=${ARM}---"
