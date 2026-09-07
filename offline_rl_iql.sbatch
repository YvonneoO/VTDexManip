#!/usr/bin/env bash
#SBATCH --job-name=vtdex_iql
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_iql_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_iql_%j.err
#
# Offline-RL launcher for BottleCap Turning: builds a d3rlpy MDPDataset from an
# already-generated episode manifest (tactile_collection/offline_rl/
# generate_bottlecap_manifest.py), then trains d3rlpy's own IQL implementation
# on it (tactile_collection/offline_rl/build_mdp_dataset.py + train_iql.py --
# NOT bidexhands' hand-rolled algorithms/offrl/iql/, which this new pipeline
# deliberately does not reuse).
#
# Uses a SEPARATE env from vtdexmanip's own IsaacGym env (VTDEX_ENV in
# vision_train.sbatch) -- d3rlpy has no IsaacGym dependency, so this reuses the
# `d3rlpy_offline` env shared with the bidexhands-side offline-RL pipeline
# (see setup_d3rlpy_env.sh under tactile_collection/offline_rl/).
#
# Required (--export=ALL,...): MANIFEST (path to a manifest JSON from
#   generate_bottlecap_manifest.py), ARM (p_only or p_gt_tac), OUT_DIR
# Optional: N_STEPS (default 100000), N_STEPS_PER_EPOCH (default 1000)

set -euo pipefail

if [[ -z "${MANIFEST:-}" || -z "${ARM:-}" || -z "${OUT_DIR:-}" ]]; then
  echo "MANIFEST, ARM, and OUT_DIR must all be set (--export=ALL,MANIFEST=...,ARM=p_gt_tac,OUT_DIR=...)" >&2
  exit 2
fi
if [[ "${ARM}" != "p_only" && "${ARM}" != "p_gt_tac" ]]; then
  echo "ARM must be p_only or p_gt_tac, got: ${ARM}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "MANIFEST not found: ${MANIFEST}" >&2
  exit 3
fi

YQQ=/scratch/project/prj-02-phai-lab/yqq
VTDEX_ROOT="${YQQ}/VTDexManip"
cd "${YQQ}"
source "${YQQ}/env.sh"
mkdir -p "${YQQ}/logs" "${YQQ}/tmp" "${OUT_DIR}"

export TMPDIR="${YQQ}/tmp"
export HF_HOME="${YQQ}/.cache/huggingface"
export TORCH_HOME="${YQQ}/.cache/torch"
export PYTHONUNBUFFERED=1

D3RLPY_ENV="${YQQ}/envs/d3rlpy_offline"
export PATH="${D3RLPY_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${D3RLPY_ENV}/lib:${LD_LIBRARY_PATH:-}"

N_STEPS="${N_STEPS:-100000}"
N_STEPS_PER_EPOCH="${N_STEPS_PER_EPOCH:-1000}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "MANIFEST=${MANIFEST}"
echo "ARM=${ARM}"
echo "OUT_DIR=${OUT_DIR}"
echo "N_STEPS=${N_STEPS}"

cd "${VTDEX_ROOT}/tactile_collection/offline_rl"

DATASET_PATH="${OUT_DIR}/dataset_${ARM}_train.h5"

"${D3RLPY_ENV}/bin/python" build_mdp_dataset.py \
  --manifest "${MANIFEST}" \
  --arm "${ARM}" \
  --split train \
  --out "${DATASET_PATH}"

"${D3RLPY_ENV}/bin/python" train_iql.py \
  --dataset "${DATASET_PATH}" \
  --out "${OUT_DIR}/iql_${ARM}" \
  --n_steps "${N_STEPS}" \
  --n_steps_per_epoch "${N_STEPS_PER_EPOCH}" \
  --device cuda:0

echo "---IQL TRAINING DONE: arm=${ARM}---"
