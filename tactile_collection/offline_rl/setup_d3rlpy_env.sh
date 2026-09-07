#!/usr/bin/env bash
# One-time setup for the `d3rlpy_offline` conda env used by BOTH this repo's
# BottleCap IQL pipeline (offline_rl_iql.sbatch) and the analogous bidexhands
# offline-RL work -- d3rlpy has no IsaacGym dependency, so one env can serve
# both, unlike VTDexManip's own `vtdexmanip` env (which vision_train.sbatch
# uses and which DOES need IsaacGym).
#
# NOT run by this session -- no cluster access here. Someone with a VISION
# shell must run this manually (following @TAMU/CLAUDE.md's VISION rules: only
# under /scratch/project/prj-02-phai-lab/yqq, never /tmp or /home, no
# containers -- Lmod modules + conda instead of docker/apptainer).
#
# d3rlpy v2.x API (MDPDataset, IQLConfig, logging adapters) used by
# build_mdp_dataset.py / train_iql.py was verified 2026-09-07 against the
# `d3rlpy` GitHub `master` branch source (this environment can't pip-install
# to check locally). Pin the exact resolved version after first install
# (`pip freeze | grep d3rlpy`) into requirements below for reproducibility --
# left unpinned here since the exact latest 2.x patch wasn't independently
# confirmed against PyPI.

set -euo pipefail

YQQ=/scratch/project/prj-02-phai-lab/yqq
ENV_DIR="${YQQ}/envs/d3rlpy_offline"

module load Miniforge3
module load CUDA/12.8.0

conda create -y -p "${ENV_DIR}" python=3.10
conda activate "${ENV_DIR}"

export PIP_CACHE_DIR="${YQQ}/.cache/pip"
export TMPDIR="${YQQ}/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

# d3rlpy 2.x requires torch>=2.5.0 (per its own setup metadata) -- install
# torch explicitly first so pip doesn't pull a CPU-only wheel by default.
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install d3rlpy      # do NOT `pip install --user`; this env is yqq-local
pip install h5py         # MDPDataset.dump()/ReplayBuffer.load() are HDF5-backed
pip install wandb        # optional: enables train_iql.py's --wandb flag
pip install tensorboard  # for TensorboardAdapterFactory's log output

echo "[setup_d3rlpy_env] done. Verify with:"
echo "  ${ENV_DIR}/bin/python -c 'import d3rlpy; print(d3rlpy.__version__)'"
echo "Record the printed version in a comment here once confirmed."
