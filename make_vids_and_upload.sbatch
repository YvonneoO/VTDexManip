#!/usr/bin/env bash
#SBATCH --job-name=make_vids
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/make_vids_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/make_vids_%j.err
#
# Render rgb+tactile side-by-side check videos for one or more already-collected
# episode dirs, then upload them to HF. A real bash script (not `sbatch --wrap`,
# which runs under /bin/sh -- `source` isn't a builtin there and brace expansion
# isn't supported, so both silently break: `source env.sh` fails with
# "source: not found" leaving HF_TOKEN unset, and `episode_{0,1}` passes through
# as one literal unexpanded path instead of two).
#
# Required (--export=ALL,...): EPISODE_DIRS (space-separated, relative to VTDexManip/), LABEL
# Optional: TITLE (default "Tactile rollout")

set -euo pipefail

if [[ -z "${EPISODE_DIRS:-}" || -z "${LABEL:-}" ]]; then
  echo "EPISODE_DIRS and LABEL must both be set" >&2
  exit 2
fi

YQQ=/scratch/project/prj-02-phai-lab/yqq
VTDEX_ROOT="${YQQ}/VTDexManip"
cd "${YQQ}"
source "${YQQ}/env.sh"

VTDEX_ENV="${YQQ}/envs/vtdexmanip"
export PATH="${VTDEX_ENV}/bin:${PATH}"

cd "${VTDEX_ROOT}"

echo "EPISODE_DIRS=${EPISODE_DIRS}"
echo "LABEL=${LABEL}"
echo "TITLE=${TITLE:-Tactile rollout}"

"${VTDEX_ENV}/bin/python" tactile_collection/make_episode_videos.py ${EPISODE_DIRS} --title "${TITLE:-Tactile rollout}"

VIDEOS=""
for d in ${EPISODE_DIRS}; do
  VIDEOS="${VIDEOS} ${d}/rgb_tactile_side_by_side.mp4"
done

"${VTDEX_ENV}/bin/python" tactile_collection/upload_episode_videos.py --label "${LABEL}" ${VIDEOS}

echo "---MAKE_VIDS_DONE: ${LABEL}---"
