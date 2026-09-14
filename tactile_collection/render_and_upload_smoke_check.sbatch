#!/usr/bin/env bash
#SBATCH --job-name=vtdex_smokevid
#SBATCH --partition=def
#SBATCH --account=prj-02-phai-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_smokevid_%j.out
#SBATCH --error=/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_smokevid_%j.err
#
# Renders rgb_tactile_side_by_side.mp4 check videos for a smoke-test batch's
# successful_episodes/ (make_episode_videos.py) and uploads them to HF
# (upload_episode_videos.py) for review before committing to a full
# 1000-trajectory collection. CPU-only (video encode), no GPU needed.
#
# Required (--export=ALL,...): OUT_DIR (the collector's OUT_DIR, containing
#   successful_episodes/), LABEL (HF path label, e.g. bottle_cap-vt_all_cls_smoke5),
#   TITLE (video overlay title), NUM_EPISODES (default 5)

set -euo pipefail

if [[ -z "${OUT_DIR:-}" || -z "${LABEL:-}" ]]; then
  echo "OUT_DIR and LABEL must be set" >&2
  exit 2
fi

YQQ=/scratch/project/prj-02-phai-lab/yqq
VTDEX_ROOT="${YQQ}/VTDexManip"
cd "${VTDEX_ROOT}"
source "${YQQ}/env.sh"

module load GCCcore/13.3.0 FFmpeg
VTDEX_ENV="${YQQ}/envs/vtdexmanip"

NUM_EPISODES="${NUM_EPISODES:-5}"
TITLE="${TITLE:-${LABEL}}"

echo "OUT_DIR=${OUT_DIR}"
echo "LABEL=${LABEL}"
echo "TITLE=${TITLE}"
echo "NUM_EPISODES=${NUM_EPISODES}"

EP_DIRS=()
for ((i = 0; i < NUM_EPISODES; i++)); do
  ep=$(printf "episode_%06d" "${i}")
  EP_DIRS+=("${OUT_DIR}/successful_episodes/${ep}")
done

"${VTDEX_ENV}/bin/python" tactile_collection/make_episode_videos.py \
  "${EP_DIRS[@]}" --title "${TITLE}"

VIDEOS=()
for d in "${EP_DIRS[@]}"; do
  VIDEOS+=("${d}/rgb_tactile_side_by_side.mp4")
done

"${VTDEX_ENV}/bin/python" tactile_collection/upload_episode_videos.py \
  --label "${LABEL}" "${VIDEOS[@]}"

echo SMOKEVID_DONE
