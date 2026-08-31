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
# Optional: SEED (default 111, selects which trained checkpoint -- seed<SEED>/checkpoint/ -- to load),
#   EVAL_SEED (default = SEED; the env/rollout seed passed to --seed, independent of which checkpoint
#   is loaded -- set this alone to re-eval the SAME checkpoint under a different env reset sequence),
#   MAX_TRAJS (default 100)
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
EVAL_SEED="${EVAL_SEED:-${SEED}}"
MAX_TRAJS="${MAX_TRAJS:-100}"

echo "HOST=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "TASK_NAME=${TASK_NAME}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SEED=${SEED}"
echo "EVAL_SEED=${EVAL_SEED}"
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
    --seed "${EVAL_SEED}" \
    --headless
else
  "${VTDEX_ENV}/bin/python" -c "
import eval_agent, sys
sys.argv = ['eval_agent.py', '--task', '${TASK_NAME}-${MODEL_NAME}', '--rl_device', 'cuda:0', '--resume_model', '${CKPT}', '--test', '--seed', '${EVAL_SEED}', '--headless']
eval_agent.eval_policy(max_trajs=${MAX_TRAJS})
"
fi

echo "---EVAL_DONE: ${TASK_NAME}-${MODEL_NAME} ckpt_seed${SEED} eval_seed${EVAL_SEED} ckpt=$(basename ${CKPT})---"

# Auto-upload: checkpoint + a per-arm eval-result JSON, straight to HF, no manual step.
# Success % is parsed from THIS job's own log (the last "Mean success:" line printed
# by ppo.py's eval() right before it breaks at len(successes_que) > num_envs*max_trajs).
LOGFILE="/scratch/project/prj-02-phai-lab/yqq/logs/vtdex_eval_${SLURM_JOB_ID}.out"
SUCCESS_FRAC=$(grep -aE "Mean success:" "${LOGFILE}" | tail -1 | sed -E 's/.*Mean success: ([0-9.eE+-]+).*/\1/')
echo "SUCCESS_FRAC=${SUCCESS_FRAC}"

"${VTDEX_ENV}/bin/python" - "${TASK_NAME}" "${MODEL_NAME}" "${SEED}" "${EVAL_SEED}" "${CKPT}" "${SUCCESS_FRAC}" "${MAX_TRAJS}" "${SLURM_JOB_ID}" <<'PYEOF'
import sys, os, json
from huggingface_hub import HfApi

task, model, seed, eval_seed, ckpt, success_frac, max_trajs, job_id = sys.argv[1:9]
repo_id = "qqyang/hora-v4-shadow-tennis"
api = HfApi(token=os.environ["HF_TOKEN"])
same_seed = (seed == eval_seed)

ckpt_name = os.path.basename(ckpt)
hf_ckpt_name = f"{task}-{model}_seed{seed}_{ckpt_name}"
api.upload_file(
    path_or_fileobj=ckpt,
    path_in_repo=f"tactile_sr_ablation/vtdexmanip/checkpoints/{hf_ckpt_name}",
    repo_id=repo_id, repo_type="model",
)

result = {
    "task": task, "arm": model, "checkpoint_seed": int(seed), "eval_seed": int(eval_seed),
    "checkpoint": hf_ckpt_name,
    "eval_job_id": job_id,
    "protocol": "eval_agent.py --test (act_inference, deterministic)",
    "num_eval_trajectories": int(max_trajs),
    "success_pct": round(float(success_frac) * 100, 2),
}
# Re-evals with a different EVAL_SEED (same checkpoint, different env reset stream) get their
# own filename so they never clobber the canonical checkpoint_seed==eval_seed result.
result_name = f"{task}-{model}_seed{seed}.json" if same_seed else f"{task}-{model}_seed{seed}_evalseed{eval_seed}.json"
local_path = os.path.join(os.path.dirname(ckpt), f"eval_result_{task}_{model}_evalseed{eval_seed}.json")
with open(local_path, "w") as f:
    json.dump(result, f, indent=2)

api.upload_file(
    path_or_fileobj=local_path,
    path_in_repo=f"tactile_sr_ablation/vtdexmanip/eval_results/{result_name}",
    repo_id=repo_id, repo_type="model",
)
print(f"UPLOADED ckpt+eval_result for {task}-{model} ckpt_seed{seed} eval_seed{eval_seed}: {result['success_pct']}%")
PYEOF

echo "---UPLOAD_DONE: ${TASK_NAME}-${MODEL_NAME}---"
