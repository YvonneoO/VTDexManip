import os
import shutil

from huggingface_hub import snapshot_download

repo_id = "qqyang/vtdexmanip-pretrained-encoders"
VTDEXMANIP_ROOT = "/scratch/project/prj-02-phai-lab/yqq/VTDexManip"

# local_dir_use_symlinks=False forces real file copies instead of HF's default
# blob-cache + relative-symlink scheme. The symlinks it creates are relative to
# the download location's own depth (e.g. "../../../.cache/huggingface/...")
# and silently break once moved to a destination at a *different* depth (the
# staging dir here is 2 levels deep, both real destinations are 3 levels deep)
# -- python's Path.exists() then returns False on the moved symlink even
# though `ls` shows the file "present". Real copies have no such fragility.
local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=VTDEXMANIP_ROOT + "/_pretrained_staging",
    local_dir_use_symlinks=False,
)
print("Downloaded to:", local_dir, flush=True)

staging = VTDEXMANIP_ROOT + "/_pretrained_staging"
targets = [
    ("model_and_config", VTDEXMANIP_ROOT + "/model/vitac/model_and_config"),
    ("pre_model_baselines", VTDEXMANIP_ROOT + "/model/backbones/pre_model_baselines"),
]
for src_name, dst in targets:
    src = os.path.join(staging, src_name)
    if os.path.islink(dst) or os.path.exists(dst):
        shutil.rmtree(dst) if os.path.isdir(dst) and not os.path.islink(dst) else os.remove(dst)
        print(f"removed stale (possibly symlink-broken) destination: {dst}", flush=True)
    shutil.move(src, dst)
    print(f"moved {src} -> {dst}", flush=True)

print("DONE", flush=True)
