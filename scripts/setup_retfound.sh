#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/third_party/RETFound_MAE"
WEIGHTS_DIR="$REPO_ROOT/models/retfound"
WEIGHTS="$WEIGHTS_DIR/RETFound_cfp_weights.pth"
WEIGHTS_FULL="$WEIGHTS_DIR/RETFound_cfp_weights.full.pth"

if [[ ! -d "$VENDOR_DIR" ]]; then
  mkdir -p "$(dirname "$VENDOR_DIR")"
  git clone --depth 1 https://github.com/rmaphoh/RETFound_MAE.git "$VENDOR_DIR"
fi

mkdir -p "$WEIGHTS_DIR"
if [[ ! -f "$WEIGHTS" ]]; then
  if [[ ! -f "$WEIGHTS_FULL" ]]; then
    curl -L --fail -o "$WEIGHTS_FULL" \
      "https://drive.usercontent.google.com/download?id=1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE&export=download&confirm=t"
  fi

  docker compose run --rm train python - <<'PY'
from pathlib import Path
import torch

full = Path("models/retfound/RETFound_cfp_weights.full.pth")
out = Path("models/retfound/RETFound_cfp_weights.pth")
state = torch.load(full, map_location="cpu")
if isinstance(state, dict) and "model" in state:
    state = {"model": state["model"]}
torch.save(state, out)
print("wrote", out)
PY

  rm -f "$WEIGHTS_FULL"
fi

echo "RETFound code: $VENDOR_DIR"
echo "RETFound weights: $WEIGHTS"
echo "Config hint: set model.retfound_ckpt: models/retfound/RETFound_cfp_weights.pth"
