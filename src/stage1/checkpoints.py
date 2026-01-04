from pathlib import Path

import torch


def strip_prefix(state, prefix="module."):
    if not any(k.startswith(prefix) for k in state):
        return state
    return {k[len(prefix):]: v for k, v in state.items()}


def sanitize_state_dict(state):
    cleaned = {}
    for key, value in state.items():
        new_key = key.replace("backbone.", "")
        new_key = new_key.replace("mlp.w12.", "mlp.fc1.")
        new_key = new_key.replace("mlp.w3.", "mlp.fc2.")
        cleaned[new_key] = value
    return cleaned


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def download_retfound_checkpoint(repo_id, cache_dir=None):
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.pth", "*.pt"],
        cache_dir=cache_dir,
    )
    candidates = list(Path(local_dir).glob("*.pth")) + list(Path(local_dir).glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {local_dir}")
    return max(candidates, key=lambda p: p.stat().st_size)


def resolve_checkpoint(path_or_repo, cache_dir=None):
    path = Path(path_or_repo)
    if path.exists():
        return path
    return download_retfound_checkpoint(path_or_repo, cache_dir=cache_dir)


def encoder_state_from_mae(state):
    encoder_state = {}
    for key, value in state.items():
        if key.startswith("decoder_") or key == "mask_token":
            continue
        encoder_state[key] = value
    return encoder_state
