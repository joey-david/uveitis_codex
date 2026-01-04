import os

import pytest
import torch

from stage1.checkpoints import load_checkpoint, sanitize_state_dict, strip_prefix
from stage1.mae import mae_vit_large_patch16


def test_checkpoint_forward():
    ckpt_path = os.environ.get("RETF_CHECKPOINT")
    if not ckpt_path or not os.path.exists(ckpt_path):
        pytest.skip("Set RETF_CHECKPOINT to a local RETFound checkpoint to run this test")
    model = mae_vit_large_patch16()
    state = sanitize_state_dict(strip_prefix(load_checkpoint(ckpt_path)))
    model.load_state_dict(state, strict=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        loss, pred, mask = model(x)
    assert pred.shape[1] == (224 // 16) ** 2
    assert mask.shape[1] == (224 // 16) ** 2
    assert loss.item() >= 0
