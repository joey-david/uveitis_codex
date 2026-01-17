import torch

from mvcavit.model import MVCAViT


def test_model_forward():
    model = MVCAViT(num_classes=3, num_boxes=5, embed_dim=64, img_size=64, vit_name="vit_tiny_patch16_224")
    x = torch.randn(2, 3, 64, 64)
    outputs = model(x, x)
    assert outputs["logits"].shape == (2, 3)
    assert outputs["boxes"].shape == (2, 5, 4)
    assert outputs["obj_logits"].shape == (2, 5)
