from __future__ import annotations

import math
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def _load_retfound_vit_l(img_size: int):
    candidates = [
        (Path(__file__).resolve().parents[2] / ".." / "RETFound").resolve(),
        (Path(__file__).resolve().parents[2] / ".." / "retfound").resolve(),
    ]
    retfound_dir = next((p for p in candidates if p.exists()), candidates[0])
    if str(retfound_dir) not in sys.path:
        sys.path.insert(0, str(retfound_dir))
    import models_vit  # type: ignore

    return models_vit.RETFound_mae(img_size=img_size, num_classes=0, global_pool=False)


def _clean_checkpoint_dict(state: dict) -> dict:
    if "model" in state:
        state = state["model"]
    elif "teacher" in state:
        state = state["teacher"]
    out = {}
    for k, v in state.items():
        k = k.replace("backbone.", "")
        k = k.replace("mlp.w12.", "mlp.fc1.")
        k = k.replace("mlp.w3.", "mlp.fc2.")
        if k.startswith("head."):
            continue
        out[k] = v
    return out


def _interpolate_pos_embed(pos_embed: torch.Tensor, h: int, w: int) -> torch.Tensor:
    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]
    n = patch_pos.shape[1]
    old = int(math.sqrt(n))
    patch_pos = patch_pos.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
    return torch.cat([cls_pos, patch_pos], dim=1)


class RetFoundSimpleFPN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        out_channels: int = 256,
        ckpt_path: str | None = None,
        freeze_blocks: int = 0,
    ):
        super().__init__()
        self.vit = backbone
        self.out_channels = out_channels
        self.proj = nn.Conv2d(self.vit.embed_dim, out_channels, kernel_size=1)

        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu")
            cleaned = _clean_checkpoint_dict(state)
            self.vit.load_state_dict(cleaned, strict=False)

        self.set_freeze_blocks(freeze_blocks)

    def set_freeze_blocks(self, n: int) -> None:
        for p in self.vit.patch_embed.parameters():
            p.requires_grad = n <= 0
        for i, blk in enumerate(self.vit.blocks):
            req = i >= n
            for p in blk.parameters():
                p.requires_grad = req

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        b, _, h_in, w_in = x.shape
        p = self.vit.patch_embed.patch_size
        ph = p[0] if isinstance(p, tuple) else p
        pw = p[1] if isinstance(p, tuple) else p

        tokens = self.vit.patch_embed(x)
        h = h_in // ph
        w = w_in // pw

        cls = self.vit.cls_token.expand(b, -1, -1)
        pos = _interpolate_pos_embed(self.vit.pos_embed, h, w)
        x_tok = torch.cat((cls, tokens), dim=1)
        x_tok = self.vit.pos_drop(x_tok + pos)

        for blk in self.vit.blocks:
            x_tok = blk(x_tok)
        if hasattr(self.vit, "norm"):
            x_tok = self.vit.norm(x_tok)

        feat = x_tok[:, 1:, :].transpose(1, 2).reshape(b, self.vit.embed_dim, h, w)
        feat = self.proj(feat)

        p3 = feat
        p2 = F.interpolate(p3, scale_factor=2.0, mode="bilinear", align_corners=False)
        p4 = F.max_pool2d(p3, kernel_size=2, stride=2)
        p5 = F.max_pool2d(p4, kernel_size=2, stride=2)
        return OrderedDict({"p2": p2, "p3": p3, "p4": p4, "p5": p5})


def build_detector(cfg: dict) -> FasterRCNN:
    model_cfg = cfg["model"]
    num_classes = int(model_cfg["num_classes"])

    backbone_name = model_cfg["backbone"]
    if backbone_name == "retfound_vit_l":
        vit = _load_retfound_vit_l(int(model_cfg.get("input_size", 1024)))
        backbone = RetFoundSimpleFPN(
            vit,
            out_channels=int(model_cfg.get("fpn_out_channels", 256)),
            ckpt_path=model_cfg.get("retfound_ckpt") or None,
            freeze_blocks=int(model_cfg.get("freeze_blocks", 0)),
        )
        feat_names = ["p2", "p3", "p4", "p5"]
    elif backbone_name == "retfound_vit_b":
        vit = torchvision.models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
        backbone = RetFoundSimpleFPN(
            vit,
            out_channels=int(model_cfg.get("fpn_out_channels", 256)),
            ckpt_path=model_cfg.get("retfound_ckpt") or None,
            freeze_blocks=int(model_cfg.get("freeze_blocks", 0)),
        )
        feat_names = ["p2", "p3", "p4", "p5"]
    else:
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone="DEFAULT",
            num_classes=num_classes,
            rpn_fg_iou_thresh=float(model_cfg.get("rpn_fg_iou_thresh", 0.7)),
            rpn_bg_iou_thresh=float(model_cfg.get("rpn_bg_iou_thresh", 0.3)),
            box_fg_iou_thresh=float(model_cfg.get("roi_fg_iou_thresh", 0.5)),
            box_bg_iou_thresh=float(model_cfg.get("roi_bg_iou_thresh", 0.5)),
            rpn_pre_nms_top_n_train=int(model_cfg.get("rpn_pre_nms_top_n_train", 2000)),
            rpn_post_nms_top_n_train=int(model_cfg.get("rpn_post_nms_top_n_train", 1000)),
            rpn_pre_nms_top_n_test=int(model_cfg.get("rpn_pre_nms_top_n_test", 1000)),
            rpn_post_nms_top_n_test=int(model_cfg.get("rpn_post_nms_top_n_test", 500)),
        )

    anchor_sizes = model_cfg.get("anchor_sizes", [[16], [32], [64], [128]])
    aspect_ratios = model_cfg.get("aspect_ratios", [[0.5, 1.0, 2.0]] * 4)

    anchor_generator = AnchorGenerator(
        sizes=tuple(tuple(int(s) for s in level) for level in anchor_sizes),
        aspect_ratios=tuple(tuple(float(r) for r in level) for level in aspect_ratios),
    )

    roi_pooler = MultiScaleRoIAlign(featmap_names=feat_names, output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        rpn_fg_iou_thresh=float(model_cfg.get("rpn_fg_iou_thresh", 0.7)),
        rpn_bg_iou_thresh=float(model_cfg.get("rpn_bg_iou_thresh", 0.3)),
        box_fg_iou_thresh=float(model_cfg.get("roi_fg_iou_thresh", 0.5)),
        box_bg_iou_thresh=float(model_cfg.get("roi_bg_iou_thresh", 0.5)),
        rpn_pre_nms_top_n_train=int(model_cfg.get("rpn_pre_nms_top_n_train", 2000)),
        rpn_post_nms_top_n_train=int(model_cfg.get("rpn_post_nms_top_n_train", 1000)),
        rpn_pre_nms_top_n_test=int(model_cfg.get("rpn_pre_nms_top_n_test", 1000)),
        rpn_post_nms_top_n_test=int(model_cfg.get("rpn_post_nms_top_n_test", 500)),
        box_detections_per_img=int(model_cfg.get("box_detections_per_img", 200)),
    )
    return model
