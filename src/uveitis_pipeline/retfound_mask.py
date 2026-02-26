"""RETFound-backed mask-first models and training utilities."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _clean_checkpoint_dict(state: dict) -> dict:
    """Normalize checkpoint keys for RETFound/MAE weights."""
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
    """Resize ViT positional embeddings to a new patch grid."""
    cls_pos = pos_embed[:, :1]
    patch_pos = pos_embed[:, 1:]
    old = int(math.sqrt(max(1, patch_pos.shape[1])))
    patch_pos = patch_pos.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
    return torch.cat([cls_pos, patch_pos], dim=1)


def load_retfound_vit(vendor_dir: str | Path, image_size: int) -> nn.Module:
    """Load RETFound ViT-L architecture from vendor repo."""
    vendor = Path(vendor_dir)
    if not vendor.exists():
        raise FileNotFoundError(
            f"RETFound vendor repo not found: {vendor}. Run scripts/setup_retfound.sh or clone RETFound_MAE there."
        )
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    import models_vit  # type: ignore

    return models_vit.RETFound_mae(img_size=int(image_size), num_classes=0, global_pool=False)


def load_retfound_weights(vit: nn.Module, ckpt_path: str | Path) -> None:
    """Load RETFound checkpoint into ViT backbone with positional interpolation."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    cleaned = _clean_checkpoint_dict(state)

    if "pos_embed" in cleaned and hasattr(vit, "pos_embed"):
        want = tuple(vit.pos_embed.shape)
        got = tuple(cleaned["pos_embed"].shape)
        if want != got:
            n = int(vit.pos_embed.shape[1] - 1)
            g = int(math.sqrt(max(n, 1)))
            cleaned["pos_embed"] = _interpolate_pos_embed(cleaned["pos_embed"], g, g)

    vit.load_state_dict(cleaned, strict=False)


def load_encoder_state(encoder: "RetFoundEncoder", state: dict) -> None:
    """Load encoder state with safe positional embedding interpolation."""
    sd = dict(state)
    key = "vit.pos_embed"
    if key in sd and tuple(sd[key].shape) != tuple(encoder.vit.pos_embed.shape):
        n = int(encoder.vit.pos_embed.shape[1] - 1)
        g = int(math.sqrt(max(n, 1)))
        sd[key] = _interpolate_pos_embed(sd[key], g, g)
    encoder.load_state_dict(sd, strict=False)


class RetFoundEncoder(nn.Module):
    """Expose RETFound token features as a 2D feature map."""

    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit = vit
        self.embed_dim = int(vit.embed_dim)

    def set_freeze_blocks(self, freeze_blocks: int) -> None:
        """Freeze the first N ViT blocks (and patch embed when N>0)."""
        n = int(freeze_blocks)
        req_patch = n <= 0
        for p in self.vit.patch_embed.parameters():
            p.requires_grad = req_patch
        for i, blk in enumerate(self.vit.blocks):
            req = i >= n
            for p in blk.parameters():
                p.requires_grad = req

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return final token features reshaped as BxCxHxW."""
        b, _, h_in, w_in = x.shape
        patch = self.vit.patch_embed.patch_size
        ph = patch[0] if isinstance(patch, tuple) else patch
        pw = patch[1] if isinstance(patch, tuple) else patch
        gh = max(1, h_in // ph)
        gw = max(1, w_in // pw)

        tokens = self.vit.patch_embed(x)
        cls = self.vit.cls_token.expand(b, -1, -1)
        pos = _interpolate_pos_embed(self.vit.pos_embed, gh, gw)

        x_tok = torch.cat((cls, tokens), dim=1)
        x_tok = self.vit.pos_drop(x_tok + pos)
        for blk in self.vit.blocks:
            x_tok = blk(x_tok)
        if hasattr(self.vit, "norm"):
            x_tok = self.vit.norm(x_tok)

        feat = x_tok[:, 1:, :].transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
        return feat


class RetFoundMaskModel(nn.Module):
    """Mask predictor with RETFound encoder and a light decoder head."""

    def __init__(self, encoder: RetFoundEncoder, num_classes: int, decoder_channels: int = 256):
        super().__init__()
        self.encoder = encoder
        c = int(decoder_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder.embed_dim, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.head = nn.Conv2d(c, int(num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return full-resolution per-class mask logits."""
        feat = self.encoder(x)
        low = self.head(self.decoder(feat))
        logits = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class ContrastiveProjectionHead(nn.Module):
    """Projection MLP for contrastive RETFound adaptation."""

    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized projections."""
        z = self.net(x)
        return F.normalize(z, dim=1)


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
    focal_alpha: float = 0.25,
) -> torch.Tensor:
    """Compute BCE/Focal + Dice with sparse-positive handling for lesion masks."""
    pos = target.sum(dim=(0, 2, 3))
    total = torch.tensor(target.shape[0] * target.shape[2] * target.shape[3], device=target.device, dtype=target.dtype)
    neg = total - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(1.0, 64.0)
    bce_raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight.view(1, -1, 1, 1),
        reduction="none",
    )
    if focal_gamma > 0:
        prob = torch.sigmoid(logits)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = focal_alpha * target + (1.0 - focal_alpha) * (1.0 - target)
        bce_map = alpha_t * ((1.0 - pt) ** focal_gamma) * bce_raw
    else:
        bce_map = bce_raw

    if class_weights is not None:
        cw = class_weights.to(logits.device, dtype=logits.dtype).view(1, -1, 1, 1)
        bce = (bce_map * cw).mean()
    else:
        bce = bce_map.mean()

    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(0, 2, 3))
    den = prob.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + 1e-6
    dice = 1.0 - ((2.0 * inter + 1e-6) / den)
    present = target.sum(dim=(0, 2, 3)) > 0
    if class_weights is not None:
        w = class_weights.to(logits.device, dtype=logits.dtype)
        w = w / (w.mean() + 1e-6)
        if torch.any(present):
            wp = w[present]
            dice_pos = (dice[present] * wp).sum() / (wp.sum() + 1e-6)
        else:
            dice_pos = torch.tensor(0.0, device=logits.device)
        if torch.any(~present):
            wn = w[~present]
            dice_neg = (dice[~present] * wn).sum() / (wn.sum() + 1e-6)
        else:
            dice_neg = torch.tensor(0.0, device=logits.device)
    else:
        dice_pos = dice[present].mean() if torch.any(present) else torch.tensor(0.0, device=logits.device)
        dice_neg = dice[~present].mean() if torch.any(~present) else torch.tensor(0.0, device=logits.device)
    return bce + dice_pos + 0.1 * dice_neg


def multi_label_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Return per-class IoU for multi-label masks."""
    pred = (torch.sigmoid(logits) >= float(threshold)).float()
    inter = (pred * target).sum(dim=(0, 2, 3))
    union = (pred + target - pred * target).sum(dim=(0, 2, 3)) + 1e-6
    return inter / union


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Compute SimCLR NT-Xent loss for two augmented batches."""
    b = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / float(temperature)

    eye = torch.eye(2 * b, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    labels = torch.arange(b, device=z.device)
    labels = torch.cat([labels + b, labels], dim=0)
    return F.cross_entropy(sim, labels)
