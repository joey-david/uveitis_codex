import torch
import torch.nn as nn
import timm
from torchvision import models


class CnnEncoder(nn.Module):
    def __init__(self, name="resnet18", embed_dim=256):
        super().__init__()
        base = getattr(models, name)(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.proj = nn.Conv2d(base.fc.in_features, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.proj(x)
        tokens = x.flatten(2).transpose(1, 2)
        return tokens


class VitEncoder(nn.Module):
    def __init__(self, name="vit_base_patch16_224", pretrained=False, img_size=224, embed_dim=256):
        super().__init__()
        self.vit = timm.create_model(
            name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,
        )
        vit_dim = self.vit.num_features
        self.proj = nn.Linear(vit_dim, embed_dim)

    def forward(self, x):
        tokens = self.vit.forward_features(x)
        if tokens.ndim == 2:
            tokens = tokens[:, None, :]
        if tokens.shape[1] > 1:
            tokens = tokens[:, 1:]
        return self.proj(tokens)


class ViewEncoder(nn.Module):
    def __init__(self, cnn_name, vit_name, embed_dim, img_size, pretrained_vit):
        super().__init__()
        self.cnn = CnnEncoder(cnn_name, embed_dim=embed_dim)
        self.vit = VitEncoder(vit_name, pretrained=pretrained_vit, img_size=img_size, embed_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        cnn_tokens = self.cnn(x)
        vit_tokens = self.vit(x)
        tokens = torch.cat([cnn_tokens, vit_tokens], dim=1)
        return self.norm(tokens)


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.attn_ab = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_ba = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fusion_logits = nn.Parameter(torch.zeros(2))

    def forward(self, tokens_a, tokens_b, weights=None):
        ab, _ = self.attn_ab(tokens_a, tokens_b, tokens_b, need_weights=False)
        ba, _ = self.attn_ba(tokens_b, tokens_a, tokens_a, need_weights=False)
        ab_pool = ab.mean(dim=1)
        ba_pool = ba.mean(dim=1)
        if weights is None:
            alpha, beta = torch.softmax(self.fusion_logits, dim=0)
        else:
            weights = torch.tensor(weights, device=ab_pool.device, dtype=ab_pool.dtype)
            weights = weights / weights.sum()
            alpha, beta = weights[0], weights[1]
        fused = alpha * ab_pool + beta * ba_pool
        return fused, (alpha, beta)


class MVCAViT(nn.Module):
    def __init__(
        self,
        num_classes=5,
        num_boxes=10,
        embed_dim=256,
        img_size=224,
        cnn_name="resnet18",
        vit_name="vit_base_patch16_224",
        pretrained_vit=False,
        num_heads=8,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.macula = ViewEncoder(cnn_name, vit_name, embed_dim, img_size, pretrained_vit)
        self.optic = ViewEncoder(cnn_name, vit_name, embed_dim, img_size, pretrained_vit)
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads)
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_boxes * 4),
        )
        self.obj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_boxes),
        )
        self._external_weights = None

    def set_fusion_weights(self, weights):
        self._external_weights = weights

    def forward(self, macula, optic):
        macula_tokens = self.macula(macula)
        optic_tokens = self.optic(optic)
        fused, weights = self.fusion(macula_tokens, optic_tokens, self._external_weights)
        logits = self.cls_head(fused)
        boxes = self.box_head(fused).view(-1, self.num_boxes, 4).sigmoid()
        obj_logits = self.obj_head(fused).view(-1, self.num_boxes)
        return {
            "logits": logits,
            "boxes": boxes,
            "obj_logits": obj_logits,
            "fusion_weights": weights,
        }
