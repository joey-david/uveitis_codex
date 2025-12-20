import torch
import torch.nn as nn
from torchvision import transforms

# Torchvision model imports guarded to avoid import errors on older versions
try:
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        resnet50, ResNet50_Weights,
        efficientnet_b0, EfficientNet_B0_Weights,
    )
except Exception:  # pragma: no cover
    resnet18 = resnet50 = efficientnet_b0 = None
    ResNet18_Weights = ResNet50_Weights = EfficientNet_B0_Weights = None
try:
    import timm  # optional, for a wide variety of backbones
    from timm.data import resolve_model_data_config as timm_resolve_cfg
except Exception:  # pragma: no cover
    timm = None
    timm_resolve_cfg = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EncoderBase(nn.Module):
    """Base class for encoders exposing a common interface.

    Required attributes:
    - embed_dim: int
    - preprocess: callable that turns a PIL image into a 4D tensor [1,C,H,W]
    """

    def __init__(self):
        super().__init__()
        self.embed_dim = None
        self.input_size = 224
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.preprocess = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TorchvisionEncoder(EncoderBase):
    def __init__(self, model: nn.Module, embed_dim: int, preprocess=None, *, input_size: int = 224, mean=None, std=None):
        super().__init__()
        self.backbone = model
        self.embed_dim = embed_dim
        self.input_size = input_size or 224
        if mean is not None:
            self.mean = list(mean)
        if std is not None:
            self.std = list(std)
        if preprocess is not None:
            self.preprocess = preprocess
        else:
            self.preprocess = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x in [B,3,H,W]; output [B, D]
        return self.backbone(x)


def _tv_meta_from_weights(weights):
    try:
        # Extract mean/std and recommended size when available
        meta = getattr(weights, 'meta', {})
        mean = meta.get('mean', IMAGENET_MEAN)
        std = meta.get('std', IMAGENET_STD)
        # torchvision uses 'min_size' or 'resize_size'; default 224
        size = meta.get('min_size', meta.get('resize_size', 224))
        if isinstance(size, (list, tuple)):
            size = size[-1]
        return int(size), mean, std
    except Exception:
        return 224, IMAGENET_MEAN, IMAGENET_STD


def _tv_preprocess_from_mean_std(size, mean, std) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_encoder(name: str) -> EncoderBase:
    """Factory for common encoders.

    Supported names:
    - "scratch_resnet18"     → ResNet-18, random init (embed_dim=512)
    - "resnet18_imagenet"    → ResNet-18, ImageNet weights (embed_dim=512)
    - "resnet50_imagenet"    → ResNet-50, ImageNet weights (embed_dim=2048)
    - "efficientnet_b0_imagenet" → EffNet-B0, ImageNet weights (embed_dim=1280)
    """

    key = (name or "").lower()

    if resnet18 is None:
        raise RuntimeError("torchvision models are unavailable; cannot build encoder")

    if key in ("scratch_resnet18", "resnet18_scratch"):
        m = resnet18(weights=None)
        m.fc = nn.Identity()
        return TorchvisionEncoder(m, embed_dim=512, input_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if key == "resnet18_imagenet":
        weights = ResNet18_Weights.IMAGENET1K_V1
        m = resnet18(weights=weights)
        m.fc = nn.Identity()  # yields [B, 512]
        size, mean, std = _tv_meta_from_weights(weights)
        pp = _tv_preprocess_from_mean_std(size, mean, std)
        return TorchvisionEncoder(m, embed_dim=512, preprocess=pp, input_size=size, mean=mean, std=std)

    if key == "resnet50_imagenet":
        weights = ResNet50_Weights.IMAGENET1K_V2
        m = resnet50(weights=weights)
        m.fc = nn.Identity()  # yields [B, 2048]
        size, mean, std = _tv_meta_from_weights(weights)
        pp = _tv_preprocess_from_mean_std(size, mean, std)
        return TorchvisionEncoder(m, embed_dim=2048, preprocess=pp, input_size=size, mean=mean, std=std)

    if key == "efficientnet_b0_imagenet":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        m = efficientnet_b0(weights=weights)
        # EfficientNet returns logits via classifier; replace it with Identity to get pooled features
        m.classifier = nn.Identity()
        # Wrap with a small module to ensure forward returns [B, 1280]
        class EffNetWrapper(nn.Module):
            def __init__(self, eff):
                super().__init__()
                self.model = eff
            def forward(self, x):
                return self.model(x)
        size, mean, std = _tv_meta_from_weights(weights)
        pp = _tv_preprocess_from_mean_std(size, mean, std)
        return TorchvisionEncoder(EffNetWrapper(m), embed_dim=1280, preprocess=pp, input_size=size, mean=mean, std=std)

    # Optional: TIMM models, e.g., "timm:convnext_tiny", "timm:vit_base_patch16_224"
    if key.startswith("timm:"):
        if timm is None:
            raise RuntimeError("Requested a timm model but timm is not installed. `pip install timm` to enable.")
        model_name = key.split(":", 1)[1]
        m = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
        # num_features is standard in timm models
        embed_dim = int(getattr(m, 'num_features', 0) or getattr(m, 'feature_info', {}).get('channels', [0])[-1] or 768)
        # Data config for mean/std/size
        if timm_resolve_cfg is not None:
            dc = timm_resolve_cfg(m)
            size = int(dc.get('input_size', (3, 224, 224))[1])
            mean = list(dc.get('mean', IMAGENET_MEAN))
            std = list(dc.get('std', IMAGENET_STD))
        else:
            size, mean, std = 224, IMAGENET_MEAN, IMAGENET_STD

        class TimmWrapper(nn.Module):
            def __init__(self, mm):
                super().__init__()
                self.model = mm
            def forward(self, x):
                return self.model(x)

        pp = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return TorchvisionEncoder(TimmWrapper(m), embed_dim=embed_dim, preprocess=pp, input_size=size, mean=mean, std=std)


# Convenience helpers used by training/inference
def build_train_transforms(encoder: EncoderBase, aug: bool = True) -> transforms.Compose:
    t = [transforms.Resize((encoder.input_size, encoder.input_size))]
    if aug:
        # Light augmentations to preserve pretraining invariances
        t += [
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
        ]
    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=encoder.mean, std=encoder.std),
    ]
    return transforms.Compose(t)


def build_eval_transforms(encoder: EncoderBase) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((encoder.input_size, encoder.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=encoder.mean, std=encoder.std),
    ])

    raise ValueError(f"Unknown encoder name: {name}")
