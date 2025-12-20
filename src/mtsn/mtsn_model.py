import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Optional
from config import load_config
from src.common.encoders import build_encoder, EncoderBase

class MTSN(nn.Module):
    def __init__(self, embedding_dim: Optional[int] = None, encoder_name: Optional[str] = None):
        super().__init__()

        # Select encoder via config unless explicitly provided
        if encoder_name is None:
            try:
                cfg = load_config()
                encoder_name = getattr(cfg.mtsn, 'encoder_name', 'scratch_resnet18')
            except Exception:
                encoder_name = 'scratch_resnet18'

        # Build encoder wrapper
        self.encoder: EncoderBase = build_encoder(encoder_name)

        # Embedding dimension from encoder factory unless overridden
        self.embedding_dim = int(embedding_dim or getattr(self.encoder, 'embed_dim', 512))

        # Head for single-feature scoring (used by OVV reference screening)
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        # Pairwise classifier over |f1 - f2| concatenated with 4 size scalars
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim + 4, 128),  # +4 for widths/heights
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)  # logit
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input image into a feature vector.
        """
        return self.encoder.encode(x)
    
    def forward_once(self, x):
        return self.encode(x)
    
    def forward(self, x1, x2, width1=None, height1=None, width2=None, height2=None):
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)
        diff = torch.abs(f1 - f2)
        
        # Concatenate size information with the difference features
        if width1 is None or height1 is None or width2 is None or height2 is None:
            raise ValueError("Width and height must be provided for both images.")
        sizes = torch.stack([width1, height1, width2, height2], dim=1)
        combined = torch.cat([diff, sizes], dim=1)
        
        out = self.classifier(combined)
        return out.squeeze(dim=1)
    
if __name__ == '__main__':
    # Test model output shape
    model = MTSN(encoder_name='scratch_resnet18')
    dummy1 = torch.randn(8, 3, 64, 64)
    dummy2 = torch.randn(8, 3, 64, 64)
    out = model(dummy1, dummy2)
    assert out.shape == (8,)
    print("Output shape is correct:", out.shape)
    
    try:
        from torchviz import make_dot
        
        # Create graph visualization for the encoder
        y1 = model.encoder(dummy1)
        encoder_viz = make_dot(y1, params=dict(model.encoder.named_parameters()))
        encoder_viz.render("mtsn_encoder", format="png")
        
        # Create graph visualization for the full model
        y = model(dummy1, dummy2)
        model_viz = make_dot(y, params=dict(model.named_parameters()))
        model_viz.render("mtsn_full_model", format="png")
        
        print("\nModel architecture visualizations saved as:")
        print("- mtsn_encoder.png")
        print("- mtsn_full_model.png")
        
    except ImportError:
        print("torchviz and graphviz not installed.")
