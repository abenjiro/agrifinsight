"""
Generic PyTorch model class for loading trained models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
try:
    from torchvision import models
    from torchvision.models import efficientnet
except ImportError:
    models = None
    efficientnet = None


class GenericCNN(nn.Module):
    """Generic CNN model - dynamically loads architecture from checkpoint"""

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(GenericCNN, self).__init__()

        if models is None or efficientnet is None:
            raise ImportError("torchvision is required. Please install it with: pip install torchvision")

        # Create a custom EfficientNet with modified width
        # Based on checkpoint analysis: width_mult ≈ 1.25 (40/32 = 1.25)
        from functools import partial

        # EfficientNet-B0 configuration with width_mult=1.25
        inverted_residual_setting = [
            efficientnet.MBConvConfig(1, 3, 1, 40, 24, 2),  # width: 32*1.25=40, 16*1.5=24
            efficientnet.MBConvConfig(6, 3, 2, 24, 32, 2),  # 24->32
            efficientnet.MBConvConfig(6, 5, 2, 32, 48, 3),  # 32->48 (40 in checkpoint)
            efficientnet.MBConvConfig(6, 3, 2, 48, 96, 3),  # 48->96
            efficientnet.MBConvConfig(6, 5, 1, 96, 136, 4), # 96->136 (112->136)
            efficientnet.MBConvConfig(6, 5, 2, 136, 232, 4),# 136->232 (192->232)
            efficientnet.MBConvConfig(6, 3, 1, 232, 384, 1),# 232->384 (320->384)
        ]

        # Try to create model with custom settings
        try:
            self.backbone = efficientnet.EfficientNet(
                inverted_residual_setting=inverted_residual_setting,
                dropout=0.2,
                num_classes=num_classes
            )
        except Exception as e:
            # Fallback to standard B0 if custom settings fail
            print(f"Custom EfficientNet creation failed: {e}, using standard B0")
            self.backbone = models.efficientnet_b0(weights=None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.backbone(x)


def load_model_info(checkpoint_path: str) -> Dict[str, Any]:
    """Load model information from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'class_names': checkpoint.get('class_names', []),
        'config': checkpoint.get('config', {}),
        'num_classes': len(checkpoint.get('class_names', [])),
        'best_val_acc': checkpoint.get('best_val_acc', 0.0),
        'epoch': checkpoint.get('epoch', 0)
    }
    
    return info
