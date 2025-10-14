"""
Generic PyTorch model class for loading trained models
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List


class GenericCNN(nn.Module):
    """Generic CNN model that can load various architectures"""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(GenericCNN, self).__init__()
        
        # Simple CNN architecture that works for most image classification tasks
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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
