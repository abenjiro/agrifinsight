"""
Wrapper to load and use pre-trained checkpoint directly
without needing to match architecture exactly
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class CheckpointWrapper(nn.Module):
    """Loads a model checkpoint and creates a forward-compatible wrapper"""

    def __init__(self, checkpoint_path: str, num_classes: int = 39):
        super(CheckpointWrapper, self).__init__()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Create a module dict to hold all the weights
        self._state_dict = state_dict
        self.num_classes = num_classes

        # Register all parameters from the state dict
        for name, param in state_dict.items():
            # Replace dots with underscores for parameter names
            param_name = name.replace('.', '_')
            self.register_parameter(param_name, nn.Parameter(param, requires_grad=False))

    def forward(self, x):
        """
        This is a placeholder - the actual model should be properly reconstructed.
        For now, this raises an error to indicate the architecture needs to be fixed.
        """
        raise NotImplementedError(
            "CheckpointWrapper cannot perform forward passes. "
            "The model architecture needs to match the checkpoint."
        )


def create_model_from_checkpoint(checkpoint_path: str) -> nn.Module:
    """
    Attempts to create the correct model architecture from a checkpoint.

    This function examines the checkpoint structure and tries to reconstruct
    the original model architecture.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Try to infer the model type from the keys
    first_key = list(state_dict.keys())[0]

    if 'backbone.features' in first_key:
        # This is an EfficientNet-based model
        from torchvision import models

        # Get number of classes from the final layer
        final_weight_key = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k][-1]
        num_classes = state_dict[final_weight_key].shape[0]

        print(f"Detected EfficientNet with {num_classes} classes")

        # Try different EfficientNet variants
        for variant_name, variant_fn in [
            ('B0', models.efficientnet_b0),
            ('B1', models.efficientnet_b1),
            ('B2', models.efficientnet_b2),
        ]:
            try:
                model = variant_fn(weights=None)
                # Modify classifier
                in_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(in_features, num_classes)
                )

                # Try to load with strict=False
                missing, unexpected = model.load_state_dict(state_dict, strict=False)

                # Check if most weights loaded successfully
                total_keys = len(state_dict)
                missing_count = len(missing)
                match_ratio = (total_keys - missing_count) / total_keys

                print(f"  EfficientNet-{variant_name}: {match_ratio*100:.1f}% weights matched")

                if match_ratio > 0.5:  # If more than 50% of weights match
                    return model

            except Exception as e:
                print(f"  EfficientNet-{variant_name} failed: {e}")
                continue

        raise RuntimeError("Could not find matching EfficientNet architecture")

    else:
        raise RuntimeError(f"Unknown model architecture (first key: {first_key})")
