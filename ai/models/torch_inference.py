"""
Torch inference utilities for image classification models.

This module provides a `TorchImageClassifier` that can load either:
- a TorchScript model via `torch.jit.load`, or
- a state_dict checkpoint given a user-defined model class that can be
  imported dynamically.

Preprocessing uses torchvision transforms suitable for standard image
classification models. Adjust normalization means/stds to match training.
"""

import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

try:
    import torchvision.transforms as T
except Exception:  # pragma: no cover - torchvision may be optional in some envs
    T = None  # type: ignore


logger = logging.getLogger(__name__)


class TorchImageClassifier:
    """Generic image classifier for PyTorch models."""

    def __init__(
        self,
        model_path: str,
        model_module: Optional[str] = None,
        model_class_name: Optional[str] = None,
        input_size: Tuple[int, int] = (224, 224),
        class_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        normalization: Optional[Tuple[List[float], List[float]]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_module = model_module
        self.model_class_name = model_class_name
        self.input_size = input_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = class_names or []
        self.normalization = normalization or (
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)

        if T is None:
            raise RuntimeError("torchvision is required for image preprocessing. Please install torchvision.")

        mean, std = self.normalization
        self.transforms = T.Compose(
            [
                T.Resize(self.input_size, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def _load_model(self):
        """Load a TorchScript or eager model with state_dict."""
        # Try TorchScript first
        try:
            model = torch.jit.load(str(self.model_path), map_location=self.device)
            logger.info(f"Loaded TorchScript model from {self.model_path}")
            return model
        except Exception as script_err:
            logger.debug(f"TorchScript load failed: {script_err}")

        # Fallback: eager model with state_dict requires module and class
        if not self.model_module or not self.model_class_name:
            raise RuntimeError(
                "State dict loading requires 'model_module' and 'model_class_name' to be provided"
            )

        try:
            module = importlib.import_module(self.model_module)
            model_class = getattr(module, self.model_class_name)
            model = model_class()
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            # Accept both {'state_dict': ...} and raw state_dict
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"Loaded PyTorch model '{self.model_class_name}' from module '{self.model_module}' with checkpoint {self.model_path}"
            )
            return model
        except Exception as e:
            logger.exception("Failed to load PyTorch model")
            raise e

    def _preprocess(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transforms(image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.inference_mode()
    def predict(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        inputs = self._preprocess(image_path)
        logits = self.model(inputs)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu()

        top_k = min(top_k, probs.numel())
        values, indices = torch.topk(probs, k=top_k)

        predictions: List[Dict[str, Any]] = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            label = self.class_names[idx] if 0 <= idx < len(self.class_names) else str(idx)
            predictions.append({"label": label, "confidence": float(score)})

        best_label = predictions[0]["label"] if predictions else None
        best_conf = predictions[0]["confidence"] if predictions else 0.0

        return {
            "predicted_label": best_label,
            "confidence_score": best_conf,
            "top_predictions": predictions,
        }


