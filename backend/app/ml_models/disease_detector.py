"""
Disease Detection Inference Service
Uses trained EfficientNet-B3 model for plant disease classification
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import logging
from typing import Dict, List, Tuple

from app.ml_models.plant_disease_model import PlantDiseaseClassifier
from app.ml_models.class_mappings import CLASS_NAMES, TREATMENT_RECOMMENDATIONS, get_class_info
from app.ml_models.model_loader import get_model_loader

logger = logging.getLogger(__name__)


class DiseaseDetector:
    """Disease detection inference service"""

    def __init__(self, model_path: str = None, device: str = None, model_url: str = None):
        """
        Initialize the disease detector

        Args:
            model_path: Path to the trained model file (local)
            device: Device to run inference on ('cuda', 'mps', or 'cpu')
            model_url: URL to download model from (S3, GCS, HTTP, etc.)
        """
        # Check if model should be downloaded from URL
        if model_url:
            logger.info(f"Loading model from URL: {model_url}")
            model_loader = get_model_loader()
            model_path = model_loader.get_model_path(model_url)
        elif model_path is None:
            # Check environment variable for model URL
            model_url_env = os.getenv('MODEL_URL')
            if model_url_env:
                logger.info(f"Loading model from MODEL_URL env: {model_url_env}")
                model_loader = get_model_loader()
                model_path = model_loader.get_model_path(model_url_env)
            else:
                # Fallback to local model
                model_path = os.path.join(
                    os.path.dirname(__file__),
                    'disease_detection_model.pth'
                )

        if device is None:
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing disease detector on device: {self.device}")

        # Model parameters (must match training configuration)
        self.num_classes = len(CLASS_NAMES)
        self.img_size = (224, 224)

        # Initialize model
        self.model = PlantDiseaseClassifier(
            num_classes=self.num_classes,
            model_name='efficientnet_b3',
            pretrained=False,  # We're loading trained weights
            dropout_rate=0.3
        )

        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model weights")

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Disease detector initialized successfully with {self.num_classes} classes")

        except Exception as e:
            logger.error(f"Failed to load disease detection model: {e}")
            raise

        # Define image preprocessing transforms (must match training)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for model inference

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Apply transforms
            image_tensor = self.transform(image)

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise

    def predict(
        self,
        image_path: str,
        top_k: int = 3
    ) -> Dict:
        """
        Predict disease from image

        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)

            # Run inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)

            # Get top k predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes), dim=1)

            # Convert to lists
            top_probs = top_probs[0].cpu().numpy().tolist()
            top_indices = top_indices[0].cpu().numpy().tolist()

            # Format predictions
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                class_name = CLASS_NAMES[idx]
                class_info = get_class_info(class_name)
                treatment_info = TREATMENT_RECOMMENDATIONS.get(class_name, {})

                predictions.append({
                    'class_name': class_name,
                    'crop': class_info['crop'],
                    'condition': class_info['condition'],
                    'is_healthy': class_info['is_healthy'],
                    'confidence': float(prob),
                    'confidence_percentage': float(prob * 100),
                    'disease_info': treatment_info
                })

            # Primary prediction
            primary_prediction = predictions[0]

            # Determine if high confidence
            is_confident = primary_prediction['confidence'] > 0.7

            result = {
                'success': True,
                'prediction': primary_prediction,
                'alternative_predictions': predictions[1:] if len(predictions) > 1 else [],
                'is_confident': is_confident,
                'model_version': 'efficientnet_b3_v1',
                'total_classes': self.num_classes
            }

            logger.info(
                f"Prediction: {primary_prediction['crop']} - {primary_prediction['condition']} "
                f"(confidence: {primary_prediction['confidence_percentage']:.2f}%)"
            )

            return result

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {
                'success': False,
                'error': str(e),
                'prediction': None
            }

    def get_treatment_recommendations(self, class_name: str) -> Dict:
        """
        Get detailed treatment recommendations for a disease

        Args:
            class_name: The disease class name

        Returns:
            Treatment recommendations
        """
        return TREATMENT_RECOMMENDATIONS.get(class_name, {
            'disease_name': 'Unknown',
            'severity': 'unknown',
            'description': 'No information available',
            'treatments': ['Consult agricultural extension service'],
            'prevention': ['Monitor plants regularly']
        })


# Singleton instance
_detector = None


def get_disease_detector() -> DiseaseDetector:
    """Get or create singleton disease detector instance"""
    global _detector

    if _detector is None:
        try:
            _detector = DiseaseDetector()
            logger.info("Disease detector singleton created")
        except Exception as e:
            logger.error(f"Failed to create disease detector: {e}")
            raise

    return _detector
