"""
Simple Model Manager for Disease Detection
Handles only the disease detection model to avoid dependency issues
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

from disease_detection import DiseaseDetectionModel

logger = logging.getLogger(__name__)

class SimpleModelManager:
    """Simple manager for disease detection model only"""
    
    def __init__(self, models_dir: str = "ai/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize disease model
        self.disease_model = None
        
        # Model path
        self.disease_model_path = self.models_dir / "crop_disease_model_final.keras"
        
        # Load model if it exists
        self._load_disease_model()
    
    def _load_disease_model(self):
        """Load disease detection model"""
        try:
            if self.disease_model_path.exists():
                self.disease_model = DiseaseDetectionModel(str(self.disease_model_path))
                logger.info(f"Disease detection model loaded from {self.disease_model_path}")
            else:
                logger.warning(f"Disease detection model not found at {self.disease_model_path}")
                self.disease_model = DiseaseDetectionModel()
                logger.info("Disease detection model initialized (new)")
        except Exception as e:
            logger.error(f"Error loading disease model: {str(e)}")
            self.disease_model = None
    
    def analyze_crop_health(self, image_path: str) -> Dict[str, Any]:
        """Analyze crop health from image"""
        if not self.disease_model:
            raise ValueError("Disease detection model not available")
        
        try:
            result = self.disease_model.predict(image_path)
            logger.info(f"Crop health analysis completed for {image_path}")
            return result
        except Exception as e:
            logger.error(f"Error analyzing crop health: {str(e)}")
            raise
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of disease detection model"""
        return {
            'disease_detection': {
                'available': self.disease_model is not None,
                'model_path': str(self.disease_model_path),
                'exists': self.disease_model_path.exists()
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the disease detection model"""
        return {
            'disease_detection': {
                'description': 'CNN-based crop disease detection',
                'input': 'Crop image (224x224x3)',
                'output': 'Disease classification with confidence scores',
                'supported_diseases': self.disease_model.class_names if self.disease_model else []
            }
        }

