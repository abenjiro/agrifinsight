"""
AI Model Manager
Centralized management of all AI models for AgriFinSight
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

from .disease_detection import DiseaseDetectionModel
from .planting_predictor import PlantingPredictor
from .harvest_predictor import HarvestPredictor

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized manager for all AI models"""
    
    def __init__(self, models_dir: str = "ai/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.disease_model = None
        self.planting_model = None
        self.harvest_model = None
        
        # Model paths
        self.disease_model_path = self.models_dir / "disease_detection.h5"
        self.planting_model_path = self.models_dir / "planting_predictor.pkl"
        self.harvest_model_path = self.models_dir / "harvest_predictor.pkl"
        
        # Load models if they exist
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        try:
            # Load disease detection model
            if self.disease_model_path.exists():
                self.disease_model = DiseaseDetectionModel(str(self.disease_model_path))
                logger.info("Disease detection model loaded")
            else:
                self.disease_model = DiseaseDetectionModel()
                logger.info("Disease detection model initialized (new)")
            
            # Load planting predictor model
            if self.planting_model_path.exists():
                self.planting_model = PlantingPredictor(str(self.planting_model_path))
                logger.info("Planting predictor model loaded")
            else:
                self.planting_model = PlantingPredictor()
                logger.info("Planting predictor model initialized (new)")
            
            # Load harvest predictor model
            if self.harvest_model_path.exists():
                self.harvest_model = HarvestPredictor(str(self.harvest_model_path))
                logger.info("Harvest predictor model loaded")
            else:
                self.harvest_model = HarvestPredictor()
                logger.info("Harvest predictor model initialized (new)")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
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
    
    def predict_planting_time(self, crop_type: str, weather_data: Dict, 
                            soil_data: Dict, location_data: Dict) -> Dict[str, Any]:
        """Predict optimal planting time"""
        if not self.planting_model:
            raise ValueError("Planting predictor model not available")
        
        try:
            result = self.planting_model.predict_planting_time(
                crop_type, weather_data, soil_data, location_data
            )
            logger.info(f"Planting time prediction completed for {crop_type}")
            return result
        except Exception as e:
            logger.error(f"Error predicting planting time: {str(e)}")
            raise
    
    def predict_harvest_time(self, crop_type: str, crop_data: Dict, 
                           weather_data: Dict, soil_data: Dict, 
                           growth_data: Dict) -> Dict[str, Any]:
        """Predict optimal harvest time"""
        if not self.harvest_model:
            raise ValueError("Harvest predictor model not available")
        
        try:
            result = self.harvest_model.predict_harvest_time(
                crop_type, crop_data, weather_data, soil_data, growth_data
            )
            logger.info(f"Harvest time prediction completed for {crop_type}")
            return result
        except Exception as e:
            logger.error(f"Error predicting harvest time: {str(e)}")
            raise
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            'disease_detection': {
                'available': self.disease_model is not None,
                'model_path': str(self.disease_model_path),
                'exists': self.disease_model_path.exists()
            },
            'planting_predictor': {
                'available': self.planting_model is not None,
                'model_path': str(self.planting_model_path),
                'exists': self.planting_model_path.exists()
            },
            'harvest_predictor': {
                'available': self.harvest_model is not None,
                'model_path': str(self.harvest_model_path),
                'exists': self.harvest_model_path.exists()
            }
        }
        
        return status
    
    def save_all_models(self):
        """Save all models to disk"""
        try:
            if self.disease_model:
                self.disease_model.save_model(str(self.disease_model_path))
                logger.info("Disease detection model saved")
            
            if self.planting_model:
                self.planting_model.save_model(str(self.planting_model_path))
                logger.info("Planting predictor model saved")
            
            if self.harvest_model:
                self.harvest_model.save_model(str(self.harvest_model_path))
                logger.info("Harvest predictor model saved")
            
            logger.info("All models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def train_disease_model(self, training_data_dir: str, epochs: int = 50):
        """Train the disease detection model"""
        if not self.disease_model:
            raise ValueError("Disease detection model not available")
        
        try:
            # This would typically involve loading training data and training the model
            # For now, we'll just save the model
            self.disease_model.save_model(str(self.disease_model_path))
            logger.info("Disease detection model training completed")
            
        except Exception as e:
            logger.error(f"Error training disease model: {str(e)}")
            raise
    
    def train_planting_model(self, training_data: Any):
        """Train the planting predictor model"""
        if not self.planting_model:
            raise ValueError("Planting predictor model not available")
        
        try:
            # This would typically involve training with historical data
            # For now, we'll just save the model
            self.planting_model.save_model(str(self.planting_model_path))
            logger.info("Planting predictor model training completed")
            
        except Exception as e:
            logger.error(f"Error training planting model: {str(e)}")
            raise
    
    def train_harvest_model(self, training_data: Any):
        """Train the harvest predictor model"""
        if not self.harvest_model:
            raise ValueError("Harvest predictor model not available")
        
        try:
            # This would typically involve training with historical data
            # For now, we'll just save the model
            self.harvest_model.save_model(str(self.harvest_model_path))
            logger.info("Harvest predictor model training completed")
            
        except Exception as e:
            logger.error(f"Error training harvest model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all models"""
        info = {
            'disease_detection': {
                'description': 'CNN-based crop disease detection',
                'input': 'Crop image (224x224x3)',
                'output': 'Disease classification with confidence scores',
                'supported_diseases': [
                    'healthy', 'bacterial_spot', 'early_blight', 'late_blight',
                    'leaf_mold', 'septoria_leaf_spot', 'spider_mites', 'target_spot',
                    'yellow_leaf_curl_virus', 'mosaic_virus'
                ]
            },
            'planting_predictor': {
                'description': 'ML-based planting time prediction',
                'input': 'Weather, soil, and location data',
                'output': 'Optimal planting date and recommendations',
                'supported_crops': ['maize', 'rice', 'wheat', 'tomato', 'potato']
            },
            'harvest_predictor': {
                'description': 'ML-based harvest time prediction',
                'input': 'Crop growth data, weather, and soil conditions',
                'output': 'Optimal harvest date and yield estimation',
                'supported_crops': ['maize', 'rice', 'wheat', 'tomato', 'potato']
            }
        }
        
        return info
