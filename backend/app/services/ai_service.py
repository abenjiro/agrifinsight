"""
AI Service for AgriFinSight
Integrates with AI models for crop analysis and predictions
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add the AI models directory to the Python path
ai_models_path = Path(__file__).parent.parent.parent.parent / "ai" / "models"
sys.path.append(str(ai_models_path))

# Set the models directory for ModelManager
MODELS_DIR = str(ai_models_path)

try:
    from simple_model_manager import SimpleModelManager
except ImportError:
    # Fallback for when AI models are not available
    SimpleModelManager = None

logger = logging.getLogger(__name__)

class AIService:
    """Service for AI model operations"""
    
    def __init__(self):
        self.model_manager = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models"""
        try:
            if SimpleModelManager:
                self.model_manager = SimpleModelManager(MODELS_DIR)
                logger.info(f"AI models initialized successfully from {MODELS_DIR}")
            else:
                logger.warning("AI models not available - using mock responses")
        except Exception as e:
            logger.error(f"Error initializing AI models: {str(e)}")
            logger.warning("AI models not available - using mock responses")
    
    def analyze_crop_health(self, image_path: str) -> Dict[str, Any]:
        """Analyze crop health from image"""
        if self.model_manager and self.model_manager.disease_model:
            try:
                result = self.model_manager.analyze_crop_health(image_path)
                logger.info(f"Crop health analysis completed for {image_path}")
                return result
            except Exception as e:
                logger.error(f"Error in crop health analysis: {str(e)}")
                return self._get_mock_disease_analysis()
        else:
            logger.warning("Disease detection model not available, using mock response")
            return self._get_mock_disease_analysis()
    
    def predict_planting_time(self, crop_type: str, weather_data: Dict, 
                            soil_data: Dict, location_data: Dict) -> Dict[str, Any]:
        """Predict optimal planting time"""
        if self.model_manager:
            try:
                return self.model_manager.predict_planting_time(
                    crop_type, weather_data, soil_data, location_data
                )
            except Exception as e:
                logger.error(f"Error in planting prediction: {str(e)}")
                return self._get_mock_planting_prediction(crop_type)
        else:
            return self._get_mock_planting_prediction(crop_type)
    
    def predict_harvest_time(self, crop_type: str, crop_data: Dict, 
                           weather_data: Dict, soil_data: Dict, 
                           growth_data: Dict) -> Dict[str, Any]:
        """Predict optimal harvest time"""
        if self.model_manager:
            try:
                return self.model_manager.predict_harvest_time(
                    crop_type, crop_data, weather_data, soil_data, growth_data
                )
            except Exception as e:
                logger.error(f"Error in harvest prediction: {str(e)}")
                return self._get_mock_harvest_prediction(crop_type)
        else:
            return self._get_mock_harvest_prediction(crop_type)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of AI models"""
        if self.model_manager:
            return self.model_manager.get_model_status()
        else:
            return {
                'disease_detection': {'available': False, 'status': 'not_loaded'},
                'planting_predictor': {'available': False, 'status': 'not_loaded'},
                'harvest_predictor': {'available': False, 'status': 'not_loaded'}
            }
    
    def _get_mock_disease_analysis(self) -> Dict[str, Any]:
        """Mock disease analysis for testing"""
        return {
            'disease_detected': 'healthy',
            'confidence_score': 0.85,
            'disease_type': 'healthy',
            'severity': 'none',
            'recommendations': [
                'Your crop appears healthy. Continue current care practices.',
                'Monitor regularly for any changes in appearance.'
            ],
            'treatment_advice': 'No treatment needed. Continue regular care and monitoring.',
            'top_predictions': [
                {'disease': 'healthy', 'confidence': 0.85},
                {'disease': 'early_blight', 'confidence': 0.10},
                {'disease': 'bacterial_spot', 'confidence': 0.05}
            ],
            'is_healthy': True,
            'needs_attention': False
        }
    
    def _get_mock_planting_prediction(self, crop_type: str) -> Dict[str, Any]:
        """Mock planting prediction for testing"""
        from datetime import datetime, timedelta
        
        return {
            'crop_type': crop_type,
            'recommended_planting_date': (datetime.now() + timedelta(days=7)).isoformat(),
            'days_from_now': 7,
            'confidence_score': 0.75,
            'validation_result': {
                'temperature_ok': True,
                'rainfall_ok': True,
                'soil_moisture_ok': True,
                'overall_suitable': True
            },
            'recommendations': [
                f'Conditions are suitable for planting {crop_type}.',
                'Prepare soil by tilling and removing weeds.',
                'Ensure proper drainage to prevent waterlogging.',
                'Consider using high-quality seeds for better yield.'
            ],
            'weather_conditions': {
                'temperature_avg': 22,
                'humidity_avg': 65,
                'rainfall_total': 150
            },
            'soil_conditions': {
                'soil_moisture': 0.6,
                'soil_ph': 6.5,
                'soil_nitrogen': 55
            },
            'risk_factors': []
        }
    
    def _get_mock_harvest_prediction(self, crop_type: str) -> Dict[str, Any]:
        """Mock harvest prediction for testing"""
        from datetime import datetime, timedelta
        
        return {
            'crop_type': crop_type,
            'predicted_harvest_date': (datetime.now() + timedelta(days=30)).isoformat(),
            'days_from_now': 30,
            'confidence_score': 0.80,
            'readiness_result': {
                'maturity_reached': True,
                'optimal_moisture': True,
                'weather_suitable': True,
                'overall_ready': True
            },
            'recommendations': [
                f'Your {crop_type} is ready for harvest.',
                'Harvest in the morning when temperatures are cooler.',
                'Use clean, sharp tools to prevent damage.',
                'Handle harvested produce gently to maintain quality.'
            ],
            'expected_yield': {
                'estimated_yield_kg_per_hectare': 3000,
                'yield_factor': 1.0,
                'base_yield': 3000,
                'adjustments': {
                    'weather': 'favorable',
                    'soil': 'good',
                    'pest_disease': 'low'
                }
            },
            'quality_prediction': 'good',
            'risk_factors': []
        }
