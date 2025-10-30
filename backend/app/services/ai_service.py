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
# Path: backend/app/services/ai_service.py -> go up to project root
ai_models_path = Path(__file__).parent.parent.parent.parent / "ai" / "models"
sys.path.append(str(ai_models_path))

# Set the models directory for ModelManager
MODELS_DIR = str(ai_models_path)

# Optional: Torch inference support
try:
    from torch_inference import TorchImageClassifier
except Exception:
    TorchImageClassifier = None  # type: ignore

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
        self.torch_classifier = None
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

        # Try to initialize Torch model if available
        try:
            if TorchImageClassifier is not None:
                best_model_path = Path(MODELS_DIR) / "best_model.pth"
                if best_model_path.exists():
                    # Load model info to get class names
                    try:
                        from generic_model import load_model_info
                        model_info = load_model_info(str(best_model_path))
                        class_names = model_info.get('class_names', [])
                        logger.info(f"Found {len(class_names)} classes: {class_names}")
                    except Exception as info_err:
                        logger.warning(f"Could not load model info: {info_err}")
                        class_names = []
                    
                    # Use generic model class
                    self.torch_classifier = TorchImageClassifier(
                        model_path=str(best_model_path),
                        model_module="generic_model",
                        model_class_name="GenericCNN",
                        class_names=class_names,
                    )
                    logger.info(f"Loaded PyTorch model from {best_model_path}")
                else:
                    logger.info(f"No PyTorch model found at {best_model_path}")
            else:
                logger.info("TorchImageClassifier not available; skipping PyTorch model init")
        except Exception as e:
            logger.error(f"Error initializing PyTorch model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def analyze_crop_health(self, image_path: str) -> Dict[str, Any]:
        """Analyze crop health from image"""
        # Prefer PyTorch classifier if available
        if self.torch_classifier is not None:
            try:
                result = self.torch_classifier.predict(image_path)
                logger.info(f"Crop health analysis (torch) completed for {image_path}")
                # Normalize result keys to match expected schema
                top_preds = result.get("top_predictions", [])
                return {
                    'disease_detected': result.get('predicted_label'),
                    'confidence_score': result.get('confidence_score', 0.0),
                    'disease_type': result.get('predicted_label'),
                    'severity': 'unknown',
                    'recommendations': [],
                    'treatment_advice': '',
                    'top_predictions': [
                        {'disease': p.get('label'), 'confidence': p.get('confidence')} for p in top_preds
                    ],
                    'is_healthy': False,
                    'needs_attention': True
                }
            except Exception as e:
                logger.error(f"Error in torch crop health analysis: {str(e)}")
                # Fallback to TF model if available, else mock

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
        status: Dict[str, Any] = {
            'disease_detection': {'available': False, 'status': 'not_loaded'},
            'planting_predictor': {'available': False, 'status': 'not_loaded'},
            'harvest_predictor': {'available': False, 'status': 'not_loaded'},
            'pytorch_classifier': {
                'available': self.torch_classifier is not None,
                'model_path': str((Path(MODELS_DIR) / 'best_model.pth')),
                'loaded': self.torch_classifier is not None
            }
        }

        if self.model_manager:
            try:
                status.update(self.model_manager.get_model_status())
            except Exception:
                pass

        return status
    
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
