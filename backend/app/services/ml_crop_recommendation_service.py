"""
ML-based Crop Recommendation Service
Uses trained PyTorch model for intelligent crop recommendations
Uses configuration from app.config for model paths
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class CropRecommendationModel(nn.Module):
    """
    Neural network for crop recommendation
    Must match the architecture used during training
    """

    def __init__(self, input_size=9, num_crops=6, hidden_sizes=[128, 64, 32]):
        super(CropRecommendationModel, self).__init__()

        # Shared layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(prev_size, num_crops)
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(prev_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.shared_layers(x)
        crop_logits = self.classifier(features)
        suitability = self.regressor(features) * 100
        return crop_logits, suitability.squeeze()


class MLCropRecommendationService:
    """ML-based crop recommendation service"""

    # Feature encoding mappings (must match training)
    SOIL_TYPE_MAPPING = {
        "sandy": 0, "sandy loam": 1, "loam": 2, "clay loam": 3,
        "clay": 4, "silty clay": 5, "silt": 6
    }

    CLIMATE_ZONE_MAPPING = {
        "tropical": 0, "subtropical": 1, "temperate": 2, "arid": 3, "semi-arid": 4,
        "tropical rainforest": 0, "tropical savanna": 0  # Map to tropical
    }

    TERRAIN_TYPE_MAPPING = {
        "flat": 0, "gently sloping": 1, "rolling": 2, "hilly": 3, "mountainous": 4, "valley": 5
    }

    # Crop information database
    CROP_INFO = {
        "Maize": {
            "water_requirements": "medium",
            "growth_duration_days": 120,
            "care_difficulty": "easy",
            "expected_yield": {"min": 2000, "max": 5000, "unit": "kg/acre"},
            "market_demand": "high",
            "profit_margin": 45,
            "benefits": [
                "Staple food crop with consistent demand",
                "Multiple varieties available",
                "Good rotation crop"
            ],
            "challenges": [
                "Susceptible to fall armyworm",
                "Requires adequate moisture during pollination"
            ],
            "tips": [
                "Plant at the onset of rains",
                "Maintain proper spacing (75cm x 25cm)",
                "Apply fertilizer in splits"
            ]
        },
        "Rice": {
            "water_requirements": "high",
            "growth_duration_days": 150,
            "care_difficulty": "moderate",
            "expected_yield": {"min": 3000, "max": 6000, "unit": "kg/acre"},
            "market_demand": "high",
            "profit_margin": 50,
            "benefits": [
                "High market value",
                "Staple food with guaranteed market",
                "Can grow in flooded conditions"
            ],
            "challenges": [
                "Requires consistent water supply",
                "Labor intensive harvesting"
            ],
            "tips": [
                "Ensure proper water management",
                "Use quality seeds",
                "Control weeds early"
            ]
        },
        "Cassava": {
            "water_requirements": "low",
            "growth_duration_days": 300,
            "care_difficulty": "easy",
            "expected_yield": {"min": 8000, "max": 15000, "unit": "kg/acre"},
            "market_demand": "medium",
            "profit_margin": 40,
            "benefits": [
                "Drought tolerant",
                "Low input requirements",
                "Long storage in ground"
            ],
            "challenges": [
                "Long maturity period",
                "Post-harvest perishability"
            ],
            "tips": [
                "Plant quality stems",
                "Ridge planting improves yield",
                "Harvest at right maturity"
            ]
        },
        "Tomato": {
            "water_requirements": "medium",
            "growth_duration_days": 90,
            "care_difficulty": "moderate",
            "expected_yield": {"min": 5000, "max": 12000, "unit": "kg/acre"},
            "market_demand": "high",
            "profit_margin": 60,
            "benefits": [
                "High returns per acre",
                "Short maturity period",
                "Year-round demand"
            ],
            "challenges": [
                "Susceptible to diseases",
                "Requires regular care",
                "Perishable product"
            ],
            "tips": [
                "Use disease-resistant varieties",
                "Stake plants for support",
                "Regular pest monitoring"
            ]
        },
        "Soybean": {
            "water_requirements": "medium",
            "growth_duration_days": 100,
            "care_difficulty": "easy",
            "expected_yield": {"min": 800, "max": 1500, "unit": "kg/acre"},
            "market_demand": "high",
            "profit_margin": 55,
            "benefits": [
                "Fixes nitrogen in soil",
                "Good rotation crop",
                "Growing export market"
            ],
            "challenges": [
                "Seed quality critical",
                "Storage pest issues"
            ],
            "tips": [
                "Inoculate seeds before planting",
                "Plant early for best yields",
                "Control weeds in first 40 days"
            ]
        },
        "Groundnut": {
            "water_requirements": "medium",
            "growth_duration_days": 120,
            "care_difficulty": "moderate",
            "expected_yield": {"min": 1000, "max": 2500, "unit": "kg/acre"},
            "market_demand": "high",
            "profit_margin": 50,
            "benefits": [
                "Nitrogen fixing legume",
                "Good market price",
                "Improves soil fertility"
            ],
            "challenges": [
                "Requires calcium",
                "Aflatoxin management needed"
            ],
            "tips": [
                "Apply gypsum for pod filling",
                "Proper drying before storage",
                "Use clean seeds"
            ]
        }
    }

    def __init__(self, model_path: str = None):
        """Initialize the ML crop recommendation service"""
        # Use GPU if configured and available
        use_gpu = settings.use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.model = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.crop_names = None
        self.encoders = None

        # Default model path from settings
        if model_path is None:
            base_dir = Path(__file__).parent.parent.parent.parent
            model_path = str(base_dir / settings.model_path / settings.crop_recommendation_model)

        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load the trained PyTorch model"""
        try:
            logger.info(f"Loading ML crop recommendation model from: {self.model_path}")

            # Check if model file exists
            if not Path(self.model_path).exists():
                logger.warning(f"Model file not found: {self.model_path}")
                logger.warning("ML recommendations will not be available. Please train the model first.")
                return

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Extract metadata
            self.crop_names = checkpoint['crop_names']
            self.scaler_mean = np.array(checkpoint['scaler_mean'])
            self.scaler_scale = np.array(checkpoint['scaler_scale'])
            self.encoders = checkpoint.get('encoders', {})

            num_crops = checkpoint['num_crops']
            input_size = checkpoint['input_size']

            # Initialize model
            self.model = CropRecommendationModel(input_size=input_size, num_crops=num_crops)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"✓ Model loaded successfully. Crops: {self.crop_names}")
            logger.info(f"✓ Test accuracy: {checkpoint.get('test_accuracy', 'N/A')}%")

        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.model = None

    def _encode_features(self, farm_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Encode farm data into model input features
        Returns normalized feature vector
        """
        try:
            # Extract and validate features
            latitude = farm_data.get('latitude')
            longitude = farm_data.get('longitude')
            altitude = farm_data.get('altitude', 0)
            avg_temperature = farm_data.get('avg_temperature')
            avg_annual_rainfall = farm_data.get('avg_annual_rainfall')
            soil_ph = farm_data.get('soil_ph')
            soil_type = farm_data.get('soil_type', '').lower()
            climate_zone = farm_data.get('climate_zone', '').lower()
            terrain_type = farm_data.get('terrain_type', 'flat').lower()

            # Check required fields
            if any(x is None for x in [latitude, longitude, avg_temperature, avg_annual_rainfall, soil_ph]):
                logger.warning("Missing required features for ML prediction")
                return None

            # Encode categorical features
            soil_type_encoded = self.SOIL_TYPE_MAPPING.get(soil_type, 2)  # Default to 'loam'
            climate_zone_encoded = self.CLIMATE_ZONE_MAPPING.get(climate_zone, 0)  # Default to 'tropical'
            terrain_type_encoded = self.TERRAIN_TYPE_MAPPING.get(terrain_type, 0)  # Default to 'flat'

            # Create feature vector
            features = np.array([
                latitude,
                longitude,
                altitude,
                avg_temperature,
                avg_annual_rainfall,
                soil_ph,
                soil_type_encoded,
                climate_zone_encoded,
                terrain_type_encoded
            ], dtype=np.float32)

            # Normalize using scaler from training
            if self.scaler_mean is not None and self.scaler_scale is not None:
                features = (features - self.scaler_mean) / self.scaler_scale

            return features

        except Exception as e:
            logger.error(f"Error encoding features: {e}")
            return None

    def predict(self, farm_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make predictions using the ML model
        Returns dict with crop probabilities and suitability scores
        """
        if self.model is None:
            return None

        try:
            # Encode features
            features = self._encode_features(farm_data)
            if features is None:
                return None

            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                crop_logits, suitability = self.model(features_tensor)

                # Get probabilities
                probabilities = torch.softmax(crop_logits, dim=1).cpu().numpy()[0]

                # Get suitability score
                suit_score = suitability.cpu().item()

            # Create results
            results = {
                'crop_probabilities': {
                    self.crop_names[i]: float(probabilities[i]) * 100
                    for i in range(len(self.crop_names))
                },
                'top_crop': self.crop_names[np.argmax(probabilities)],
                'top_crop_probability': float(np.max(probabilities)) * 100,
                'predicted_suitability': float(suit_score)
            }

            return results

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None

    def generate_recommendations(
        self,
        farm_data: Dict[str, Any],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate crop recommendations using ML model
        """
        logger.info(f"Generating ML-based recommendations for farm at ({farm_data.get('latitude')}, {farm_data.get('longitude')})")

        # Get ML predictions
        ml_results = self.predict(farm_data)

        if ml_results is None:
            logger.warning("ML prediction failed, falling back to rule-based recommendations")
            return []

        # Get crop probabilities
        crop_probs = ml_results['crop_probabilities']

        # Sort crops by probability
        sorted_crops = sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)

        recommendations = []

        for i, (crop_name, probability) in enumerate(sorted_crops[:top_n]):
            crop_info = self.CROP_INFO.get(crop_name, {})

            # Calculate confidence score (based on probability)
            confidence = min(0.95, probability / 100)

            recommendation = {
                "recommended_crop": crop_name,
                "suitability_score": round(probability, 2),
                "confidence_score": round(confidence, 2),
                "prediction_method": "ml_model",
                "model_probability": round(probability, 2),
                "climate_factors": self._get_climate_factors(farm_data),
                "soil_factors": self._get_soil_factors(farm_data),
                "geographic_factors": {
                    "elevation": farm_data.get('altitude'),
                    "terrain": farm_data.get('terrain_type')
                },
                "market_factors": {
                    "demand": crop_info.get('market_demand'),
                    "profit_margin": crop_info.get('profit_margin')
                },
                "planting_season": self._determine_planting_season(farm_data, crop_info),
                "expected_yield_range": crop_info.get('expected_yield'),
                "water_requirements": crop_info.get('water_requirements'),
                "care_difficulty": crop_info.get('care_difficulty'),
                "growth_duration_days": crop_info.get('growth_duration_days'),
                "estimated_profit_margin": crop_info.get('profit_margin'),
                "market_demand": crop_info.get('market_demand'),
                "benefits": crop_info.get('benefits', []),
                "challenges": crop_info.get('challenges', []),
                "tips": crop_info.get('tips', []),
                "model_version": "ml_v1.0"
            }

            # Add alternative crops
            alternatives = [crop for crop, _ in sorted_crops[i+1:i+4]]
            recommendation['alternative_crops'] = alternatives

            recommendations.append(recommendation)

        return recommendations

    def _get_climate_factors(self, farm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract climate factors from farm data"""
        return {
            "temperature": farm_data.get('avg_temperature'),
            "rainfall": farm_data.get('avg_annual_rainfall'),
            "climate_zone": farm_data.get('climate_zone')
        }

    def _get_soil_factors(self, farm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract soil factors from farm data"""
        return {
            "ph": farm_data.get('soil_ph'),
            "type": farm_data.get('soil_type'),
            "composition": farm_data.get('soil_composition')
        }

    def _determine_planting_season(self, farm_data: Dict[str, Any], crop_info: Dict[str, Any]) -> str:
        """Determine optimal planting season"""
        climate_zone = farm_data.get('climate_zone', '').lower()

        if 'tropical' in climate_zone:
            if crop_info.get('water_requirements') == 'high':
                return "Beginning of rainy season (April-June)"
            else:
                return "Rainy season (May-July) or with irrigation"
        elif 'temperate' in climate_zone:
            return "Spring (March-May)"
        else:
            return "Consult local agricultural extension"


# Create singleton instance
ml_crop_recommendation_service = MLCropRecommendationService()
