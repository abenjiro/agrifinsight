"""
AI-Powered Harvest Time Predictor
Loads trained ML model and makes predictions
"""

import joblib
import numpy as np
import os
from typing import Dict, Optional
from datetime import datetime, timedelta

# Path to trained model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'ai', 'models', 'harvest_predictor.joblib'
)


class HarvestPredictor:
    """AI model for predicting harvest timing"""

    def __init__(self):
        self.model_data = None
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(MODEL_PATH):
                self.model_data = joblib.load(MODEL_PATH)
                self.model = self.model_data['model']
                self.scaler = self.model_data['scaler']
                self.feature_names = self.model_data['feature_names']
                print(f"✓ AI Harvest Model loaded successfully")
                print(f"  Model accuracy: R² = {self.model_data['metrics']['r2']:.4f}")
            else:
                print(f"⚠ AI Harvest Model not found at {MODEL_PATH}")
                print(f"  Run: python ai/training/train_harvest_model.py")
        except Exception as e:
            print(f"✗ Error loading AI Harvest Model: {e}")
            self.model = None

    def predict(self, features: Dict[str, float]) -> Optional[Dict]:
        """
        Predict harvest timing

        Args:
            features: Dictionary containing:
                - days_since_planting
                - plant_height, leaf_count
                - flowering_stage, fruit_development, growth_stage
                - temperature_avg, humidity_avg, rainfall_total
                - soil_moisture, soil_nitrogen, soil_phosphorus, soil_potassium
                - disease_pressure, pest_pressure

        Returns:
            Dictionary with prediction and confidence
        """
        if self.model is None:
            return None

        try:
            # Extract features in correct order
            feature_values = []
            for feature_name in self.feature_names:
                value = features.get(feature_name, 0.0)
                feature_values.append(value)

            # Convert to numpy array and reshape
            X = np.array(feature_values).reshape(1, -1)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make prediction
            days_to_harvest = self.model.predict(X_scaled)[0]

            # Calculate harvest date
            planting_date = features.get('planting_date')
            if planting_date:
                if isinstance(planting_date, str):
                    planting_date = datetime.fromisoformat(planting_date)
                harvest_date = datetime.now() + timedelta(days=int(days_to_harvest))
            else:
                harvest_date = None

            # Calculate confidence
            confidence = self._calculate_confidence(features, days_to_harvest)

            return {
                'days_to_harvest': max(0, round(days_to_harvest, 1)),
                'harvest_date': harvest_date.isoformat() if harvest_date else None,
                'confidence': round(confidence, 2),
                'model_version': 'random_forest_v1',
                'prediction_type': 'ai_ml_model'
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def _calculate_confidence(self, features: Dict[str, float], predicted_days: float) -> float:
        """
        Calculate prediction confidence based on feature quality and growth stage

        High confidence if:
        - Clear growth stage data available
        - Flowering/fruit development stages are clear
        - Low disease/pest pressure
        - Prediction is reasonable (not extreme)
        """
        confidence = 0.85  # Base confidence

        # Growth stage bonus (higher stage = more confident)
        growth_stage = features.get('growth_stage', 0)
        if growth_stage >= 4:  # Fruit development or maturation
            confidence += 0.08
        elif growth_stage >= 3:  # Flowering
            confidence += 0.05

        # Flowering stage bonus
        flowering = features.get('flowering_stage', 0)
        if flowering >= 0.8:
            confidence += 0.03

        # Low stress bonus
        disease = features.get('disease_pressure', 0)
        pest = features.get('pest_pressure', 0)
        if disease < 0.3 and pest < 0.3:
            confidence += 0.04

        # Reasonable prediction bonus (not too extreme)
        if 5 <= predicted_days <= 45:
            confidence += 0.03

        return min(0.99, confidence)


# Global instance
_predictor = None


def get_harvest_predictor() -> HarvestPredictor:
    """Get or create singleton predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = HarvestPredictor()
    return _predictor


def predict_harvest_time(crop_data: Dict, weather_data: Dict, soil_data: Dict,
                         health_data: Dict = None) -> Optional[Dict]:
    """
    Convenience function to predict harvest time

    Args:
        crop_data: Dict with growth metrics (days_since_planting, height, etc.)
        weather_data: Dict with weather conditions
        soil_data: Dict with soil conditions
        health_data: Dict with disease/pest pressure (optional)

    Returns:
        Prediction dictionary or None
    """
    predictor = get_harvest_predictor()

    if predictor.model is None:
        return None

    # Combine all features
    features = {
        **crop_data,
        **weather_data,
        **soil_data,
        **(health_data or {})
    }

    return predictor.predict(features)
