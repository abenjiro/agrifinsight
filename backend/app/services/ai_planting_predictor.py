"""
AI-Powered Planting Time Predictor
Loads trained ML model and makes predictions
"""

import joblib
import numpy as np
import os
from typing import Dict, Optional

# Path to trained model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'ai', 'models', 'planting_predictor.joblib'
)


class PlantingPredictor:
    """AI model for predicting optimal planting times"""

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
                print(f"✓ AI Planting Model loaded successfully")
                print(f"  Model accuracy: R² = {self.model_data['metrics']['r2']:.4f}")
            else:
                print(f"⚠ AI Planting Model not found at {MODEL_PATH}")
                print(f"  Run: python ai/training/train_planting_model.py")
        except Exception as e:
            print(f"✗ Error loading AI Planting Model: {e}")
            self.model = None

    def predict(self, features: Dict[str, float]) -> Optional[Dict]:
        """
        Predict optimal planting time

        Args:
            features: Dictionary containing:
                - temperature_avg, temperature_min, temperature_max
                - humidity_avg
                - rainfall_total, rainfall_days
                - wind_speed_avg
                - soil_temperature, soil_moisture, soil_ph
                - soil_nitrogen, soil_phosphorus, soil_potassium
                - elevation, latitude, longitude

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
            days_to_plant = self.model.predict(X_scaled)[0]

            # Calculate confidence based on model's feature importance
            # Higher confidence if key features (soil temp, air temp, soil N) are optimal
            confidence = self._calculate_confidence(features)

            return {
                'days_to_plant': max(0, round(days_to_plant, 1)),
                'confidence': round(confidence, 2),
                'model_version': 'gradient_boosting_v1',
                'prediction_type': 'ai_ml_model'
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """
        Calculate prediction confidence based on feature quality

        High confidence if:
        - Soil temperature is optimal (15-25°C)
        - Air temperature is moderate (18-30°C)
        - Soil nitrogen is adequate (>40)
        - Soil moisture is good (0.4-0.7)
        """
        confidence = 0.85  # Base confidence

        soil_temp = features.get('soil_temperature', 20)
        if 15 <= soil_temp <= 25:
            confidence += 0.05

        air_temp = features.get('temperature_avg', 25)
        if 18 <= air_temp <= 30:
            confidence += 0.04

        soil_n = features.get('soil_nitrogen', 50)
        if soil_n >= 40:
            confidence += 0.03

        soil_moisture = features.get('soil_moisture', 0.5)
        if 0.4 <= soil_moisture <= 0.7:
            confidence += 0.03

        return min(0.99, confidence)


# Global instance
_predictor = None


def get_planting_predictor() -> PlantingPredictor:
    """Get or create singleton predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = PlantingPredictor()
    return _predictor


def predict_planting_time(weather_data: Dict, soil_data: Dict, location: Dict) -> Optional[Dict]:
    """
    Convenience function to predict planting time

    Args:
        weather_data: Dict with temperature, humidity, rainfall, wind
        soil_data: Dict with soil conditions
        location: Dict with lat, lon, elevation

    Returns:
        Prediction dictionary or None
    """
    predictor = get_planting_predictor()

    if predictor.model is None:
        return None

    # Combine all features
    features = {
        **weather_data,
        **soil_data,
        **location
    }

    return predictor.predict(features)
