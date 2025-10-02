"""
Planting Time Prediction Model
Uses weather data, soil conditions, and historical data to predict optimal planting times
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

class PlantingPredictor:
    """Predict optimal planting times based on environmental conditions"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'temperature_avg', 'temperature_min', 'temperature_max',
            'humidity_avg', 'rainfall_total', 'rainfall_days',
            'wind_speed_avg', 'soil_temperature', 'soil_moisture',
            'soil_ph', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium',
            'elevation', 'latitude', 'longitude'
        ]
        self.crop_requirements = {
            'maize': {
                'min_temp': 10, 'max_temp': 35, 'opt_temp': 25,
                'min_rainfall': 500, 'opt_rainfall': 800,
                'min_soil_moisture': 0.3, 'opt_soil_moisture': 0.6
            },
            'rice': {
                'min_temp': 20, 'max_temp': 40, 'opt_temp': 30,
                'min_rainfall': 1000, 'opt_rainfall': 1500,
                'min_soil_moisture': 0.7, 'opt_soil_moisture': 0.9
            },
            'wheat': {
                'min_temp': 5, 'max_temp': 25, 'opt_temp': 15,
                'min_rainfall': 300, 'opt_rainfall': 600,
                'min_soil_moisture': 0.4, 'opt_soil_moisture': 0.7
            },
            'tomato': {
                'min_temp': 15, 'max_temp': 30, 'opt_temp': 22,
                'min_rainfall': 400, 'opt_rainfall': 700,
                'min_soil_moisture': 0.5, 'opt_soil_moisture': 0.8
            },
            'potato': {
                'min_temp': 8, 'max_temp': 25, 'opt_temp': 18,
                'min_rainfall': 500, 'opt_rainfall': 800,
                'min_soil_moisture': 0.6, 'opt_soil_moisture': 0.8
            }
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build the planting prediction model"""
        # Use Gradient Boosting for better performance on tabular data
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        logger.info("Planting prediction model built successfully")
    
    def prepare_features(self, weather_data: Dict, soil_data: Dict, location_data: Dict) -> np.ndarray:
        """Prepare features for model prediction"""
        features = []
        
        # Weather features
        features.extend([
            weather_data.get('temperature_avg', 20),
            weather_data.get('temperature_min', 15),
            weather_data.get('temperature_max', 25),
            weather_data.get('humidity_avg', 60),
            weather_data.get('rainfall_total', 100),
            weather_data.get('rainfall_days', 10),
            weather_data.get('wind_speed_avg', 5)
        ])
        
        # Soil features
        features.extend([
            soil_data.get('soil_temperature', 18),
            soil_data.get('soil_moisture', 0.5),
            soil_data.get('soil_ph', 6.5),
            soil_data.get('soil_nitrogen', 50),
            soil_data.get('soil_phosphorus', 30),
            soil_data.get('soil_potassium', 40)
        ])
        
        # Location features
        features.extend([
            location_data.get('elevation', 100),
            location_data.get('latitude', 0),
            location_data.get('longitude', 0)
        ])
        
        return np.array(features).reshape(1, -1)
    
    def predict_planting_time(self, crop_type: str, weather_data: Dict, 
                            soil_data: Dict, location_data: Dict) -> Dict:
        """Predict optimal planting time for a crop"""
        try:
            # Check if crop is supported
            if crop_type.lower() not in self.crop_requirements:
                raise ValueError(f"Crop type '{crop_type}' not supported")
            
            # Prepare features
            features = self.prepare_features(weather_data, soil_data, location_data)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Make prediction (days from now)
            days_to_plant = self.model.predict(features_scaled)[0]
            
            # Calculate planting date
            planting_date = datetime.now() + timedelta(days=int(days_to_plant))
            
            # Validate against crop requirements
            validation_result = self._validate_planting_conditions(
                crop_type, weather_data, soil_data
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                crop_type, weather_data, soil_data, validation_result
            )
            
            # Generate recommendations
            recommendations = self._generate_planting_recommendations(
                crop_type, weather_data, soil_data, validation_result
            )
            
            result = {
                'crop_type': crop_type,
                'recommended_planting_date': planting_date.isoformat(),
                'days_from_now': int(days_to_plant),
                'confidence_score': confidence,
                'validation_result': validation_result,
                'recommendations': recommendations,
                'weather_conditions': weather_data,
                'soil_conditions': soil_data,
                'risk_factors': self._identify_risk_factors(crop_type, weather_data, soil_data)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting planting time: {str(e)}")
            raise
    
    def _validate_planting_conditions(self, crop_type: str, weather_data: Dict, 
                                    soil_data: Dict) -> Dict:
        """Validate planting conditions against crop requirements"""
        requirements = self.crop_requirements[crop_type.lower()]
        
        validation = {
            'temperature_ok': False,
            'rainfall_ok': False,
            'soil_moisture_ok': False,
            'overall_suitable': False
        }
        
        # Check temperature
        temp_avg = weather_data.get('temperature_avg', 20)
        if requirements['min_temp'] <= temp_avg <= requirements['max_temp']:
            validation['temperature_ok'] = True
        
        # Check rainfall
        rainfall = weather_data.get('rainfall_total', 100)
        if rainfall >= requirements['min_rainfall']:
            validation['rainfall_ok'] = True
        
        # Check soil moisture
        soil_moisture = soil_data.get('soil_moisture', 0.5)
        if soil_moisture >= requirements['min_soil_moisture']:
            validation['soil_moisture_ok'] = True
        
        # Overall suitability
        validation['overall_suitable'] = all([
            validation['temperature_ok'],
            validation['rainfall_ok'],
            validation['soil_moisture_ok']
        ])
        
        return validation
    
    def _calculate_confidence(self, crop_type: str, weather_data: Dict, 
                            soil_data: Dict, validation_result: Dict) -> float:
        """Calculate confidence score for planting prediction"""
        base_confidence = 0.5
        
        # Increase confidence based on validation results
        if validation_result['temperature_ok']:
            base_confidence += 0.2
        if validation_result['rainfall_ok']:
            base_confidence += 0.2
        if validation_result['soil_moisture_ok']:
            base_confidence += 0.1
        
        # Ensure confidence is between 0 and 1
        return min(1.0, max(0.0, base_confidence))
    
    def _generate_planting_recommendations(self, crop_type: str, weather_data: Dict, 
                                         soil_data: Dict, validation_result: Dict) -> List[str]:
        """Generate planting recommendations"""
        recommendations = []
        
        if validation_result['overall_suitable']:
            recommendations.append(f"Conditions are suitable for planting {crop_type}.")
        else:
            recommendations.append(f"Conditions may not be optimal for planting {crop_type}.")
        
        if not validation_result['temperature_ok']:
            temp_avg = weather_data.get('temperature_avg', 20)
            requirements = self.crop_requirements[crop_type.lower()]
            if temp_avg < requirements['min_temp']:
                recommendations.append("Temperature is too low. Wait for warmer weather.")
            elif temp_avg > requirements['max_temp']:
                recommendations.append("Temperature is too high. Consider planting in cooler conditions.")
        
        if not validation_result['rainfall_ok']:
            rainfall = weather_data.get('rainfall_total', 100)
            requirements = self.crop_requirements[crop_type.lower()]
            recommendations.append(f"Insufficient rainfall. Consider irrigation or wait for more rain.")
        
        if not validation_result['soil_moisture_ok']:
            recommendations.append("Soil moisture is low. Consider irrigation before planting.")
        
        # General recommendations
        recommendations.append("Prepare soil by tilling and removing weeds.")
        recommendations.append("Ensure proper drainage to prevent waterlogging.")
        recommendations.append("Consider using high-quality seeds for better yield.")
        
        return recommendations
    
    def _identify_risk_factors(self, crop_type: str, weather_data: Dict, 
                             soil_data: Dict) -> List[str]:
        """Identify potential risk factors for planting"""
        risk_factors = []
        
        # Weather risks
        if weather_data.get('wind_speed_avg', 0) > 15:
            risk_factors.append("High wind speed may damage young plants")
        
        if weather_data.get('rainfall_days', 0) > 20:
            risk_factors.append("Excessive rainy days may cause waterlogging")
        
        # Soil risks
        soil_ph = soil_data.get('soil_ph', 6.5)
        if soil_ph < 5.5 or soil_ph > 8.0:
            risk_factors.append(f"Soil pH ({soil_ph}) is outside optimal range (5.5-8.0)")
        
        # Temperature risks
        temp_avg = weather_data.get('temperature_avg', 20)
        requirements = self.crop_requirements[crop_type.lower()]
        if temp_avg < requirements['min_temp']:
            risk_factors.append("Temperature below minimum requirement")
        elif temp_avg > requirements['max_temp']:
            risk_factors.append("Temperature above maximum requirement")
        
        return risk_factors
    
    def train_model(self, training_data: pd.DataFrame):
        """Train the model with historical data"""
        try:
            # Prepare features and target
            X = training_data[self.feature_names]
            y = training_data['days_to_plant']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model trained successfully. MSE: {mse:.2f}, R²: {r2:.2f}")
            
            return {
                'mse': mse,
                'r2_score': r2,
                'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        if self.model:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a pre-trained model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
