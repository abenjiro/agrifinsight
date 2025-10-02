"""
Harvest Time Prediction Model
Predicts optimal harvest timing based on crop growth stages and environmental conditions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HarvestPredictor:
    """Predict optimal harvest times based on crop growth and environmental conditions"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'days_since_planting', 'temperature_avg', 'humidity_avg',
            'rainfall_total', 'soil_moisture', 'soil_nitrogen',
            'soil_phosphorus', 'soil_potassium', 'growth_stage',
            'plant_height', 'leaf_count', 'flowering_stage',
            'fruit_development', 'disease_pressure', 'pest_pressure'
        ]
        
        # Crop-specific growth parameters
        self.crop_growth_params = {
            'maize': {
                'maturity_days': 90,
                'flowering_days': 60,
                'grain_fill_days': 30,
                'optimal_harvest_moisture': 0.25
            },
            'rice': {
                'maturity_days': 120,
                'flowering_days': 80,
                'grain_fill_days': 40,
                'optimal_harvest_moisture': 0.20
            },
            'wheat': {
                'maturity_days': 150,
                'flowering_days': 100,
                'grain_fill_days': 50,
                'optimal_harvest_moisture': 0.14
            },
            'tomato': {
                'maturity_days': 75,
                'flowering_days': 45,
                'fruit_development_days': 30,
                'optimal_harvest_moisture': 0.85
            },
            'potato': {
                'maturity_days': 100,
                'flowering_days': 60,
                'tuber_development_days': 40,
                'optimal_harvest_moisture': 0.75
            }
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self):
        """Build the harvest prediction model"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        logger.info("Harvest prediction model built successfully")
    
    def prepare_features(self, crop_data: Dict, weather_data: Dict, 
                        soil_data: Dict, growth_data: Dict) -> np.ndarray:
        """Prepare features for model prediction"""
        features = []
        
        # Crop growth features
        features.extend([
            crop_data.get('days_since_planting', 0),
            crop_data.get('plant_height', 0),
            crop_data.get('leaf_count', 0),
            crop_data.get('flowering_stage', 0),
            crop_data.get('fruit_development', 0),
            crop_data.get('disease_pressure', 0),
            crop_data.get('pest_pressure', 0)
        ])
        
        # Weather features
        features.extend([
            weather_data.get('temperature_avg', 20),
            weather_data.get('humidity_avg', 60),
            weather_data.get('rainfall_total', 100)
        ])
        
        # Soil features
        features.extend([
            soil_data.get('soil_moisture', 0.5),
            soil_data.get('soil_nitrogen', 50),
            soil_data.get('soil_phosphorus', 30),
            soil_data.get('soil_potassium', 40)
        ])
        
        return np.array(features).reshape(1, -1)
    
    def predict_harvest_time(self, crop_type: str, crop_data: Dict, 
                           weather_data: Dict, soil_data: Dict, 
                           growth_data: Dict) -> Dict:
        """Predict optimal harvest time for a crop"""
        try:
            # Check if crop is supported
            if crop_type.lower() not in self.crop_growth_params:
                raise ValueError(f"Crop type '{crop_type}' not supported")
            
            # Prepare features
            features = self.prepare_features(crop_data, weather_data, soil_data, growth_data)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Make prediction (days from now)
            days_to_harvest = self.model.predict(features_scaled)[0]
            
            # Calculate harvest date
            harvest_date = datetime.now() + timedelta(days=int(days_to_harvest))
            
            # Validate harvest readiness
            readiness_result = self._validate_harvest_readiness(
                crop_type, crop_data, weather_data, soil_data
            )
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                crop_type, crop_data, readiness_result
            )
            
            # Generate recommendations
            recommendations = self._generate_harvest_recommendations(
                crop_type, crop_data, readiness_result
            )
            
            # Calculate expected yield
            expected_yield = self._estimate_yield(
                crop_type, crop_data, weather_data, soil_data
            )
            
            result = {
                'crop_type': crop_type,
                'predicted_harvest_date': harvest_date.isoformat(),
                'days_from_now': int(days_to_harvest),
                'confidence_score': confidence,
                'readiness_result': readiness_result,
                'recommendations': recommendations,
                'expected_yield': expected_yield,
                'quality_prediction': self._predict_quality(crop_type, crop_data, weather_data),
                'risk_factors': self._identify_harvest_risks(crop_type, crop_data, weather_data)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting harvest time: {str(e)}")
            raise
    
    def _validate_harvest_readiness(self, crop_type: str, crop_data: Dict, 
                                  weather_data: Dict, soil_data: Dict) -> Dict:
        """Validate if crop is ready for harvest"""
        params = self.crop_growth_params[crop_type.lower()]
        days_since_planting = crop_data.get('days_since_planting', 0)
        
        readiness = {
            'maturity_reached': False,
            'optimal_moisture': False,
            'weather_suitable': False,
            'overall_ready': False
        }
        
        # Check maturity
        if days_since_planting >= params['maturity_days']:
            readiness['maturity_reached'] = True
        
        # Check moisture content
        current_moisture = crop_data.get('moisture_content', 0.3)
        optimal_moisture = params['optimal_harvest_moisture']
        if abs(current_moisture - optimal_moisture) <= 0.05:
            readiness['optimal_moisture'] = True
        
        # Check weather suitability
        temperature = weather_data.get('temperature_avg', 20)
        humidity = weather_data.get('humidity_avg', 60)
        if 15 <= temperature <= 30 and humidity < 80:
            readiness['weather_suitable'] = True
        
        # Overall readiness
        readiness['overall_ready'] = all([
            readiness['maturity_reached'],
            readiness['optimal_moisture'],
            readiness['weather_suitable']
        ])
        
        return readiness
    
    def _calculate_confidence(self, crop_type: str, crop_data: Dict, 
                            readiness_result: Dict) -> float:
        """Calculate confidence score for harvest prediction"""
        base_confidence = 0.5
        
        # Increase confidence based on readiness
        if readiness_result['maturity_reached']:
            base_confidence += 0.3
        if readiness_result['optimal_moisture']:
            base_confidence += 0.2
        if readiness_result['weather_suitable']:
            base_confidence += 0.2
        
        # Adjust based on crop age
        days_since_planting = crop_data.get('days_since_planting', 0)
        params = self.crop_growth_params[crop_type.lower()]
        maturity_ratio = min(1.0, days_since_planting / params['maturity_days'])
        base_confidence += maturity_ratio * 0.2
        
        # Ensure confidence is between 0 and 1
        return min(1.0, max(0.0, base_confidence))
    
    def _generate_harvest_recommendations(self, crop_type: str, crop_data: Dict, 
                                        readiness_result: Dict) -> List[str]:
        """Generate harvest recommendations"""
        recommendations = []
        
        if readiness_result['overall_ready']:
            recommendations.append(f"Your {crop_type} is ready for harvest.")
        else:
            recommendations.append(f"Your {crop_type} may need more time to mature.")
        
        if not readiness_result['maturity_reached']:
            days_since_planting = crop_data.get('days_since_planting', 0)
            params = self.crop_growth_params[crop_type.lower()]
            remaining_days = params['maturity_days'] - days_since_planting
            recommendations.append(f"Wait approximately {remaining_days} more days for maturity.")
        
        if not readiness_result['optimal_moisture']:
            recommendations.append("Monitor moisture content. Harvest when optimal moisture is reached.")
        
        if not readiness_result['weather_suitable']:
            recommendations.append("Wait for better weather conditions for harvesting.")
        
        # General recommendations
        recommendations.append("Harvest in the morning when temperatures are cooler.")
        recommendations.append("Use clean, sharp tools to prevent damage.")
        recommendations.append("Handle harvested produce gently to maintain quality.")
        recommendations.append("Store harvested produce in appropriate conditions immediately.")
        
        return recommendations
    
    def _estimate_yield(self, crop_type: str, crop_data: Dict, 
                       weather_data: Dict, soil_data: Dict) -> Dict:
        """Estimate expected yield based on current conditions"""
        # Base yield estimates (kg per hectare)
        base_yields = {
            'maize': 3000,
            'rice': 4000,
            'wheat': 2500,
            'tomato': 50000,
            'potato': 20000
        }
        
        base_yield = base_yields.get(crop_type.lower(), 2000)
        
        # Adjust based on conditions
        yield_factor = 1.0
        
        # Weather adjustment
        temperature = weather_data.get('temperature_avg', 20)
        if 18 <= temperature <= 25:
            yield_factor += 0.2
        elif temperature < 15 or temperature > 30:
            yield_factor -= 0.3
        
        # Soil adjustment
        soil_nitrogen = soil_data.get('soil_nitrogen', 50)
        if soil_nitrogen > 60:
            yield_factor += 0.1
        elif soil_nitrogen < 30:
            yield_factor -= 0.2
        
        # Disease/pest adjustment
        disease_pressure = crop_data.get('disease_pressure', 0)
        pest_pressure = crop_data.get('pest_pressure', 0)
        yield_factor -= (disease_pressure + pest_pressure) * 0.1
        
        # Calculate final yield
        estimated_yield = base_yield * yield_factor
        
        return {
            'estimated_yield_kg_per_hectare': int(estimated_yield),
            'yield_factor': round(yield_factor, 2),
            'base_yield': base_yield,
            'adjustments': {
                'weather': 'favorable' if yield_factor > 1.0 else 'unfavorable',
                'soil': 'good' if soil_nitrogen > 50 else 'needs_improvement',
                'pest_disease': 'low' if (disease_pressure + pest_pressure) < 0.3 else 'high'
            }
        }
    
    def _predict_quality(self, crop_type: str, crop_data: Dict, 
                        weather_data: Dict) -> str:
        """Predict harvest quality"""
        quality_score = 0.5
        
        # Weather impact on quality
        temperature = weather_data.get('temperature_avg', 20)
        humidity = weather_data.get('humidity_avg', 60)
        
        if 18 <= temperature <= 25 and 40 <= humidity <= 70:
            quality_score += 0.3
        elif temperature > 30 or humidity > 80:
            quality_score -= 0.2
        
        # Disease/pest impact
        disease_pressure = crop_data.get('disease_pressure', 0)
        pest_pressure = crop_data.get('pest_pressure', 0)
        quality_score -= (disease_pressure + pest_pressure) * 0.2
        
        # Determine quality grade
        if quality_score >= 0.8:
            return 'excellent'
        elif quality_score >= 0.6:
            return 'good'
        elif quality_score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _identify_harvest_risks(self, crop_type: str, crop_data: Dict, 
                               weather_data: Dict) -> List[str]:
        """Identify potential risks for harvest"""
        risks = []
        
        # Weather risks
        temperature = weather_data.get('temperature_avg', 20)
        humidity = weather_data.get('humidity_avg', 60)
        
        if temperature > 30:
            risks.append("High temperatures may reduce quality")
        if humidity > 80:
            risks.append("High humidity may cause mold and spoilage")
        if weather_data.get('rainfall_total', 0) > 200:
            risks.append("Excessive rainfall may cause waterlogging")
        
        # Disease/pest risks
        disease_pressure = crop_data.get('disease_pressure', 0)
        pest_pressure = crop_data.get('pest_pressure', 0)
        
        if disease_pressure > 0.5:
            risks.append("High disease pressure may affect quality")
        if pest_pressure > 0.5:
            risks.append("Pest damage may reduce market value")
        
        # Timing risks
        days_since_planting = crop_data.get('days_since_planting', 0)
        params = self.crop_growth_params[crop_type.lower()]
        
        if days_since_planting > params['maturity_days'] + 30:
            risks.append("Over-mature crops may have reduced quality")
        
        return risks
    
    def train_model(self, training_data: pd.DataFrame):
        """Train the model with historical data"""
        try:
            # Prepare features and target
            X = training_data[self.feature_names]
            y = training_data['days_to_harvest']
            
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
