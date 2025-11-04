"""
Harvest Prediction Service
Provides intelligent harvest timing recommendations and yield predictions
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from app.services.weather_service import weather_service

logger = logging.getLogger(__name__)


class HarvestService:
    """Service for harvest timing predictions and yield estimation"""

    # Crop maturity indicators
    CROP_MATURITY_INFO = {
        'maize': {
            'days_to_maturity': {'min': 90, 'typical': 120, 'max': 150},
            'maturity_indicators': [
                'Husks turn brown and dry',
                'Kernels hard and dent when pressed',
                'Black layer forms at kernel base',
                'Moisture content below 25%'
            ],
            'harvest_window_days': 14,
            'optimal_moisture': 20,
            'yield_per_acre': {'min': 2000, 'typical': 3500, 'max': 5000},  # kg
            'post_harvest_care': [
                'Dry to 13-14% moisture for storage',
                'Shell and clean immediately',
                'Store in dry, ventilated area'
            ]
        },
        'rice': {
            'days_to_maturity': {'min': 120, 'typical': 150, 'max': 180},
            'maturity_indicators': [
                'Grains turn golden yellow',
                '80% of grains are hard',
                'Moisture content 20-25%',
                'Panicles bend downward'
            ],
            'harvest_window_days': 7,
            'optimal_moisture': 22,
            'yield_per_acre': {'min': 3000, 'typical': 4500, 'max': 6000},
            'post_harvest_care': [
                'Dry to 14% moisture',
                'Thresh and winnow promptly',
                'Store in sealed containers'
            ]
        },
        'cassava': {
            'days_to_maturity': {'min': 270, 'typical': 300, 'max': 365},
            'maturity_indicators': [
                'Leaves turn yellow and fall',
                'Stems become woody',
                'Tubers are firm and well-developed',
                'Plants are 10-12 months old'
            ],
            'harvest_window_days': 30,
            'optimal_moisture': None,
            'yield_per_acre': {'min': 8000, 'typical': 12000, 'max': 15000},
            'post_harvest_care': [
                'Process within 24-48 hours',
                'Store in cool, humid conditions',
                'Remove damaged tubers'
            ]
        },
        'tomato': {
            'days_to_maturity': {'min': 70, 'typical': 90, 'max': 120},
            'maturity_indicators': [
                'Fruits show full color (red/yellow)',
                'Fruits firm but slightly soft',
                'Easy to detach from stem',
                'Characteristic aroma present'
            ],
            'harvest_window_days': 21,
            'optimal_moisture': None,
            'yield_per_acre': {'min': 5000, 'typical': 8500, 'max': 12000},
            'post_harvest_care': [
                'Handle gently to avoid bruising',
                'Cool immediately after harvest',
                'Sort by ripeness stage',
                'Use within 5-7 days or refrigerate'
            ]
        },
        'soybean': {
            'days_to_maturity': {'min': 90, 'typical': 100, 'max': 120},
            'maturity_indicators': [
                'Leaves turn yellow and drop',
                'Pods turn brown/tan',
                'Seeds rattle in pods',
                'Moisture content 13-15%'
            ],
            'harvest_window_days': 10,
            'optimal_moisture': 13,
            'yield_per_acre': {'min': 800, 'typical': 1200, 'max': 1500},
            'post_harvest_care': [
                'Dry to 12% moisture',
                'Clean and sort',
                'Treat against storage pests',
                'Store in sealed containers'
            ]
        },
        'groundnut': {
            'days_to_maturity': {'min': 100, 'typical': 120, 'max': 140},
            'maturity_indicators': [
                'Leaves turn yellow',
                'Inner pod walls show dark color',
                'Veins in pod darken',
                'Kernels fill pods completely'
            ],
            'harvest_window_days': 14,
            'optimal_moisture': 10,
            'yield_per_acre': {'min': 1000, 'typical': 1800, 'max': 2500},
            'post_harvest_care': [
                'Dry to 8-10% moisture',
                'Cure properly to prevent aflatoxin',
                'Shell and sort',
                'Store in dry, cool conditions'
            ]
        }
    }

    async def get_harvest_prediction(
        self,
        crop_type: str,
        planting_date: str,
        farm_data: Dict,
        current_growth_stage: Optional[str] = None
    ) -> Dict:
        """
        Get comprehensive harvest prediction and recommendations
        """
        logger.info(f"Generating harvest prediction for {crop_type}")

        latitude = farm_data.get('latitude')
        longitude = farm_data.get('longitude')

        if not latitude or not longitude:
            raise ValueError("Farm location (latitude/longitude) is required")

        # Parse planting date
        try:
            plant_date = datetime.strptime(planting_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid planting date format. Use YYYY-MM-DD")

        # Calculate maturity dates
        maturity_info = self._calculate_maturity_dates(crop_type, plant_date)

        # Get current crop age
        crop_age_days = (datetime.now() - plant_date).days

        # Estimate current growth stage if not provided
        if not current_growth_stage:
            current_growth_stage = self._estimate_growth_stage(crop_type, crop_age_days)

        # Get weather forecast for harvest window
        weather_forecast = None
        if maturity_info['days_until_maturity'] <= 14:
            weather_forecast = await weather_service.get_harvest_weather_advice(
                latitude, longitude, crop_type, maturity_info['estimated_harvest_date']
            )

        # Calculate yield prediction
        yield_prediction = self._predict_yield(
            crop_type,
            farm_data,
            crop_age_days,
            current_growth_stage
        )

        # Generate harvest recommendations
        harvest_advice = self._generate_harvest_advice(
            crop_type,
            maturity_info,
            weather_forecast,
            current_growth_stage
        )

        return {
            'crop_type': crop_type,
            'planting_date': planting_date,
            'crop_age_days': crop_age_days,
            'current_growth_stage': current_growth_stage,
            'maturity_timeline': maturity_info,
            'harvest_readiness': self._assess_harvest_readiness(
                crop_age_days, maturity_info
            ),
            'yield_prediction': yield_prediction,
            'weather_forecast': weather_forecast,
            'harvest_recommendations': harvest_advice,
            'maturity_indicators': self.CROP_MATURITY_INFO.get(
                crop_type.lower(), {}
            ).get('maturity_indicators', []),
            'post_harvest_care': self.CROP_MATURITY_INFO.get(
                crop_type.lower(), {}
            ).get('post_harvest_care', []),
            'generated_at': datetime.now().isoformat()
        }

    def _calculate_maturity_dates(self, crop_type: str, planting_date: datetime) -> Dict:
        """
        Calculate various maturity dates for the crop
        """
        crop_info = self.CROP_MATURITY_INFO.get(crop_type.lower(), {})
        maturity_days = crop_info.get('days_to_maturity', {'typical': 120})

        min_harvest = planting_date + timedelta(days=maturity_days.get('min', 90))
        typical_harvest = planting_date + timedelta(days=maturity_days.get('typical', 120))
        max_harvest = planting_date + timedelta(days=maturity_days.get('max', 150))

        today = datetime.now()
        days_until = (typical_harvest - today).days

        return {
            'earliest_harvest_date': min_harvest.strftime('%Y-%m-%d'),
            'estimated_harvest_date': typical_harvest.strftime('%Y-%m-%d'),
            'latest_harvest_date': max_harvest.strftime('%Y-%m-%d'),
            'days_until_maturity': max(0, days_until),
            'harvest_window_days': crop_info.get('harvest_window_days', 14),
            'is_overdue': days_until < -7  # More than a week past typical harvest
        }

    def _estimate_growth_stage(self, crop_type: str, crop_age_days: int) -> str:
        """
        Estimate current growth stage based on crop age
        """
        crop_info = self.CROP_MATURITY_INFO.get(crop_type.lower(), {})
        typical_maturity = crop_info.get('days_to_maturity', {}).get('typical', 120)

        progress = (crop_age_days / typical_maturity) * 100

        if progress < 10:
            return 'germination'
        elif progress < 25:
            return 'seedling'
        elif progress < 50:
            return 'vegetative'
        elif progress < 75:
            return 'flowering'
        elif progress < 95:
            return 'fruit_development'
        elif progress < 110:
            return 'maturation'
        else:
            return 'overdue'

    def _assess_harvest_readiness(self, crop_age_days: int, maturity_info: Dict) -> Dict:
        """
        Assess how ready the crop is for harvest
        """
        days_until = maturity_info['days_until_maturity']
        is_overdue = maturity_info['is_overdue']

        if is_overdue:
            status = 'overdue'
            message = "Crop is past optimal harvest time. Harvest immediately to prevent losses"
            readiness_percentage = 100
            urgency = 'critical'

        elif days_until <= 0:
            status = 'ready'
            message = "Crop has reached maturity. Harvest when conditions are favorable"
            readiness_percentage = 100
            urgency = 'high'

        elif days_until <= 7:
            status = 'almost_ready'
            message = f"Crop will be ready in {days_until} days. Start harvest preparations"
            readiness_percentage = 95
            urgency = 'medium'

        elif days_until <= 21:
            status = 'approaching'
            message = f"Harvest expected in {days_until} days. Monitor crop development"
            readiness_percentage = 75
            urgency = 'low'

        else:
            status = 'developing'
            message = f"Crop still developing. Harvest expected in {days_until} days"
            readiness_percentage = min(70, (crop_age_days / 120) * 100)
            urgency = 'none'

        return {
            'status': status,
            'message': message,
            'readiness_percentage': round(readiness_percentage, 1),
            'urgency': urgency,
            'days_until_harvest': max(0, days_until)
        }

    def _predict_yield(
        self,
        crop_type: str,
        farm_data: Dict,
        crop_age_days: int,
        growth_stage: str
    ) -> Dict:
        """
        Predict yield based on crop info and farm conditions
        """
        crop_info = self.CROP_MATURITY_INFO.get(crop_type.lower(), {})
        base_yield = crop_info.get('yield_per_acre', {'typical': 3000})

        # Start with typical yield
        predicted_yield = base_yield['typical']

        # Adjust based on farm conditions
        multiplier = 1.0
        factors = []

        # Soil pH factor
        soil_ph = farm_data.get('soil_ph')
        if soil_ph:
            if 6.0 <= soil_ph <= 6.8:
                multiplier *= 1.1
                factors.append("Optimal soil pH (+10%)")
            elif soil_ph < 5.5 or soil_ph > 7.5:
                multiplier *= 0.85
                factors.append("Suboptimal soil pH (-15%)")

        # Climate zone factor
        climate_zone = farm_data.get('climate_zone', '').lower()
        if 'tropical' in climate_zone and crop_type.lower() in ['cassava', 'rice', 'maize']:
            multiplier *= 1.05
            factors.append("Favorable climate (+5%)")

        # Temperature factor
        avg_temp = farm_data.get('avg_temperature')
        if avg_temp:
            if crop_type.lower() == 'tomato' and (avg_temp < 15 or avg_temp > 30):
                multiplier *= 0.9
                factors.append("Temperature stress (-10%)")
            elif 20 <= avg_temp <= 28:
                multiplier *= 1.05
                factors.append("Optimal temperature (+5%)")

        # Growth stage factor
        if growth_stage in ['overdue', 'maturation']:
            multiplier *= 0.95
            factors.append("Late harvest timing (-5%)")

        predicted_yield = predicted_yield * multiplier

        # Get farm size for total yield estimate
        farm_size = farm_data.get('size', 1)  # Default to 1 acre
        total_yield = predicted_yield * farm_size

        return {
            'predicted_yield_per_acre': round(predicted_yield, 0),
            'unit': 'kg',
            'yield_range': {
                'minimum': round(base_yield['min'] * multiplier, 0),
                'expected': round(predicted_yield, 0),
                'maximum': round(base_yield['max'] * multiplier, 0)
            },
            'total_farm_yield': round(total_yield, 0),
            'farm_size_acres': farm_size,
            'confidence': 'medium',
            'yield_factors': factors,
            'note': 'Prediction based on typical conditions. Actual yield may vary based on farm management practices'
        }

    def _generate_harvest_advice(
        self,
        crop_type: str,
        maturity_info: Dict,
        weather_forecast: Optional[Dict],
        growth_stage: str
    ) -> Dict:
        """
        Generate actionable harvest advice
        """
        advice = {
            'recommendations': [],
            'warnings': [],
            'optimal_timing': None,
            'weather_considerations': []
        }

        days_until = maturity_info['days_until_maturity']

        # General harvest recommendations
        if days_until <= 0:
            advice['recommendations'].append("Crop is ready for harvest")
            advice['recommendations'].append("Choose dry weather for harvesting")

            if weather_forecast:
                weather_rec = weather_forecast.get('harvest_advice', {})
                if weather_rec.get('recommendation') == 'favorable':
                    advice['optimal_timing'] = "Harvest in next 2-3 days (dry weather expected)"
                    advice['weather_considerations'].append("Favorable weather window available")
                elif weather_rec.get('recommendation') == 'delay':
                    advice['warnings'].append("Heavy rain expected - delay harvest")
                    advice['weather_considerations'].append("Wait for drier conditions")

        elif days_until <= 7:
            advice['recommendations'].append(f"Prepare for harvest in {days_until} days")
            advice['recommendations'].append("Arrange for labor and equipment")
            advice['recommendations'].append("Prepare storage facilities")

        # Crop-specific advice
        if crop_type.lower() == 'maize':
            if days_until <= 0:
                advice['recommendations'].append("Check moisture content (should be below 25%)")
                advice['recommendations'].append("Harvest early morning for best quality")

        elif crop_type.lower() == 'rice':
            if days_until <= 0:
                advice['recommendations'].append("Drain field 7-10 days before harvest")
                advice['recommendations'].append("Harvest at 20-25% moisture")

        elif crop_type.lower() == 'tomato':
            advice['recommendations'].append("Harvest fruits as they ripen (multiple pickings)")
            advice['recommendations'].append("Pick early morning for longer shelf life")

        # Warnings
        if maturity_info['is_overdue']:
            advice['warnings'].append("URGENT: Crop is overdue for harvest - quality may be deteriorating")

        if growth_stage == 'overdue':
            advice['warnings'].append("Risk of yield loss and pest damage increases with delay")

        return advice


# Create singleton instance
harvest_service = HarvestService()
