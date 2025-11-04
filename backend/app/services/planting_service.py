"""
Planting Prediction Service
Provides intelligent planting time recommendations based on weather, soil, and climate data
"""

import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from app.services.weather_service import weather_service
from app.services.ml_crop_recommendation_service import ml_crop_recommendation_service

logger = logging.getLogger(__name__)


class PlantingService:
    """Service for planting time predictions and recommendations"""

    # Crop planting calendars (month-based for tropical/subtropical regions)
    CROP_PLANTING_WINDOWS = {
        'maize': {
            'primary_season': [3, 4, 5],  # March-May (main rainy season)
            'secondary_season': [9, 10],  # Sept-Oct (minor season)
            'min_soil_temp': 16,
            'optimal_soil_temp': 21,
            'min_rainfall_7days': 50,
            'days_to_maturity': 120
        },
        'rice': {
            'primary_season': [4, 5, 6],  # April-June
            'secondary_season': [10, 11],
            'min_soil_temp': 18,
            'optimal_soil_temp': 25,
            'min_rainfall_7days': 100,
            'days_to_maturity': 150
        },
        'cassava': {
            'primary_season': [4, 5, 6, 7],  # Can be planted longer window
            'secondary_season': [9, 10, 11],
            'min_soil_temp': 18,
            'optimal_soil_temp': 27,
            'min_rainfall_7days': 30,
            'days_to_maturity': 300
        },
        'tomato': {
            'primary_season': [2, 3, 4],
            'secondary_season': [9, 10],
            'min_soil_temp': 15,
            'optimal_soil_temp': 21,
            'min_rainfall_7days': 40,
            'days_to_maturity': 90
        },
        'soybean': {
            'primary_season': [5, 6, 7],
            'secondary_season': [],
            'min_soil_temp': 18,
            'optimal_soil_temp': 25,
            'min_rainfall_7days': 50,
            'days_to_maturity': 100
        },
        'groundnut': {
            'primary_season': [4, 5, 6],
            'secondary_season': [9, 10],
            'min_soil_temp': 18,
            'optimal_soil_temp': 25,
            'min_rainfall_7days': 50,
            'days_to_maturity': 120
        }
    }

    async def get_planting_recommendation(
        self,
        farm_data: Dict,
        crop_type: str
    ) -> Dict:
        """
        Get comprehensive planting recommendations for a specific crop
        """
        logger.info(f"Generating planting recommendation for {crop_type}")

        latitude = farm_data.get('latitude')
        longitude = farm_data.get('longitude')

        if not latitude or not longitude:
            raise ValueError("Farm location (latitude/longitude) is required")

        # Get weather-based advice
        weather_advice = await weather_service.get_planting_weather_advice(
            latitude, longitude, crop_type
        )

        # Check seasonal appropriateness
        seasonal_advice = self._check_seasonal_timing(crop_type, farm_data)

        # Get soil suitability
        soil_suitability = self._check_soil_suitability(crop_type, farm_data)

        # Calculate estimated planting date
        planting_date = self._calculate_optimal_planting_date(
            crop_type,
            weather_advice,
            seasonal_advice
        )

        # Get ML-based crop suitability if available
        ml_suitability = self._get_ml_suitability(farm_data, crop_type)

        # Calculate overall recommendation score
        overall_score = self._calculate_overall_score(
            weather_advice,
            seasonal_advice,
            soil_suitability,
            ml_suitability
        )

        # Generate final recommendation
        recommendation = {
            'crop_type': crop_type,
            'overall_recommendation': overall_score['recommendation'],
            'suitability_score': overall_score['score'],
            'confidence': overall_score['confidence'],
            'summary': overall_score['summary'],
            'planting_window': {
                'recommended_date': planting_date['recommended'],
                'earliest_date': planting_date['earliest'],
                'latest_date': planting_date['latest'],
                'reason': planting_date['reason']
            },
            'weather_analysis': weather_advice['planting_advice'],
            'seasonal_analysis': seasonal_advice,
            'soil_analysis': soil_suitability,
            'estimated_harvest_date': self._calculate_harvest_date(
                planting_date['recommended'],
                crop_type
            ),
            'preparation_checklist': self._get_preparation_checklist(crop_type),
            'generated_at': datetime.now().isoformat()
        }

        return recommendation

    async def get_multi_crop_comparison(
        self,
        farm_data: Dict,
        crop_types: List[str]
    ) -> Dict:
        """
        Compare planting recommendations for multiple crops
        """
        recommendations = []

        for crop in crop_types:
            try:
                rec = await self.get_planting_recommendation(farm_data, crop)
                recommendations.append({
                    'crop': crop,
                    'suitability_score': rec['suitability_score'],
                    'recommendation': rec['overall_recommendation'],
                    'planting_date': rec['planting_window']['recommended_date'],
                    'harvest_date': rec['estimated_harvest_date'],
                    'summary': rec['summary']
                })
            except Exception as e:
                logger.error(f"Error getting recommendation for {crop}: {e}")

        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)

        return {
            'farm_location': {
                'latitude': farm_data.get('latitude'),
                'longitude': farm_data.get('longitude')
            },
            'comparison': recommendations,
            'top_recommendation': recommendations[0] if recommendations else None,
            'generated_at': datetime.now().isoformat()
        }

    def _check_seasonal_timing(self, crop_type: str, farm_data: Dict) -> Dict:
        """
        Check if current month is appropriate for planting
        """
        current_month = datetime.now().month
        crop_windows = self.CROP_PLANTING_WINDOWS.get(crop_type.lower(), {})

        primary_season = crop_windows.get('primary_season', [])
        secondary_season = crop_windows.get('secondary_season', [])

        is_primary = current_month in primary_season
        is_secondary = current_month in secondary_season

        if is_primary:
            status = 'optimal'
            message = f"Currently in primary planting season for {crop_type}"
        elif is_secondary:
            status = 'good'
            message = f"Currently in secondary planting season for {crop_type}"
        else:
            # Find next planting window
            all_months = sorted(primary_season + secondary_season)
            next_month = None
            for month in all_months:
                if month > current_month:
                    next_month = month
                    break
            if not next_month and all_months:
                next_month = all_months[0]

            status = 'off_season'
            if next_month:
                months_to_wait = (next_month - current_month) % 12
                message = f"Off-season for {crop_type}. Next planting window in {months_to_wait} months"
            else:
                message = f"Planting calendar not available for {crop_type}"

        return {
            'status': status,
            'message': message,
            'current_month': current_month,
            'primary_months': primary_season,
            'secondary_months': secondary_season
        }

    def _check_soil_suitability(self, crop_type: str, farm_data: Dict) -> Dict:
        """
        Check soil conditions for crop suitability
        """
        soil_ph = farm_data.get('soil_ph')
        soil_type = farm_data.get('soil_type', '').lower()

        # Optimal pH ranges for crops
        ph_requirements = {
            'maize': (5.5, 7.0),
            'rice': (5.0, 6.5),
            'cassava': (4.5, 7.5),
            'tomato': (6.0, 6.8),
            'soybean': (6.0, 7.0),
            'groundnut': (5.9, 6.3)
        }

        # Preferred soil types
        soil_preferences = {
            'maize': ['loam', 'sandy loam', 'clay loam'],
            'rice': ['clay', 'clay loam', 'silty clay'],
            'cassava': ['sandy', 'sandy loam', 'loam'],
            'tomato': ['loam', 'sandy loam'],
            'soybean': ['loam', 'clay loam'],
            'groundnut': ['sandy loam', 'loam']
        }

        issues = []
        strengths = []

        # Check pH
        if soil_ph:
            req_ph = ph_requirements.get(crop_type.lower(), (5.5, 7.0))
            if soil_ph < req_ph[0]:
                issues.append(f"Soil too acidic (pH {soil_ph}, need {req_ph[0]}-{req_ph[1]}). Consider liming")
            elif soil_ph > req_ph[1]:
                issues.append(f"Soil too alkaline (pH {soil_ph}, need {req_ph[0]}-{req_ph[1]})")
            else:
                strengths.append(f"Soil pH optimal ({soil_ph})")

        # Check soil type
        if soil_type:
            preferred = soil_preferences.get(crop_type.lower(), [])
            if soil_type in preferred:
                strengths.append(f"Soil type suitable ({soil_type})")
            else:
                issues.append(f"Soil type not ideal ({soil_type}). Preferred: {', '.join(preferred)}")

        # Overall assessment
        if len(issues) == 0:
            suitability = 'high'
            message = "Soil conditions are excellent for this crop"
        elif len(issues) == 1:
            suitability = 'medium'
            message = "Soil conditions acceptable with amendments"
        else:
            suitability = 'low'
            message = "Soil may need significant amendments"

        return {
            'suitability': suitability,
            'message': message,
            'strengths': strengths,
            'concerns': issues,
            'soil_ph': soil_ph,
            'soil_type': soil_type
        }

    def _get_ml_suitability(self, farm_data: Dict, crop_type: str) -> Optional[Dict]:
        """
        Get ML-based crop suitability score
        """
        try:
            recommendations = ml_crop_recommendation_service.generate_recommendations(
                farm_data, top_n=5
            )

            # Find the crop in recommendations
            for rec in recommendations:
                if rec['recommended_crop'].lower() == crop_type.lower():
                    return {
                        'suitability_score': rec['suitability_score'],
                        'confidence': rec['confidence_score'],
                        'method': 'ml_model'
                    }

            return None

        except Exception as e:
            logger.warning(f"ML suitability check failed: {e}")
            return None

    def _calculate_overall_score(
        self,
        weather_advice: Dict,
        seasonal_advice: Dict,
        soil_suitability: Dict,
        ml_suitability: Optional[Dict]
    ) -> Dict:
        """
        Calculate overall planting recommendation score (0-100)
        """
        score = 0
        factors = []

        # Weather score (40%)
        weather_rec = weather_advice['planting_advice']['recommendation']
        if weather_rec == 'plant_now':
            score += 40
            factors.append("Weather optimal")
        elif weather_rec == 'plant_with_caution':
            score += 25
            factors.append("Weather acceptable")
        else:
            score += 10
            factors.append("Weather unfavorable")

        # Seasonal score (30%)
        seasonal_status = seasonal_advice['status']
        if seasonal_status == 'optimal':
            score += 30
            factors.append("Peak planting season")
        elif seasonal_status == 'good':
            score += 20
            factors.append("Acceptable season")
        else:
            score += 5
            factors.append("Off-season")

        # Soil score (20%)
        soil_suit = soil_suitability['suitability']
        if soil_suit == 'high':
            score += 20
            factors.append("Soil ideal")
        elif soil_suit == 'medium':
            score += 12
            factors.append("Soil acceptable")
        else:
            score += 5
            factors.append("Soil needs amendment")

        # ML model score (10% - if available)
        if ml_suitability:
            ml_score = ml_suitability['suitability_score']
            score += (ml_score / 100) * 10
            factors.append(f"ML model: {ml_score:.0f}% suitable")
        else:
            score += 5  # Default if ML not available

        # Determine recommendation
        if score >= 75:
            recommendation = 'highly_recommended'
            confidence = 'high'
            summary = f"Excellent time to plant ({score:.0f}/100). " + ". ".join(factors)
        elif score >= 50:
            recommendation = 'recommended'
            confidence = 'medium'
            summary = f"Good time to plant with proper care ({score:.0f}/100). " + ". ".join(factors)
        else:
            recommendation = 'not_recommended'
            confidence = 'high'
            summary = f"Wait for better conditions ({score:.0f}/100). " + ". ".join(factors)

        return {
            'score': round(score, 1),
            'recommendation': recommendation,
            'confidence': confidence,
            'summary': summary,
            'contributing_factors': factors
        }

    def _calculate_optimal_planting_date(
        self,
        crop_type: str,
        weather_advice: Dict,
        seasonal_advice: Dict
    ) -> Dict:
        """
        Calculate optimal planting date
        """
        today = datetime.now()

        weather_rec = weather_advice['planting_advice']['recommendation']
        seasonal_status = seasonal_advice['status']

        if weather_rec == 'plant_now' and seasonal_status in ['optimal', 'good']:
            recommended = today + timedelta(days=2)
            earliest = today
            latest = today + timedelta(days=7)
            reason = "Current conditions are favorable for immediate planting"

        elif seasonal_status == 'off_season':
            # Find next planting window
            current_month = today.month
            primary_months = seasonal_advice['primary_months']

            if primary_months:
                next_month = None
                for month in sorted(primary_months):
                    if month > current_month:
                        next_month = month
                        break
                if not next_month:
                    next_month = primary_months[0]

                # Calculate date in next planting window
                if next_month > current_month:
                    months_diff = next_month - current_month
                else:
                    months_diff = (12 - current_month) + next_month

                recommended = today + timedelta(days=30 * months_diff)
                earliest = recommended - timedelta(days=7)
                latest = recommended + timedelta(days=14)
                reason = f"Wait for primary planting season (next window: Month {next_month})"
            else:
                recommended = today + timedelta(days=30)
                earliest = today + timedelta(days=21)
                latest = today + timedelta(days=45)
                reason = "Planting calendar unavailable - general estimate provided"

        else:
            # Wait for better weather
            recommended = today + timedelta(days=14)
            earliest = today + timedelta(days=7)
            latest = today + timedelta(days=21)
            reason = "Wait for improved weather conditions"

        return {
            'recommended': recommended.strftime('%Y-%m-%d'),
            'earliest': earliest.strftime('%Y-%m-%d'),
            'latest': latest.strftime('%Y-%m-%d'),
            'reason': reason
        }

    def _calculate_harvest_date(self, planting_date_str: str, crop_type: str) -> str:
        """
        Calculate estimated harvest date based on planting date
        """
        try:
            planting_date = datetime.strptime(planting_date_str, '%Y-%m-%d')
            crop_windows = self.CROP_PLANTING_WINDOWS.get(crop_type.lower(), {})
            days_to_maturity = crop_windows.get('days_to_maturity', 120)

            harvest_date = planting_date + timedelta(days=days_to_maturity)
            return harvest_date.strftime('%Y-%m-%d')

        except Exception as e:
            logger.error(f"Error calculating harvest date: {e}")
            return "Unknown"

    def _get_preparation_checklist(self, crop_type: str) -> List[str]:
        """
        Get pre-planting preparation checklist
        """
        general_checklist = [
            "Clear and prepare land (remove weeds and debris)",
            "Test soil pH and nutrient levels",
            "Prepare planting beds or ridges",
            "Ensure quality seed availability",
            "Check irrigation/water availability",
            "Prepare necessary farming tools"
        ]

        crop_specific = {
            'maize': [
                "Apply basal fertilizer (NPK)",
                "Ensure proper spacing plan (75cm x 25cm)",
                "Prepare for pest management (armyworm control)"
            ],
            'rice': [
                "Prepare nursery beds",
                "Ensure water supply for flooding",
                "Plan for transplanting labor"
            ],
            'cassava': [
                "Source quality stem cuttings",
                "Prepare ridges/mounds",
                "Plan for weed control in first 3 months"
            ],
            'tomato': [
                "Prepare nursery for seedlings",
                "Install staking/support structures",
                "Ensure disease management supplies ready"
            ],
            'soybean': [
                "Inoculate seeds with rhizobium",
                "Prepare for early weed control",
                "Plan for storage pest management"
            ],
            'groundnut': [
                "Source quality seeds",
                "Plan for gypsum application (calcium source)",
                "Prepare for proper drying after harvest"
            ]
        }

        specific = crop_specific.get(crop_type.lower(), [])
        return general_checklist + specific


# Create singleton instance
planting_service = PlantingService()
