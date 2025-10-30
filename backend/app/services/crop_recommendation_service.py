"""
Crop Recommendation Service
AI-powered crop recommendations based on geospatial and farm data
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CropRecommendationService:
    """Service for generating AI-powered crop recommendations"""

    # Crop database with climate and soil requirements
    CROP_DATABASE = {
        "Maize": {
            "climate_requirements": {
                "min_temp": 18, "max_temp": 32,
                "min_rainfall": 500, "max_rainfall": 1000,
                "climate_zones": ["tropical", "subtropical", "temperate"]
            },
            "soil_requirements": {
                "ph_min": 5.5, "ph_max": 7.5,
                "types": ["loam", "sandy loam", "clay loam"],
                "drainage": "well-drained"
            },
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
            "climate_requirements": {
                "min_temp": 20, "max_temp": 38,
                "min_rainfall": 1000, "max_rainfall": 3000,
                "climate_zones": ["tropical", "subtropical"]
            },
            "soil_requirements": {
                "ph_min": 5.0, "ph_max": 7.0,
                "types": ["clay", "clay loam", "silty clay"],
                "drainage": "poor to moderate"
            },
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
            "climate_requirements": {
                "min_temp": 20, "max_temp": 35,
                "min_rainfall": 500, "max_rainfall": 1500,
                "climate_zones": ["tropical", "subtropical"]
            },
            "soil_requirements": {
                "ph_min": 4.5, "ph_max": 7.0,
                "types": ["sandy loam", "loam", "sandy"],
                "drainage": "well-drained"
            },
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
            "climate_requirements": {
                "min_temp": 18, "max_temp": 27,
                "min_rainfall": 600, "max_rainfall": 1300,
                "climate_zones": ["tropical", "subtropical", "temperate"]
            },
            "soil_requirements": {
                "ph_min": 6.0, "ph_max": 7.0,
                "types": ["loam", "sandy loam"],
                "drainage": "well-drained"
            },
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
            "climate_requirements": {
                "min_temp": 20, "max_temp": 30,
                "min_rainfall": 500, "max_rainfall": 900,
                "climate_zones": ["tropical", "subtropical", "temperate"]
            },
            "soil_requirements": {
                "ph_min": 6.0, "ph_max": 7.5,
                "types": ["loam", "sandy loam", "clay loam"],
                "drainage": "well-drained"
            },
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
            "climate_requirements": {
                "min_temp": 20, "max_temp": 30,
                "min_rainfall": 500, "max_rainfall": 1000,
                "climate_zones": ["tropical", "subtropical"]
            },
            "soil_requirements": {
                "ph_min": 5.5, "ph_max": 6.5,
                "types": ["sandy loam", "loam", "sandy"],
                "drainage": "well-drained"
            },
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

    def __init__(self):
        """Initialize the crop recommendation service"""
        self.model_version = "1.0.0"

    def generate_recommendations(
        self,
        farm_data: Dict[str, Any],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate crop recommendations based on farm geospatial data

        Args:
            farm_data: Dictionary containing farm information
            top_n: Number of recommendations to return

        Returns:
            List of crop recommendation dictionaries
        """
        logger.info(f"Generating crop recommendations for farm at ({farm_data.get('latitude')}, {farm_data.get('longitude')})")

        recommendations = []

        for crop_name, crop_info in self.CROP_DATABASE.items():
            suitability = self._calculate_suitability(farm_data, crop_info)

            if suitability['score'] > 30:  # Only include crops with >30% suitability
                recommendation = {
                    "recommended_crop": crop_name,
                    "suitability_score": suitability['score'],
                    "confidence_score": suitability['confidence'],
                    "climate_factors": suitability['climate_match'],
                    "soil_factors": suitability['soil_match'],
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
                    "model_version": self.model_version
                }
                recommendations.append(recommendation)

        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)

        # Add alternative crops to top recommendations
        for i, rec in enumerate(recommendations[:top_n]):
            alternatives = [r['recommended_crop'] for r in recommendations[i+1:i+4]]
            rec['alternative_crops'] = alternatives

        return recommendations[:top_n]

    def _calculate_suitability(
        self,
        farm_data: Dict[str, Any],
        crop_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate crop suitability score based on farm conditions"""

        climate_score = self._match_climate(farm_data, crop_info['climate_requirements'])
        soil_score = self._match_soil(farm_data, crop_info['soil_requirements'])

        # Overall suitability (weighted average)
        overall_score = (climate_score['score'] * 0.5 + soil_score['score'] * 0.5)

        # Confidence based on data availability
        data_completeness = self._assess_data_completeness(farm_data)
        confidence = min(0.95, overall_score / 100 * data_completeness)

        return {
            "score": round(overall_score, 2),
            "confidence": round(confidence, 2),
            "climate_match": climate_score,
            "soil_match": soil_score
        }

    def _match_climate(
        self,
        farm_data: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Match farm climate with crop requirements"""

        score = 0
        factors = {}

        # Temperature match
        farm_temp = farm_data.get('avg_temperature')
        if farm_temp:
            if requirements['min_temp'] <= farm_temp <= requirements['max_temp']:
                score += 40
                factors['temperature'] = "optimal"
            elif (requirements['min_temp'] - 5) <= farm_temp <= (requirements['max_temp'] + 5):
                score += 20
                factors['temperature'] = "acceptable"
            else:
                factors['temperature'] = "unsuitable"

        # Rainfall match
        farm_rainfall = farm_data.get('avg_annual_rainfall')
        if farm_rainfall:
            if requirements['min_rainfall'] <= farm_rainfall <= requirements['max_rainfall']:
                score += 40
                factors['rainfall'] = "optimal"
            elif (requirements['min_rainfall'] - 200) <= farm_rainfall <= (requirements['max_rainfall'] + 200):
                score += 20
                factors['rainfall'] = "acceptable"
            else:
                factors['rainfall'] = "unsuitable"

        # Climate zone match
        farm_climate = farm_data.get('climate_zone')
        if farm_climate and farm_climate.lower() in requirements.get('climate_zones', []):
            score += 20
            factors['climate_zone'] = "suitable"
        elif farm_climate:
            factors['climate_zone'] = "not_optimal"

        return {"score": score, "factors": factors}

    def _match_soil(
        self,
        farm_data: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Match farm soil with crop requirements"""

        score = 0
        factors = {}

        # pH match
        farm_ph = farm_data.get('soil_ph')
        if farm_ph:
            if requirements['ph_min'] <= farm_ph <= requirements['ph_max']:
                score += 50
                factors['ph'] = "optimal"
            elif (requirements['ph_min'] - 0.5) <= farm_ph <= (requirements['ph_max'] + 0.5):
                score += 25
                factors['ph'] = "acceptable"
            else:
                factors['ph'] = "unsuitable"

        # Soil type match
        farm_soil_type = farm_data.get('soil_type')
        if farm_soil_type:
            if any(soil_type.lower() in farm_soil_type.lower() for soil_type in requirements.get('types', [])):
                score += 50
                factors['type'] = "suitable"
            else:
                score += 10  # Still give some points
                factors['type'] = "acceptable"

        return {"score": score, "factors": factors}

    def _assess_data_completeness(self, farm_data: Dict[str, Any]) -> float:
        """Assess how complete the farm data is"""

        required_fields = [
            'latitude', 'longitude', 'avg_temperature',
            'avg_annual_rainfall', 'soil_type', 'soil_ph', 'climate_zone'
        ]

        available = sum(1 for field in required_fields if farm_data.get(field) is not None)
        completeness = available / len(required_fields)

        return completeness

    def _determine_planting_season(
        self,
        farm_data: Dict[str, Any],
        crop_info: Dict[str, Any]
    ) -> str:
        """Determine optimal planting season based on location"""

        # Simplified logic - can be enhanced with more sophisticated weather patterns
        climate_zone = farm_data.get('climate_zone', '').lower()

        if climate_zone in ['tropical', 'subtropical']:
            if crop_info.get('water_requirements') == 'high':
                return "Beginning of rainy season (April-June)"
            else:
                return "Rainy season (May-July) or with irrigation"
        elif climate_zone == 'temperate':
            return "Spring (March-May)"
        else:
            return "Consult local agricultural extension"


# Create service instance
crop_recommendation_service = CropRecommendationService()
