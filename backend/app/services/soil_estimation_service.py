"""
Soil Estimation Service
Estimates soil type and pH based on location, climate, and regional data
"""

import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class SoilEstimationService:
    """
    Estimates soil characteristics based on geographic and climate data
    Uses regional soil patterns for Ghana and West Africa
    """

    # Regional soil patterns for Ghana
    # Based on latitude bands and climate zones
    GHANA_SOIL_PATTERNS = {
        'coastal': {
            'lat_range': (4.0, 6.0),
            'soil_types': ['Sandy', 'Sandy Loam'],
            'typical_ph': (5.5, 6.5),
            'description': 'Coastal sandy soils, well-drained'
        },
        'forest': {
            'lat_range': (6.0, 8.0),
            'soil_types': ['Clay Loam', 'Loam', 'Silty Clay'],
            'typical_ph': (5.0, 6.0),
            'description': 'Forest zone with clay-rich soils'
        },
        'transition': {
            'lat_range': (8.0, 9.5),
            'soil_types': ['Loam', 'Sandy Loam', 'Clay Loam'],
            'typical_ph': (5.5, 6.5),
            'description': 'Transition zone with mixed soils'
        },
        'savanna': {
            'lat_range': (9.5, 11.5),
            'soil_types': ['Sandy Loam', 'Loam'],
            'typical_ph': (6.0, 7.0),
            'description': 'Guinea savanna with sandy loam soils'
        }
    }

    # Rainfall influence on soil pH
    # Higher rainfall = more leaching = lower pH
    RAINFALL_PH_ADJUSTMENT = {
        'very_high': -0.5,  # >1500mm
        'high': -0.3,       # 1200-1500mm
        'moderate': 0.0,    # 800-1200mm
        'low': 0.3          # <800mm
    }

    def estimate_soil_data(
        self,
        latitude: float,
        longitude: float,
        avg_rainfall: Optional[float] = None,
        climate_zone: Optional[str] = None
    ) -> Dict:
        """
        Estimate soil type and pH based on location and climate

        Args:
            latitude: Farm latitude
            longitude: Farm longitude
            avg_rainfall: Average annual rainfall in mm
            climate_zone: Climate zone if known

        Returns:
            Dict with soil_type, soil_ph, confidence, and notes
        """
        try:
            # Determine regional soil pattern
            soil_pattern = self._get_soil_pattern_by_latitude(latitude)

            if not soil_pattern:
                logger.warning(f"Location ({latitude}, {longitude}) outside known regions")
                return self._get_default_estimation()

            # Select most appropriate soil type
            soil_type = self._select_soil_type(
                soil_pattern['soil_types'],
                avg_rainfall,
                climate_zone
            )

            # Estimate pH
            soil_ph = self._estimate_ph(
                soil_pattern['typical_ph'],
                avg_rainfall
            )

            # Calculate confidence level
            confidence = self._calculate_confidence(
                latitude, longitude, avg_rainfall
            )

            return {
                'soil_type': soil_type,
                'soil_ph': soil_ph,
                'confidence': confidence,
                'method': 'regional_estimation',
                'region': soil_pattern['description'],
                'notes': self._generate_notes(soil_type, soil_ph, confidence)
            }

        except Exception as e:
            logger.error(f"Error estimating soil data: {e}")
            return self._get_default_estimation()

    def _get_soil_pattern_by_latitude(self, latitude: float) -> Optional[Dict]:
        """Get soil pattern based on latitude"""
        for pattern_name, pattern_data in self.GHANA_SOIL_PATTERNS.items():
            lat_min, lat_max = pattern_data['lat_range']
            if lat_min <= latitude <= lat_max:
                return pattern_data
        return None

    def _select_soil_type(
        self,
        possible_types: list,
        avg_rainfall: Optional[float],
        climate_zone: Optional[str]
    ) -> str:
        """
        Select most appropriate soil type from possibilities
        Based on rainfall patterns
        """
        if not avg_rainfall or len(possible_types) == 1:
            return possible_types[0]

        # Higher rainfall -> more clay content (leaching and clay formation)
        # Lower rainfall -> more sandy soils
        if avg_rainfall > 1200:
            # Prefer clay-rich soils in high rainfall areas
            for soil_type in ['Clay Loam', 'Silty Clay', 'Clay']:
                if soil_type in possible_types:
                    return soil_type
        elif avg_rainfall < 800:
            # Prefer sandy soils in low rainfall areas
            for soil_type in ['Sandy Loam', 'Sandy', 'Loam']:
                if soil_type in possible_types:
                    return soil_type

        # Default to first option (most common)
        return possible_types[0]

    def _estimate_ph(
        self,
        typical_ph_range: Tuple[float, float],
        avg_rainfall: Optional[float]
    ) -> float:
        """
        Estimate soil pH within typical range
        Adjusted by rainfall (more rain = lower pH due to leaching)
        """
        # Start with middle of typical range
        base_ph = (typical_ph_range[0] + typical_ph_range[1]) / 2

        if not avg_rainfall:
            return round(base_ph, 1)

        # Adjust based on rainfall
        if avg_rainfall > 1500:
            adjustment = self.RAINFALL_PH_ADJUSTMENT['very_high']
        elif avg_rainfall > 1200:
            adjustment = self.RAINFALL_PH_ADJUSTMENT['high']
        elif avg_rainfall > 800:
            adjustment = self.RAINFALL_PH_ADJUSTMENT['moderate']
        else:
            adjustment = self.RAINFALL_PH_ADJUSTMENT['low']

        estimated_ph = base_ph + adjustment

        # Clamp to typical range
        estimated_ph = max(typical_ph_range[0], min(typical_ph_range[1], estimated_ph))

        return round(estimated_ph, 1)

    def _calculate_confidence(
        self,
        latitude: float,
        longitude: float,
        avg_rainfall: Optional[float]
    ) -> str:
        """
        Calculate confidence level of estimation
        """
        # Higher confidence if we have more data
        confidence_score = 0

        # Base confidence for location
        if 4.0 <= latitude <= 11.5 and -3.5 <= longitude <= 1.5:
            confidence_score += 60  # Within Ghana's approximate bounds
        else:
            confidence_score += 30  # Outside main region

        # Bonus for having rainfall data
        if avg_rainfall:
            confidence_score += 20

        # Determine confidence level
        if confidence_score >= 75:
            return 'medium'  # We're estimating, not measuring
        elif confidence_score >= 50:
            return 'low-medium'
        else:
            return 'low'

    def _generate_notes(
        self,
        soil_type: str,
        soil_ph: float,
        confidence: str
    ) -> str:
        """Generate explanatory notes about the estimation"""
        notes = []

        notes.append(f"Estimated based on regional soil patterns")

        if confidence in ['low', 'low-medium']:
            notes.append("Recommendation: Conduct soil test for accurate results")

        # pH interpretation
        if soil_ph < 5.5:
            notes.append("Acidic soil - consider liming for most crops")
        elif soil_ph > 7.0:
            notes.append("Alkaline soil - may need sulfur amendments")
        else:
            notes.append("pH suitable for most crops")

        return "; ".join(notes)

    def _get_default_estimation(self) -> Dict:
        """Return default soil estimation when location is unknown"""
        return {
            'soil_type': 'Loam',
            'soil_ph': 6.0,
            'confidence': 'low',
            'method': 'default',
            'region': 'Unknown region - using defaults',
            'notes': 'Unable to estimate from location. Using typical agricultural soil values. Please conduct soil test for accurate data.'
        }


# Create singleton instance
soil_estimation_service = SoilEstimationService()
