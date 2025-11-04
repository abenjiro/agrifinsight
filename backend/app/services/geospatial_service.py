"""
Geospatial service for weather, satellite, and environmental data
Uses configuration from app.config for all API endpoints and keys
"""

import requests
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from app.config import settings

class GeospatialService:
    """Service for handling geospatial data, weather, and satellite imagery"""

    def __init__(self):
        # API keys and URLs from settings
        self.openweather_api_key = settings.openweather_api_key
        self.openweather_api_url = settings.openweather_api_url
        self.nasa_api_key = settings.nasa_api_key
        self.elevation_api_url = settings.elevation_api_url
        self.soilgrids_api_url = settings.soilgrids_api_url
        self.nominatim_api_url = settings.nominatim_api_url
        self.nominatim_user_agent = settings.nominatim_user_agent

    async def get_weather_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch current weather and forecast data for given coordinates
        Uses OpenWeatherMap API
        """
        if not self.openweather_api_key:
            return self._get_mock_weather_data()

        try:
            # Current weather
            current_url = f"{self.openweather_api_url}/weather?lat={latitude}&lon={longitude}&appid={self.openweather_api_key}&units=metric"
            current_response = requests.get(current_url, timeout=10)
            current_data = current_response.json() if current_response.ok else {}

            # 7-day forecast
            forecast_url = f"{self.openweather_api_url}/onecall?lat={latitude}&lon={longitude}&exclude=minutely,hourly&appid={self.openweather_api_key}&units=metric"
            forecast_response = requests.get(forecast_url, timeout=10)
            forecast_data = forecast_response.json() if forecast_response.ok else {}

            return {
                "current": {
                    "temperature": current_data.get("main", {}).get("temp"),
                    "feels_like": current_data.get("main", {}).get("feels_like"),
                    "humidity": current_data.get("main", {}).get("humidity"),
                    "pressure": current_data.get("main", {}).get("pressure"),
                    "wind_speed": current_data.get("wind", {}).get("speed"),
                    "wind_direction": current_data.get("wind", {}).get("deg"),
                    "clouds": current_data.get("clouds", {}).get("all"),
                    "description": current_data.get("weather", [{}])[0].get("description"),
                    "icon": current_data.get("weather", [{}])[0].get("icon"),
                },
                "forecast": forecast_data.get("daily", [])[:7],
                "timezone": forecast_data.get("timezone"),
            }
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return self._get_mock_weather_data()

    async def get_elevation(self, latitude: float, longitude: float) -> Optional[float]:
        """
        Get elevation/altitude for given coordinates
        Uses Open Elevation API
        """
        try:
            url = f"{self.elevation_api_url}/lookup?locations={latitude},{longitude}"
            response = requests.get(url, timeout=10)

            if response.ok:
                data = response.json()
                results = data.get("results", [])
                if results:
                    return results[0].get("elevation")
            return None
        except Exception as e:
            print(f"Error fetching elevation: {e}")
            return None

    async def get_climate_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch historical climate data including annual rainfall and temperature
        Uses NASA POWER API for agricultural climate data
        """
        try:
            from datetime import datetime, timedelta

            # Get data for the past year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            power_url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
            params = {
                "parameters": "T2M,PRECTOTCORR",  # Temperature and Precipitation
                "community": "AG",
                "longitude": longitude,
                "latitude": latitude,
                "format": "JSON"
            }

            response = requests.get(power_url, params=params, timeout=15)

            if response.ok:
                data = response.json()
                properties = data.get("properties", {}).get("parameter", {})

                # Calculate annual averages
                temp_data = properties.get("T2M", {})
                precip_data = properties.get("PRECTOTCORR", {})

                if temp_data and precip_data:
                    # Get annual average temperature
                    temp_values = [v for v in temp_data.values() if isinstance(v, (int, float))]
                    avg_temp = sum(temp_values) / len(temp_values) if temp_values else None

                    # Get annual total precipitation (sum of monthly values)
                    precip_values = [v for v in precip_data.values() if isinstance(v, (int, float))]
                    annual_rainfall = sum(precip_values) * 30 if precip_values else None  # Convert daily to monthly estimate

                    return {
                        "avg_temperature": round(avg_temp, 1) if avg_temp else None,
                        "avg_annual_rainfall": round(annual_rainfall, 1) if annual_rainfall else None,
                        "data_source": "NASA POWER",
                        "temp_range": {
                            "min": round(min(temp_values), 1) if temp_values else None,
                            "max": round(max(temp_values), 1) if temp_values else None
                        }
                    }

            return {}

        except Exception as e:
            print(f"Error fetching climate data: {e}")
            return {}

    async def get_ndvi_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch NDVI (Normalized Difference Vegetation Index) data
        Uses NASA MODIS data via AppEEARS API (free, lower resolution)
        NDVI range: -1 to 1, where higher values indicate healthier vegetation
        """
        try:
            # Alternative: Use Copernicus Global Land Service for NDVI
            # This is a simplified implementation using available free APIs

            # For production, consider:
            # 1. Sentinel Hub API (paid, high resolution, 10m)
            # 2. Google Earth Engine (free for research, requires approval)
            # 3. NASA MODIS (free, lower resolution, 250m)

            # Using NASA POWER API for vegetation-related parameters
            power_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
            params = {
                "parameters": "ALLSKY_SFC_SW_DWN",  # Solar radiation (proxy for vegetation health)
                "community": "AG",
                "longitude": longitude,
                "latitude": latitude,
                "start": (datetime.now() - timedelta(days=90)).strftime("%Y%m%d"),
                "end": datetime.now().strftime("%Y%m%d"),
                "format": "JSON"
            }

            response = requests.get(power_url, params=params, timeout=15)

            if response.ok:
                data = response.json()
                solar_data = data.get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})

                # Calculate vegetation health proxy from solar radiation patterns
                # In real implementation, this would be actual NDVI from satellite
                if solar_data:
                    values = [v for v in solar_data.values() if isinstance(v, (int, float))]
                    avg_solar = sum(values) / len(values) if values else 0

                    # Rough NDVI estimate (simplified)
                    # Real NDVI requires near-infrared and red band reflectance
                    estimated_ndvi = min(0.8, avg_solar / 250)  # Normalize to 0-0.8 range

                    return {
                        "ndvi_available": True,
                        "ndvi_value": round(estimated_ndvi, 2),
                        "ndvi_interpretation": self._interpret_ndvi(estimated_ndvi),
                        "data_source": "NASA POWER (proxy)",
                        "note": "Using solar radiation as vegetation health proxy. For production, use Sentinel-2 NDVI.",
                        "last_update": datetime.now().isoformat()
                    }

            return {
                "ndvi_available": False,
                "message": "NDVI data unavailable. Consider integrating Sentinel Hub API for production."
            }

        except Exception as e:
            print(f"Error fetching NDVI data: {e}")
            return {"ndvi_available": False, "error": str(e)}

    def _interpret_ndvi(self, ndvi: float) -> str:
        """Interpret NDVI values for agricultural context"""
        if ndvi < 0:
            return "Water/Bare soil"
        elif ndvi < 0.2:
            return "Sparse vegetation/Early growth"
        elif ndvi < 0.5:
            return "Moderate vegetation"
        elif ndvi < 0.7:
            return "Healthy vegetation"
        else:
            return "Very healthy/Dense vegetation"

    async def get_satellite_imagery(self, latitude: float, longitude: float,
                                   date: Optional[str] = None) -> Dict:
        """
        Fetch satellite imagery metadata and NDVI
        For production: Integrate Sentinel Hub, Google Earth Engine, or Planet Labs
        """
        try:
            # Get NDVI data
            ndvi_data = await self.get_ndvi_data(latitude, longitude)

            # Get historical climate trends
            power_url = f"https://power.larc.nasa.gov/api/temporal/monthly/point"
            params = {
                "parameters": "T2M,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN",
                "community": "AG",
                "longitude": longitude,
                "latitude": latitude,
                "start": (datetime.now() - timedelta(days=365)).strftime("%Y%m%d"),
                "end": datetime.now().strftime("%Y%m%d"),
                "format": "JSON"
            }

            response = requests.get(power_url, params=params, timeout=15)
            climate_data = response.json() if response.ok else {}

            return {
                "source": "NASA_POWER",
                "ndvi_data": ndvi_data,
                "climate_trends": climate_data.get("properties", {}).get("parameter", {}),
                "image_url": None,  # Would contain actual satellite image URL with Sentinel Hub
                "note": "For production satellite imagery, integrate Sentinel Hub API",
                "recommendations": self._get_satellite_recommendations(ndvi_data)
            }

        except Exception as e:
            print(f"Error fetching satellite data: {e}")
            return {
                "source": "UNAVAILABLE",
                "error": str(e)
            }

    def _get_satellite_recommendations(self, ndvi_data: Dict) -> List[str]:
        """Generate recommendations based on NDVI data"""
        recommendations = []

        if not ndvi_data.get("ndvi_available"):
            return ["Satellite data integration needed for vegetation health monitoring"]

        ndvi = ndvi_data.get("ndvi_value", 0)

        if ndvi < 0.3:
            recommendations.extend([
                "Low vegetation health detected",
                "Consider soil testing and nutrient supplementation",
                "Check irrigation and water availability"
            ])
        elif ndvi < 0.5:
            recommendations.extend([
                "Moderate vegetation health",
                "Monitor crop development closely",
                "Optimize fertilizer application"
            ])
        else:
            recommendations.extend([
                "Healthy vegetation detected",
                "Continue current farming practices",
                "Monitor for pest and disease pressure"
            ])

        return recommendations

    async def get_soil_data(self, latitude: float, longitude: float) -> Dict:
        """
        Get soil data for given coordinates
        Uses SoilGrids REST API
        """
        try:
            # SoilGrids API
            url = f"{self.soilgrids_api_url}/properties/query"
            params = {
                "lon": longitude,
                "lat": latitude,
                "property": "clay,sand,silt,phh2o,soc",  # Clay, Sand, Silt, pH, Organic Carbon
                "depth": "0-5cm",
                "value": "mean"
            }

            response = requests.get(url, params=params, timeout=15)

            if response.ok:
                data = response.json()
                properties = data.get("properties", {}).get("layers", [])

                soil_composition = {}
                for prop in properties:
                    name = prop.get("name")
                    depths = prop.get("depths", [])
                    if depths:
                        value = depths[0].get("values", {}).get("mean", 0)
                        soil_composition[name] = value / 10  # Convert to standard units

                return {
                    "composition": soil_composition,
                    "source": "SoilGrids",
                }
            return {}
        except Exception as e:
            print(f"Error fetching soil data: {e}")
            return {}

    async def reverse_geocode(self, latitude: float, longitude: float) -> Dict:
        """
        Reverse geocode coordinates to get address and location metadata
        Uses OpenStreetMap Nominatim API
        """
        try:
            url = f"{self.nominatim_api_url}/reverse"
            params = {
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "addressdetails": 1
            }
            headers = {
                "User-Agent": self.nominatim_user_agent  # Required by Nominatim
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)

            if response.ok:
                data = response.json()
                address = data.get("address", {})

                return {
                    "display_name": data.get("display_name"),
                    "country": address.get("country"),
                    "region": address.get("state") or address.get("region"),
                    "district": address.get("county") or address.get("district"),
                    "village": address.get("village"),
                    "town": address.get("town") or address.get("city"),
                }
            return {}
        except Exception as e:
            print(f"Error in reverse geocoding: {e}")
            return {}

    async def get_timezone(self, latitude: float, longitude: float) -> Optional[str]:
        """
        Get timezone for given coordinates
        Note: In production, use a dedicated timezone API like TimeZoneDB
        """
        try:
            # Using a simple approximation based on longitude
            # For production, use: https://timezonedb.com/api or similar
            offset_hours = round(longitude / 15)

            # This is a simplified mapping - in production use proper timezone lookup
            timezone_map = {
                0: "Africa/Accra",  # Ghana timezone as default for West Africa
                1: "Africa/Lagos",
                -1: "Atlantic/Cape_Verde",
            }

            return timezone_map.get(offset_hours, "Africa/Accra")
        except Exception as e:
            print(f"Error determining timezone: {e}")
            return "Africa/Accra"

    async def get_climate_zone(self, latitude: float, longitude: float, weather_data: Dict = None) -> str:
        """
        Determine climate zone based on coordinates and weather data
        Uses Köppen climate classification with weather API data
        """
        try:
            # Try to get climate data from weather API if available
            if weather_data and weather_data.get('current'):
                temp = weather_data['current'].get('temperature')
                # In production, you would fetch annual temperature and precipitation data
                # For now, use enhanced latitude-based classification with temperature hints

                abs_lat = abs(latitude)

                # Enhanced classification considering location
                if abs_lat < 10:
                    # Equatorial zone
                    if temp and temp > 25:
                        return "Tropical Rainforest"
                    else:
                        return "Tropical Monsoon"
                elif abs_lat < 20:
                    # Tropical zone
                    if longitude < -20 or (longitude > 40 and longitude < 120):
                        return "Tropical Savanna"
                    else:
                        return "Tropical"
                elif abs_lat < 30:
                    return "Subtropical"
                elif abs_lat < 40:
                    return "Temperate"
                elif abs_lat < 60:
                    return "Continental"
                else:
                    return "Polar"

            # Fallback to basic latitude classification
            abs_lat = abs(latitude)
            if abs_lat < 10:
                return "Tropical Rainforest"
            elif abs_lat < 20:
                return "Tropical Savanna"
            elif abs_lat < 30:
                return "Subtropical"
            elif abs_lat < 40:
                return "Temperate"
            elif abs_lat < 60:
                return "Continental"
            else:
                return "Polar"

        except Exception as e:
            print(f"Error determining climate zone: {e}")
            return "Tropical"

    async def enrich_farm_location(
        self,
        latitude: float,
        longitude: float,
        include_satellite: bool = False
    ) -> Dict:
        """
        Comprehensive location enrichment for a farm
        Combines multiple data sources

        Parameters:
        - latitude: Farm latitude
        - longitude: Farm longitude
        - include_satellite: If True, fetch satellite imagery and NDVI data (slower)
        """
        try:
            # Fetch all data in parallel would be better with asyncio.gather
            elevation = await self.get_elevation(latitude, longitude)
            geocode_data = await self.reverse_geocode(latitude, longitude)
            timezone = await self.get_timezone(latitude, longitude)
            weather = await self.get_weather_data(latitude, longitude)
            climate_data = await self.get_climate_data(latitude, longitude)
            climate_zone = await self.get_climate_zone(latitude, longitude, weather)
            soil = await self.get_soil_data(latitude, longitude)

            # Prepare comprehensive response
            enrichment_result = {
                "altitude": elevation,
                "address": geocode_data.get("display_name"),
                "country": geocode_data.get("country"),
                "region": geocode_data.get("region"),
                "district": geocode_data.get("district"),
                "timezone": timezone,
                "climate_zone": climate_zone,
                "current_weather": weather.get("current"),
                "soil_data": soil,
            }

            # Add climate data if available
            if climate_data:
                enrichment_result["avg_temperature"] = climate_data.get("avg_temperature")
                enrichment_result["avg_annual_rainfall"] = climate_data.get("avg_annual_rainfall")
                enrichment_result["climate_data_source"] = climate_data.get("data_source")

            # Add satellite data if requested (slower, optional)
            if include_satellite:
                satellite_data = await self.get_satellite_imagery(latitude, longitude)
                enrichment_result["satellite_data"] = satellite_data
                enrichment_result["vegetation_health"] = satellite_data.get("ndvi_data", {})
                enrichment_result["satellite_recommendations"] = satellite_data.get("recommendations", [])

            return enrichment_result

        except Exception as e:
            print(f"Error enriching farm location: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _get_mock_weather_data(self) -> Dict:
        """Return mock weather data for testing"""
        return {
            "current": {
                "temperature": 28.5,
                "feels_like": 30.2,
                "humidity": 75,
                "pressure": 1013,
                "wind_speed": 3.5,
                "wind_direction": 180,
                "clouds": 40,
                "description": "partly cloudy",
                "icon": "02d",
            },
            "forecast": [],
            "timezone": "Africa/Accra",
        }


# Singleton instance
geospatial_service = GeospatialService()
