"""
Geospatial service for weather, satellite, and environmental data
"""

import requests
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class GeospatialService:
    """Service for handling geospatial data, weather, and satellite imagery"""

    def __init__(self):
        # API keys (set in environment variables)
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "")
        self.nasa_api_key = os.getenv("NASA_API_KEY", "DEMO_KEY")
        self.elevation_api_url = "https://api.open-elevation.com/api/v1/lookup"

    async def get_weather_data(self, latitude: float, longitude: float) -> Dict:
        """
        Fetch current weather and forecast data for given coordinates
        Uses OpenWeatherMap API
        """
        if not self.openweather_api_key:
            return self._get_mock_weather_data()

        try:
            # Current weather
            current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={self.openweather_api_key}&units=metric"
            current_response = requests.get(current_url, timeout=10)
            current_data = current_response.json() if current_response.ok else {}

            # 7-day forecast
            forecast_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={latitude}&lon={longitude}&exclude=minutely,hourly&appid={self.openweather_api_key}&units=metric"
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
            url = f"{self.elevation_api_url}?locations={latitude},{longitude}"
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

    async def get_satellite_imagery(self, latitude: float, longitude: float,
                                   date: Optional[str] = None) -> Dict:
        """
        Fetch satellite imagery from NASA POWER API or Sentinel
        Returns URLs to satellite images and NDVI data
        """
        try:
            # NASA POWER API for agricultural data
            if not date:
                date = datetime.now().strftime("%Y%m%d")

            # This is a simplified example - in production, you'd use:
            # - Sentinel Hub API for actual satellite imagery
            # - Google Earth Engine API for NDVI and land use
            # - Planet Labs API for high-resolution imagery

            power_url = f"https://power.larc.nasa.gov/api/temporal/daily/point"
            params = {
                "parameters": "T2M,PRECTOTCORR,RH2M",  # Temperature, Precipitation, Humidity
                "community": "AG",
                "longitude": longitude,
                "latitude": latitude,
                "start": (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                "end": datetime.now().strftime("%Y%m%d"),
                "format": "JSON"
            }

            response = requests.get(power_url, params=params, timeout=15)
            data = response.json() if response.ok else {}

            return {
                "source": "NASA_POWER",
                "climate_data": data.get("properties", {}).get("parameter", {}),
                "ndvi_available": False,  # Would be true with Sentinel/GEE
                "last_image_date": datetime.now().isoformat(),
                "image_url": None,  # Would contain actual URL with commercial API
            }
        except Exception as e:
            print(f"Error fetching satellite data: {e}")
            return {
                "source": "UNAVAILABLE",
                "error": str(e)
            }

    async def get_soil_data(self, latitude: float, longitude: float) -> Dict:
        """
        Get soil data for given coordinates
        Uses SoilGrids REST API
        """
        try:
            # SoilGrids API
            url = f"https://rest.isric.org/soilgrids/v2.0/properties/query"
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
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "addressdetails": 1
            }
            headers = {
                "User-Agent": "AgriFinSight/1.0"  # Required by Nominatim
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

    async def get_climate_zone(self, latitude: float, longitude: float) -> str:
        """
        Determine climate zone based on coordinates
        Uses Köppen climate classification
        """
        # Simplified Köppen classification based on latitude
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

    async def enrich_farm_location(self, latitude: float, longitude: float) -> Dict:
        """
        Comprehensive location enrichment for a farm
        Combines multiple data sources
        """
        try:
            # Fetch all data in parallel would be better with asyncio.gather
            elevation = await self.get_elevation(latitude, longitude)
            geocode_data = await self.reverse_geocode(latitude, longitude)
            timezone = await self.get_timezone(latitude, longitude)
            climate_zone = await self.get_climate_zone(latitude, longitude)
            weather = await self.get_weather_data(latitude, longitude)
            soil = await self.get_soil_data(latitude, longitude)

            return {
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
        except Exception as e:
            print(f"Error enriching farm location: {e}")
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
