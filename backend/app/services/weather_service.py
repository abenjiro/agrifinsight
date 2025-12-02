"""
Weather Service for AgriFinSight
Handles weather data fetching, forecasting, and agricultural recommendations
"""

import requests
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from app.config import settings

logger = logging.getLogger(__name__)


class WeatherService:
    """Service for weather data and agricultural weather recommendations"""

    def __init__(self):
        self.api_key = settings.openweather_api_key
        self.api_url = settings.openweather_api_url

        # Validate API key is configured
        if not self.api_key:
            raise ValueError(
                "OpenWeather API key is required. "
                "Please set OPENWEATHER_API_KEY in your .env file. "
                "Get a free key at: https://openweathermap.org/api"
            )

    async def get_current_weather(self, latitude: float, longitude: float) -> Dict:
        """
        Get current weather conditions for a location
        Always fetches live data from OpenWeather API
        """
        try:
            url = f"{self.api_url}/weather"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                'temperature': round(data['main']['temp'], 1),
                'feels_like': round(data['main']['feels_like'], 1),
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': round(data['wind']['speed'], 1),
                'wind_direction': data['wind'].get('deg'),
                'clouds': data['clouds']['all'],
                'visibility': data.get('visibility'),
                'description': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon'],
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat(),
                'timestamp': datetime.now().isoformat()
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching current weather from OpenWeather API: {e}")
            raise Exception(
                f"Failed to fetch weather data from OpenWeather API. "
                f"Error: {str(e)}. Please check your API key and internet connection."
            )
        except Exception as e:
            logger.error(f"Unexpected error in get_current_weather: {e}")
            raise

    async def get_forecast(self, latitude: float, longitude: float, days: int = 7) -> Dict:
        """
        Get weather forecast for the next N days
        Uses free tier 5-day forecast API (forecast/daily is deprecated)
        Always fetches live data from OpenWeather API
        """
        try:
            # Use free tier forecast API (5 day / 3 hour forecast)
            url = f"{self.api_url}/forecast"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': min(days * 8, 40)  # 8 forecasts per day, max 40 (5 days)
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Group forecasts by day and aggregate
            daily_forecast = []
            current_date = None
            day_temps = []
            day_data = {}

            for forecast in data.get('list', []):
                forecast_date = datetime.fromtimestamp(forecast['dt']).date()

                if current_date != forecast_date:
                    # Save previous day if exists
                    if current_date and day_temps:
                        daily_forecast.append({
                            'date': datetime.combine(current_date, datetime.min.time()).isoformat(),
                            'temp_min': round(min(day_temps), 1),
                            'temp_max': round(max(day_temps), 1),
                            'temp_day': round(sum(day_temps) / len(day_temps), 1),
                            'temp_night': round(day_temps[0] if len(day_temps) > 0 else 20, 1),
                            'humidity': day_data.get('humidity', 65),
                            'pressure': day_data.get('pressure', 1013),
                            'wind_speed': round(day_data.get('wind_speed', 3.0), 1),
                            'clouds': day_data.get('clouds', 40),
                            'pop': round(day_data.get('pop', 0) * 100),
                            'rain': day_data.get('rain', 0),
                            'description': day_data.get('description', 'partly cloudy'),
                            'icon': day_data.get('icon', '02d')
                        })

                    # Start new day
                    current_date = forecast_date
                    day_temps = []
                    day_data = {}

                # Collect data for current day
                day_temps.append(forecast['main']['temp'])
                day_data = {
                    'humidity': forecast['main']['humidity'],
                    'pressure': forecast['main']['pressure'],
                    'wind_speed': forecast['wind']['speed'],
                    'clouds': forecast['clouds']['all'],
                    'pop': forecast.get('pop', 0),
                    'rain': forecast.get('rain', {}).get('3h', 0),
                    'description': forecast['weather'][0]['description'],
                    'icon': forecast['weather'][0]['icon']
                }

                # Stop if we have enough days
                if len(daily_forecast) >= days:
                    break

            # Add the last day
            if current_date and day_temps and len(daily_forecast) < days:
                daily_forecast.append({
                    'date': datetime.combine(current_date, datetime.min.time()).isoformat(),
                    'temp_min': round(min(day_temps), 1),
                    'temp_max': round(max(day_temps), 1),
                    'temp_day': round(sum(day_temps) / len(day_temps), 1),
                    'temp_night': round(day_temps[0] if len(day_temps) > 0 else 20, 1),
                    'humidity': day_data.get('humidity', 65),
                    'pressure': day_data.get('pressure', 1013),
                    'wind_speed': round(day_data.get('wind_speed', 3.0), 1),
                    'clouds': day_data.get('clouds', 40),
                    'pop': round(day_data.get('pop', 0) * 100),
                    'rain': day_data.get('rain', 0),
                    'description': day_data.get('description', 'partly cloudy'),
                    'icon': day_data.get('icon', '02d')
                })

            return {
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'timezone': data.get('city', {}).get('timezone', 'Africa/Accra')
                },
                'forecast': daily_forecast[:days],
                'generated_at': datetime.now().isoformat()
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching forecast from OpenWeather API: {e}")
            raise Exception(
                f"Failed to fetch weather forecast from OpenWeather API. "
                f"Error: {str(e)}. Please check your API key and internet connection."
            )
        except Exception as e:
            logger.error(f"Unexpected error in get_forecast: {e}")
            logger.error(f"Error details: {str(e)}")
            raise

    async def get_planting_weather_advice(
        self,
        latitude: float,
        longitude: float,
        crop_type: str
    ) -> Dict:
        """
        Get weather-based planting advice for a specific crop
        """
        # Get current weather and forecast
        current = await self.get_current_weather(latitude, longitude)
        forecast = await self.get_forecast(latitude, longitude, days=14)

        # Analyze conditions for planting
        advice = self._analyze_planting_conditions(current, forecast, crop_type)

        return {
            'crop_type': crop_type,
            'current_conditions': current,
            'forecast_summary': self._summarize_forecast(forecast),
            'planting_advice': advice,
            'generated_at': datetime.now().isoformat()
        }

    async def get_harvest_weather_advice(
        self,
        latitude: float,
        longitude: float,
        crop_type: str,
        estimated_harvest_date: str
    ) -> Dict:
        """
        Get weather-based harvest timing advice
        """
        forecast = await self.get_forecast(latitude, longitude, days=14)

        advice = self._analyze_harvest_conditions(forecast, crop_type, estimated_harvest_date)

        return {
            'crop_type': crop_type,
            'estimated_harvest_date': estimated_harvest_date,
            'forecast_summary': self._summarize_forecast(forecast),
            'harvest_advice': advice,
            'generated_at': datetime.now().isoformat()
        }

    def _analyze_planting_conditions(
        self,
        current: Dict,
        forecast: Dict,
        crop_type: str
    ) -> Dict:
        """
        Analyze weather conditions to determine if it's a good time to plant
        """
        # Get forecast data
        forecast_days = forecast.get('forecast', [])

        if not forecast_days:
            return {
                'recommendation': 'wait',
                'reason': 'Unable to fetch weather forecast',
                'confidence': 'low'
            }

        # Calculate averages for next 7 days
        avg_temp = sum(d['temp_day'] for d in forecast_days[:7]) / min(7, len(forecast_days))
        total_rain = sum(d.get('rain', 0) for d in forecast_days[:7])
        rain_days = sum(1 for d in forecast_days[:7] if d.get('pop', 0) > 50)

        # Crop-specific thresholds
        crop_requirements = {
            'maize': {'min_temp': 18, 'max_temp': 35, 'min_rain': 50, 'max_rain': 300},
            'rice': {'min_temp': 20, 'max_temp': 38, 'min_rain': 100, 'max_rain': 500},
            'cassava': {'min_temp': 20, 'max_temp': 35, 'min_rain': 30, 'max_rain': 250},
            'tomato': {'min_temp': 18, 'max_temp': 30, 'min_rain': 40, 'max_rain': 200},
            'soybean': {'min_temp': 20, 'max_temp': 35, 'min_rain': 50, 'max_rain': 300},
            'groundnut': {'min_temp': 20, 'max_temp': 35, 'min_rain': 50, 'max_rain': 250}
        }

        requirements = crop_requirements.get(crop_type.lower(), {
            'min_temp': 18, 'max_temp': 35, 'min_rain': 50, 'max_rain': 300
        })

        # Determine recommendation
        issues = []
        strengths = []

        # Temperature check
        if avg_temp < requirements['min_temp']:
            issues.append(f"Temperature too low ({avg_temp:.1f}°C, need >{requirements['min_temp']}°C)")
        elif avg_temp > requirements['max_temp']:
            issues.append(f"Temperature too high ({avg_temp:.1f}°C, need <{requirements['max_temp']}°C)")
        else:
            strengths.append(f"Temperature optimal ({avg_temp:.1f}°C)")

        # Rainfall check
        if total_rain < requirements['min_rain']:
            issues.append(f"Insufficient rainfall expected ({total_rain:.0f}mm, need >{requirements['min_rain']}mm)")
        elif total_rain > requirements['max_rain']:
            issues.append(f"Excessive rainfall expected ({total_rain:.0f}mm, may cause waterlogging)")
        else:
            strengths.append(f"Rainfall adequate ({total_rain:.0f}mm over next 7 days)")

        # Determine overall recommendation
        if len(issues) == 0:
            recommendation = 'plant_now'
            confidence = 'high'
            reason = "Conditions are optimal for planting. " + " ".join(strengths)
        elif len(issues) == 1:
            recommendation = 'plant_with_caution'
            confidence = 'medium'
            reason = f"Planting possible but: {issues[0]}. Consider irrigation or protection measures."
        else:
            recommendation = 'wait'
            confidence = 'high'
            reason = "Wait for better conditions: " + "; ".join(issues)

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'reason': reason,
            'conditions': {
                'avg_temperature': round(avg_temp, 1),
                'total_rainfall_7days': round(total_rain, 1),
                'rainy_days': rain_days
            },
            'optimal_ranges': requirements,
            'strengths': strengths,
            'concerns': issues
        }

    def _analyze_harvest_conditions(
        self,
        forecast: Dict,
        crop_type: str,
        estimated_harvest_date: str
    ) -> Dict:
        """
        Analyze weather conditions for harvest timing
        """
        forecast_days = forecast.get('forecast', [])

        if not forecast_days:
            return {
                'recommendation': 'monitor',
                'reason': 'Unable to fetch weather forecast'
            }

        # Look for dry weather window
        dry_days = []
        for day in forecast_days:
            if day.get('pop', 100) < 30 and day.get('rain', 10) < 5:
                dry_days.append(day['date'])

        if len(dry_days) >= 3:
            recommendation = 'favorable'
            reason = f"Good harvest window: {len(dry_days)} dry days forecasted in next 2 weeks"
        elif len(dry_days) >= 1:
            recommendation = 'proceed_with_caution'
            reason = f"Limited dry days ({len(dry_days)} days). Consider early harvest if crop is ready"
        else:
            recommendation = 'delay'
            reason = "Heavy rain expected. Wait for drier conditions to prevent crop damage"

        return {
            'recommendation': recommendation,
            'reason': reason,
            'dry_days_available': len(dry_days),
            'optimal_harvest_dates': dry_days[:3]
        }

    def _summarize_forecast(self, forecast: Dict) -> Dict:
        """
        Create a summary of the forecast
        """
        forecast_days = forecast.get('forecast', [])

        if not forecast_days:
            return {}

        return {
            'avg_temp': round(sum(d['temp_day'] for d in forecast_days) / len(forecast_days), 1),
            'min_temp': round(min(d['temp_min'] for d in forecast_days), 1),
            'max_temp': round(max(d['temp_max'] for d in forecast_days), 1),
            'total_rainfall': round(sum(d.get('rain', 0) for d in forecast_days), 1),
            'rainy_days': sum(1 for d in forecast_days if d.get('pop', 0) > 50),
            'avg_humidity': round(sum(d['humidity'] for d in forecast_days) / len(forecast_days), 0)
        }


# Create singleton instance
weather_service = WeatherService()
