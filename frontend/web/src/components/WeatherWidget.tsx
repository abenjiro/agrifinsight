import { useEffect, useState } from 'react'
import { Cloud, Droplets, Wind, Gauge, Sunrise, Sunset, Calendar } from 'lucide-react'

interface CurrentWeather {
  temperature: number
  feels_like: number
  humidity: number
  pressure: number
  wind_speed: number
  wind_direction?: number
  clouds: number
  visibility?: number
  description: string
  icon: string
  sunrise: string
  sunset: string
  timestamp: string
  is_mock_data?: boolean
}

interface ForecastDay {
  date: string
  temp_min: number
  temp_max: number
  temp_day: number
  humidity: number
  wind_speed: number
  clouds: number
  pop: number
  rain: number
  description: string
  icon: string
}

interface WeatherData {
  current_weather: CurrentWeather
  forecast: {
    forecast: ForecastDay[]
  }
}

interface WeatherWidgetProps {
  farmId: number
  compact?: boolean
  showForecast?: boolean
  days?: number
}

export function WeatherWidget({ farmId, compact = false, showForecast = true, days = 7 }: WeatherWidgetProps) {
  const [weather, setWeather] = useState<WeatherData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchWeather()
  }, [farmId, days])

  const fetchWeather = async () => {
    setLoading(true)
    setError(null)

    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch(
        `http://localhost:8000/api/recommendations/weather/${farmId}?days=${days}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to fetch weather data')
      }

      const data = await response.json()
      setWeather(data)
    } catch (err: any) {
      console.error('Weather fetch error:', err)
      setError(err.message || 'Failed to load weather')
    } finally {
      setLoading(false)
    }
  }

  const formatTime = (isoString: string) => {
    return new Date(isoString).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const formatDate = (isoString: string) => {
    const date = new Date(isoString)
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric'
    })
  }

  const getWeatherIcon = (iconCode: string) => {
    // Map OpenWeatherMap icons to emoji or use icon code
    const iconMap: { [key: string]: string } = {
      '01d': '☀️', '01n': '🌙',
      '02d': '⛅', '02n': '☁️',
      '03d': '☁️', '03n': '☁️',
      '04d': '☁️', '04n': '☁️',
      '09d': '🌧️', '09n': '🌧️',
      '10d': '🌦️', '10n': '🌧️',
      '11d': '⛈️', '11n': '⛈️',
      '13d': '🌨️', '13n': '🌨️',
      '50d': '🌫️', '50n': '🌫️'
    }
    return iconMap[iconCode] || '🌤️'
  }

  if (loading) {
    return (
      <div className="bg-white rounded-2xl shadow-soft-xl p-6">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-sm text-gray-600">Loading weather...</span>
        </div>
      </div>
    )
  }

  if (error || !weather) {
    return (
      <div className="bg-white rounded-2xl shadow-soft-xl p-6">
        <div className="text-center py-8">
          <Cloud className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <p className="text-sm text-gray-600">{error || 'Weather data unavailable'}</p>
          <button
            onClick={fetchWeather}
            className="mt-3 text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  const current = weather.current_weather

  if (compact) {
    return (
      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl p-4 border border-blue-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="text-4xl">{getWeatherIcon(current.icon)}</div>
            <div>
              <div className="text-2xl font-bold text-gray-900">{Math.round(current.temperature)}°C</div>
              <div className="text-xs text-gray-600 capitalize">{current.description}</div>
            </div>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-1 text-sm text-gray-600">
              <Droplets className="w-3 h-3" />
              <span>{current.humidity}%</span>
            </div>
            <div className="flex items-center gap-1 text-sm text-gray-600">
              <Wind className="w-3 h-3" />
              <span>{current.wind_speed} m/s</span>
            </div>
          </div>
        </div>
        {current.is_mock_data && (
          <div className="mt-2 text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded">
            ⚠️ Mock data - Add OpenWeatherMap API key for real-time weather
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="bg-white rounded-2xl shadow-soft-xl overflow-hidden">
      {/* Current Weather */}
      <div className="bg-gradient-to-br from-blue-500 to-cyan-600 text-white p-6">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold mb-1">Current Weather</h3>
            <p className="text-blue-100 text-sm">Updated: {formatTime(current.timestamp)}</p>
          </div>
          {current.is_mock_data && (
            <div className="bg-amber-500 text-white px-2 py-1 rounded text-xs">
              Mock Data
            </div>
          )}
        </div>

        <div className="mt-6 grid grid-cols-2 gap-6">
          <div>
            <div className="flex items-center gap-4">
              <div className="text-6xl">{getWeatherIcon(current.icon)}</div>
              <div>
                <div className="text-5xl font-bold">{Math.round(current.temperature)}°</div>
                <div className="text-blue-100 text-sm">Feels like {Math.round(current.feels_like)}°</div>
              </div>
            </div>
            <p className="mt-3 text-lg capitalize">{current.description}</p>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Droplets className="w-5 h-5 text-blue-200" />
              <span className="text-sm">Humidity: <strong>{current.humidity}%</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <Wind className="w-5 h-5 text-blue-200" />
              <span className="text-sm">Wind: <strong>{current.wind_speed} m/s</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <Gauge className="w-5 h-5 text-blue-200" />
              <span className="text-sm">Pressure: <strong>{current.pressure} hPa</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <Cloud className="w-5 h-5 text-blue-200" />
              <span className="text-sm">Clouds: <strong>{current.clouds}%</strong></span>
            </div>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-blue-400 flex items-center justify-between text-sm">
          <div className="flex items-center gap-2">
            <Sunrise className="w-4 h-4" />
            <span>Sunrise: {formatTime(current.sunrise)}</span>
          </div>
          <div className="flex items-center gap-2">
            <Sunset className="w-4 h-4" />
            <span>Sunset: {formatTime(current.sunset)}</span>
          </div>
        </div>
      </div>

      {/* Forecast */}
      {showForecast && weather.forecast.forecast && weather.forecast.forecast.length > 0 && (
        <div className="p-6">
          <h4 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Calendar className="w-4 h-4" />
            {days}-Day Forecast
          </h4>

          <div className="space-y-3">
            {weather.forecast.forecast.map((day, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition"
              >
                <div className="flex items-center gap-4 flex-1">
                  <div className="text-2xl">{getWeatherIcon(day.icon)}</div>
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">{formatDate(day.date)}</div>
                    <div className="text-xs text-gray-600 capitalize">{day.description}</div>
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  <div className="text-center">
                    <div className="text-xs text-gray-500">Temp</div>
                    <div className="font-semibold text-gray-900">
                      {Math.round(day.temp_max)}° / {Math.round(day.temp_min)}°
                    </div>
                  </div>

                  <div className="text-center">
                    <div className="text-xs text-gray-500">Rain</div>
                    <div className="font-semibold text-blue-600">{day.pop}%</div>
                  </div>

                  <div className="text-center">
                    <div className="text-xs text-gray-500">Wind</div>
                    <div className="font-semibold text-gray-600">{day.wind_speed.toFixed(1)} m/s</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
