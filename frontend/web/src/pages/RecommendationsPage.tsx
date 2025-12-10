import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Lightbulb,
  TrendingUp,
  CheckCircle,
  Calendar,
  Droplets,
  Sun,
  Cloud,
  Wind,
  RefreshCw,
  Sprout,
  CloudRain,
  Thermometer,
  Loader,
  AlertCircle,
  ChevronRight,
  MapPin
} from 'lucide-react'

interface Farm {
  id: number
  name: string
  latitude?: number
  longitude?: number
  address?: string
}

interface CropRecommendation {
  crop: string
  suitability_score: number
  recommendation: string
  planting_date: string
  harvest_date: string
  summary: string
}

interface WeatherForecast {
  date: string
  temp_min: number
  temp_max: number
  temp_day: number
  humidity: number
  wind_speed: number
  pop: number
  rain: number
  description: string
  icon: string
}

export function RecommendationsPage() {
  const navigate = useNavigate()
  const [farms, setFarms] = useState<Farm[]>([])
  const [selectedFarm, setSelectedFarm] = useState<Farm | null>(null)
  const [cropRecommendations, setCropRecommendations] = useState<CropRecommendation[]>([])
  const [weatherForecast, setWeatherForecast] = useState<WeatherForecast[]>([])
  const [loading, setLoading] = useState(true)
  const [weatherLoading, setWeatherLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    const loadFarms = async () => {
      if (!isMounted) return
      setLoading(true)
      try {
        const token = localStorage.getItem('auth_token')
        const response = await fetch('http://localhost:8000/api/farms/', {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        })

        if (!response.ok) throw new Error('Failed to fetch farms')
        if (!isMounted) return

        const data = await response.json()
        const farmsList = data.data || data
        setFarms(farmsList)

        // Auto-select first farm with coordinates
        const farmWithCoords = farmsList.find((f: Farm) => f.latitude && f.longitude)
        if (farmWithCoords && isMounted) {
          setSelectedFarm(farmWithCoords)
        }
      } catch (err: any) {
        if (isMounted) {
          console.error('Error fetching farms:', err)
          setError(err.message)
        }
      } finally {
        if (isMounted) {
          setLoading(false)
        }
      }
    }

    loadFarms()

    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    const abortController = new AbortController()

    const loadData = async () => {
      if (selectedFarm && selectedFarm.latitude && selectedFarm.longitude) {
        await fetchRecommendations(abortController.signal)
        if (!abortController.signal.aborted) {
          await fetchWeather(abortController.signal)
        }
      }
    }

    loadData()

    return () => {
      abortController.abort()
    }
  }, [selectedFarm])

  const fetchFarms = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch('http://localhost:8000/api/farms/', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) throw new Error('Failed to fetch farms')

      const data = await response.json()
      const farmsList = data.data || data
      setFarms(farmsList)

      // Auto-select first farm with coordinates
      const farmWithCoords = farmsList.find((f: Farm) => f.latitude && f.longitude)
      if (farmWithCoords) {
        setSelectedFarm(farmWithCoords)
      }
    } catch (err: any) {
      console.error('Error fetching farms:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const fetchRecommendations = async (signal?: AbortSignal) => {
    if (!selectedFarm) return

    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch(
        `http://localhost:8000/api/recommendations/planting/${selectedFarm.id}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          signal
        }
      )

      if (!response.ok) throw new Error('Failed to fetch recommendations')

      const data = await response.json()
      setCropRecommendations(data.comparison.comparison || [])
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('Error fetching recommendations:', err)
      }
    }
  }

  const fetchWeather = async (signal?: AbortSignal) => {
    if (!selectedFarm) return

    setWeatherLoading(true)
    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch(
        `http://localhost:8000/api/recommendations/weather/${selectedFarm.id}?days=7`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          },
          signal
        }
      )

      if (!response.ok) throw new Error('Failed to fetch weather')

      const data = await response.json()
      setWeatherForecast(data.forecast.forecast || [])
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('Error fetching weather:', err)
      }
    } finally {
      setWeatherLoading(false)
    }
  }

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case 'highly_recommended':
        return 'from-green-600 to-lime-400'
      case 'recommended':
        return 'from-blue-600 to-cyan-400'
      case 'not_recommended':
        return 'from-red-600 to-rose-400'
      default:
        return 'from-gray-600 to-gray-400'
    }
  }

  const getRecommendationBg = (rec: string) => {
    switch (rec) {
      case 'highly_recommended':
        return 'bg-green-100 text-green-800'
      case 'recommended':
        return 'bg-blue-100 text-blue-800'
      case 'not_recommended':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getRecommendationText = (rec: string) => {
    switch (rec) {
      case 'highly_recommended':
        return 'Highly Recommended'
      case 'recommended':
        return 'Recommended'
      case 'not_recommended':
        return 'Not Recommended'
      default:
        return 'Review Needed'
    }
  }

  const getWeatherIcon = (iconCode: string) => {
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
      <div className="flex items-center justify-center h-64">
        <Loader className="h-12 w-12 animate-spin text-green-600" />
        <span className="ml-3 text-gray-600">Loading recommendations...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="w-full max-w-full px-3">
        <div className="bg-red-50 border border-red-200 rounded-2xl p-6 text-center">
          <AlertCircle className="w-12 h-12 text-red-600 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-red-900 mb-2">Error Loading Recommendations</h3>
          <p className="text-red-700 mb-4">{error}</p>
          <button
            onClick={fetchFarms}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  if (farms.length === 0) {
    return (
      <div className="w-full max-w-full px-3">
        <div className="bg-white border-0 shadow-soft-xl rounded-2xl p-8 text-center">
          <Sprout className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No Farms Yet</h3>
          <p className="text-gray-600 mb-6">Create a farm first to get personalized recommendations</p>
          <button
            onClick={() => navigate('/farms')}
            className="px-6 py-3 bg-gradient-to-tl from-green-600 to-lime-400 text-white rounded-lg font-semibold hover:scale-102 transition"
          >
            Go to Farms
          </button>
        </div>
      </div>
    )
  }

  return (
    <>
      {/* Farm Selector */}
      <div className="w-full max-w-full px-3 mb-6">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h6 className="mb-0 font-bold text-gray-900">AI-Powered Recommendations</h6>
                <p className="text-sm text-gray-600">Select a farm to view personalized insights</p>
              </div>
              <button
                onClick={() => {
                  fetchRecommendations()
                  fetchWeather()
                }}
                className="flex items-center gap-2 px-4 py-2 bg-gradient-to-tl from-purple-700 to-pink-500 text-white rounded-lg hover:scale-102 transition text-xs font-bold uppercase"
              >
                <RefreshCw className="w-3 h-3" />
                Refresh
              </button>
            </div>

            <div className="flex flex-wrap gap-3">
              {farms.map((farm) => (
                <button
                  key={farm.id}
                  onClick={() => setSelectedFarm(farm)}
                  className={`px-4 py-2 rounded-lg font-medium text-sm transition ${
                    selectedFarm?.id === farm.id
                      ? 'bg-gradient-to-tl from-green-600 to-lime-400 text-white shadow-md'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <MapPin className="w-3 h-3 inline mr-1" />
                  {farm.name}
                </button>
              ))}
            </div>

            {selectedFarm && !selectedFarm.latitude && (
              <div className="mt-4 bg-amber-50 border border-amber-200 rounded-lg p-4">
                <AlertCircle className="w-5 h-5 text-amber-600 inline mr-2" />
                <span className="text-sm text-amber-800">
                  This farm needs GPS coordinates for weather and planting recommendations.
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {selectedFarm && selectedFarm.latitude && selectedFarm.longitude && (
        <>
          {/* Crop Recommendations */}
          <div className="w-full max-w-full px-3 mb-6 lg:w-8/12 lg:flex-none">
            <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border mb-6">
              <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
                <div className="flex items-center justify-between">
                  <div>
                    <h6 className="mb-0 font-bold">Planting Recommendations</h6>
                    <p className="leading-normal text-sm text-gray-600">
                      Top crops for {selectedFarm.name}
                    </p>
                  </div>
                  <button
                    onClick={() => navigate(`/farms/${selectedFarm.id}/planting`)}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-green-700 bg-green-50 hover:bg-green-100 rounded-lg transition"
                  >
                    View All Crops
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <div className="flex-auto p-4 space-y-4">
                {cropRecommendations.length === 0 ? (
                  <div className="text-center py-8">
                    <Loader className="w-8 h-8 animate-spin text-green-600 mx-auto mb-3" />
                    <p className="text-gray-600">Loading crop recommendations...</p>
                  </div>
                ) : (
                  cropRecommendations.slice(0, 5).map((crop) => (
                    <div
                      key={crop.crop}
                      className="relative flex flex-col min-w-0 break-words bg-gray-50 border-0 shadow-soft-xs rounded-xl bg-clip-border hover:shadow-soft-lg transition-all cursor-pointer"
                      onClick={() => navigate(`/farms/${selectedFarm.id}/planting`)}
                    >
                      <div className="flex-auto p-4">
                        <div className="flex items-start space-x-4">
                          <div className={`w-12 h-12 bg-gradient-to-tl ${getRecommendationColor(crop.recommendation)} rounded-lg flex items-center justify-center flex-shrink-0 shadow-soft-md`}>
                            <Sprout className="w-6 h-6 text-white" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between mb-2">
                              <h6 className="mb-0 font-semibold text-gray-900">{crop.crop}</h6>
                              <div className="flex items-center gap-3">
                                <span className="text-2xl font-bold text-green-600">
                                  {crop.suitability_score.toFixed(0)}
                                </span>
                                <span className={`inline-block px-2.5 py-1 text-xs font-semibold rounded-lg ${getRecommendationBg(crop.recommendation)}`}>
                                  {getRecommendationText(crop.recommendation)}
                                </span>
                              </div>
                            </div>
                            <p className="mb-3 leading-normal text-sm text-gray-700 line-clamp-2">
                              {crop.summary}
                            </p>
                            <div className="flex items-center space-x-4 text-xs text-gray-500">
                              <span className="flex items-center gap-1">
                                <Calendar className="w-3 h-3" />
                                Plant: {new Date(crop.planting_date).toLocaleDateString()}
                              </span>
                              <span>•</span>
                              <span className="flex items-center gap-1">
                                <TrendingUp className="w-3 h-3" />
                                Harvest: {new Date(crop.harvest_date).toLocaleDateString()}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Action Button */}
            <button
              onClick={() => navigate(`/farms/${selectedFarm.id}/planting`)}
              className="inline-block w-full px-6 py-3 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
            >
              <Lightbulb className="w-4 h-4 inline mr-2" />
              View Detailed Analysis
            </button>
          </div>

          {/* Sidebar */}
          <div className="w-full max-w-full px-3 lg:w-4/12 lg:flex-none">
            {/* Weather Forecast Card */}
            <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border mb-6">
              <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
                <div className="flex items-center justify-between">
                  <div>
                    <h6 className="mb-0 font-bold">Weather Forecast</h6>
                    <p className="leading-normal text-sm text-gray-600">Next 7 days</p>
                  </div>
                  {weatherLoading && <Loader className="w-4 h-4 animate-spin text-blue-600" />}
                </div>
              </div>
              <div className="flex-auto p-4">
                <div className="space-y-3">
                  {weatherForecast.length === 0 ? (
                    <div className="text-center py-4 text-gray-500 text-sm">
                      Loading weather...
                    </div>
                  ) : (
                    weatherForecast.map((weather, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-3 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg border border-blue-100 hover:shadow-md transition"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="text-3xl">
                            {getWeatherIcon(weather.icon)}
                          </div>
                          <div>
                            <p className="text-xs font-semibold text-gray-900">
                              {new Date(weather.date).toLocaleDateString('en-US', {
                                weekday: 'short',
                                month: 'short',
                                day: 'numeric'
                              })}
                            </p>
                            <p className="text-xs text-gray-600 capitalize">{weather.description}</p>
                            <div className="flex items-center gap-2 mt-1">
                              <span className="text-xs text-blue-600 flex items-center gap-1">
                                <Droplets className="w-3 h-3" />
                                {weather.pop}%
                              </span>
                              {weather.rain > 0 && (
                                <span className="text-xs text-cyan-600">
                                  {weather.rain.toFixed(1)}mm
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-lg font-bold text-gray-900">
                            {Math.round(weather.temp_max)}°
                          </p>
                          <p className="text-xs text-gray-500">
                            {Math.round(weather.temp_min)}°
                          </p>
                          <p className="text-xs text-gray-500 flex items-center justify-end gap-1 mt-1">
                            <Wind className="w-3 h-3" />
                            {weather.wind_speed.toFixed(1)} m/s
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Crop Suitability Stats Card */}
            <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border mb-6">
              <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
                <h6 className="mb-0 font-bold">Crop Suitability</h6>
                <p className="leading-normal text-sm text-gray-600">Based on suitability scores</p>
              </div>
              <div className="flex-auto p-4">
                <div className="space-y-3">
                  {/* Highly Suitable (80-100%) */}
                  <button
                    onClick={() => setSelectedCategory(selectedCategory === 'high' ? null : 'high')}
                    className="w-full flex items-center justify-between p-3 bg-green-50 rounded-lg border border-green-200 hover:bg-green-100 transition cursor-pointer"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                      <span className="text-sm font-medium text-gray-700">Highly Suitable (≥80%)</span>
                    </div>
                    <span className="text-lg font-bold text-green-600">
                      {cropRecommendations.filter((r) => r.suitability_score >= 80).length}
                    </span>
                  </button>
                  {selectedCategory === 'high' && cropRecommendations.filter((r) => r.suitability_score >= 80).length > 0 && (
                    <div className="ml-5 pl-4 border-l-2 border-green-300 space-y-2">
                      {cropRecommendations.filter((r) => r.suitability_score >= 80).map((crop) => (
                        <div key={crop.crop} className="flex items-center justify-between py-2 px-3 bg-white rounded border border-green-100 hover:border-green-300 transition">
                          <span className="text-sm font-medium text-gray-800">{crop.crop}</span>
                          <span className="text-sm font-bold text-green-600">{crop.suitability_score.toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Suitable (60-79%) */}
                  <button
                    onClick={() => setSelectedCategory(selectedCategory === 'medium' ? null : 'medium')}
                    className="w-full flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200 hover:bg-blue-100 transition cursor-pointer"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                      <span className="text-sm font-medium text-gray-700">Suitable (60-79%)</span>
                    </div>
                    <span className="text-lg font-bold text-blue-600">
                      {cropRecommendations.filter((r) => r.suitability_score >= 60 && r.suitability_score < 80).length}
                    </span>
                  </button>
                  {selectedCategory === 'medium' && cropRecommendations.filter((r) => r.suitability_score >= 60 && r.suitability_score < 80).length > 0 && (
                    <div className="ml-5 pl-4 border-l-2 border-blue-300 space-y-2">
                      {cropRecommendations.filter((r) => r.suitability_score >= 60 && r.suitability_score < 80).map((crop) => (
                        <div key={crop.crop} className="flex items-center justify-between py-2 px-3 bg-white rounded border border-blue-100 hover:border-blue-300 transition">
                          <span className="text-sm font-medium text-gray-800">{crop.crop}</span>
                          <span className="text-sm font-bold text-blue-600">{crop.suitability_score.toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Moderate (<60%) */}
                  <button
                    onClick={() => setSelectedCategory(selectedCategory === 'low' ? null : 'low')}
                    className="w-full flex items-center justify-between p-3 bg-amber-50 rounded-lg border border-amber-200 hover:bg-amber-100 transition cursor-pointer"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-amber-600 rounded-full"></div>
                      <span className="text-sm font-medium text-gray-700">Moderate (&lt;60%)</span>
                    </div>
                    <span className="text-lg font-bold text-amber-600">
                      {cropRecommendations.filter((r) => r.suitability_score < 60).length}
                    </span>
                  </button>
                  {selectedCategory === 'low' && cropRecommendations.filter((r) => r.suitability_score < 60).length > 0 && (
                    <div className="ml-5 pl-4 border-l-2 border-amber-300 space-y-2">
                      {cropRecommendations.filter((r) => r.suitability_score < 60).map((crop) => (
                        <div key={crop.crop} className="flex items-center justify-between py-2 px-3 bg-white rounded border border-amber-100 hover:border-amber-300 transition">
                          <span className="text-sm font-medium text-gray-800">{crop.crop}</span>
                          <span className="text-sm font-bold text-amber-600">{crop.suitability_score.toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Quick Actions Card */}
            <div className="relative flex flex-col min-w-0 break-words bg-gradient-to-tl from-purple-700 to-pink-500 border-0 shadow-soft-xl rounded-2xl bg-clip-border">
              <div className="flex-auto p-4">
                <div className="text-white">
                  <h6 className="mb-2 text-white font-semibold">Ready to Plant?</h6>
                  <p className="mb-4 text-xs font-normal leading-tight text-white opacity-80">
                    View detailed analysis including weather, seasonal timing, and soil suitability
                  </p>
                  <button
                    onClick={() => navigate(`/farms/${selectedFarm.id}/planting`)}
                    className="inline-block w-full px-6 py-2.5 mb-0 font-bold text-center uppercase align-middle transition-all bg-white rounded-lg cursor-pointer leading-normal text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 text-purple-700"
                  >
                    <Calendar className="w-3 h-3 inline mr-2" />
                    View Full Analysis
                  </button>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  )
}
