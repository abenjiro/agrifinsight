import { useState, useEffect } from 'react'
import {
  Lightbulb,
  TrendingUp,
  CheckCircle,
  Calendar,
  Droplets,
  Sun,
  Cloud,
  Wind,
  RefreshCw
} from 'lucide-react'
import type { Recommendation, WeatherData } from '../types'

export function RecommendationsPage() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [weatherData, setWeatherData] = useState<WeatherData[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Simulate loading data
    setTimeout(() => {
      setRecommendations([
        {
          id: '1',
          farm_id: '1',
          type: 'fertilizer',
          title: 'Apply Nitrogen Fertilizer',
          description: 'Your corn crop shows signs of nitrogen deficiency. Apply 50 lbs of nitrogen per acre.',
          priority: 'high',
          created_at: '2024-01-15'
        },
        {
          id: '2',
          farm_id: '1',
          type: 'irrigation',
          title: 'Increase Irrigation Frequency',
          description: 'Soil moisture levels are below optimal. Water every 3 days instead of weekly.',
          priority: 'medium',
          created_at: '2024-01-14'
        },
        {
          id: '3',
          farm_id: '2',
          type: 'harvest',
          title: 'Prepare for Harvest',
          description: 'Soybean crop is ready for harvest. Optimal timing is within the next 5-7 days.',
          priority: 'high',
          created_at: '2024-01-13'
        }
      ])
      setWeatherData([
        {
          temperature: 22,
          humidity: 65,
          rainfall: 0,
          wind_speed: 8,
          date: '2024-01-15'
        },
        {
          temperature: 24,
          humidity: 70,
          rainfall: 5,
          wind_speed: 12,
          date: '2024-01-16'
        }
      ])
      setLoading(false)
    }, 1000)
  }, [])

  const getPriorityGradient = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'from-red-600 to-rose-400'
      case 'medium':
        return 'from-yellow-600 to-yellow-400'
      case 'low':
        return 'from-green-600 to-lime-400'
      default:
        return 'from-gray-600 to-gray-400'
    }
  }

  const getPriorityBg = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'bg-red-100 text-red-800'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800'
      case 'low':
        return 'bg-green-100 text-green-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'fertilizer':
        return TrendingUp
      case 'irrigation':
        return Droplets
      case 'harvest':
        return Calendar
      case 'planting':
        return Sun
      default:
        return Lightbulb
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-pink-600"></div>
      </div>
    )
  }

  return (
    <>
      {/* Recommendations List */}
      <div className="w-full max-w-full px-3 mb-6 lg:w-8/12 lg:flex-none">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border mb-6">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <div className="flex items-center justify-between">
              <div>
                <h6 className="mb-0 font-bold">Active Recommendations</h6>
                <p className="leading-normal text-sm text-gray-600">{recommendations.length} AI-powered insights</p>
              </div>
              <button className="inline-block px-4 py-2 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-purple-700 to-pink-500 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85">
                <RefreshCw className="w-3 h-3 inline mr-1" />
                Refresh
              </button>
            </div>
          </div>
          <div className="flex-auto p-4 space-y-4">
            {recommendations.map((rec) => {
              const Icon = getTypeIcon(rec.type)
              return (
                <div
                  key={rec.id}
                  className="relative flex flex-col min-w-0 break-words bg-gray-50 border-0 shadow-soft-xs rounded-xl bg-clip-border hover:shadow-soft-lg transition-all"
                >
                  <div className="flex-auto p-4">
                    <div className="flex items-start space-x-4">
                      <div className={`w-12 h-12 bg-gradient-to-tl ${getPriorityGradient(rec.priority)} rounded-lg flex items-center justify-center flex-shrink-0 shadow-soft-md`}>
                        <Icon className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-2">
                          <h6 className="mb-0 font-semibold text-gray-900">{rec.title}</h6>
                          <span className={`inline-block px-2.5 py-1 text-xs font-semibold rounded-lg ${getPriorityBg(rec.priority)}`}>
                            {rec.priority}
                          </span>
                        </div>
                        <p className="mb-3 leading-normal text-sm text-gray-700">{rec.description}</p>
                        <div className="flex items-center space-x-4 text-xs text-gray-500">
                          <span className="capitalize">{rec.type}</span>
                          <span>•</span>
                          <span>{new Date(rec.created_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Action Button */}
        <button className="inline-block w-full px-6 py-3 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85">
          <Lightbulb className="w-4 h-4 inline mr-2" />
          Get New Recommendations
        </button>
      </div>

      {/* Sidebar */}
      <div className="w-full max-w-full px-3 lg:w-4/12 lg:flex-none">
        {/* Weather Forecast Card */}
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border mb-6">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <h6 className="mb-0 font-bold">Weather Forecast</h6>
            <p className="leading-normal text-sm text-gray-600">Next 7 days</p>
          </div>
          <div className="flex-auto p-4">
            <div className="space-y-3">
              {weatherData.map((weather, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-tl from-blue-600 to-cyan-400 rounded-lg flex items-center justify-center">
                      {weather.rainfall > 0 ? (
                        <Cloud className="w-5 h-5 text-white" />
                      ) : (
                        <Sun className="w-5 h-5 text-white" />
                      )}
                    </div>
                    <div>
                      <p className="text-xs font-semibold text-gray-900">
                        {new Date(weather.date).toLocaleDateString('en-US', {
                          weekday: 'short',
                          month: 'short',
                          day: 'numeric'
                        })}
                      </p>
                      <p className="text-xs text-gray-500">{weather.humidity}% humidity</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-bold text-gray-900">{weather.temperature}°C</p>
                    <p className="text-xs text-gray-500">
                      <Wind className="w-3 h-3 inline mr-1" />
                      {weather.wind_speed} km/h
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Priority Stats Card */}
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border mb-6">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl">
            <h6 className="mb-0 font-bold">Priority Stats</h6>
            <p className="leading-normal text-sm text-gray-600">Task breakdown</p>
          </div>
          <div className="flex-auto p-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-red-600 rounded-full"></div>
                  <span className="text-sm font-medium text-gray-700">High Priority</span>
                </div>
                <span className="text-lg font-bold text-red-600">
                  {recommendations.filter((r) => r.priority === 'high').length}
                </span>
              </div>

              <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-yellow-600 rounded-full"></div>
                  <span className="text-sm font-medium text-gray-700">Medium Priority</span>
                </div>
                <span className="text-lg font-bold text-yellow-600">
                  {recommendations.filter((r) => r.priority === 'medium').length}
                </span>
              </div>

              <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-2 h-2 bg-green-600 rounded-full"></div>
                  <span className="text-sm font-medium text-gray-700">Low Priority</span>
                </div>
                <span className="text-lg font-bold text-green-600">
                  {recommendations.filter((r) => r.priority === 'low').length}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions Card */}
        <div className="relative flex flex-col min-w-0 break-words bg-gradient-to-tl from-purple-700 to-pink-500 border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="flex-auto p-4">
            <div className="text-white">
              <h6 className="mb-2 text-white font-semibold">Need Help?</h6>
              <p className="mb-4 text-xs font-normal leading-tight text-white opacity-80">
                Mark all recommendations as completed or get expert advice
              </p>
              <button className="inline-block w-full px-6 py-2.5 mb-0 font-bold text-center uppercase align-middle transition-all bg-white rounded-lg cursor-pointer leading-normal text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 text-purple-700">
                <CheckCircle className="w-3 h-3 inline mr-2" />
                Mark All Complete
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
