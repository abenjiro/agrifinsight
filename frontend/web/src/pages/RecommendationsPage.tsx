import { useState, useEffect } from 'react'
import { 
  Lightbulb, 
  TrendingUp, 
  CheckCircle, 
  Calendar,
  Droplets,
  Sun
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

  const getPriorityColor = (priority: string) => {
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
        return <TrendingUp className="w-5 h-5" />
      case 'irrigation':
        return <Droplets className="w-5 h-5" />
      case 'harvest':
        return <Calendar className="w-5 h-5" />
      case 'planting':
        return <Sun className="w-5 h-5" />
      default:
        return <Lightbulb className="w-5 h-5" />
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Recommendations</h1>
        <p className="text-gray-600">AI-powered farming recommendations based on your data</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recommendations List */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">Active Recommendations</h2>
            <span className="text-sm text-gray-500">{recommendations.length} items</span>
          </div>

          {recommendations.map((rec) => (
            <div key={rec.id} className="card p-6">
              <div className="flex items-start space-x-4">
                <div className="flex-shrink-0">
                  <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center text-primary-600">
                    {getTypeIcon(rec.type)}
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-medium text-gray-900">{rec.title}</h3>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getPriorityColor(rec.priority)}`}>
                      {rec.priority}
                    </span>
                  </div>
                  <p className="text-gray-600 mb-4">{rec.description}</p>
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <span>Type: {rec.type}</span>
                    <span>•</span>
                    <span>{new Date(rec.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Weather & Quick Stats */}
        <div className="space-y-6">
          {/* Weather Card */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Weather Forecast</h3>
            <div className="space-y-3">
              {weatherData.map((weather, index) => (
                <div key={index} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-b-0">
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      {new Date(weather.date).toLocaleDateString()}
                    </p>
                    <p className="text-xs text-gray-500">
                      {weather.temperature}°C • {weather.humidity}% humidity
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">
                      {weather.rainfall}mm rain
                    </p>
                    <p className="text-xs text-gray-500">
                      {weather.wind_speed} km/h wind
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Stats</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">High Priority</span>
                <span className="text-lg font-semibold text-red-600">
                  {recommendations.filter(r => r.priority === 'high').length}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Medium Priority</span>
                <span className="text-lg font-semibold text-yellow-600">
                  {recommendations.filter(r => r.priority === 'medium').length}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Low Priority</span>
                <span className="text-lg font-semibold text-green-600">
                  {recommendations.filter(r => r.priority === 'low').length}
                </span>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="space-y-3">
            <button className="btn btn-primary w-full">
              <CheckCircle className="w-4 h-4 mr-2" />
              Mark All as Read
            </button>
            <button className="btn btn-outline w-full">
              <Lightbulb className="w-4 h-4 mr-2" />
              Get New Recommendations
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

