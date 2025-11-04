import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Sprout, Calendar, TrendingUp, AlertCircle, CheckCircle2, Clock,
  Thermometer, Droplets, Cloud, ArrowLeft, Loader, Info, ChevronRight,
  MapPin, Beaker
} from 'lucide-react'
import { showError } from '../utils/sweetalert'

interface PlantingRecommendation {
  crop_type: string
  overall_recommendation: string
  suitability_score: number
  confidence: string
  summary: string
  planting_window: {
    recommended_date: string
    earliest_date: string
    latest_date: string
    reason: string
  }
  weather_analysis: {
    recommendation: string
    confidence: string
    reason: string
    conditions: {
      avg_temperature: number
      total_rainfall_7days: number
      rainy_days: number
    }
    strengths: string[]
    concerns: string[]
  }
  seasonal_analysis: {
    status: string
    message: string
    current_month: number
    primary_months: number[]
    secondary_months: number[]
  }
  soil_analysis: {
    suitability: string
    message: string
    strengths: string[]
    concerns: string[]
  }
  estimated_harvest_date: string
  preparation_checklist: string[]
}

interface CropComparison {
  crop: string
  suitability_score: number
  recommendation: string
  planting_date: string
  harvest_date: string
  summary: string
}

export function PlantingRecommendationsPage() {
  const { farmId } = useParams<{ farmId: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [selectedCrop, setSelectedCrop] = useState<string | null>(null)
  const [recommendation, setRecommendation] = useState<PlantingRecommendation | null>(null)
  const [comparison, setComparison] = useState<CropComparison[]>([])
  const [farmName, setFarmName] = useState<string>('')
  const [view, setView] = useState<'comparison' | 'detailed'>('comparison')

  useEffect(() => {
    if (farmId) {
      fetchComparison()
    }
  }, [farmId])

  const fetchComparison = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch(
        `http://localhost:8000/api/recommendations/planting/${farmId}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations')
      }

      const data = await response.json()
      setFarmName(data.farm_name)
      setComparison(data.comparison.comparison)
    } catch (error: any) {
      console.error('Error fetching recommendations:', error)
      showError(error.message || 'Failed to load recommendations')
    } finally {
      setLoading(false)
    }
  }

  const fetchDetailedRecommendation = async (cropType: string) => {
    setLoading(true)
    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch(
        `http://localhost:8000/api/recommendations/planting/${farmId}?crop_type=${cropType}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to fetch detailed recommendation')
      }

      const data = await response.json()
      setRecommendation(data.recommendation)
      setSelectedCrop(cropType)
      setView('detailed')
    } catch (error: any) {
      console.error('Error fetching detailed recommendation:', error)
      showError(error.message || 'Failed to load detailed recommendation')
    } finally {
      setLoading(false)
    }
  }

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case 'highly_recommended': return 'bg-green-100 text-green-800 border-green-200'
      case 'recommended': return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'not_recommended': return 'bg-red-100 text-red-800 border-red-200'
      default: return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const getRecommendationIcon = (rec: string) => {
    switch (rec) {
      case 'highly_recommended': return <CheckCircle2 className="w-5 h-5" />
      case 'recommended': return <Info className="w-5 h-5" />
      case 'not_recommended': return <AlertCircle className="w-5 h-5" />
      default: return <Clock className="w-5 h-5" />
    }
  }

  const getRecommendationText = (rec: string) => {
    switch (rec) {
      case 'highly_recommended': return 'Highly Recommended'
      case 'recommended': return 'Recommended'
      case 'not_recommended': return 'Not Recommended'
      default: return 'Review Needed'
    }
  }

  if (loading && !recommendation && comparison.length === 0) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader className="w-8 h-8 animate-spin text-green-600" />
        <span className="ml-3 text-gray-600">Loading recommendations...</span>
      </div>
    )
  }

  return (
    <div className="w-full max-w-full px-3">
      {/* Header */}
      <div className="mb-6">
        <button
          onClick={() => navigate('/farms')}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Farms
        </button>

        <div className="bg-white rounded-2xl shadow-soft-xl p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
                <Sprout className="w-8 h-8 text-green-600" />
                Planting Recommendations
              </h1>
              <p className="text-gray-600 mt-1">{farmName}</p>
            </div>

            {view === 'detailed' && (
              <button
                onClick={() => {
                  setView('comparison')
                  setRecommendation(null)
                  setSelectedCrop(null)
                }}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium transition"
              >
                View All Crops
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Crop Comparison View */}
      {view === 'comparison' && (
        <div className="bg-white rounded-2xl shadow-soft-xl p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">Compare Crops for Your Farm</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {comparison.map((crop) => (
              <div
                key={crop.crop}
                onClick={() => fetchDetailedRecommendation(crop.crop)}
                className="border border-gray-200 rounded-xl p-5 hover:shadow-lg hover:border-green-300 transition cursor-pointer group"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-bold text-gray-900 text-lg">{crop.crop}</h3>
                    <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium mt-2 border ${getRecommendationColor(crop.recommendation)}`}>
                      {getRecommendationIcon(crop.recommendation)}
                      {getRecommendationText(crop.recommendation)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-green-600">{crop.suitability_score.toFixed(0)}</div>
                    <div className="text-xs text-gray-500">Suitability</div>
                  </div>
                </div>

                <div className="space-y-2 text-sm text-gray-600 mb-4">
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    <span>Plant: {new Date(crop.planting_date).toLocaleDateString()}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-gray-400" />
                    <span>Harvest: {new Date(crop.harvest_date).toLocaleDateString()}</span>
                  </div>
                </div>

                <p className="text-xs text-gray-600 line-clamp-2 mb-3">
                  {crop.summary}
                </p>

                <div className="flex items-center justify-end text-green-600 text-sm font-medium group-hover:gap-2 transition-all">
                  View Details
                  <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detailed View */}
      {view === 'detailed' && recommendation && (
        <div className="space-y-6">
          {/* Overall Score Card */}
          <div className="bg-gradient-to-br from-green-500 to-emerald-600 text-white rounded-2xl shadow-soft-xl p-6">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-3xl font-bold mb-2">{selectedCrop}</h2>
                <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium bg-white/20`}>
                  {getRecommendationIcon(recommendation.overall_recommendation)}
                  {getRecommendationText(recommendation.overall_recommendation)}
                </div>
              </div>
              <div className="text-right">
                <div className="text-5xl font-bold">{recommendation.suitability_score.toFixed(1)}</div>
                <div className="text-green-100 text-sm">Suitability Score</div>
                <div className="mt-1 text-xs text-green-200">Confidence: {recommendation.confidence}</div>
              </div>
            </div>

            <p className="mt-4 text-green-50 text-sm leading-relaxed">
              {recommendation.summary}
            </p>
          </div>

          {/* Planting Window */}
          <div className="bg-white rounded-2xl shadow-soft-xl p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Calendar className="w-5 h-5 text-green-600" />
              Planting Window
            </h3>

            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <div className="text-xs text-blue-600 font-medium mb-1">Earliest</div>
                <div className="text-lg font-bold text-gray-900">
                  {new Date(recommendation.planting_window.earliest_date).toLocaleDateString()}
                </div>
              </div>
              <div className="bg-green-50 rounded-lg p-4 text-center border-2 border-green-200">
                <div className="text-xs text-green-600 font-medium mb-1">Recommended</div>
                <div className="text-lg font-bold text-gray-900">
                  {new Date(recommendation.planting_window.recommended_date).toLocaleDateString()}
                </div>
              </div>
              <div className="bg-amber-50 rounded-lg p-4 text-center">
                <div className="text-xs text-amber-600 font-medium mb-1">Latest</div>
                <div className="text-lg font-bold text-gray-900">
                  {new Date(recommendation.planting_window.latest_date).toLocaleDateString()}
                </div>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm text-gray-700">
                <strong>Reason:</strong> {recommendation.planting_window.reason}
              </p>
            </div>

            <div className="mt-4 flex items-center gap-2 text-sm text-gray-600">
              <TrendingUp className="w-4 h-4 text-green-600" />
              <span>Expected Harvest: <strong>{new Date(recommendation.estimated_harvest_date).toLocaleDateString()}</strong></span>
            </div>
          </div>

          {/* Weather Analysis */}
          <div className="bg-white rounded-2xl shadow-soft-xl p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Cloud className="w-5 h-5 text-blue-600" />
              Weather Analysis
            </h3>

            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-lg p-4">
                <div className="flex items-center gap-2 text-orange-600 mb-1">
                  <Thermometer className="w-4 h-4" />
                  <span className="text-xs font-medium">Temperature</span>
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {recommendation.weather_analysis.conditions.avg_temperature.toFixed(1)}°C
                </div>
                <div className="text-xs text-gray-600">7-day average</div>
              </div>

              <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-4">
                <div className="flex items-center gap-2 text-blue-600 mb-1">
                  <Droplets className="w-4 h-4" />
                  <span className="text-xs font-medium">Rainfall</span>
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {recommendation.weather_analysis.conditions.total_rainfall_7days.toFixed(0)}mm
                </div>
                <div className="text-xs text-gray-600">Expected in 7 days</div>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-indigo-50 rounded-lg p-4">
                <div className="flex items-center gap-2 text-purple-600 mb-1">
                  <Cloud className="w-4 h-4" />
                  <span className="text-xs font-medium">Rainy Days</span>
                </div>
                <div className="text-2xl font-bold text-gray-900">
                  {recommendation.weather_analysis.conditions.rainy_days}
                </div>
                <div className="text-xs text-gray-600">In next 7 days</div>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <p className="text-sm text-gray-700">
                <strong>Assessment:</strong> {recommendation.weather_analysis.reason}
              </p>
            </div>

            {recommendation.weather_analysis.strengths.length > 0 && (
              <div className="mb-3">
                <h4 className="text-sm font-semibold text-gray-900 mb-2">Favorable Conditions:</h4>
                <ul className="space-y-1">
                  {recommendation.weather_analysis.strengths.map((strength, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-green-700">
                      <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      {strength}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {recommendation.weather_analysis.concerns.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-gray-900 mb-2">Concerns:</h4>
                <ul className="space-y-1">
                  {recommendation.weather_analysis.concerns.map((concern, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-amber-700">
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      {concern}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Seasonal & Soil Analysis */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Seasonal */}
            <div className="bg-white rounded-2xl shadow-soft-xl p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-purple-600" />
                Seasonal Analysis
              </h3>

              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-4 ${
                recommendation.seasonal_analysis.status === 'optimal' ? 'bg-green-100 text-green-800' :
                recommendation.seasonal_analysis.status === 'good' ? 'bg-blue-100 text-blue-800' :
                'bg-amber-100 text-amber-800'
              }`}>
                {recommendation.seasonal_analysis.status === 'optimal' ? '🌟 Optimal Season' :
                 recommendation.seasonal_analysis.status === 'good' ? '✓ Good Season' :
                 '⏰ Off Season'}
              </div>

              <p className="text-sm text-gray-700 mb-4">{recommendation.seasonal_analysis.message}</p>

              <div className="bg-gray-50 rounded-lg p-3">
                <div className="text-xs text-gray-600 mb-2">Planting Months:</div>
                <div className="flex flex-wrap gap-2">
                  {recommendation.seasonal_analysis.primary_months.map((month) => (
                    <span key={month} className="px-2 py-1 bg-green-100 text-green-700 rounded text-xs font-medium">
                      Month {month}
                    </span>
                  ))}
                  {recommendation.seasonal_analysis.secondary_months.map((month) => (
                    <span key={month} className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                      Month {month}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Soil */}
            <div className="bg-white rounded-2xl shadow-soft-xl p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Beaker className="w-5 h-5 text-amber-600" />
                Soil Analysis
              </h3>

              <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium mb-4 ${
                recommendation.soil_analysis.suitability === 'high' ? 'bg-green-100 text-green-800' :
                recommendation.soil_analysis.suitability === 'medium' ? 'bg-blue-100 text-blue-800' :
                'bg-amber-100 text-amber-800'
              }`}>
                {recommendation.soil_analysis.suitability === 'high' ? '✓ Highly Suitable' :
                 recommendation.soil_analysis.suitability === 'medium' ? '~ Acceptable' :
                 '⚠ Needs Amendment'}
              </div>

              <p className="text-sm text-gray-700 mb-4">{recommendation.soil_analysis.message}</p>

              {recommendation.soil_analysis.strengths.length > 0 && (
                <ul className="space-y-1 mb-3">
                  {recommendation.soil_analysis.strengths.map((strength, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-green-700">
                      <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      {strength}
                    </li>
                  ))}
                </ul>
              )}

              {recommendation.soil_analysis.concerns.length > 0 && (
                <ul className="space-y-1">
                  {recommendation.soil_analysis.concerns.map((concern, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm text-amber-700">
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      {concern}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {/* Preparation Checklist */}
          <div className="bg-white rounded-2xl shadow-soft-xl p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-green-600" />
              Preparation Checklist
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {recommendation.preparation_checklist.map((item, index) => (
                <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition">
                  <div className="mt-0.5 w-5 h-5 rounded border-2 border-gray-300 flex-shrink-0"></div>
                  <span className="text-sm text-gray-700">{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
