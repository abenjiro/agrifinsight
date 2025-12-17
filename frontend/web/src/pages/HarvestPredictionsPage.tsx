import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Sprout, Calendar, TrendingUp, AlertCircle, CheckCircle2, Clock,
  ArrowLeft, Loader, Droplets, Thermometer, Cloud, Gauge,
  Target, Award, ClipboardList, Info, AlertTriangle,
  Package, Truck, Briefcase, Sun, CloudRain
} from 'lucide-react'
import { showError, showSuccess } from '../utils/sweetalert'

interface HarvestPrediction {
  crop_type: string
  crop_age_days: number
  current_growth_stage: string
  harvest_readiness: {
    status: string
    message: string
    readiness_percentage: number
    urgency?: string
    days_until_harvest?: number
  }
  maturity_timeline: {
    estimated_harvest_date: string
    earliest_harvest_date: string
    latest_harvest_date: string
    days_until_maturity: number
    harvest_window_days?: number
    is_overdue?: boolean
  }
  yield_prediction: {
    predicted_yield_per_acre: number
    yield_range: {
      minimum: number
      expected: number
      maximum: number
    }
    confidence: string
    unit?: string
    total_farm_yield?: number
    farm_size_acres?: number
    yield_factors?: string[]
  }
  weather_forecast?: {
    current?: {
      temperature: number
      humidity: number
      rainfall: number
      conditions?: string
    }
    forecast?: Array<{
      date: string
      temperature_max: number
      temperature_min: number
      rainfall: number
      conditions: string
    }>
  }
  maturity_indicators: string[]
  harvest_recommendations: {
    recommendations: string[]
    warnings: string[]
    optimal_timing: string | null
    weather_considerations: string[]
  }
  post_harvest_care: string[]
}

export function HarvestPredictionsPage() {
  const { cropId } = useParams<{ cropId: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [prediction, setPrediction] = useState<HarvestPrediction | null>(null)
  const [cropName, setCropName] = useState<string>('')
  const [farmName, setFarmName] = useState<string>('')

  useEffect(() => {
    if (cropId) {
      fetchHarvestPrediction()
    }
  }, [cropId])

  const fetchHarvestPrediction = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch(
        `http://localhost:8000/api/recommendations/harvest/${cropId}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch harvest predictions')
      }

      const data = await response.json()
      console.log('Harvest prediction data:', data)

      setPrediction(data.prediction || data)
      setCropName(data.crop_name || data.prediction?.crop_type || 'Crop')
      setFarmName(data.farm_name || 'Farm')
    } catch (error: any) {
      console.error('Error fetching harvest predictions:', error)
      showError(error.message || 'Failed to load harvest predictions')
    } finally {
      setLoading(false)
    }
  }

  const getReadinessColor = (status: string) => {
    const statusMap: Record<string, string> = {
      'ready': 'text-green-600 bg-green-50 border-green-200',
      'almost_ready': 'text-yellow-600 bg-yellow-50 border-yellow-200',
      'developing': 'text-blue-600 bg-blue-50 border-blue-200',
      'early': 'text-gray-600 bg-gray-50 border-gray-200',
      'overdue': 'text-red-600 bg-red-50 border-red-200'
    }
    return statusMap[status] || 'text-gray-600 bg-gray-50 border-gray-200'
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <Loader className="w-12 h-12 animate-spin text-green-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading harvest predictions...</p>
        </div>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-600 mx-auto mb-4" />
          <p className="text-gray-600">No harvest prediction data available</p>
          <button
            onClick={() => navigate(-1)}
            className="mt-4 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
          >
            Go Back
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full">
      {/* Header */}
      <div className="w-full px-3 mb-6">
          <button
            onClick={() => navigate(-1)}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-4"
          >
            <ArrowLeft className="w-5 h-5" />
            Back
          </button>

          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <Sprout className="w-8 h-8 text-green-600" />
                Harvest Predictions
              </h1>
              <p className="text-gray-600 mt-1">
                {cropName} • {farmName}
              </p>
            </div>

            <div className="text-right">
              <div className="text-sm text-gray-500">
                {prediction.crop_age_days < 0 ? 'Planting In' : 'Crop Age'}
              </div>
              <div className="text-2xl font-bold text-green-600">
                {prediction.crop_age_days < 0
                  ? `${Math.abs(prediction.crop_age_days)} days`
                  : `${prediction.crop_age_days} days`
                }
              </div>
            </div>
          </div>
        </div>

      {/* Harvest Readiness Status */}
      <div className="w-full px-3 mb-6">
        <div className={`p-6 rounded-xl border-2 ${getReadinessColor(prediction.harvest_readiness.status)}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-white rounded-full">
                {prediction.harvest_readiness.status === 'ready' ? (
                  <CheckCircle2 className="w-8 h-8 text-green-600" />
                ) : prediction.harvest_readiness.status === 'almost_ready' ? (
                  <Clock className="w-8 h-8 text-yellow-600" />
                ) : (
                  <Sprout className="w-8 h-8 text-blue-600" />
                )}
              </div>
              <div>
                <div className="text-sm font-medium opacity-75">Harvest Readiness</div>
                <div className="text-2xl font-bold capitalize">
                  {prediction.harvest_readiness.status.replace('_', ' ')}
                </div>
                <p className="text-sm mt-1">{prediction.harvest_readiness.message}</p>
              </div>
            </div>

            <div className="text-center">
              {prediction.harvest_readiness.readiness_percentage < 0 ? (
                <>
                  <div className="text-3xl font-bold">
                    Not Planted
                  </div>
                  <div className="text-xs opacity-75 mt-1">
                    {Math.abs(prediction.harvest_readiness.readiness_percentage)}% before planting
                  </div>
                </>
              ) : (
                <>
                  <div className="text-4xl font-bold">
                    {prediction.harvest_readiness.readiness_percentage}%
                  </div>
                  <div className="text-sm opacity-75">Ready</div>
                </>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-4 bg-white bg-opacity-50 rounded-full h-3 overflow-hidden">
            <div
              className="h-full bg-current transition-all duration-500"
              style={{
                width: `${Math.max(0, Math.min(100, prediction.harvest_readiness.readiness_percentage))}%`
              }}
            />
          </div>
        </div>
      </div>

      {/* Maturity and Yield Grid */}
      <div className="w-full px-3 mb-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Maturity Timeline */}
          <div className="lg:col-span-2 bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Calendar className="w-6 h-6 text-green-600" />
              Maturity Timeline
            </h2>

            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg border border-green-200">
                <div>
                  <div className="text-sm text-gray-600">Estimated Harvest Date</div>
                  <div className="text-lg font-bold text-green-700">
                    {formatDate(prediction.maturity_timeline.estimated_harvest_date)}
                  </div>
                  <div className="text-sm text-gray-500 mt-1">
                    {prediction.maturity_timeline.days_until_maturity < 0
                      ? `Overdue by ${Math.abs(prediction.maturity_timeline.days_until_maturity)} days`
                      : `In ${prediction.maturity_timeline.days_until_maturity} days`
                    }
                  </div>
                </div>
                <Target className="w-8 h-8 text-green-600" />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="text-sm text-gray-600">Earliest Date</div>
                  <div className="text-md font-semibold text-gray-700">
                    {formatDate(prediction.maturity_timeline.earliest_harvest_date)}
                  </div>
                </div>

                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="text-sm text-gray-600">Latest Date</div>
                  <div className="text-md font-semibold text-gray-700">
                    {formatDate(prediction.maturity_timeline.latest_harvest_date)}
                  </div>
                </div>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                <div className="text-sm font-medium text-blue-700 mb-2">Current Growth Stage</div>
                <div className="text-lg font-bold text-blue-900 capitalize">
                  {prediction.current_growth_stage.replace('_', ' ')}
                </div>
              </div>
            </div>
          </div>

          {/* Yield Prediction */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-green-600" />
              Yield Prediction
            </h2>

            <div className="text-center mb-4">
              <div className="text-4xl font-bold text-green-600">
                {prediction.yield_prediction.predicted_yield_per_acre.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">kg per acre</div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">Minimum</span>
                <span className="font-semibold text-gray-700">
                  {prediction.yield_prediction.yield_range.minimum.toLocaleString()} kg
                </span>
              </div>

              <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">Maximum</span>
                <span className="font-semibold text-gray-700">
                  {prediction.yield_prediction.yield_range.maximum.toLocaleString()} kg
                </span>
              </div>

              <div className="p-3 bg-blue-50 rounded-lg border border-blue-200 text-center">
                <div className="text-xs text-blue-600">Confidence</div>
                <div className="text-sm font-bold text-blue-700 capitalize">
                  {prediction.yield_prediction.confidence}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Weather Forecast */}
      {prediction.weather_forecast?.current && (
        <div className="w-full px-3 mb-6">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Cloud className="w-6 h-6 text-blue-600" />
              Current Weather Conditions
            </h2>

            <div className="grid grid-cols-3 gap-3">
              <div className="p-4 bg-orange-50 rounded-lg text-center">
                <Thermometer className="w-6 h-6 text-orange-600 mx-auto mb-2" />
                <div className="text-xs text-gray-600">Temperature</div>
                <div className="text-2xl font-bold text-orange-700">
                  {prediction.weather_forecast.current.temperature}°C
                </div>
              </div>

              <div className="p-4 bg-blue-50 rounded-lg text-center">
                <Droplets className="w-6 h-6 text-blue-600 mx-auto mb-2" />
                <div className="text-xs text-gray-600">Humidity</div>
                <div className="text-2xl font-bold text-blue-700">
                  {prediction.weather_forecast.current.humidity}%
                </div>
              </div>

              <div className="p-4 bg-cyan-50 rounded-lg text-center">
                <CloudRain className="w-6 h-6 text-cyan-600 mx-auto mb-2" />
                <div className="text-xs text-gray-600">Rainfall</div>
                <div className="text-2xl font-bold text-cyan-700">
                  {prediction.weather_forecast.current.rainfall}mm
                </div>
              </div>
            </div>

            {prediction.weather_forecast.current.conditions && (
              <div className="mt-4 p-3 bg-gray-50 rounded-lg text-center">
                <div className="text-sm text-gray-600">Conditions</div>
                <div className="text-lg font-semibold text-gray-800 capitalize">
                  {prediction.weather_forecast.current.conditions}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Maturity Indicators */}
      <div className="w-full px-3 mb-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <ClipboardList className="w-6 h-6 text-green-600" />
            Maturity Indicators
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {prediction.maturity_indicators.map((indicator, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition">
                <CheckCircle2 className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                <span className="text-sm text-gray-700">{indicator}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Harvest Recommendations */}
      <div className="w-full px-3 mb-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <Info className="w-6 h-6 text-blue-600" />
            Harvest Recommendations
          </h2>

          {/* Check if there's any content to display */}
          {!prediction.harvest_recommendations.optimal_timing &&
           prediction.harvest_recommendations.warnings.length === 0 &&
           prediction.harvest_recommendations.recommendations.length === 0 &&
           prediction.harvest_recommendations.weather_considerations.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Info className="w-12 h-12 mx-auto mb-3 text-gray-400" />
              <p>No specific recommendations available at this time.</p>
              <p className="text-sm mt-2">Check back as your crop matures for harvest timing advice.</p>
            </div>
          ) : (
            <div className="space-y-4">
            {/* Optimal Timing */}
            {prediction.harvest_recommendations.optimal_timing && (
              <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 className="w-5 h-5 text-green-600" />
                  <span className="font-semibold text-green-800">Optimal Timing</span>
                </div>
                <p className="text-sm text-green-700 ml-7">{prediction.harvest_recommendations.optimal_timing}</p>
              </div>
            )}

            {/* Warnings */}
            {prediction.harvest_recommendations.warnings.length > 0 && (
              <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-600" />
                  <span className="font-semibold text-yellow-800">Important Warnings</span>
                </div>
                <ul className="space-y-1 ml-7">
                  {prediction.harvest_recommendations.warnings.map((warning, index) => (
                    <li key={index} className="text-sm text-yellow-800">• {warning}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Recommendations */}
            {prediction.harvest_recommendations.recommendations.length > 0 && (
              <div className="space-y-2">
                <div className="font-semibold text-gray-700">Action Items:</div>
                {prediction.harvest_recommendations.recommendations.map((rec, index) => (
                  <div key={index} className="flex items-start gap-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      {index + 1}
                    </div>
                    <span className="text-sm text-gray-700">{rec}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Weather Considerations */}
            {prediction.harvest_recommendations.weather_considerations.length > 0 && (
              <div className="p-4 bg-cyan-50 rounded-lg border border-cyan-200">
                <div className="flex items-center gap-2 mb-2">
                  <Cloud className="w-5 h-5 text-cyan-600" />
                  <span className="font-semibold text-cyan-800">Weather Considerations</span>
                </div>
                <ul className="space-y-1 ml-7">
                  {prediction.harvest_recommendations.weather_considerations.map((consideration, index) => (
                    <li key={index} className="text-sm text-cyan-800">• {consideration}</li>
                  ))}
                </ul>
              </div>
            )}
            </div>
          )}
        </div>
      </div>

      {/* Post-Harvest Care */}
      <div className="w-full px-3 mb-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <Package className="w-6 h-6 text-purple-600" />
            Post-Harvest Care
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {prediction.post_harvest_care.map((care, index) => (
              <div key={index} className="p-4 bg-purple-50 rounded-lg border border-purple-200 hover:shadow-md transition">
                <div className="flex items-start gap-2">
                  <Briefcase className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-gray-700">{care}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
