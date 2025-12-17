import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Sprout, Calendar, TrendingUp, AlertCircle, CheckCircle2, Clock,
  Thermometer, Droplets, Cloud, ArrowLeft, Loader, Info, ChevronRight,
  Beaker, Plus, X
} from 'lucide-react'
import { showError, showSuccess } from '../utils/sweetalert'
import { cropService } from '../services/api'

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
  const [apiErrors, setApiErrors] = useState<any[]>([])

  // Quick Add Crop Modal State
  const [showAddCropModal, setShowAddCropModal] = useState(false)
  const [addingCrop, setAddingCrop] = useState(false)
  const [cropToAdd, setCropToAdd] = useState<CropComparison | null>(null)
  const [quickAddForm, setQuickAddForm] = useState({
    quantity: '',
    quantity_unit: 'acres',
    notes: ''
  })

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
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch recommendations')
      }

      const data = await response.json()
      console.log('Planting recommendations data:', data)
      setFarmName(data.farm_name)

      // Store any errors from the API
      if (data.comparison?.errors) {
        setApiErrors(data.comparison.errors)
        console.error('API Errors:', data.comparison.errors)
      }

      // Check if comparison data exists and has the right structure
      if (data.comparison && data.comparison.comparison) {
        setComparison(data.comparison.comparison)
      } else if (data.comparison && Array.isArray(data.comparison)) {
        setComparison(data.comparison)
      } else {
        console.warn('Unexpected data structure:', data)
        setComparison([])
      }
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

  const openQuickAddCropModal = (crop: CropComparison) => {
    setCropToAdd(crop)
    setQuickAddForm({
      quantity: '',
      quantity_unit: 'acres',
      notes: `Recommended planting date: ${new Date(crop.planting_date).toLocaleDateString()}`
    })
    setShowAddCropModal(true)
  }

  const handleQuickAddCrop = async () => {
    if (!cropToAdd || !farmId) return

    if (!quickAddForm.quantity || parseFloat(quickAddForm.quantity) <= 0) {
      showError('Please enter a valid area')
      return
    }

    setAddingCrop(true)
    try {
      await cropService.createCrop(parseInt(farmId), {
        farm_id: parseInt(farmId),
        crop_type: cropToAdd.crop,
        planting_date: cropToAdd.planting_date,
        expected_harvest_date: cropToAdd.harvest_date,
        quantity: parseFloat(quickAddForm.quantity),
        quantity_unit: quickAddForm.quantity_unit,
        notes: quickAddForm.notes || undefined
      })

      showSuccess(`${cropToAdd.crop} has been added to your farm!`)
      setShowAddCropModal(false)
      setCropToAdd(null)
      setQuickAddForm({ quantity: '', quantity_unit: 'acres', notes: '' })

      // Navigate to farm detail page after a brief delay
      setTimeout(() => {
        navigate(`/farms/${farmId}`)
      }, 1500)
    } catch (error: any) {
      console.error('Error adding crop:', error)
      showError(error.response?.data?.detail || 'Failed to add crop')
    } finally {
      setAddingCrop(false)
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
                Planting Calendar
              </h1>
              <p className="text-gray-600 mt-1">{farmName}</p>
              <p className="text-sm text-gray-500 mt-1">When to plant different crops based on current conditions</p>
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

      {/* Info Notice */}
      <div className="bg-blue-50 border border-blue-200 rounded-2xl p-4 mb-6">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-blue-900">
            <p className="font-medium mb-1">Personalized Recommendations</p>
            <p className="text-blue-700">
              These recommendations are tailored to your farm's specific conditions including location (GPS),
              climate (temperature & rainfall), soil type, and current weather patterns.
              <span className="font-medium"> Update your farm's soil data for even more accurate recommendations.</span>
            </p>
          </div>
        </div>
      </div>

      {/* Crop Comparison View */}
      {view === 'comparison' && (
        <div className="bg-white rounded-2xl shadow-soft-xl p-6">
          <h2 className="text-lg font-bold text-gray-900 mb-4">Compare Crops for Your Farm</h2>

          {comparison.length === 0 && !loading ? (
            <div className="text-center py-12">
              <Calendar className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-900 mb-2">No Recommendations Available</h3>
              <p className="text-gray-600 mb-2">
                Unable to generate planting recommendations for this farm.
              </p>
              <p className="text-sm text-gray-500 max-w-md mx-auto mb-4">
                Make sure your farm has GPS coordinates (latitude/longitude) set.
                Check the browser console for more details about the error.
              </p>

              {apiErrors.length > 0 && (
                <div className="mt-6 max-w-2xl mx-auto">
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-left">
                    <h4 className="font-semibold text-red-800 mb-2">Error Details:</h4>
                    <ul className="space-y-2 text-sm text-red-700">
                      {apiErrors.map((err, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <span className="font-medium">{err.crop}:</span>
                          <span className="flex-1">{err.error}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {comparison.map((crop) => (
              <div
                key={crop.crop}
                className="border border-gray-200 rounded-xl p-5 hover:shadow-lg hover:border-green-300 transition group"
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

                <p className="text-xs text-gray-600 line-clamp-2 mb-4">
                  {crop.summary}
                </p>

                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      openQuickAddCropModal(crop)
                    }}
                    className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition text-sm font-medium"
                  >
                    <Plus className="w-4 h-4" />
                    Add to Farm
                  </button>
                  <button
                    onClick={() => fetchDetailedRecommendation(crop.crop)}
                    className="px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-gray-700 transition"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
            </div>
          )}
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

            {/* Quick Add Button in Detailed View */}
            <div className="mt-4 pt-4 border-t border-white/20">
              <button
                onClick={() => {
                  const cropComparison = comparison.find(c => c.crop === selectedCrop)
                  if (cropComparison) {
                    openQuickAddCropModal(cropComparison)
                  }
                }}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-white text-green-700 rounded-lg hover:bg-green-50 transition font-semibold"
              >
                <Plus className="w-5 h-5" />
                Add {selectedCrop} to Farm
              </button>
            </div>
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

      {/* Quick Add Crop Modal */}
      {showAddCropModal && cropToAdd && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1000] p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full">
            {/* Header */}
            <div className="bg-gradient-to-r from-green-600 to-emerald-600 text-white p-6 rounded-t-2xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="bg-white/20 p-2 rounded-lg">
                    <Sprout className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">Add {cropToAdd.crop}</h3>
                    <p className="text-green-100 text-sm">Quick add to your farm</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowAddCropModal(false)}
                  className="text-white/80 hover:text-white transition"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="p-6 space-y-4">
              {/* Pre-filled Information */}
              <div className="bg-green-50 border border-green-200 rounded-lg p-4 space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Recommended Planting:</span>
                  <span className="font-semibold text-gray-900">
                    {new Date(cropToAdd.planting_date).toLocaleDateString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Expected Harvest:</span>
                  <span className="font-semibold text-gray-900">
                    {new Date(cropToAdd.harvest_date).toLocaleDateString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Suitability Score:</span>
                  <span className="font-semibold text-green-600">
                    {cropToAdd.suitability_score.toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Area Input (Required) */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Area to Plant <span className="text-red-500">*</span>
                </label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    placeholder="0.0"
                    value={quickAddForm.quantity}
                    onChange={(e) => setQuickAddForm({ ...quickAddForm, quantity: e.target.value })}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  />
                  <select
                    value={quickAddForm.quantity_unit}
                    onChange={(e) => setQuickAddForm({ ...quickAddForm, quantity_unit: e.target.value })}
                    className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  >
                    <option value="acres">Acres</option>
                    <option value="hectares">Hectares</option>
                  </select>
                </div>
              </div>

              {/* Notes (Optional) */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Notes (Optional)
                </label>
                <textarea
                  rows={3}
                  placeholder="Add any notes about this crop..."
                  value={quickAddForm.notes}
                  onChange={(e) => setQuickAddForm({ ...quickAddForm, notes: e.target.value })}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none"
                />
              </div>
            </div>

            {/* Footer */}
            <div className="bg-gray-50 px-6 py-4 rounded-b-2xl flex gap-3">
              <button
                onClick={() => setShowAddCropModal(false)}
                disabled={addingCrop}
                className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-100 transition disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleQuickAddCrop}
                disabled={addingCrop || !quickAddForm.quantity}
                className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {addingCrop ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Adding...
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4" />
                    Add Crop
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
