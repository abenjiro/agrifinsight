import { useState, useEffect } from 'react'
import { Sparkles, TrendingUp, Droplet, Calendar, ChevronDown, ChevronUp, Loader } from 'lucide-react'
import { cropRecommendationService } from '../services/api'
import type { CropRecommendation } from '../types'

interface CropRecommendationsProps {
  farmId: number
}

// Pure helper functions - moved outside component for better performance
const getSuitabilityColor = (score: number) => {
  if (score >= 75) return 'text-green-600 bg-green-50'
  if (score >= 50) return 'text-yellow-600 bg-yellow-50'
  return 'text-orange-600 bg-orange-50'
}

const getDifficultyBadge = (difficulty?: string) => {
  const colors = {
    easy: 'bg-green-100 text-green-800',
    moderate: 'bg-yellow-100 text-yellow-800',
    difficult: 'bg-red-100 text-red-800'
  }
  return colors[difficulty as keyof typeof colors] || 'bg-gray-100 text-gray-800'
}

export default function CropRecommendations({ farmId }: CropRecommendationsProps) {
  const [recommendations, setRecommendations] = useState<CropRecommendation[]>([])
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const [generating, setGenerating] = useState(false)
  const [generateError, setGenerateError] = useState('')

  // Derive the error to display from both error states
  const error = loadError || generateError

  useEffect(() => {
    loadRecommendations()
  }, [farmId])

  const loadRecommendations = async () => {
    setLoading(true)
    setLoadError('')
    try {
      const data = await cropRecommendationService.getRecommendations(farmId)
      setRecommendations(data)
    } catch (err: any) {
      console.error('Failed to load recommendations:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateRecommendations = async () => {
    setGenerating(true)
    setGenerateError('')
    try {
      const data = await cropRecommendationService.generateRecommendations(farmId)
      setRecommendations(data)
    } catch (err: any) {
      setGenerateError(err.response?.data?.detail || 'Failed to generate recommendations')
    } finally {
      setGenerating(false)
    }
  }

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6 flex items-center justify-center">
        <Loader className="h-6 w-6 animate-spin text-green-600" />
        <span className="ml-2 text-gray-600">Loading recommendations...</span>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Compact Header */}
      <div className="px-4 py-3 border-b border-gray-200 bg-gradient-to-r from-green-50 to-blue-50">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-green-600" />
            <h3 className="text-sm font-semibold text-gray-900">Crop Recommendations</h3>
            {recommendations.length > 0 && (
              <span className="text-xs text-gray-600">({recommendations.length} crops)</span>
            )}
          </div>
          <button
            onClick={handleGenerateRecommendations}
            disabled={generating}
            className="px-3 py-1.5 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-400 flex items-center gap-1.5 text-xs"
          >
            {generating ? (
              <>
                <Loader className="h-3 w-3 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="h-3 w-3" />
                Generate
              </>
            )}
          </button>
        </div>
        <p className="text-xs text-gray-600">AI-powered suggestions for the best crops for your farm</p>
      </div>

      {error && (
        <div className="mx-4 mt-3 bg-red-50 border border-red-200 text-red-700 px-3 py-2 rounded text-xs">
          {error}
        </div>
      )}

      <div className="p-4">
        {recommendations.length === 0 ? (
          <div className="text-center py-6">
            <Sparkles className="h-8 w-8 text-gray-400 mx-auto mb-2" />
            <p className="text-sm text-gray-600">
              Click Generate to get AI crop recommendations
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {recommendations.map((rec) => (
              <div key={rec.id} className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow">
                {/* Compact Summary */}
                <div
                  className="p-3 flex items-center justify-between cursor-pointer hover:bg-gray-50"
                  onClick={() => setExpandedId(expandedId === rec.id ? null : rec.id)}
                >
                  <div className="flex items-center gap-3 flex-1">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-semibold text-gray-900 text-sm">{rec.recommended_crop}</h4>
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getSuitabilityColor(rec.suitability_score)}`}>
                          {rec.suitability_score.toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-gray-600">
                        <span className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          {rec.growth_duration_days}d
                        </span>
                        <span className="flex items-center gap-1">
                          <TrendingUp className="h-3 w-3" />
                          {rec.estimated_profit_margin}%
                        </span>
                        <span className="flex items-center gap-1 capitalize">
                          <Droplet className="h-3 w-3" />
                          {rec.water_requirements}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button className="text-gray-400 hover:text-gray-600">
                    {expandedId === rec.id ? (
                      <ChevronUp className="h-4 w-4" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </button>
                </div>

                {/* Expanded Details */}
                {expandedId === rec.id && (
                  <div className="px-3 py-3 border-t border-gray-200 bg-gray-50 space-y-3 text-xs">
                    <div className="grid grid-cols-2 gap-3">
                      {/* Planting Season */}
                      {rec.planting_season && (
                        <div>
                          <span className="font-medium text-gray-700">Season: </span>
                          <span className="text-gray-600">{rec.planting_season}</span>
                        </div>
                      )}

                      {/* Expected Yield */}
                      {rec.expected_yield_range && (
                        <div>
                          <span className="font-medium text-gray-700">Yield: </span>
                          <span className="text-gray-600">
                            {rec.expected_yield_range.min.toLocaleString()} - {rec.expected_yield_range.max.toLocaleString()} {rec.expected_yield_range.unit}
                          </span>
                        </div>
                      )}

                      {/* Market Demand */}
                      <div>
                        <span className="font-medium text-gray-700">Demand: </span>
                        <span className="text-gray-600 capitalize">{rec.market_demand}</span>
                      </div>

                      {/* Care Difficulty */}
                      {rec.care_difficulty && (
                        <div>
                          <span className="font-medium text-gray-700">Care: </span>
                          <span className={`capitalize ${getDifficultyBadge(rec.care_difficulty)}`}>
                            {rec.care_difficulty}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Benefits */}
                    {rec.benefits && rec.benefits.length > 0 && (
                      <div>
                        <div className="font-medium text-gray-700 mb-1">✓ Benefits</div>
                        <ul className="text-gray-600 space-y-0.5 ml-4">
                          {rec.benefits.slice(0, 2).map((benefit, idx) => (
                            <li key={idx}>• {benefit}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Top Tips */}
                    {rec.tips && rec.tips.length > 0 && (
                      <div>
                        <div className="font-medium text-gray-700 mb-1">💡 Tips</div>
                        <ul className="text-gray-600 space-y-0.5 ml-4">
                          {rec.tips.slice(0, 2).map((tip, idx) => (
                            <li key={idx}>• {tip}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Alternative Crops */}
                    {rec.alternative_crops && rec.alternative_crops.length > 0 && (
                      <div>
                        <span className="font-medium text-gray-700">Alternatives: </span>
                        <span className="text-gray-600">{rec.alternative_crops.join(', ')}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
