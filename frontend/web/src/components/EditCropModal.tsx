import { useState, useEffect } from 'react'
import { X, Loader, Sprout, Sparkles } from 'lucide-react'
import { cropService } from '../services/api'
import type { Crop } from '../types'

interface EditCropModalProps {
  crop: Crop
  onClose: () => void
  onUpdate: () => void
}

// Crop growth durations (in days)
const CROP_GROWTH_DURATION: Record<string, number> = {
  'maize': 120,
  'corn': 120,
  'rice': 150,
  'cassava': 300,
  'tomato': 90,
  'soybean': 100,
  'groundnut': 120,
  'beans': 90,
  'cowpea': 75,
  'pepper': 90,
  'okra': 60,
  'onion': 100,
  'cabbage': 80,
  'lettuce': 45,
  'carrot': 75,
  'potato': 90,
  'sweet potato': 120,
  'yam': 240,
  'plantain': 365,
  'banana': 365,
  'pineapple': 480,
  'mango': 365,
  'orange': 365,
  'cocoa': 365,
  'coffee': 365,
  'cotton': 150,
  'sugarcane': 365
}

export default function EditCropModal({ crop, onClose, onUpdate }: EditCropModalProps) {
  const [formData, setFormData] = useState({
    crop_type: crop.crop_type,
    variety: crop.variety || '',
    quantity: crop.quantity?.toString() || '',
    quantity_unit: crop.quantity_unit || 'acres',
    planting_date: crop.planting_date ? crop.planting_date.split('T')[0] : '',
    expected_harvest_date: crop.expected_harvest_date ? crop.expected_harvest_date.split('T')[0] : '',
    growth_stage: crop.growth_stage || '',
    health_status: crop.health_status || 'healthy',
    expected_yield: crop.expected_yield?.toString() || '',
    yield_unit: crop.yield_unit || 'kg',
    notes: crop.notes || '',
    irrigation_method: crop.irrigation_method || ''
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [isHarvestDateAuto, setIsHarvestDateAuto] = useState(false)

  // Auto-calculate harvest date when planting date or crop type changes
  useEffect(() => {
    if (formData.planting_date && formData.crop_type) {
      calculateHarvestDate()
    }
  }, [formData.planting_date, formData.crop_type])

  const calculateHarvestDate = () => {
    if (!formData.planting_date || !formData.crop_type) return

    const cropType = formData.crop_type.toLowerCase()
    const growthDuration = CROP_GROWTH_DURATION[cropType] || 90 // Default 90 days

    const plantDate = new Date(formData.planting_date)
    const harvestDate = new Date(plantDate)
    harvestDate.setDate(harvestDate.getDate() + growthDuration)

    const harvestDateString = harvestDate.toISOString().split('T')[0]

    setFormData(prev => ({
      ...prev,
      expected_harvest_date: harvestDateString
    }))
    setIsHarvestDateAuto(true)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target

    // If user manually changes harvest date, disable auto-calculation
    if (name === 'expected_harvest_date') {
      setIsHarvestDateAuto(false)
    }

    setFormData({
      ...formData,
      [name]: value
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      await cropService.updateCrop(crop.id, {
        crop_type: formData.crop_type,
        variety: formData.variety || undefined,
        quantity: formData.quantity ? parseFloat(formData.quantity) : undefined,
        quantity_unit: formData.quantity_unit || undefined,
        planting_date: formData.planting_date || undefined,
        expected_harvest_date: formData.expected_harvest_date || undefined,
        growth_stage: formData.growth_stage || undefined,
        health_status: formData.health_status || undefined,
        expected_yield: formData.expected_yield ? parseFloat(formData.expected_yield) : undefined,
        yield_unit: formData.yield_unit || undefined,
        notes: formData.notes || undefined,
        irrigation_method: formData.irrigation_method || undefined
      })

      onUpdate()
      onClose()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update crop')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1000] p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6 rounded-t-2xl sticky top-0 z-10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-white/20 p-2 rounded-lg">
                <Sprout className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-xl font-bold">Edit Crop</h3>
                <p className="text-blue-100 text-sm">{crop.crop_type}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-white/80 hover:text-white transition"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Crop Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Crop Type <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                name="crop_type"
                value={formData.crop_type}
                onChange={handleChange}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Variety */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Variety
              </label>
              <input
                type="text"
                name="variety"
                value={formData.variety}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Quantity */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Area
              </label>
              <div className="flex gap-2">
                <input
                  type="number"
                  step="0.01"
                  name="quantity"
                  value={formData.quantity}
                  onChange={handleChange}
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <select
                  name="quantity_unit"
                  value={formData.quantity_unit}
                  onChange={handleChange}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="acres">Acres</option>
                  <option value="hectares">Hectares</option>
                </select>
              </div>
            </div>

            {/* Growth Stage */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Growth Stage
              </label>
              <select
                name="growth_stage"
                value={formData.growth_stage}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select stage</option>
                <option value="seedling">Seedling</option>
                <option value="vegetative">Vegetative</option>
                <option value="flowering">Flowering</option>
                <option value="fruiting">Fruiting</option>
                <option value="mature">Mature</option>
              </select>
            </div>

            {/* Health Status */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Health Status
              </label>
              <select
                name="health_status"
                value={formData.health_status}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="healthy">Healthy</option>
                <option value="stressed">Stressed</option>
                <option value="diseased">Diseased</option>
              </select>
            </div>

            {/* Irrigation Method */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Irrigation Method
              </label>
              <select
                name="irrigation_method"
                value={formData.irrigation_method}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select method</option>
                <option value="drip">Drip Irrigation</option>
                <option value="sprinkler">Sprinkler</option>
                <option value="flood">Flood/Furrow</option>
                <option value="rain_fed">Rain-fed</option>
              </select>
            </div>

            {/* Planting Date */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Planting Date
              </label>
              <input
                type="date"
                name="planting_date"
                value={formData.planting_date}
                onChange={handleChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Expected Harvest Date */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1 flex items-center gap-2">
                Expected Harvest Date
                {isHarvestDateAuto && (
                  <span className="flex items-center gap-1 text-xs text-green-600 font-normal">
                    <Sparkles className="w-3 h-3" />
                    Auto-calculated
                  </span>
                )}
              </label>
              <input
                type="date"
                name="expected_harvest_date"
                value={formData.expected_harvest_date}
                onChange={handleChange}
                className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                  isHarvestDateAuto ? 'border-green-300 bg-green-50' : 'border-gray-300'
                }`}
              />
              {isHarvestDateAuto && (
                <p className="text-xs text-green-600 mt-1">
                  Based on typical growth duration for {formData.crop_type}
                </p>
              )}
            </div>

            {/* Expected Yield */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Expected Yield
              </label>
              <div className="flex gap-2">
                <input
                  type="number"
                  step="0.01"
                  name="expected_yield"
                  value={formData.expected_yield}
                  onChange={handleChange}
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <select
                  name="yield_unit"
                  value={formData.yield_unit}
                  onChange={handleChange}
                  className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="kg">kg</option>
                  <option value="tons">Tons</option>
                  <option value="bags">Bags</option>
                </select>
              </div>
            </div>
          </div>

          {/* Notes */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Notes
            </label>
            <textarea
              name="notes"
              rows={3}
              value={formData.notes}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              placeholder="Add any notes about this crop..."
            />
          </div>

          {/* Footer */}
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              disabled={loading}
              className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  Updating...
                </>
              ) : (
                'Update Crop'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
