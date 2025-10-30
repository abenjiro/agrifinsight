import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import { cropService } from '../services/api'
import type { CropCreate } from '../types'

interface AddCropModalProps {
  isOpen: boolean
  onClose: () => void
  farmId: number
  onSuccess: () => void
}

const COMMON_CROP_TYPES = [
  'Maize',
  'Rice',
  'Cassava',
  'Tomato',
  'Soybean',
  'Groundnut',
  'Yam',
  'Plantain',
  'Cocoa',
  'Coffee',
  'Pepper',
  'Onion',
  'Cabbage',
  'Carrot',
  'Beans',
  'Peas',
  'Okra',
  'Garden Egg',
  'Cucumber',
  'Watermelon',
  'Pineapple',
  'Mango',
  'Orange',
  'Banana',
  'Coconut',
  'Palm Oil',
  'Rubber',
  'Cashew',
  'Cotton',
  'Wheat',
  'Barley',
  'Sorghum',
  'Millet',
  'Sweet Potato',
  'Potato',
  'Ginger',
  'Garlic',
  'Sugarcane',
  'Tea',
  'Tobacco'
]

const IRRIGATION_METHODS = [
  'rain-fed',
  'drip',
  'sprinkler',
  'flood',
  'furrow',
  'manual'
]

const GROWTH_STAGES = [
  'seedling',
  'vegetative',
  'flowering',
  'fruiting',
  'mature'
]

// Crop intelligence database for predictions
const CROP_DATA: Record<string, {
  growthDurationDays: number
  waterRequirement: 'low' | 'medium' | 'high'
  recommendedIrrigation: string
  avgYield: { min: number, max: number, unit: string }
}> = {
  'Maize': { growthDurationDays: 120, waterRequirement: 'medium', recommendedIrrigation: 'rain-fed', avgYield: { min: 2000, max: 5000, unit: 'kg/acre' } },
  'Rice': { growthDurationDays: 150, waterRequirement: 'high', recommendedIrrigation: 'flood', avgYield: { min: 3000, max: 6000, unit: 'kg/acre' } },
  'Cassava': { growthDurationDays: 300, waterRequirement: 'low', recommendedIrrigation: 'rain-fed', avgYield: { min: 8000, max: 15000, unit: 'kg/acre' } },
  'Tomato': { growthDurationDays: 90, waterRequirement: 'medium', recommendedIrrigation: 'drip', avgYield: { min: 5000, max: 12000, unit: 'kg/acre' } },
  'Soybean': { growthDurationDays: 100, waterRequirement: 'medium', recommendedIrrigation: 'rain-fed', avgYield: { min: 800, max: 1500, unit: 'kg/acre' } },
  'Groundnut': { growthDurationDays: 120, waterRequirement: 'medium', recommendedIrrigation: 'rain-fed', avgYield: { min: 1000, max: 2500, unit: 'kg/acre' } },
  'Yam': { growthDurationDays: 240, waterRequirement: 'medium', recommendedIrrigation: 'rain-fed', avgYield: { min: 6000, max: 12000, unit: 'kg/acre' } },
  'Plantain': { growthDurationDays: 365, waterRequirement: 'high', recommendedIrrigation: 'manual', avgYield: { min: 8000, max: 15000, unit: 'kg/acre' } },
  'Cocoa': { growthDurationDays: 1095, waterRequirement: 'high', recommendedIrrigation: 'rain-fed', avgYield: { min: 400, max: 1000, unit: 'kg/acre' } },
  'Pepper': { growthDurationDays: 90, waterRequirement: 'medium', recommendedIrrigation: 'drip', avgYield: { min: 2000, max: 5000, unit: 'kg/acre' } },
  'Onion': { growthDurationDays: 110, waterRequirement: 'medium', recommendedIrrigation: 'drip', avgYield: { min: 8000, max: 15000, unit: 'kg/acre' } },
  'Cabbage': { growthDurationDays: 75, waterRequirement: 'medium', recommendedIrrigation: 'drip', avgYield: { min: 10000, max: 20000, unit: 'kg/acre' } },
}

export default function AddCropModal({ isOpen, onClose, farmId, onSuccess }: AddCropModalProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [cropTypes, setCropTypes] = useState<string[]>([])
  const [filteredCrops, setFilteredCrops] = useState<string[]>([])
  const [formData, setFormData] = useState<CropCreate>({
    farm_id: farmId,
    crop_type: '',
    variety: '',
    quantity: undefined,
    quantity_unit: 'acres',
    planting_date: '',
    expected_harvest_date: '',
    growth_stage: 'seedling',
    health_status: 'healthy',
    expected_yield: undefined,
    yield_unit: 'kg',
    notes: '',
    irrigation_method: 'rain-fed',
  })

  // Fetch crop types from API when modal opens
  useEffect(() => {
    if (isOpen) {
      loadCropTypes()
    }
  }, [isOpen])

  const loadCropTypes = async () => {
    try {
      const types = await cropService.getCropTypes()
      setCropTypes(types)
    } catch (err) {
      console.error('Failed to load crop types:', err)
      // Fallback to common crops if API fails
      setCropTypes(COMMON_CROP_TYPES)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleCropTypeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setFormData(prev => ({
      ...prev,
      crop_type: value
    }))

    // Filter crops based on input
    if (value.trim()) {
      const filtered = cropTypes.filter(crop =>
        crop.toLowerCase().includes(value.toLowerCase())
      )
      setFilteredCrops(filtered)
      setShowSuggestions(true)
    } else {
      setFilteredCrops([])
      setShowSuggestions(false)
    }
  }

  const calculateHarvestDate = (plantingDate: string, cropType: string): string => {
    if (!plantingDate || !cropType) return ''

    const cropData = CROP_DATA[cropType]
    if (!cropData) return ''

    const planting = new Date(plantingDate)
    const harvest = new Date(planting)
    harvest.setDate(harvest.getDate() + cropData.growthDurationDays)

    return harvest.toISOString().split('T')[0]
  }

  const autoPredictCropData = (cropType: string, plantingDate?: string) => {
    const cropData = CROP_DATA[cropType]
    if (!cropData) return

    const updates: Partial<CropCreate> = {
      irrigation_method: cropData.recommendedIrrigation,
      yield_unit: cropData.avgYield.unit.split('/')[0], // Extract 'kg' from 'kg/acre'
      expected_yield: Math.round((cropData.avgYield.min + cropData.avgYield.max) / 2)
    }

    // Calculate harvest date if planting date exists
    if (plantingDate || formData.planting_date) {
      const harvestDate = calculateHarvestDate(
        plantingDate || formData.planting_date || '',
        cropType
      )
      if (harvestDate) {
        updates.expected_harvest_date = harvestDate
      }
    }

    setFormData(prev => ({
      ...prev,
      ...updates
    }))
  }

  const handleCropSelect = (crop: string) => {
    setFormData(prev => ({
      ...prev,
      crop_type: crop
    }))
    setShowSuggestions(false)

    // Auto-predict crop data
    autoPredictCropData(crop)
  }

  const handlePlantingDateChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const plantingDate = e.target.value
    setFormData(prev => ({
      ...prev,
      planting_date: plantingDate
    }))

    // Auto-calculate harvest date
    if (plantingDate && formData.crop_type) {
      const harvestDate = calculateHarvestDate(plantingDate, formData.crop_type)
      if (harvestDate) {
        setFormData(prev => ({
          ...prev,
          expected_harvest_date: harvestDate
        }))
      }
    }
  }

  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value ? parseFloat(value) : undefined
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      await cropService.createCrop(farmId, formData)
      onSuccess()
      onClose()
      // Reset form
      setFormData({
        farm_id: farmId,
        crop_type: '',
        variety: '',
        quantity: undefined,
        quantity_unit: 'acres',
        planting_date: '',
        expected_harvest_date: '',
        growth_stage: 'seedling',
        health_status: 'healthy',
        expected_yield: undefined,
        yield_unit: 'kg',
        notes: '',
        irrigation_method: 'rain-fed',
      })
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to add crop')
    } finally {
      setLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-[1000] overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        {/* Background overlay */}
        <div
          className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity"
          onClick={onClose}
        />

        {/* Modal panel */}
        <div className="relative inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-2xl sm:w-full z-[1001]">
          <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Add New Crop</h3>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-500"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            {/* Smart Predictions Banner */}
            <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-md">
              <p className="text-sm text-green-800">
                <span className="font-semibold">🤖 Smart Predictions:</span> Select a crop type and planting date - we'll automatically predict harvest date, yield, and irrigation needs!
              </p>
            </div>

            {error && (
              <div className="mb-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Crop Type */}
                <div className="relative">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Crop Type *
                  </label>
                  <input
                    type="text"
                    name="crop_type"
                    value={formData.crop_type}
                    onChange={handleCropTypeChange}
                    onFocus={() => {
                      if (formData.crop_type.trim()) {
                        const filtered = cropTypes.filter(crop =>
                          crop.toLowerCase().includes(formData.crop_type.toLowerCase())
                        )
                        setFilteredCrops(filtered)
                        setShowSuggestions(true)
                      }
                    }}
                    onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                    required
                    placeholder="Type to search crops..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                  {/* Suggestions Dropdown */}
                  {showSuggestions && filteredCrops.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
                      {filteredCrops.map((crop) => (
                        <div
                          key={crop}
                          onClick={() => handleCropSelect(crop)}
                          className="px-3 py-2 hover:bg-green-50 cursor-pointer text-sm"
                        >
                          {crop}
                        </div>
                      ))}
                    </div>
                  )}
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
                    placeholder="e.g., Hybrid DKC-8081"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>

                {/* Quantity */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Quantity
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      name="quantity"
                      value={formData.quantity || ''}
                      onChange={handleNumberChange}
                      step="0.1"
                      placeholder="5"
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                    />
                    <select
                      name="quantity_unit"
                      value={formData.quantity_unit}
                      onChange={handleChange}
                      className="w-24 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                    >
                      <option value="acres">acres</option>
                      <option value="hectares">hectares</option>
                      <option value="kg">kg</option>
                      <option value="bags">bags</option>
                    </select>
                  </div>
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
                    onChange={handlePlantingDateChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>

                {/* Expected Harvest Date */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Expected Harvest Date
                    {formData.expected_harvest_date && formData.crop_type && CROP_DATA[formData.crop_type] && (
                      <span className="ml-2 text-xs text-green-600">✓ Auto-calculated</span>
                    )}
                  </label>
                  <input
                    type="date"
                    name="expected_harvest_date"
                    value={formData.expected_harvest_date}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 bg-green-50"
                    readOnly={!!formData.crop_type && !!CROP_DATA[formData.crop_type]}
                  />
                  {formData.crop_type && CROP_DATA[formData.crop_type] && (
                    <p className="text-xs text-gray-500 mt-1">
                      Based on {CROP_DATA[formData.crop_type].growthDurationDays} days growth period
                    </p>
                  )}
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  >
                    {GROWTH_STAGES.map(stage => (
                      <option key={stage} value={stage}>
                        {stage.charAt(0).toUpperCase() + stage.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Expected Yield */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Expected Yield
                    {formData.expected_yield && formData.crop_type && CROP_DATA[formData.crop_type] && (
                      <span className="ml-2 text-xs text-green-600">✓ Auto-predicted</span>
                    )}
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      name="expected_yield"
                      value={formData.expected_yield || ''}
                      onChange={handleNumberChange}
                      step="0.1"
                      placeholder="4000"
                      className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 bg-green-50"
                    />
                    <select
                      name="yield_unit"
                      value={formData.yield_unit}
                      onChange={handleChange}
                      className="w-24 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                    >
                      <option value="kg">kg</option>
                      <option value="tons">tons</option>
                      <option value="bags">bags</option>
                    </select>
                  </div>
                </div>

                {/* Irrigation Method */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Irrigation Method
                    {formData.irrigation_method && formData.crop_type && CROP_DATA[formData.crop_type] && (
                      <span className="ml-2 text-xs text-green-600">✓ Recommended</span>
                    )}
                  </label>
                  <select
                    name="irrigation_method"
                    value={formData.irrigation_method}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 bg-green-50"
                  >
                    {IRRIGATION_METHODS.map(method => (
                      <option key={method} value={method}>
                        {method.charAt(0).toUpperCase() + method.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Notes */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Notes
                </label>
                <textarea
                  name="notes"
                  value={formData.notes}
                  onChange={handleChange}
                  rows={3}
                  placeholder="Additional information about this crop..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                />
              </div>

              {/* Actions */}
              <div className="flex gap-3 justify-end pt-4">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-400"
                >
                  {loading ? 'Adding...' : 'Add Crop'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}
