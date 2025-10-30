import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Plus, MapPin, Ruler, Sprout, Edit2, Trash2, X, Navigation, Cloud, Droplets, Thermometer, Map, ChevronRight, ChevronLeft, Check, Eye } from 'lucide-react'
import { showSuccess, showError, showConfirm } from '../utils/sweetalert'
import { farmService } from '../services/api'

interface Farm {
  id: number
  name: string
  address?: string
  latitude?: number
  longitude?: number
  altitude?: number
  size?: number
  size_unit?: string
  soil_type?: string
  soil_ph?: number
  terrain_type?: string
  climate_zone?: string
  avg_annual_rainfall?: number
  avg_temperature?: number
  country?: string
  region?: string
  district?: string
  timezone?: string
  created_at: string
  updated_at?: string
}

export function FarmsPage() {
  const navigate = useNavigate()
  const [farms, setFarms] = useState<Farm[]>([])
  const [loading, setLoading] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [editingFarm, setEditingFarm] = useState<Farm | null>(null)
  const [enriching, setEnriching] = useState(false)
  const [currentStep, setCurrentStep] = useState(1)

  const [formData, setFormData] = useState({
    name: '',
    address: '',
    latitude: '',
    longitude: '',
    altitude: '',
    size: '',
    size_unit: 'acres',
    soil_type: '',
    soil_ph: '',
    terrain_type: '',
    climate_zone: '',
    avg_annual_rainfall: '',
    avg_temperature: '',
    country: '',
    region: '',
    district: '',
    timezone: ''
  })

  const steps = [
    { number: 1, title: 'Basic Info', description: 'Farm details' },
    { number: 2, title: 'Location', description: 'GPS & Address' },
    { number: 3, title: 'Review', description: 'Confirm details' }
  ]

  useEffect(() => {
    fetchFarms()
  }, [])

  const fetchFarms = async () => {
    setLoading(true)
    try {
      const response : any = await farmService.getFarms()
      console.log(response)
      setFarms(response.data || response)
    } catch (error) {
      console.error('Error fetching farms:', error)
      showError('Failed to load farms. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const getCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          setFormData(prev => ({
            ...prev,
            latitude: position.coords.latitude.toString(),
            longitude: position.coords.longitude.toString(),
            altitude: position.coords.altitude?.toString() || ''
          }))

          // Automatically enrich location data
          await enrichLocationData(position.coords.latitude, position.coords.longitude)
        },
        (error) => {
          showError('Unable to get your location: ' + error.message, 'Location Error')
        }
      )
    } else {
      showError('Geolocation is not supported by your browser', 'Not Supported')
    }
  }

  const enrichLocationData = async (lat?: number, lon?: number) => {
    const latitude = lat || parseFloat(formData.latitude)
    const longitude = lon || parseFloat(formData.longitude)

    if (!latitude || !longitude) {
      return
    }

    setEnriching(true)
    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch(`http://localhost:8000/api/farms/enrich-location?latitude=${latitude}&longitude=${longitude}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error('Failed to enrich location')
      }

      const data = await response.json()

      // Update form with ALL enriched data including environment
      setFormData(prev => ({
        ...prev,
        address: data.address || prev.address,
        altitude: data.altitude?.toString() || prev.altitude,
        country: data.country || prev.country,
        region: data.region || prev.region,
        district: data.district || prev.district,
        timezone: data.timezone || prev.timezone,
        climate_zone: data.climate_zone || prev.climate_zone,
        avg_temperature: data.avg_temperature?.toString() || data.current_weather?.temperature?.toFixed(1) || prev.avg_temperature,
        avg_annual_rainfall: data.avg_annual_rainfall?.toString() || prev.avg_annual_rainfall,
        soil_ph: data.soil_data?.composition?.phh2o ? (data.soil_data.composition.phh2o / 10).toFixed(1) : prev.soil_ph
      }))

    } catch (error: any) {
      console.error('Error enriching location:', error)
      showError(error.message || 'Failed to auto-capture environment data. You can still continue.', 'Enrichment Failed')
    } finally {
      setEnriching(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!formData.name.trim()) {
      showError('Please enter a farm name', 'Validation Error')
      return
    }

    try {
      const token = localStorage.getItem('auth_token')
      const payload = {
        name: formData.name,
        address: formData.address || null,
        latitude: formData.latitude ? parseFloat(formData.latitude) : null,
        longitude: formData.longitude ? parseFloat(formData.longitude) : null,
        altitude: formData.altitude ? parseFloat(formData.altitude) : null,
        size: formData.size ? parseFloat(formData.size) : null,
        size_unit: formData.size_unit || 'acres',
        soil_type: formData.soil_type || null,
        soil_ph: formData.soil_ph ? parseFloat(formData.soil_ph) : null,
        terrain_type: formData.terrain_type || null,
        climate_zone: formData.climate_zone || null,
        avg_annual_rainfall: formData.avg_annual_rainfall ? parseFloat(formData.avg_annual_rainfall) : null,
        avg_temperature: formData.avg_temperature ? parseFloat(formData.avg_temperature) : null,
        country: formData.country || null,
        region: formData.region || null,
        district: formData.district || null,
        timezone: formData.timezone || null
      }

      const url = editingFarm
        ? `http://localhost:8000/api/farms/${editingFarm.id}`
        : 'http://localhost:8000/api/farms/'

      const response = await fetch(url, {
        method: editingFarm ? 'PUT' : 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Operation failed' }))
        throw new Error(errorData.detail || 'Operation failed')
      }

      resetForm()
      fetchFarms()
      showSuccess(editingFarm ? 'Farm updated successfully!' : 'Farm created successfully!', 'Success')
    } catch (error: any) {
      console.error('Error saving farm:', error)
      showError(error.message || 'Failed to save farm. Please try again.', 'Save Failed')
    }
  }

  const handleEdit = (farm: Farm) => {
    setEditingFarm(farm)
    setFormData({
      name: farm.name,
      address: farm.address || '',
      latitude: farm.latitude?.toString() || '',
      longitude: farm.longitude?.toString() || '',
      altitude: farm.altitude?.toString() || '',
      size: farm.size?.toString() || '',
      size_unit: farm.size_unit || 'acres',
      soil_type: farm.soil_type || '',
      soil_ph: farm.soil_ph?.toString() || '',
      terrain_type: farm.terrain_type || '',
      climate_zone: farm.climate_zone || '',
      avg_annual_rainfall: farm.avg_annual_rainfall?.toString() || '',
      avg_temperature: farm.avg_temperature?.toString() || '',
      country: farm.country || '',
      region: farm.region || '',
      district: farm.district || '',
      timezone: farm.timezone || ''
    })
    setShowCreateModal(true)
  }

  const handleDelete = async (farmId: number) => {
    const result = await showConfirm(
      'This action cannot be undone. All data associated with this farm will be permanently deleted.',
      'Delete this farm?',
      'Yes, delete it',
      'Cancel'
    )

    if (!result.isConfirmed) {
      return
    }

    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch(`http://localhost:8000/api/farms/${farmId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (!response.ok) {
        throw new Error('Failed to delete farm')
      }

      fetchFarms()
      showSuccess('Farm deleted successfully!', 'Deleted')
    } catch (error) {
      console.error('Error deleting farm:', error)
      showError('Failed to delete farm. Please try again.', 'Delete Failed')
    }
  }

  const resetForm = () => {
    setFormData({
      name: '',
      address: '',
      latitude: '',
      longitude: '',
      altitude: '',
      size: '',
      size_unit: 'acres',
      soil_type: '',
      soil_ph: '',
      terrain_type: '',
      climate_zone: '',
      avg_annual_rainfall: '',
      avg_temperature: '',
      country: '',
      region: '',
      district: '',
      timezone: ''
    })
    setShowCreateModal(false)
    setEditingFarm(null)
    setCurrentStep(1)
  }

  const nextStep = () => {
    if (currentStep === 1 && !formData.name.trim()) {
      showError('Please enter a farm name', 'Validation Error')
      return
    }
    if (currentStep === 2 && (!formData.latitude || !formData.longitude)) {
      showError('Please add GPS coordinates or use your location', 'Validation Error')
      return
    }
    setCurrentStep(prev => Math.min(prev + 1, steps.length))
  }

  const prevStep = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1))
  }

  return (
    <>
      {/* Header Section */}
      <div className="w-full max-w-full px-3 mb-6">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-4 pb-0 mb-0 bg-white border-b-0 rounded-t-2xl flex justify-between items-center">
            <div>
              <h6 className="mb-0 font-bold">My Farms</h6>
              <p className="leading-normal text-sm">Manage your farms with geospatial data</p>
            </div>
            <button
              onClick={() => setShowCreateModal(true)}
              className="inline-block px-6 py-2.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
            >
              <Plus className="w-3 h-3 inline mr-2" />
              Add Farm
            </button>
          </div>
        </div>
      </div>

      {/* Farms Grid */}
      <div className="w-full max-w-full px-3">
        {loading ? (
          <div className="flex items-center justify-center py-16">
            <div className="text-center">
              <div className="inline-block w-12 h-12 text-center rounded-xl bg-gradient-to-tl from-green-600 to-lime-400 mb-4 animate-pulse">
                <Sprout className="w-6 h-6 text-white relative top-3 left-3" />
              </div>
              <p className="text-sm font-semibold text-gray-900">Loading farms...</p>
            </div>
          </div>
        ) : farms.length === 0 ? (
          <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
            <div className="flex-auto p-4">
              <div className="text-center py-12">
                <div className="inline-block w-16 h-16 text-center rounded-xl bg-gray-100 mb-4">
                  <Sprout className="w-8 h-8 text-gray-300 relative top-4 left-4" />
                </div>
                <p className="text-sm font-medium text-gray-500">No farms yet</p>
                <p className="text-xs text-gray-400 mt-1">Create your first farm to get started</p>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="mt-4 inline-block px-6 py-2.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85"
                >
                  <Plus className="w-3 h-3 inline mr-2" />
                  Add Your First Farm
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {farms.map((farm) => (
              <div
                key={farm.id}
                className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border hover:shadow-soft-2xl transition"
              >
                <div className="flex-auto p-4">
                  <div className="flex justify-between items-start mb-3">
                    <h5 className="font-bold text-gray-900 text-lg">{farm.name}</h5>
                    <div className="flex gap-2">
                      <button
                        onClick={() => navigate(`/farms/${farm.id}`)}
                        className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition"
                        title="View Details"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleEdit(farm)}
                        className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete(farm.id)}
                        className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    {farm.address && (
                      <div className="flex items-center text-sm text-gray-600">
                        <MapPin className="w-4 h-4 mr-2 text-gray-400 flex-shrink-0" />
                        <span className="truncate">{farm.address}</span>
                      </div>
                    )}
                    {farm.latitude && farm.longitude && (
                      <div className="flex items-center text-sm text-gray-600">
                        <Navigation className="w-4 h-4 mr-2 text-gray-400" />
                        <span>{farm.latitude.toFixed(4)}°, {farm.longitude.toFixed(4)}°</span>
                      </div>
                    )}
                    {farm.size && (
                      <div className="flex items-center text-sm text-gray-600">
                        <Ruler className="w-4 h-4 mr-2 text-gray-400" />
                        <span>{farm.size} {farm.size_unit || 'acres'}</span>
                      </div>
                    )}
                    {farm.climate_zone && (
                      <div className="flex items-center text-sm text-gray-600">
                        <Cloud className="w-4 h-4 mr-2 text-gray-400" />
                        <span>{farm.climate_zone}</span>
                      </div>
                    )}
                    {farm.avg_temperature && (
                      <div className="flex items-center text-sm text-gray-600">
                        <Thermometer className="w-4 h-4 mr-2 text-gray-400" />
                        <span>{farm.avg_temperature}°C avg</span>
                      </div>
                    )}
                    {farm.avg_annual_rainfall && (
                      <div className="flex items-center text-sm text-gray-600">
                        <Droplets className="w-4 h-4 mr-2 text-gray-400" />
                        <span>{farm.avg_annual_rainfall}mm rainfall/year</span>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 pt-4 border-t border-gray-100">
                    <p className="text-xs text-gray-500">
                      Created {new Date(farm.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Wizard Modal */}
      {showCreateModal && (
        <>
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-[1000]"
            onClick={resetForm}
          />
          <div className="fixed inset-0 z-[1001] flex items-center justify-center p-4 overflow-y-auto">
            <div className="relative w-full max-w-3xl bg-white rounded-2xl shadow-soft-2xl my-8 z-[1002]">
              <div className="p-6">
                {/* Header */}
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">
                      {editingFarm ? 'Edit Farm' : 'Create New Farm'}
                    </h3>
                    <p className="text-sm text-gray-600 mt-1">
                      {steps[currentStep - 1].description}
                    </p>
                  </div>
                  <button
                    onClick={resetForm}
                    className="p-2 text-gray-400 hover:text-gray-600 transition"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>

                {/* Progress Steps */}
                <div className="mb-8">
                  <div className="flex items-center justify-between">
                    {steps.map((step, index) => (
                      <div key={step.number} className="flex items-center flex-1">
                        <div className="flex flex-col items-center flex-1">
                          <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition ${
                            currentStep > step.number
                              ? 'bg-green-600 text-white'
                              : currentStep === step.number
                              ? 'bg-blue-600 text-white'
                              : 'bg-gray-200 text-gray-600'
                          }`}>
                            {currentStep > step.number ? <Check className="w-5 h-5" /> : step.number}
                          </div>
                          <p className={`text-xs mt-2 font-medium ${
                            currentStep >= step.number ? 'text-gray-900' : 'text-gray-500'
                          }`}>
                            {step.title}
                          </p>
                        </div>
                        {index < steps.length - 1 && (
                          <div className={`h-1 flex-1 mx-2 rounded ${
                            currentStep > step.number ? 'bg-green-600' : 'bg-gray-200'
                          }`} />
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                <form onSubmit={handleSubmit}>
                  {/* Step 1: Basic Info */}
                  {currentStep === 1 && (
                    <div className="space-y-4">
                      <div>
                        <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                          Farm Name *
                        </label>
                        <input
                          type="text"
                          id="name"
                          name="name"
                          required
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                          placeholder="Enter farm name"
                          value={formData.name}
                          onChange={handleInputChange}
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label htmlFor="size" className="block text-sm font-medium text-gray-700 mb-1">
                            Farm Size
                          </label>
                          <input
                            type="number"
                            id="size"
                            name="size"
                            step="0.01"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                            placeholder="Enter size"
                            value={formData.size}
                            onChange={handleInputChange}
                          />
                        </div>
                        <div>
                          <label htmlFor="size_unit" className="block text-sm font-medium text-gray-700 mb-1">
                            Unit
                          </label>
                          <select
                            id="size_unit"
                            name="size_unit"
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                            value={formData.size_unit}
                            onChange={handleInputChange}
                          >
                            <option value="acres">Acres</option>
                            <option value="hectares">Hectares</option>
                          </select>
                        </div>
                      </div>

                      <div>
                        <label htmlFor="soil_type" className="block text-sm font-medium text-gray-700 mb-1">
                          Soil Type (Optional)
                        </label>
                        <select
                          id="soil_type"
                          name="soil_type"
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                          value={formData.soil_type}
                          onChange={handleInputChange}
                        >
                          <option value="">Select soil type</option>
                          <option value="clay">Clay</option>
                          <option value="sandy">Sandy</option>
                          <option value="loam">Loam</option>
                          <option value="silt">Silt</option>
                          <option value="peat">Peat</option>
                          <option value="chalky">Chalky</option>
                        </select>
                      </div>
                    </div>
                  )}

                  {/* Step 2: Location - Auto-captures environment */}
                  {currentStep === 2 && (
                    <div className="space-y-4">
                      <div className="bg-gradient-to-r from-blue-50 to-green-50 border border-blue-200 rounded-lg p-4">
                        <div className="flex items-start gap-3">
                          <Map className="w-6 h-6 text-blue-600 mt-0.5 flex-shrink-0" />
                          <div className="flex-1">
                            <p className="text-sm font-bold text-blue-900 mb-2">
                              📍 Add Farm Location
                            </p>
                            <p className="text-xs text-blue-800 mb-3">
                              Click "Use My Location" to automatically capture GPS coordinates and environmental data (weather, climate, soil, elevation). Or enter coordinates manually.
                            </p>
                            <button
                              type="button"
                              onClick={getCurrentLocation}
                              disabled={enriching}
                              className="inline-block px-4 py-2 text-xs font-bold text-white bg-gradient-to-r from-blue-600 to-green-600 rounded-lg hover:from-blue-700 hover:to-green-700 transition disabled:opacity-50"
                            >
                              {enriching ? 'Capturing Data...' : '📍 Use My Location'}
                            </button>
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <label htmlFor="latitude" className="block text-sm font-medium text-gray-700 mb-1">
                            Latitude *
                          </label>
                          <input
                            type="number"
                            id="latitude"
                            name="latitude"
                            step="any"
                            required
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                            placeholder="e.g., 5.6037"
                            value={formData.latitude}
                            onChange={handleInputChange}
                            onBlur={() => {
                              if (formData.latitude && formData.longitude && !enriching) {
                                enrichLocationData()
                              }
                            }}
                          />
                        </div>
                        <div>
                          <label htmlFor="longitude" className="block text-sm font-medium text-gray-700 mb-1">
                            Longitude *
                          </label>
                          <input
                            type="number"
                            id="longitude"
                            name="longitude"
                            step="any"
                            required
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                            placeholder="e.g., -0.1870"
                            value={formData.longitude}
                            onChange={handleInputChange}
                            onBlur={() => {
                              if (formData.latitude && formData.longitude && !enriching) {
                                enrichLocationData()
                              }
                            }}
                          />
                        </div>
                        <div>
                          <label htmlFor="altitude" className="block text-sm font-medium text-gray-700 mb-1">
                            Altitude (m)
                          </label>
                          <input
                            type="number"
                            id="altitude"
                            name="altitude"
                            step="0.1"
                            disabled
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50"
                            placeholder="Auto"
                            value={formData.altitude}
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-1 gap-4">
                        <div>
                          <label htmlFor="address" className="block text-sm font-medium text-gray-700 mb-1">
                            Address
                          </label>
                          <input
                            type="text"
                            id="address"
                            name="address"
                            disabled
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50"
                            placeholder="Auto-captured from GPS"
                            value={formData.address}
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Country
                          </label>
                          <input
                            type="text"
                            disabled
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50"
                            value={formData.country}
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Region
                          </label>
                          <input
                            type="text"
                            disabled
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50"
                            value={formData.region}
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            District
                          </label>
                          <input
                            type="text"
                            disabled
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-50"
                            value={formData.district}
                          />
                        </div>
                      </div>

                      {enriching && (
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                          <div className="flex items-center text-blue-900">
                            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
                            <span className="text-sm font-medium">
                              Capturing environment data from GPS coordinates...
                            </span>
                          </div>
                        </div>
                      )}

                      {!enriching && formData.climate_zone && (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                          <p className="text-sm font-medium text-green-900 mb-3">
                            ✅ Environment Data Auto-Captured
                          </p>
                          <div className="grid grid-cols-2 gap-3 text-xs">
                            <div className="flex items-center text-green-800">
                              <Cloud className="w-4 h-4 mr-2 flex-shrink-0" />
                              <span>Climate: {formData.climate_zone}</span>
                            </div>
                            {formData.avg_temperature && (
                              <div className="flex items-center text-green-800">
                                <Thermometer className="w-4 h-4 mr-2 flex-shrink-0" />
                                <span>Temp: {formData.avg_temperature}°C</span>
                              </div>
                            )}
                            {formData.soil_ph && (
                              <div className="flex items-center text-green-800">
                                <Sprout className="w-4 h-4 mr-2 flex-shrink-0" />
                                <span>Soil pH: {formData.soil_ph}</span>
                              </div>
                            )}
                            {formData.timezone && (
                              <div className="flex items-center text-green-800">
                                <Navigation className="w-4 h-4 mr-2 flex-shrink-0" />
                                <span>Timezone: {formData.timezone}</span>
                              </div>
                            )}
                            {formData.country && (
                              <div className="flex items-center text-green-800">
                                <MapPin className="w-4 h-4 mr-2 flex-shrink-0" />
                                <span>{formData.country}</span>
                              </div>
                            )}
                            {formData.altitude && (
                              <div className="flex items-center text-green-800">
                                <Map className="w-4 h-4 mr-2 flex-shrink-0" />
                                <span>Elevation: {parseFloat(formData.altitude).toFixed(0)}m</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Step 3: Review */}
                  {currentStep === 3 && (
                    <div className="space-y-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h4 className="font-bold text-gray-900 mb-3">Review Farm Details</h4>

                        <div className="space-y-3">
                          <div>
                            <p className="text-xs text-gray-600">Farm Name</p>
                            <p className="text-sm font-semibold text-gray-900">{formData.name}</p>
                          </div>

                          {formData.size && (
                            <div>
                              <p className="text-xs text-gray-600">Size</p>
                              <p className="text-sm font-semibold text-gray-900">
                                {formData.size} {formData.size_unit}
                              </p>
                            </div>
                          )}

                          {formData.address && (
                            <div>
                              <p className="text-xs text-gray-600">Location</p>
                              <p className="text-sm font-semibold text-gray-900">{formData.address}</p>
                              <p className="text-xs text-gray-600 mt-1">
                                {formData.latitude}, {formData.longitude}
                              </p>
                            </div>
                          )}

                          {formData.climate_zone && (
                            <div className="border-t border-gray-200 pt-3">
                              <p className="text-xs text-gray-600 mb-2">Environmental Data</p>
                              <div className="grid grid-cols-2 gap-2 text-sm">
                                <div>
                                  <span className="text-gray-600">Climate:</span>
                                  <span className="ml-2 font-medium">{formData.climate_zone}</span>
                                </div>
                                {formData.avg_temperature && (
                                  <div>
                                    <span className="text-gray-600">Avg Temp:</span>
                                    <span className="ml-2 font-medium">{formData.avg_temperature}°C</span>
                                  </div>
                                )}
                                {formData.soil_type && (
                                  <div>
                                    <span className="text-gray-600">Soil:</span>
                                    <span className="ml-2 font-medium capitalize">{formData.soil_type}</span>
                                  </div>
                                )}
                                {formData.soil_ph && (
                                  <div>
                                    <span className="text-gray-600">Soil pH:</span>
                                    <span className="ml-2 font-medium">{formData.soil_ph}</span>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <p className="text-sm text-blue-900">
                          <strong>Ready to create?</strong> Your farm data will be saved with all geospatial information for accurate crop predictions and weather monitoring.
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Navigation Buttons */}
                  <div className="flex gap-3 pt-6 mt-6 border-t border-gray-200">
                    {currentStep > 1 && (
                      <button
                        type="button"
                        onClick={prevStep}
                        className="flex-1 px-6 py-2.5 font-bold text-center text-gray-700 uppercase align-middle transition-all bg-transparent border border-solid rounded-lg cursor-pointer border-gray-400 leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 flex items-center justify-center"
                      >
                        <ChevronLeft className="w-4 h-4 mr-2" />
                        Back
                      </button>
                    )}

                    {currentStep < steps.length ? (
                      <button
                        type="button"
                        onClick={nextStep}
                        className="flex-1 px-6 py-2.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-blue-600 to-cyan-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 flex items-center justify-center"
                      >
                        Next
                        <ChevronRight className="w-4 h-4 ml-2" />
                      </button>
                    ) : (
                      <button
                        type="submit"
                        className="flex-1 px-6 py-2.5 font-bold text-center text-white uppercase align-middle transition-all bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg cursor-pointer leading-pro text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 flex items-center justify-center"
                      >
                        <Check className="w-4 h-4 mr-2" />
                        {editingFarm ? 'Update Farm' : 'Create Farm'}
                      </button>
                    )}
                  </div>
                </form>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  )
}
