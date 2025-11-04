import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { MapPin, TrendingUp, Sprout, PawPrint, Plus, Trash2, Edit2, Loader, Cloud, Calendar } from 'lucide-react'
import { DashboardLayout } from '../components/DashboardLayout'
import { WeatherWidget } from '../components/WeatherWidget'
import AddCropModal from '../components/AddCropModal'
import AddAnimalModal from '../components/AddAnimalModal'
import CropRecommendations from '../components/CropRecommendations'
import { farmService, cropService, animalService } from '../services/api'
import type { Farm, Crop, Animal } from '../types'

export default function FarmDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [farm, setFarm] = useState<Farm | null>(null)
  const [crops, setCrops] = useState<Crop[]>([])
  const [animals, setAnimals] = useState<Animal[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddCropModal, setShowAddCropModal] = useState(false)
  const [showAddAnimalModal, setShowAddAnimalModal] = useState(false)

  useEffect(() => {
    if (id) {
      loadFarmData()
    }
  }, [id])

  const loadFarmData = async () => {
    if (!id) return

    setLoading(true)
    try {
      const [farmData, cropsData, animalsData] = await Promise.all([
        farmService.getFarm(id),
        cropService.getFarmCrops(parseInt(id)),
        animalService.getFarmAnimals(parseInt(id))
      ])

      // The backend returns the farm object directly in farmData.data
      // But if farmData has a .data property, use it, otherwise use farmData directly
      setFarm(farmData.data || farmData)
      setCrops(cropsData)
      setAnimals(animalsData)
    } catch (error: any) {
      console.error('Failed to load farm data:', error)
      // Log more details for debugging
      console.error('Error details:', error.response?.data)
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteCrop = async (cropId: number) => {
    if (!confirm('Are you sure you want to delete this crop?')) return

    try {
      await cropService.deleteCrop(cropId)
      setCrops(crops.filter(c => c.id !== cropId))
    } catch (error) {
      console.error('Failed to delete crop:', error)
      alert('Failed to delete crop')
    }
  }

  const handleDeleteAnimal = async (animalId: number) => {
    if (!confirm('Are you sure you want to delete this animal record?')) return

    try {
      await animalService.deleteAnimal(animalId)
      setAnimals(animals.filter(a => a.id !== animalId))
    } catch (error) {
      console.error('Failed to delete animal:', error)
      alert('Failed to delete animal record')
    }
  }

  const getHealthStatusColor = (status: string) => {
    const colors = {
      healthy: 'bg-green-100 text-green-800',
      stressed: 'bg-yellow-100 text-yellow-800',
      diseased: 'bg-red-100 text-red-800',
      sick: 'bg-red-100 text-red-800',
      under_treatment: 'bg-orange-100 text-orange-800'
    }
    return colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800'
  }

  const getGrowthStageColor = (stage?: string) => {
    const colors = {
      seedling: 'bg-blue-100 text-blue-800',
      vegetative: 'bg-green-100 text-green-800',
      flowering: 'bg-purple-100 text-purple-800',
      fruiting: 'bg-yellow-100 text-yellow-800',
      mature: 'bg-orange-100 text-orange-800'
    }
    return colors[stage as keyof typeof colors] || 'bg-gray-100 text-gray-800'
  }

  if (loading) {
    return (
      <DashboardLayout>
        <div className="flex items-center justify-center h-64">
          <Loader className="h-8 w-8 animate-spin text-green-600" />
          <span className="ml-3 text-lg text-gray-600">Loading farm details...</span>
        </div>
      </DashboardLayout>
    )
  }

  if (!farm) {
    return (
      <DashboardLayout>
        <div className="text-center py-12">
          <h2 className="text-2xl font-semibold text-gray-900">Farm not found</h2>
          <button
            onClick={() => navigate('/farms')}
            className="mt-4 text-green-600 hover:text-green-700"
          >
            Back to farms
          </button>
        </div>
      </DashboardLayout>
    )
  }

  return (
    <DashboardLayout>
      <div className="w-full space-y-6">
        {/* Farm Header */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 mb-2">{farm.name}</h1>
              {farm.location && (
                <div className="flex items-center text-gray-600">
                  <MapPin className="h-5 w-5 mr-2" />
                  <span>{farm.location}</span>
                </div>
              )}
            </div>
            <button
              onClick={() => navigate('/farms')}
              className="text-gray-600 hover:text-gray-800"
            >
              Back to Farms
            </button>
          </div>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-sm text-green-600 font-medium">Farm Size</div>
              <div className="text-2xl font-bold text-gray-900">{farm.size} acres</div>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-sm text-blue-600 font-medium">Crops</div>
              <div className="text-2xl font-bold text-gray-900">{crops.length}</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="text-sm text-purple-600 font-medium">Animals</div>
              <div className="text-2xl font-bold text-gray-900">
                {animals.reduce((sum, a) => sum + a.quantity, 0)}
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="mt-6 flex flex-wrap gap-3">
            <button
              onClick={() => navigate(`/farms/${id}/planting`)}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg hover:from-green-700 hover:to-emerald-700 transition shadow-md"
            >
              <Calendar className="w-4 h-4" />
              Planting Recommendations
            </button>
            <button
              onClick={() => {/* TODO: Add weather modal or navigate to weather page */}}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg hover:from-blue-700 hover:to-cyan-700 transition shadow-md"
            >
              <Cloud className="w-4 h-4" />
              View Weather Forecast
            </button>
          </div>
        </div>

        {/* Weather Widget */}
        {id && farm.latitude && farm.longitude && (
          <WeatherWidget farmId={parseInt(id)} compact={true} showForecast={false} />
        )}

        {/* Main Content Grid: Crops & Animals (Main Focus) + Recommendations (Sidebar) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Crops & Animals (2/3 width) */}
          <div className="lg:col-span-2 space-y-6">
            {/* Crops Section */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-6 border-b border-gray-200 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sprout className="h-6 w-6 text-green-600" />
                  <h2 className="text-xl font-semibold text-gray-900">Crops</h2>
                </div>
                <button
                  onClick={() => setShowAddCropModal(true)}
                  className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-2"
                >
                  <Plus className="h-4 w-4" />
                  Add Crop
                </button>
              </div>

              <div className="p-6">
                {crops.length === 0 ? (
                  <div className="text-center py-12">
                    <Sprout className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Crops Yet</h3>
                    <p className="text-gray-600 mb-4">Add crops to start tracking your farming activities</p>
                    <button
                      onClick={() => setShowAddCropModal(true)}
                      className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                    >
                      Add Your First Crop
                    </button>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {crops.map((crop) => (
                      <div key={crop.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <h3 className="font-semibold text-gray-900">{crop.crop_type}</h3>
                            {crop.variety && (
                              <p className="text-sm text-gray-600">{crop.variety}</p>
                            )}
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleDeleteCrop(crop.id)}
                              className="text-red-600 hover:text-red-700"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>

                        <div className="space-y-2">
                          {crop.growth_stage && (
                            <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${getGrowthStageColor(crop.growth_stage)}`}>
                              {crop.growth_stage}
                            </span>
                          )}
                          <span className={`inline-block ml-2 px-2 py-1 rounded text-xs font-medium ${getHealthStatusColor(crop.health_status)}`}>
                            {crop.health_status}
                          </span>

                          {crop.quantity && (
                            <div className="text-sm text-gray-600">
                              {crop.quantity} {crop.quantity_unit}
                            </div>
                          )}

                          {crop.planting_date && (
                            <div className="text-sm text-gray-600">
                              Planted: {new Date(crop.planting_date).toLocaleDateString()}
                            </div>
                          )}

                          {crop.expected_harvest_date && (
                            <div className="text-sm text-gray-600">
                              Expected Harvest: {new Date(crop.expected_harvest_date).toLocaleDateString()}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Animals Section */}
            <div className="bg-white rounded-lg shadow">
              <div className="p-6 border-b border-gray-200 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <PawPrint className="h-6 w-6 text-green-600" />
                  <h2 className="text-xl font-semibold text-gray-900">Animals</h2>
                </div>
                <button
                  onClick={() => setShowAddAnimalModal(true)}
                  className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-2"
                >
                  <Plus className="h-4 w-4" />
                  Add Animals
                </button>
              </div>

              <div className="p-6">
                {animals.length === 0 ? (
                  <div className="text-center py-12">
                    <PawPrint className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">No Animals Yet</h3>
                    <p className="text-gray-600 mb-4">Add animals to track your livestock</p>
                    <button
                      onClick={() => setShowAddAnimalModal(true)}
                      className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                    >
                      Add Your First Animals
                    </button>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {animals.map((animal) => (
                      <div key={animal.id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <h3 className="font-semibold text-gray-900 capitalize">{animal.animal_type}</h3>
                            {animal.breed && (
                              <p className="text-sm text-gray-600">{animal.breed}</p>
                            )}
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleDeleteAnimal(animal.id)}
                              className="text-red-600 hover:text-red-700"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>

                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600">Quantity:</span>
                            <span className="font-medium">{animal.quantity}</span>
                          </div>

                          {animal.gender_distribution && (
                            <div className="text-sm text-gray-600">
                              Male: {animal.gender_distribution.male || 0}, Female: {animal.gender_distribution.female || 0}
                            </div>
                          )}

                          {animal.purpose && (
                            <div className="text-sm text-gray-600 capitalize">
                              Purpose: {animal.purpose}
                            </div>
                          )}

                          <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${getHealthStatusColor(animal.health_status)}`}>
                            {animal.health_status}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column: AI Recommendations (1/3 width - Compact Sidebar) */}
          <div className="lg:col-span-1">
            {id && <CropRecommendations farmId={parseInt(id)} />}
          </div>
        </div>
      </div>

      {/* Modals */}
      {id && (
        <>
          <AddCropModal
            isOpen={showAddCropModal}
            onClose={() => setShowAddCropModal(false)}
            farmId={parseInt(id)}
            onSuccess={loadFarmData}
          />
          <AddAnimalModal
            isOpen={showAddAnimalModal}
            onClose={() => setShowAddAnimalModal(false)}
            farmId={parseInt(id)}
            onSuccess={loadFarmData}
          />
        </>
      )}
    </DashboardLayout>
  )
}
