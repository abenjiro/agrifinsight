import { useState } from 'react'
import { X } from 'lucide-react'
import { animalService } from '../services/api'
import type { AnimalCreate } from '../types'

interface AddAnimalModalProps {
  isOpen: boolean
  onClose: () => void
  farmId: number
  onSuccess: () => void
}

const ANIMAL_TYPES = [
  'Cattle',
  'Goat',
  'Sheep',
  'Pig',
  'Chicken',
  'Duck',
  'Turkey',
  'Rabbit',
  'Fish',
  'Other'
]

const PURPOSE_OPTIONS = [
  'meat',
  'dairy',
  'eggs',
  'breeding',
  'draft',
  'wool',
  'mixed'
]

const HOUSING_TYPES = [
  'free-range',
  'pen',
  'barn',
  'coop',
  'cage',
  'pond',
  'pasture'
]

const FEEDING_TYPES = [
  'grazing',
  'supplemented',
  'intensive',
  'organic'
]

export default function AddAnimalModal({ isOpen, onClose, farmId, onSuccess }: AddAnimalModalProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [formData, setFormData] = useState<AnimalCreate>({
    farm_id: farmId,
    animal_type: '',
    breed: '',
    quantity: 1,
    age_group: 'adult',
    gender_distribution: { male: 0, female: 0 },
    health_status: 'healthy',
    purpose: 'meat',
    housing_type: 'pen',
    feeding_type: 'grazing',
    notes: '',
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value ? parseInt(value) : 0
    }))
  }

  const handleGenderChange = (gender: 'male' | 'female', value: string) => {
    setFormData(prev => ({
      ...prev,
      gender_distribution: {
        ...prev.gender_distribution,
        [gender]: value ? parseInt(value) : 0
      }
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      await animalService.createAnimal(farmId, formData)
      onSuccess()
      onClose()
      // Reset form
      setFormData({
        farm_id: farmId,
        animal_type: '',
        breed: '',
        quantity: 1,
        age_group: 'adult',
        gender_distribution: { male: 0, female: 0 },
        health_status: 'healthy',
        purpose: 'meat',
        housing_type: 'pen',
        feeding_type: 'grazing',
        notes: '',
      })
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to add animal')
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
              <h3 className="text-lg font-medium text-gray-900">Add Animals</h3>
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-500"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            {error && (
              <div className="mb-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Animal Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Animal Type *
                  </label>
                  <select
                    name="animal_type"
                    value={formData.animal_type}
                    onChange={handleChange}
                    required
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  >
                    <option value="">Select animal type</option>
                    {ANIMAL_TYPES.map(animal => (
                      <option key={animal} value={animal.toLowerCase()}>{animal}</option>
                    ))}
                  </select>
                </div>

                {/* Breed */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Breed
                  </label>
                  <input
                    type="text"
                    name="breed"
                    value={formData.breed}
                    onChange={handleChange}
                    placeholder="e.g., Friesian, Sasso"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>

                {/* Quantity */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Total Quantity *
                  </label>
                  <input
                    type="number"
                    name="quantity"
                    value={formData.quantity}
                    onChange={handleNumberChange}
                    required
                    min="1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>

                {/* Age Group */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Age Group
                  </label>
                  <select
                    name="age_group"
                    value={formData.age_group}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  >
                    <option value="young">Young</option>
                    <option value="adult">Adult</option>
                    <option value="senior">Senior</option>
                  </select>
                </div>

                {/* Male Count */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Male Count
                  </label>
                  <input
                    type="number"
                    value={formData.gender_distribution?.male || 0}
                    onChange={(e) => handleGenderChange('male', e.target.value)}
                    min="0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>

                {/* Female Count */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Female Count
                  </label>
                  <input
                    type="number"
                    value={formData.gender_distribution?.female || 0}
                    onChange={(e) => handleGenderChange('female', e.target.value)}
                    min="0"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  />
                </div>

                {/* Purpose */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Purpose
                  </label>
                  <select
                    name="purpose"
                    value={formData.purpose}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  >
                    {PURPOSE_OPTIONS.map(purpose => (
                      <option key={purpose} value={purpose}>
                        {purpose.charAt(0).toUpperCase() + purpose.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Housing Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Housing Type
                  </label>
                  <select
                    name="housing_type"
                    value={formData.housing_type}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  >
                    {HOUSING_TYPES.map(housing => (
                      <option key={housing} value={housing}>
                        {housing.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Feeding Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Feeding Type
                  </label>
                  <select
                    name="feeding_type"
                    value={formData.feeding_type}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  >
                    {FEEDING_TYPES.map(feeding => (
                      <option key={feeding} value={feeding}>
                        {feeding.charAt(0).toUpperCase() + feeding.slice(1)}
                      </option>
                    ))}
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
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  >
                    <option value="healthy">Healthy</option>
                    <option value="sick">Sick</option>
                    <option value="under_treatment">Under Treatment</option>
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
                  placeholder="Additional information about these animals..."
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
                  {loading ? 'Adding...' : 'Add Animals'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}
