import axios from 'axios'
import type {
  User,
  Farm,
  AnalysisResult,
  Recommendation,
  WeatherData,
  ApiResponse,
  LoginCredentials,
  RegisterData,
  Crop,
  CropCreate,
  Animal,
  AnimalCreate,
  CropRecommendation
} from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export const authService = {
  async login(credentials: LoginCredentials): Promise<ApiResponse<{ user: User; token: string }>> {
    const response = await api.post('/auth/login', credentials)
    return response.data
  },

  async register(data: RegisterData): Promise<ApiResponse<{ user: User; token: string }>> {
    const response = await api.post('/auth/register', data)
    return response.data
  },

  async logout(): Promise<void> {
    await api.post('/auth/logout')
    localStorage.removeItem('auth_token')
  },

  async getCurrentUser(): Promise<ApiResponse<User>> {
    const response = await api.get('/auth/me')
    return response.data
  },
}

export const farmService = {
  async getFarms(): Promise<ApiResponse<Farm[]>> {
    const response = await api.get('/farms')
    return response.data
  },

  async getFarm(id: string): Promise<ApiResponse<Farm>> {
    const response = await api.get(`/farms/${id}`)
    return response.data
  },

  async createFarm(farm: Omit<Farm, 'id' | 'created_at' | 'updated_at'>): Promise<ApiResponse<Farm>> {
    const response = await api.post('/farms', farm)
    return response.data
  },

  async updateFarm(id: string, farm: Partial<Farm>): Promise<ApiResponse<Farm>> {
    const response = await api.put(`/farms/${id}`, farm)
    return response.data
  },

  async deleteFarm(id: string): Promise<void> {
    await api.delete(`/farms/${id}`)
  },
}

export const analysisService = {
  async uploadImage(file: File, farmId: string): Promise<ApiResponse<AnalysisResult>> {
    const formData = new FormData()
    formData.append('image', file)
    formData.append('farm_id', farmId)

    const response = await api.post('/analysis/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  async getAnalysisHistory(farmId: string): Promise<ApiResponse<AnalysisResult[]>> {
    const response = await api.get(`/analysis/history/${farmId}`)
    return response.data
  },
}

export const recommendationService = {
  async getRecommendations(farmId: string): Promise<ApiResponse<Recommendation[]>> {
    const response = await api.get(`/recommendations/${farmId}`)
    return response.data
  },

  async getWeatherData(farmId: string): Promise<ApiResponse<WeatherData[]>> {
    const response = await api.get(`/recommendations/weather/${farmId}`)
    return response.data
  },
}

export const cropService = {
  async getCropTypes(): Promise<string[]> {
    const response = await api.get('/crops/types')
    return response.data.crop_types
  },

  async getFarmCrops(farmId: number): Promise<Crop[]> {
    const response = await api.get(`/farms/${farmId}/crops`)
    return response.data
  },

  async createCrop(farmId: number, data: CropCreate): Promise<Crop> {
    const response = await api.post(`/farms/${farmId}/crops`, data)
    return response.data
  },

  async getCrop(cropId: number): Promise<Crop> {
    const response = await api.get(`/crops/${cropId}`)
    return response.data
  },

  async updateCrop(cropId: number, data: Partial<CropCreate>): Promise<Crop> {
    const response = await api.put(`/crops/${cropId}`, data)
    return response.data
  },

  async deleteCrop(cropId: number): Promise<void> {
    await api.delete(`/crops/${cropId}`)
  },
}

export const animalService = {
  async getFarmAnimals(farmId: number): Promise<Animal[]> {
    const response = await api.get(`/farms/${farmId}/animals`)
    return response.data
  },

  async createAnimal(farmId: number, data: AnimalCreate): Promise<Animal> {
    const response = await api.post(`/farms/${farmId}/animals`, data)
    return response.data
  },

  async getAnimal(animalId: number): Promise<Animal> {
    const response = await api.get(`/animals/${animalId}`)
    return response.data
  },

  async updateAnimal(animalId: number, data: Partial<AnimalCreate>): Promise<Animal> {
    const response = await api.put(`/animals/${animalId}`, data)
    return response.data
  },

  async deleteAnimal(animalId: number): Promise<void> {
    await api.delete(`/animals/${animalId}`)
  },
}

export const cropRecommendationService = {
  async generateRecommendations(farmId: number): Promise<CropRecommendation[]> {
    const response = await api.post(`/farms/${farmId}/crop-recommendations`)
    return response.data
  },

  async getRecommendations(farmId: number): Promise<CropRecommendation[]> {
    const response = await api.get(`/farms/${farmId}/crop-recommendations`)
    return response.data
  },
}

export default api




