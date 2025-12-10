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
    // Only redirect to login if we're authenticated but token is invalid
    // Don't redirect on login/register failures (those are expected)
    const isAuthEndpoint = error.config?.url?.includes('/auth/login') ||
                          error.config?.url?.includes('/auth/register')

    if (error.response?.status === 401 && !isAuthEndpoint) {
      // User's token is invalid, clear it and redirect
      localStorage.removeItem('auth_token')
      localStorage.removeItem('user')

      // Only redirect if not already on login page
      if (!window.location.pathname.includes('/login')) {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

// Generic API wrapper helper
type UrlFunction<TArgs extends any[]> = (...args: TArgs) => string
type ResponseTransformer<TData, TResult> = (response: TData) => TResult

function wrapApi<TArgs extends any[], TResult>(
  urlOrFunction: string | UrlFunction<TArgs>,
  transformer?: ResponseTransformer<any, TResult>,
  method: 'get' | 'post' | 'put' | 'delete' = 'get'
) {
  return async (...args: TArgs): Promise<TResult> => {
    const url = typeof urlOrFunction === 'function' ? urlOrFunction(...args) : urlOrFunction
    const response = await api[method](url)
    return transformer ? transformer(response) : response.data
  }
}

function wrapApiWithBody<TArgs extends any[], TBody, TResult>(
  urlOrFunction: string | ((...args: TArgs) => string),
  transformer?: ResponseTransformer<any, TResult>,
  method: 'post' | 'put' = 'post'
) {
  return async (...args: [...TArgs, TBody]): Promise<TResult> => {
    const bodyIndex = args.length - 1
    const body = args[bodyIndex]
    const urlArgs = args.slice(0, bodyIndex) as TArgs
    const url = typeof urlOrFunction === 'function' ? urlOrFunction(...urlArgs) : urlOrFunction
    const response = await api[method](url, body)
    return transformer ? transformer(response) : response.data
  }
}

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

export const harvestService = {
  async getHarvestPrediction(cropId: string): Promise<ApiResponse<any>> {
    const response = await api.get(`/recommendations/harvest/${cropId}`)
    return response.data
  },

  async getHarvestCalendar(farmId: string): Promise<ApiResponse<any>> {
    const response = await api.get(`/recommendations/harvest/calendar/${farmId}`)
    return response.data
  },
}

export const careService = {
  async getCareRecommendations(cropId: string): Promise<ApiResponse<any>> {
    const response = await api.get(`/recommendations/care/${cropId}`)
    return response.data
  },
}

export interface CropTypeData {
  name: string
  category: string
  scientific_name: string | null
  description: string | null
  growth_duration_days: number | null
  water_requirement: string | null
  recommended_irrigation: string | null
  min_yield_per_acre: number | null
  max_yield_per_acre: number | null
  avg_yield_per_acre: number | null
  yield_unit: string | null
}

export const cropService = {
  getCropTypes: wrapApi<[], CropTypeData[]>(
    '/crops/types',
    (response) => response.data.crop_types
  ),

  getFarmCrops: wrapApi<[number], Crop[]>(
    (farmId) => `/farms/${farmId}/crops`,
    (response) => response.data
  ),

  createCrop: wrapApiWithBody<[number], CropCreate, Crop>(
    (farmId) => `/farms/${farmId}/crops`,
    (response) => response.data,
    'post'
  ),

  getCrop: wrapApi<[number], Crop>(
    (cropId) => `/crops/${cropId}`,
    (response) => response.data
  ),

  updateCrop: wrapApiWithBody<[number], Partial<CropCreate>, Crop>(
    (cropId) => `/crops/${cropId}`,
    (response) => response.data,
    'put'
  ),

  deleteCrop: wrapApi<[number], void>(
    (cropId) => `/crops/${cropId}`,
    undefined,
    'delete'
  ),
}

export const animalService = {
  getFarmAnimals: wrapApi<[number], Animal[]>(
    (farmId) => `/farms/${farmId}/animals`,
    (response) => response.data
  ),

  createAnimal: wrapApiWithBody<[number], AnimalCreate, Animal>(
    (farmId) => `/farms/${farmId}/animals`,
    (response) => response.data,
    'post'
  ),

  getAnimal: wrapApi<[number], Animal>(
    (animalId) => `/animals/${animalId}`,
    (response) => response.data
  ),

  updateAnimal: wrapApiWithBody<[number], Partial<AnimalCreate>, Animal>(
    (animalId) => `/animals/${animalId}`,
    (response) => response.data,
    'put'
  ),

  deleteAnimal: wrapApi<[number], void>(
    (animalId) => `/animals/${animalId}`,
    undefined,
    'delete'
  ),
}

export const cropRecommendationService = {
  generateRecommendations: wrapApi<[number], CropRecommendation[]>(
    (farmId) => `/farms/${farmId}/crop-recommendations`,
    (response) => response.data,
    'post'
  ),

  getRecommendations: wrapApi<[number], CropRecommendation[]>(
    (farmId) => `/farms/${farmId}/crop-recommendations`,
    (response) => response.data
  ),
}

export default api




