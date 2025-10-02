export interface User {
  id: string
  email: string
  name: string
  role: 'farmer' | 'admin' | 'analyst'
  created_at: string
  updated_at: string
}

export interface Farm {
  id: string
  name: string
  location: string
  size: number
  crop_type: string
  soil_type: string
  user_id: string
  created_at: string
  updated_at: string
}

export interface AnalysisResult {
  id: string
  farm_id: string
  image_url: string
  disease_detected: string
  confidence: number
  recommendations: string[]
  created_at: string
}

export interface Recommendation {
  id: string
  farm_id: string
  type: 'fertilizer' | 'pesticide' | 'irrigation' | 'harvest' | 'planting'
  title: string
  description: string
  priority: 'low' | 'medium' | 'high'
  created_at: string
}

export interface WeatherData {
  temperature: number
  humidity: number
  rainfall: number
  wind_speed: number
  date: string
}

export interface ApiResponse<T> {
  data: T
  message: string
  success: boolean
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface RegisterData {
  name: string
  email: string
  password: string
  confirmPassword: string
  role: 'farmer' | 'analyst'
}

