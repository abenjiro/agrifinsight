export interface User {
  id: string
  email: string
  name: string
  phone: string
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

export interface Crop {
  id: number
  farm_id: number
  field_id?: number
  crop_type: string
  variety?: string
  quantity?: number
  quantity_unit?: string
  planting_date?: string
  expected_harvest_date?: string
  actual_harvest_date?: string
  growth_stage?: string
  health_status: string
  expected_yield?: number
  actual_yield?: number
  yield_unit?: string
  notes?: string
  irrigation_method?: string
  fertilizer_used?: string[]
  pesticides_used?: string[]
  is_active: boolean
  is_harvested: boolean
  created_at: string
  updated_at?: string
}

export interface CropCreate {
  farm_id: number
  field_id?: number
  crop_type: string
  variety?: string
  quantity?: number
  quantity_unit?: string
  planting_date?: string
  expected_harvest_date?: string
  growth_stage?: string
  health_status?: string
  expected_yield?: number
  yield_unit?: string
  notes?: string
  irrigation_method?: string
  fertilizer_used?: string[]
  pesticides_used?: string[]
}

export interface Animal {
  id: number
  farm_id: number
  animal_type: string
  breed?: string
  quantity: number
  tag_numbers?: string[]
  age_group?: string
  gender_distribution?: { male?: number; female?: number }
  health_status: string
  vaccination_records?: Array<{
    date: string
    vaccine: string
    veterinarian?: string
  }>
  last_health_checkup?: string
  veterinary_notes?: string
  purpose?: string
  production_data?: Record<string, number>
  housing_type?: string
  feeding_type?: string
  feed_consumption?: Record<string, number>
  acquisition_date?: string
  acquisition_cost?: number
  current_value?: number
  notes?: string
  is_active: boolean
  created_at: string
  updated_at?: string
}

export interface AnimalCreate {
  farm_id: number
  animal_type: string
  breed?: string
  quantity: number
  tag_numbers?: string[]
  age_group?: string
  gender_distribution?: { male?: number; female?: number }
  health_status?: string
  vaccination_records?: Array<{
    date: string
    vaccine: string
    veterinarian?: string
  }>
  last_health_checkup?: string
  veterinary_notes?: string
  purpose?: string
  production_data?: Record<string, number>
  housing_type?: string
  feeding_type?: string
  feed_consumption?: Record<string, number>
  acquisition_date?: string
  acquisition_cost?: number
  current_value?: number
  notes?: string
}

export interface CropRecommendation {
  id: number
  farm_id: number
  recommended_crop: string
  confidence_score: number
  suitability_score: number
  climate_factors?: {
    score: number
    factors: Record<string, string>
  }
  soil_factors?: {
    score: number
    factors: Record<string, string>
  }
  geographic_factors?: {
    elevation?: number
    terrain?: string
  }
  market_factors?: {
    demand: string
    profit_margin: number
  }
  planting_season?: string
  expected_yield_range?: {
    min: number
    max: number
    unit: string
  }
  water_requirements?: string
  care_difficulty?: string
  growth_duration_days?: number
  estimated_profit_margin?: number
  market_demand?: string
  selling_price_range?: Record<string, number>
  benefits?: string[]
  challenges?: string[]
  tips?: string[]
  alternative_crops?: string[]
  model_version?: string
  recommendation_date: string
  is_active: boolean
  created_at: string
}




