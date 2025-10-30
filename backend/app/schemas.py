"""
Pydantic schemas for request/response models
"""

from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    phone: Optional[str] = None
    role: Optional[str] = "farmer"

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Farm schemas
class FarmBase(BaseModel):
    name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    boundary_coordinates: Optional[dict] = None
    size: Optional[float] = None
    size_unit: Optional[str] = "acres"
    soil_type: Optional[str] = None
    soil_ph: Optional[float] = None
    soil_composition: Optional[dict] = None
    terrain_type: Optional[str] = None
    elevation_profile: Optional[dict] = None
    climate_zone: Optional[str] = None
    avg_annual_rainfall: Optional[float] = None
    avg_temperature: Optional[float] = None
    water_sources: Optional[list] = None
    timezone: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    district: Optional[str] = None

class FarmCreate(FarmBase):
    pass

class FarmUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    boundary_coordinates: Optional[dict] = None
    size: Optional[float] = None
    size_unit: Optional[str] = None
    soil_type: Optional[str] = None
    soil_ph: Optional[float] = None
    soil_composition: Optional[dict] = None
    terrain_type: Optional[str] = None
    elevation_profile: Optional[dict] = None
    climate_zone: Optional[str] = None
    avg_annual_rainfall: Optional[float] = None
    avg_temperature: Optional[float] = None
    water_sources: Optional[list] = None
    timezone: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    district: Optional[str] = None

class Farm(FarmBase):
    id: int
    user_id: int
    last_satellite_image_date: Optional[datetime] = None
    satellite_image_url: Optional[str] = None
    ndvi_data: Optional[dict] = None
    land_use_history: Optional[dict] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Field schemas
class FieldBase(BaseModel):
    name: str
    crop_type: Optional[str] = None
    planting_date: Optional[datetime] = None
    expected_harvest_date: Optional[datetime] = None
    coordinates: Optional[str] = None

class FieldCreate(FieldBase):
    pass

class FieldUpdate(BaseModel):
    name: Optional[str] = None
    crop_type: Optional[str] = None
    planting_date: Optional[datetime] = None
    expected_harvest_date: Optional[datetime] = None
    coordinates: Optional[str] = None

class Field(FieldBase):
    id: int
    farm_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Crop Image schemas
class CropImageBase(BaseModel):
    image_url: str
    filename: Optional[str] = None
    file_size: Optional[int] = None

class CropImageCreate(CropImageBase):
    farm_id: int
    field_id: Optional[int] = None

class CropImage(CropImageBase):
    id: int
    farm_id: int
    field_id: Optional[int] = None
    user_id: int
    analysis_status: str
    uploaded_at: datetime

    class Config:
        from_attributes = True

# Analysis Result schemas
class AnalysisResultBase(BaseModel):
    disease_detected: Optional[str] = None
    confidence_score: Optional[float] = None
    disease_type: Optional[str] = None
    severity: Optional[str] = None
    recommendations: Optional[str] = None
    treatment_advice: Optional[str] = None
    growth_stage: Optional[str] = None
    health_score: Optional[float] = None

class AnalysisResultCreate(AnalysisResultBase):
    image_id: int
    user_id: int

class AnalysisResult(AnalysisResultBase):
    id: int
    image_id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Weather Data schemas
class WeatherDataBase(BaseModel):
    location: str
    date: datetime
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    pressure: Optional[float] = None

class WeatherDataCreate(WeatherDataBase):
    pass

class WeatherData(WeatherDataBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Planting Recommendation schemas
class PlantingRecommendationBase(BaseModel):
    crop_type: str
    recommended_planting_date: Optional[datetime] = None
    confidence_score: Optional[float] = None
    weather_conditions: Optional[dict] = None
    soil_conditions: Optional[dict] = None
    risk_factors: Optional[dict] = None

class PlantingRecommendationCreate(PlantingRecommendationBase):
    farm_id: int
    field_id: Optional[int] = None

class PlantingRecommendation(PlantingRecommendationBase):
    id: int
    user_id: int
    farm_id: int
    field_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Harvest Prediction schemas
class HarvestPredictionBase(BaseModel):
    predicted_harvest_date: Optional[datetime] = None
    confidence_score: Optional[float] = None
    expected_yield: Optional[float] = None
    quality_prediction: Optional[str] = None
    factors: Optional[dict] = None

class HarvestPredictionCreate(HarvestPredictionBase):
    farm_id: int
    field_id: Optional[int] = None

class HarvestPrediction(HarvestPredictionBase):
    id: int
    user_id: int
    farm_id: int
    field_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Token schemas
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Crop schemas
class CropBase(BaseModel):
    crop_type: str
    variety: Optional[str] = None
    quantity: Optional[float] = None
    quantity_unit: Optional[str] = None
    planting_date: Optional[datetime] = None
    expected_harvest_date: Optional[datetime] = None
    growth_stage: Optional[str] = None
    health_status: Optional[str] = "healthy"
    expected_yield: Optional[float] = None
    yield_unit: Optional[str] = None
    notes: Optional[str] = None
    irrigation_method: Optional[str] = None
    fertilizer_used: Optional[list] = None
    pesticides_used: Optional[list] = None
    is_active: Optional[bool] = True

class CropCreate(CropBase):
    farm_id: int
    field_id: Optional[int] = None

    @field_validator('planting_date', 'expected_harvest_date', mode='before')
    @classmethod
    def empty_str_to_none(cls, v):
        """Convert empty strings to None for datetime fields"""
        if v == '' or v is None:
            return None
        return v

class CropUpdate(BaseModel):
    crop_type: Optional[str] = None
    variety: Optional[str] = None
    quantity: Optional[float] = None
    quantity_unit: Optional[str] = None
    planting_date: Optional[datetime] = None
    expected_harvest_date: Optional[datetime] = None
    actual_harvest_date: Optional[datetime] = None
    growth_stage: Optional[str] = None
    health_status: Optional[str] = None
    expected_yield: Optional[float] = None
    actual_yield: Optional[float] = None
    yield_unit: Optional[str] = None
    notes: Optional[str] = None
    irrigation_method: Optional[str] = None
    fertilizer_used: Optional[list] = None
    pesticides_used: Optional[list] = None
    is_active: Optional[bool] = None
    is_harvested: Optional[bool] = None

class Crop(CropBase):
    id: int
    farm_id: int
    field_id: Optional[int] = None
    actual_harvest_date: Optional[datetime] = None
    actual_yield: Optional[float] = None
    is_harvested: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Animal schemas
class AnimalBase(BaseModel):
    animal_type: str
    breed: Optional[str] = None
    quantity: int
    tag_numbers: Optional[list] = None
    age_group: Optional[str] = None
    gender_distribution: Optional[dict] = None
    health_status: Optional[str] = "healthy"
    vaccination_records: Optional[list] = None
    last_health_checkup: Optional[datetime] = None
    veterinary_notes: Optional[str] = None
    purpose: Optional[str] = None
    production_data: Optional[dict] = None
    housing_type: Optional[str] = None
    feeding_type: Optional[str] = None
    feed_consumption: Optional[dict] = None
    acquisition_date: Optional[datetime] = None
    acquisition_cost: Optional[float] = None
    current_value: Optional[float] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = True

class AnimalCreate(AnimalBase):
    farm_id: int

class AnimalUpdate(BaseModel):
    animal_type: Optional[str] = None
    breed: Optional[str] = None
    quantity: Optional[int] = None
    tag_numbers: Optional[list] = None
    age_group: Optional[str] = None
    gender_distribution: Optional[dict] = None
    health_status: Optional[str] = None
    vaccination_records: Optional[list] = None
    last_health_checkup: Optional[datetime] = None
    veterinary_notes: Optional[str] = None
    purpose: Optional[str] = None
    production_data: Optional[dict] = None
    housing_type: Optional[str] = None
    feeding_type: Optional[str] = None
    feed_consumption: Optional[dict] = None
    acquisition_date: Optional[datetime] = None
    acquisition_cost: Optional[float] = None
    current_value: Optional[float] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None

class Animal(AnimalBase):
    id: int
    farm_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Crop Recommendation schemas
class CropRecommendationBase(BaseModel):
    recommended_crop: str
    confidence_score: Optional[float] = None
    suitability_score: Optional[float] = None
    climate_factors: Optional[dict] = None
    soil_factors: Optional[dict] = None
    geographic_factors: Optional[dict] = None
    market_factors: Optional[dict] = None
    planting_season: Optional[str] = None
    expected_yield_range: Optional[dict] = None
    water_requirements: Optional[str] = None
    care_difficulty: Optional[str] = None
    growth_duration_days: Optional[int] = None
    estimated_profit_margin: Optional[float] = None
    market_demand: Optional[str] = None
    selling_price_range: Optional[dict] = None
    benefits: Optional[list] = None
    challenges: Optional[list] = None
    tips: Optional[list] = None
    alternative_crops: Optional[list] = None

class CropRecommendation(CropRecommendationBase):
    id: int
    farm_id: int
    user_id: int
    model_version: Optional[str] = None
    recommendation_date: datetime
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# Response schemas
class MessageResponse(BaseModel):
    message: str
    success: bool = True

class ErrorResponse(BaseModel):
    message: str
    success: bool = False
    error_code: Optional[str] = None
