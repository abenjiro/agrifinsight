"""
Pydantic schemas for request/response models
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    email: EmailStr
    phone: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    password: Optional[str] = None

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
    location: Optional[str] = None
    size: Optional[float] = None
    soil_type: Optional[str] = None

class FarmCreate(FarmBase):
    pass

class FarmUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    size: Optional[float] = None
    soil_type: Optional[str] = None

class Farm(FarmBase):
    id: int
    user_id: int
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

# Response schemas
class MessageResponse(BaseModel):
    message: str
    success: bool = True

class ErrorResponse(BaseModel):
    message: str
    success: bool = False
    error_code: Optional[str] = None
