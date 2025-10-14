"""
Database models for AgriFinSight
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User model"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    phone = Column(String(20), unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    farms = relationship("Farm", back_populates="owner")
    crop_images = relationship("CropImage", back_populates="user")
    analysis_results = relationship("AnalysisResult", back_populates="user")

class Farm(Base):
    """Farm model"""
    __tablename__ = "farms"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    location = Column(String(255))  # GPS coordinates or address
    size = Column(Float)  # Size in acres/hectares
    soil_type = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="farms")
    fields = relationship("Field", back_populates="farm")
    crop_images = relationship("CropImage", back_populates="farm")

class Field(Base):
    """Field model"""
    __tablename__ = "fields"
    
    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"), nullable=False)
    name = Column(String(255), nullable=False)
    crop_type = Column(String(100))
    planting_date = Column(DateTime(timezone=True))
    expected_harvest_date = Column(DateTime(timezone=True))
    coordinates = Column(String(255))  # GPS coordinates
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    farm = relationship("Farm", back_populates="fields")
    crop_images = relationship("CropImage", back_populates="field")

class CropImage(Base):
    """Crop image model"""
    __tablename__ = "crop_images"
    
    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"), nullable=False)
    field_id = Column(Integer, ForeignKey("fields.id"))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_url = Column(String(500), nullable=False)
    filename = Column(String(255))
    file_size = Column(Integer)
    analysis_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    farm = relationship("Farm", back_populates="crop_images")
    field = relationship("Field", back_populates="crop_images")
    user = relationship("User", back_populates="crop_images")
    analysis_results = relationship("AnalysisResult", back_populates="image")

class AnalysisResult(Base):
    """Analysis result model"""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("crop_images.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    disease_detected = Column(String(100))
    confidence_score = Column(Float)
    disease_type = Column(String(100))
    severity = Column(String(50))  # low, medium, high
    recommendations = Column(Text)
    treatment_advice = Column(Text)
    growth_stage = Column(String(50))
    health_score = Column(Float)  # 0-100
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    image = relationship("CropImage", back_populates="analysis_results")
    user = relationship("User", back_populates="analysis_results")

class WeatherData(Base):
    """Weather data model"""
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String(255), nullable=False)
    date = Column(DateTime(timezone=True), nullable=False)
    temperature = Column(Float)
    humidity = Column(Float)
    rainfall = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    pressure = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PlantingRecommendation(Base):
    """Planting recommendation model"""
    __tablename__ = "planting_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    farm_id = Column(Integer, ForeignKey("farms.id"), nullable=False)
    field_id = Column(Integer, ForeignKey("fields.id"))
    crop_type = Column(String(100), nullable=False)
    recommended_planting_date = Column(DateTime(timezone=True))
    confidence_score = Column(Float)
    weather_conditions = Column(JSON)
    soil_conditions = Column(JSON)
    risk_factors = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    farm = relationship("Farm")
    field = relationship("Field")

class HarvestPrediction(Base):
    """Harvest prediction model"""
    __tablename__ = "harvest_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    farm_id = Column(Integer, ForeignKey("farms.id"), nullable=False)
    field_id = Column(Integer, ForeignKey("fields.id"))
    predicted_harvest_date = Column(DateTime(timezone=True))
    confidence_score = Column(Float)
    expected_yield = Column(Float)
    quality_prediction = Column(String(50))
    factors = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    farm = relationship("Farm")
    field = relationship("Field")
