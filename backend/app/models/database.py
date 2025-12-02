"""
Database models for AgriFinSight
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class CropType(Base):
    """Crop type master list with growth predictions"""
    __tablename__ = "crop_types"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50))  # grains, vegetables, fruits, legumes, cash_crops, tubers
    scientific_name = Column(String(200))
    description = Column(Text)
    common_varieties = Column(Text)  # Comma-separated list

    # Growth and prediction data
    growth_duration_days = Column(Integer)  # Typical days to maturity
    water_requirement = Column(String(20))  # low, medium, high
    recommended_irrigation = Column(String(50))  # rain-fed, drip, sprinkler, flood, furrow, manual
    min_yield_per_acre = Column(Float)  # Minimum yield in kg per acre
    max_yield_per_acre = Column(Float)  # Maximum yield in kg per acre
    avg_yield_per_acre = Column(Float)  # Average yield in kg per acre
    yield_unit = Column(String(20), default='kg')  # kg, tons, bags

    # Additional metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class User(Base):
    """User model"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    phone = Column(String(20), unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="farmer")  # farmer, analyst, admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    farms = relationship("Farm", back_populates="owner")

class TokenBlacklist(Base):
    """Token blacklist for logout functionality"""
    __tablename__ = "token_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(500), unique=True, index=True, nullable=False)
    blacklisted_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)  # Token expiry time

class Farm(Base):
    """Farm model with comprehensive geospatial data"""
    __tablename__ = "farms"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)

    # Location data
    address = Column(String(500))  # Human-readable address
    latitude = Column(Float, index=True)  # Decimal degrees
    longitude = Column(Float, index=True)  # Decimal degrees
    altitude = Column(Float)  # Meters above sea level
    boundary_coordinates = Column(JSON)  # GeoJSON polygon for farm boundaries

    # Farm properties
    size = Column(Float)  # Size in acres/hectares
    size_unit = Column(String(20), default="acres")  # acres or hectares

    # Soil and environmental data
    soil_type = Column(String(100))  # Primary soil type
    soil_ph = Column(Float)  # Soil pH level
    soil_composition = Column(JSON)  # Detailed soil analysis {sand: %, clay: %, silt: %, organic_matter: %}
    terrain_type = Column(String(100))  # flat, hilly, mountainous, etc.
    elevation_profile = Column(JSON)  # Elevation variations across farm

    # Climate and weather
    climate_zone = Column(String(100))  # tropical, temperate, arid, etc.
    avg_annual_rainfall = Column(Float)  # mm
    avg_temperature = Column(Float)  # Celsius
    water_sources = Column(JSON)  # Array of water sources {type: 'river/well/irrigation', location: coords}

    # Historical and satellite data references
    last_satellite_image_date = Column(DateTime(timezone=True))
    satellite_image_url = Column(String(500))  # URL to latest satellite imagery
    ndvi_data = Column(JSON)  # Normalized Difference Vegetation Index data
    land_use_history = Column(JSON)  # Historical land use data

    # Additional metadata
    timezone = Column(String(50))  # e.g., "Africa/Accra"
    country = Column(String(100))
    region = Column(String(100))  # State/Province
    district = Column(String(100))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    owner = relationship("User", back_populates="farms")
    fields = relationship("Field", back_populates="farm", cascade="all, delete-orphan")
    crop_images = relationship("CropImage", back_populates="farm", cascade="all, delete-orphan")
    crops = relationship("Crop", back_populates="farm", cascade="all, delete-orphan")
    animals = relationship("Animal", back_populates="farm", cascade="all, delete-orphan")
    crop_recommendations = relationship("CropRecommendation", back_populates="farm", cascade="all, delete-orphan")

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
    # user_id removed: can be derived via farm_id → farms → user_id
    image_url = Column(String(500), nullable=False)
    filename = Column(String(255))
    file_size = Column(Integer)
    analysis_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    farm = relationship("Farm", back_populates="crop_images")
    field = relationship("Field", back_populates="crop_images")
    analysis_results = relationship("AnalysisResult", back_populates="image", cascade="all, delete-orphan")

class AnalysisResult(Base):
    """Analysis result model"""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("crop_images.id"), nullable=False)
    # user_id removed: can be derived via image_id → crop_images → farm_id → farms → user_id
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
    # user_id removed: can be derived via farm_id → farms → user_id
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
    farm = relationship("Farm")
    field = relationship("Field")

class HarvestPrediction(Base):
    """Harvest prediction model"""
    __tablename__ = "harvest_predictions"

    id = Column(Integer, primary_key=True, index=True)
    # user_id removed: can be derived via farm_id → farms → user_id
    farm_id = Column(Integer, ForeignKey("farms.id"), nullable=False)
    field_id = Column(Integer, ForeignKey("fields.id"))
    predicted_harvest_date = Column(DateTime(timezone=True))
    confidence_score = Column(Float)
    expected_yield = Column(Float)
    quality_prediction = Column(String(50))
    factors = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    farm = relationship("Farm")
    field = relationship("Field")

class Crop(Base):
    """Crop model - tracks crops grown on farms"""
    __tablename__ = "crops"

    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"), nullable=False)
    field_id = Column(Integer, ForeignKey("fields.id"))

    # Crop information
    crop_type = Column(String(100), nullable=False)  # e.g., "Maize", "Rice", "Tomato"
    variety = Column(String(100))  # Specific variety/cultivar
    quantity = Column(Float)  # Amount planted
    quantity_unit = Column(String(20))  # kg, bags, acres, etc.

    # Planting and growth tracking
    planting_date = Column(DateTime(timezone=True))
    expected_harvest_date = Column(DateTime(timezone=True))
    actual_harvest_date = Column(DateTime(timezone=True))
    growth_stage = Column(String(50))  # seedling, vegetative, flowering, fruiting, mature
    health_status = Column(String(50), default="healthy")  # healthy, stressed, diseased

    # Yield and production
    expected_yield = Column(Float)  # Expected production
    actual_yield = Column(Float)  # Actual production after harvest
    yield_unit = Column(String(20))  # kg, tons, bags, etc.

    # Additional details
    notes = Column(Text)  # Farmer's notes
    irrigation_method = Column(String(50))  # rain-fed, drip, sprinkler, etc.
    fertilizer_used = Column(JSON)  # Array of fertilizers applied
    pesticides_used = Column(JSON)  # Array of pesticides applied

    # Status
    is_active = Column(Boolean, default=True)  # Currently being grown
    is_harvested = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    farm = relationship("Farm", back_populates="crops")
    field = relationship("Field")

class Animal(Base):
    """Animal model - tracks livestock/animals on farms"""
    __tablename__ = "animals"

    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"), nullable=False)

    # Animal information
    animal_type = Column(String(100), nullable=False)  # cattle, goat, sheep, pig, chicken, etc.
    breed = Column(String(100))  # Specific breed
    quantity = Column(Integer, nullable=False)  # Number of animals

    # Identification
    tag_numbers = Column(JSON)  # Array of individual tag/ID numbers
    age_group = Column(String(50))  # young, adult, senior
    gender_distribution = Column(JSON)  # {"male": 10, "female": 20}

    # Health and care
    health_status = Column(String(50), default="healthy")  # healthy, sick, under_treatment
    vaccination_records = Column(JSON)  # Array of vaccination records
    last_health_checkup = Column(DateTime(timezone=True))
    veterinary_notes = Column(Text)

    # Production tracking (for productive animals)
    purpose = Column(String(50))  # meat, dairy, eggs, breeding, draft, etc.
    production_data = Column(JSON)  # {"milk_per_day": 5, "eggs_per_week": 20}

    # Housing and feeding
    housing_type = Column(String(50))  # free-range, pen, barn, coop, etc.
    feeding_type = Column(String(50))  # grazing, supplemented, intensive
    feed_consumption = Column(JSON)  # Feed requirements and consumption

    # Acquisition and status
    acquisition_date = Column(DateTime(timezone=True))
    acquisition_cost = Column(Float)
    current_value = Column(Float)
    is_active = Column(Boolean, default=True)  # Currently on farm

    # Additional details
    notes = Column(Text)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    farm = relationship("Farm", back_populates="animals")

class CropRecommendation(Base):
    """AI-powered crop recommendations based on geospatial data"""
    __tablename__ = "crop_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    farm_id = Column(Integer, ForeignKey("farms.id"), nullable=False)
    # user_id removed: can be derived via farm_id → farms → user_id

    # Recommendation details
    recommended_crop = Column(String(100), nullable=False)
    confidence_score = Column(Float)  # 0-1 confidence in recommendation
    suitability_score = Column(Float)  # 0-100 how suitable the crop is

    # Factors considered
    climate_factors = Column(JSON)  # Temperature, rainfall patterns
    soil_factors = Column(JSON)  # pH, nutrients, type
    geographic_factors = Column(JSON)  # Elevation, terrain
    market_factors = Column(JSON)  # Demand, prices, profitability

    # Recommendations and guidance
    planting_season = Column(String(100))  # Best time to plant
    expected_yield_range = Column(JSON)  # {"min": 1000, "max": 1500, "unit": "kg/acre"}
    water_requirements = Column(String(100))  # low, medium, high
    care_difficulty = Column(String(50))  # easy, moderate, difficult
    growth_duration_days = Column(Integer)  # Days to maturity

    # Economic analysis
    estimated_profit_margin = Column(Float)  # Estimated profit percentage
    market_demand = Column(String(50))  # low, medium, high
    selling_price_range = Column(JSON)  # Price range data

    # Additional information
    benefits = Column(JSON)  # Array of benefits for this crop
    challenges = Column(JSON)  # Array of potential challenges
    tips = Column(JSON)  # Array of cultivation tips
    alternative_crops = Column(JSON)  # Alternative crop suggestions

    # AI model metadata
    model_version = Column(String(50))
    recommendation_date = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    farm = relationship("Farm", back_populates="crop_recommendations")
