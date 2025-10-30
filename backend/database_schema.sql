-- AgriFinSight Database Schema
-- PostgreSQL Database Schema for AgriFinSight Platform
-- Generated: 2025-10-30
-- Last Updated: 2025-10-30
-- Description: Complete database schema for agricultural analytics platform with geospatial capabilities
--
-- Key Features:
-- - Multi-farm management with geospatial data (coordinates, elevation, boundaries)
-- - Satellite imagery and NDVI (vegetation health) tracking
-- - Historical climate data (temperature, rainfall, patterns)
-- - Soil composition and analysis
-- - Crop and animal management
-- - AI-powered crop recommendations (ML + rule-based)
-- - Disease detection and analysis
-- - Weather forecasting integration
--
-- Data Sources:
-- - NASA POWER API: Climate and historical weather data
-- - OpenWeatherMap: Real-time weather and forecasts
-- - SoilGrids API: Soil composition and properties
-- - Open Elevation API: Terrain elevation data
-- - OpenStreetMap: Reverse geocoding and addresses
-- - PyTorch ML Model: Crop recommendations (49% accuracy, in training)

-- Drop existing tables (in reverse order of dependencies)
DROP TABLE IF EXISTS crop_recommendations CASCADE;
DROP TABLE IF EXISTS animals CASCADE;
DROP TABLE IF EXISTS crops CASCADE;
DROP TABLE IF EXISTS harvest_predictions CASCADE;
DROP TABLE IF EXISTS planting_recommendations CASCADE;
DROP TABLE IF EXISTS weather_data CASCADE;
DROP TABLE IF EXISTS analysis_results CASCADE;
DROP TABLE IF EXISTS crop_images CASCADE;
DROP TABLE IF EXISTS fields CASCADE;
DROP TABLE IF EXISTS farms CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS crop_types CASCADE;
DROP TABLE IF EXISTS alembic_version CASCADE;

-- ============================================================================
-- CROP TYPES TABLE (Master list of crop types)
-- ============================================================================
CREATE TABLE crop_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50),  -- grains, vegetables, fruits, legumes, cash_crops, tubers
    scientific_name VARCHAR(200),
    description TEXT,
    common_varieties TEXT,  -- Comma-separated list of common varieties
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for crop_types table
CREATE INDEX idx_crop_types_name ON crop_types(name);
CREATE INDEX idx_crop_types_category ON crop_types(category);

-- Insert default crop types
INSERT INTO crop_types (name, category, description) VALUES
('Maize', 'grains', 'Staple cereal crop widely grown for food and feed'),
('Rice', 'grains', 'Staple grain crop grown in wetland and dryland conditions'),
('Cassava', 'tubers', 'Drought-tolerant root crop, staple food in tropical regions'),
('Tomato', 'vegetables', 'High-value vegetable crop with year-round demand'),
('Soybean', 'legumes', 'Protein-rich legume crop with nitrogen-fixing properties'),
('Groundnut', 'legumes', 'Oilseed and protein crop, also called peanut'),
('Yam', 'tubers', 'Starchy tuber crop important in West African cuisine'),
('Plantain', 'fruits', 'Cooking banana variety, staple food crop'),
('Cocoa', 'cash_crops', 'Cash crop for chocolate production'),
('Coffee', 'cash_crops', 'Cash crop for beverage production'),
('Pepper', 'vegetables', 'Spice and vegetable crop with multiple varieties'),
('Onion', 'vegetables', 'Bulb vegetable crop used as seasoning'),
('Cabbage', 'vegetables', 'Leafy vegetable crop'),
('Carrot', 'vegetables', 'Root vegetable crop rich in beta-carotene'),
('Beans', 'legumes', 'Protein-rich legume with multiple varieties'),
('Peas', 'legumes', 'Legume crop eaten fresh or dried'),
('Okra', 'vegetables', 'Warm-season vegetable crop'),
('Garden Egg', 'vegetables', 'African eggplant variety'),
('Cucumber', 'vegetables', 'Vine crop grown for fresh consumption'),
('Watermelon', 'fruits', 'Large fruit crop with high water content'),
('Pineapple', 'fruits', 'Tropical fruit crop'),
('Mango', 'fruits', 'Tree fruit crop popular in tropical regions'),
('Orange', 'fruits', 'Citrus fruit crop'),
('Banana', 'fruits', 'Popular fruit crop eaten fresh'),
('Coconut', 'cash_crops', 'Tree crop with multiple uses'),
('Palm Oil', 'cash_crops', 'Oil palm for edible oil production'),
('Rubber', 'cash_crops', 'Tree crop for latex production'),
('Cashew', 'cash_crops', 'Tree crop for nuts and cashew apple'),
('Cotton', 'cash_crops', 'Fiber crop for textile industry'),
('Wheat', 'grains', 'Cereal crop for bread and flour'),
('Barley', 'grains', 'Cereal crop used for malting and feed'),
('Sorghum', 'grains', 'Drought-tolerant cereal crop'),
('Millet', 'grains', 'Small-seeded cereal crop'),
('Sweet Potato', 'tubers', 'Nutritious root crop'),
('Potato', 'tubers', 'Widely grown tuber crop'),
('Ginger', 'vegetables', 'Rhizome crop used as spice'),
('Garlic', 'vegetables', 'Bulb crop used as seasoning'),
('Sugarcane', 'cash_crops', 'Crop for sugar production'),
('Tea', 'cash_crops', 'Perennial crop for beverage production'),
('Tobacco', 'cash_crops', 'Cash crop for smoking products');

-- ============================================================================
-- USERS TABLE
-- ============================================================================
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'farmer',  -- farmer, analyst, admin
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for users table
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_phone ON users(phone);

-- ============================================================================
-- FARMS TABLE (with comprehensive geospatial data)
-- ============================================================================
CREATE TABLE farms (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,

    -- Location data
    address VARCHAR(500),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    altitude DOUBLE PRECISION,  -- meters above sea level
    boundary_coordinates JSONB,  -- GeoJSON polygon for farm boundaries

    -- Farm properties
    size DOUBLE PRECISION,  -- Size in acres/hectares
    size_unit VARCHAR(20) DEFAULT 'acres',  -- acres or hectares

    -- Soil and environmental data
    soil_type VARCHAR(100),  -- Primary soil type (clay, sandy, loam, silt, peat, chalky)
    soil_ph DOUBLE PRECISION,  -- Soil pH level
    soil_composition JSONB,  -- Detailed soil analysis {sand: %, clay: %, silt: %, organic_matter: %}
    terrain_type VARCHAR(100),  -- flat, hilly, mountainous, etc.
    elevation_profile JSONB,  -- Elevation variations across farm

    -- Climate and weather
    climate_zone VARCHAR(100),  -- tropical, temperate, arid, etc.
    avg_annual_rainfall DOUBLE PRECISION,  -- mm
    avg_temperature DOUBLE PRECISION,  -- Celsius
    water_sources JSONB,  -- Array of water sources {type: 'river/well/irrigation', location: coords}

    -- Historical and satellite data references
    last_satellite_image_date TIMESTAMP WITH TIME ZONE,
    satellite_image_url VARCHAR(500),  -- URL to latest satellite imagery
    ndvi_data JSONB,  -- Normalized Difference Vegetation Index data
    land_use_history JSONB,  -- Historical land use data

    -- Additional metadata
    timezone VARCHAR(50),  -- e.g., "Africa/Accra"
    country VARCHAR(100),
    region VARCHAR(100),  -- State/Province
    district VARCHAR(100),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for farms table
CREATE INDEX idx_farms_user_id ON farms(user_id);
CREATE INDEX idx_farms_latitude ON farms(latitude);
CREATE INDEX idx_farms_longitude ON farms(longitude);

-- ============================================================================
-- FIELDS TABLE
-- ============================================================================
CREATE TABLE fields (
    id SERIAL PRIMARY KEY,
    farm_id INTEGER NOT NULL REFERENCES farms(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    crop_type VARCHAR(100),
    planting_date TIMESTAMP WITH TIME ZONE,
    expected_harvest_date TIMESTAMP WITH TIME ZONE,
    coordinates VARCHAR(255),  -- GPS coordinates
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for fields table
CREATE INDEX idx_fields_farm_id ON fields(farm_id);

-- ============================================================================
-- CROP IMAGES TABLE
-- ============================================================================
CREATE TABLE crop_images (
    id SERIAL PRIMARY KEY,
    farm_id INTEGER NOT NULL REFERENCES farms(id) ON DELETE CASCADE,
    field_id INTEGER REFERENCES fields(id) ON DELETE SET NULL,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    image_url VARCHAR(500) NOT NULL,
    filename VARCHAR(255),
    file_size INTEGER,
    analysis_status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for crop_images table
CREATE INDEX idx_crop_images_farm_id ON crop_images(farm_id);
CREATE INDEX idx_crop_images_field_id ON crop_images(field_id);
CREATE INDEX idx_crop_images_user_id ON crop_images(user_id);
CREATE INDEX idx_crop_images_status ON crop_images(analysis_status);

-- ============================================================================
-- ANALYSIS RESULTS TABLE
-- ============================================================================
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES crop_images(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    disease_detected VARCHAR(100),
    confidence_score DOUBLE PRECISION,
    disease_type VARCHAR(100),
    severity VARCHAR(50),  -- low, medium, high
    recommendations TEXT,
    treatment_advice TEXT,
    growth_stage VARCHAR(50),
    health_score DOUBLE PRECISION,  -- 0-100
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for analysis_results table
CREATE INDEX idx_analysis_results_image_id ON analysis_results(image_id);
CREATE INDEX idx_analysis_results_user_id ON analysis_results(user_id);

-- ============================================================================
-- WEATHER DATA TABLE
-- ============================================================================
CREATE TABLE weather_data (
    id SERIAL PRIMARY KEY,
    location VARCHAR(255) NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    rainfall DOUBLE PRECISION,
    wind_speed DOUBLE PRECISION,
    wind_direction DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for weather_data table
CREATE INDEX idx_weather_data_location ON weather_data(location);
CREATE INDEX idx_weather_data_date ON weather_data(date);

-- ============================================================================
-- PLANTING RECOMMENDATIONS TABLE
-- ============================================================================
CREATE TABLE planting_recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    farm_id INTEGER NOT NULL REFERENCES farms(id) ON DELETE CASCADE,
    field_id INTEGER REFERENCES fields(id) ON DELETE SET NULL,
    crop_type VARCHAR(100) NOT NULL,
    recommended_planting_date TIMESTAMP WITH TIME ZONE,
    confidence_score DOUBLE PRECISION,
    weather_conditions JSONB,
    soil_conditions JSONB,
    risk_factors JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for planting_recommendations table
CREATE INDEX idx_planting_recommendations_user_id ON planting_recommendations(user_id);
CREATE INDEX idx_planting_recommendations_farm_id ON planting_recommendations(farm_id);
CREATE INDEX idx_planting_recommendations_field_id ON planting_recommendations(field_id);

-- ============================================================================
-- HARVEST PREDICTIONS TABLE
-- ============================================================================
CREATE TABLE harvest_predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    farm_id INTEGER NOT NULL REFERENCES farms(id) ON DELETE CASCADE,
    field_id INTEGER REFERENCES fields(id) ON DELETE SET NULL,
    predicted_harvest_date TIMESTAMP WITH TIME ZONE,
    confidence_score DOUBLE PRECISION,
    expected_yield DOUBLE PRECISION,
    quality_prediction VARCHAR(50),
    factors JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for harvest_predictions table
CREATE INDEX idx_harvest_predictions_user_id ON harvest_predictions(user_id);
CREATE INDEX idx_harvest_predictions_farm_id ON harvest_predictions(farm_id);
CREATE INDEX idx_harvest_predictions_field_id ON harvest_predictions(field_id);

-- ============================================================================
-- CROPS TABLE
-- ============================================================================
CREATE TABLE crops (
    id SERIAL PRIMARY KEY,
    farm_id INTEGER NOT NULL REFERENCES farms(id) ON DELETE CASCADE,
    field_id INTEGER REFERENCES fields(id) ON DELETE SET NULL,

    -- Crop information
    crop_type VARCHAR(100) NOT NULL,
    variety VARCHAR(100),
    quantity DOUBLE PRECISION,
    quantity_unit VARCHAR(20),

    -- Planting and growth tracking
    planting_date TIMESTAMP WITH TIME ZONE,
    expected_harvest_date TIMESTAMP WITH TIME ZONE,
    actual_harvest_date TIMESTAMP WITH TIME ZONE,
    growth_stage VARCHAR(50),  -- seedling, vegetative, flowering, fruiting, mature
    health_status VARCHAR(50) DEFAULT 'healthy',  -- healthy, stressed, diseased

    -- Yield and production
    expected_yield DOUBLE PRECISION,
    actual_yield DOUBLE PRECISION,
    yield_unit VARCHAR(20),

    -- Additional details
    notes TEXT,
    irrigation_method VARCHAR(50),  -- rain-fed, drip, sprinkler, etc.
    fertilizer_used JSONB,  -- Array of fertilizers applied
    pesticides_used JSONB,  -- Array of pesticides applied

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_harvested BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for crops table
CREATE INDEX idx_crops_farm_id ON crops(farm_id);
CREATE INDEX idx_crops_field_id ON crops(field_id);
CREATE INDEX idx_crops_crop_type ON crops(crop_type);
CREATE INDEX idx_crops_health_status ON crops(health_status);

-- ============================================================================
-- ANIMALS TABLE
-- ============================================================================
CREATE TABLE animals (
    id SERIAL PRIMARY KEY,
    farm_id INTEGER NOT NULL REFERENCES farms(id) ON DELETE CASCADE,

    -- Animal information
    animal_type VARCHAR(100) NOT NULL,  -- cattle, goat, sheep, pig, chicken, etc.
    breed VARCHAR(100),
    quantity INTEGER NOT NULL,

    -- Identification
    tag_numbers JSONB,  -- Array of individual tag/ID numbers
    age_group VARCHAR(50),  -- young, adult, senior
    gender_distribution JSONB,  -- {"male": 10, "female": 20}

    -- Health and care
    health_status VARCHAR(50) DEFAULT 'healthy',  -- healthy, sick, under_treatment
    vaccination_records JSONB,  -- Array of vaccination records
    last_health_checkup TIMESTAMP WITH TIME ZONE,
    veterinary_notes TEXT,

    -- Production tracking
    purpose VARCHAR(50),  -- meat, dairy, eggs, breeding, draft, etc.
    production_data JSONB,  -- {"milk_per_day": 5, "eggs_per_week": 20}

    -- Housing and feeding
    housing_type VARCHAR(50),  -- free-range, pen, barn, coop, etc.
    feeding_type VARCHAR(50),  -- grazing, supplemented, intensive
    feed_consumption JSONB,

    -- Acquisition and status
    acquisition_date TIMESTAMP WITH TIME ZONE,
    acquisition_cost DOUBLE PRECISION,
    current_value DOUBLE PRECISION,
    is_active BOOLEAN DEFAULT TRUE,

    -- Additional details
    notes TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for animals table
CREATE INDEX idx_animals_farm_id ON animals(farm_id);
CREATE INDEX idx_animals_animal_type ON animals(animal_type);
CREATE INDEX idx_animals_health_status ON animals(health_status);

-- ============================================================================
-- CROP RECOMMENDATIONS TABLE (AI-powered)
-- ============================================================================
CREATE TABLE crop_recommendations (
    id SERIAL PRIMARY KEY,
    farm_id INTEGER NOT NULL REFERENCES farms(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Recommendation details
    recommended_crop VARCHAR(100) NOT NULL,
    confidence_score DOUBLE PRECISION,  -- 0-1 confidence in recommendation
    suitability_score DOUBLE PRECISION,  -- 0-100 how suitable the crop is

    -- Factors considered
    climate_factors JSONB,  -- Temperature, rainfall patterns
    soil_factors JSONB,  -- pH, nutrients, type
    geographic_factors JSONB,  -- Elevation, terrain
    market_factors JSONB,  -- Demand, prices, profitability

    -- Recommendations and guidance
    planting_season VARCHAR(100),
    expected_yield_range JSONB,  -- {"min": 1000, "max": 1500, "unit": "kg/acre"}
    water_requirements VARCHAR(100),  -- low, medium, high
    care_difficulty VARCHAR(50),  -- easy, moderate, difficult
    growth_duration_days INTEGER,

    -- Economic analysis
    estimated_profit_margin DOUBLE PRECISION,
    market_demand VARCHAR(50),  -- low, medium, high
    selling_price_range JSONB,

    -- Additional information
    benefits JSONB,  -- Array of benefits
    challenges JSONB,  -- Array of challenges
    tips JSONB,  -- Array of tips
    alternative_crops JSONB,  -- Alternative crop suggestions

    -- AI model metadata
    model_version VARCHAR(50),
    prediction_method VARCHAR(50),  -- ml_model, rule_based, hybrid
    model_probability DOUBLE PRECISION,
    recommendation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for crop_recommendations table
CREATE INDEX idx_crop_recommendations_farm_id ON crop_recommendations(farm_id);
CREATE INDEX idx_crop_recommendations_user_id ON crop_recommendations(user_id);
CREATE INDEX idx_crop_recommendations_crop ON crop_recommendations(recommended_crop);
CREATE INDEX idx_crop_recommendations_date ON crop_recommendations(recommendation_date);

-- ============================================================================
-- ALEMBIC VERSION TABLE (for migration tracking)
-- ============================================================================
CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Insert current migration version
INSERT INTO alembic_version (version_num) VALUES ('6a0b20a77723');

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE crop_types IS 'Master list of all crop types available in the system';
COMMENT ON TABLE users IS 'User accounts with role-based access (farmer, analyst, admin)';
COMMENT ON TABLE farms IS 'Farms with comprehensive geospatial, environmental, and climate data';
COMMENT ON TABLE fields IS 'Individual fields within farms for crop-specific tracking';
COMMENT ON TABLE crop_images IS 'Uploaded crop images for disease detection and analysis';
COMMENT ON TABLE analysis_results IS 'AI-powered analysis results for crop health and diseases';
COMMENT ON TABLE weather_data IS 'Historical and current weather data for locations';
COMMENT ON TABLE planting_recommendations IS 'AI-generated planting recommendations based on conditions';
COMMENT ON TABLE harvest_predictions IS 'Predictive analytics for harvest timing and yield';
COMMENT ON TABLE crops IS 'Crops grown on farms with tracking for growth, health, and yield';
COMMENT ON TABLE animals IS 'Livestock and animals on farms with health and production tracking';
COMMENT ON TABLE crop_recommendations IS 'AI/ML-powered crop recommendations based on geospatial and environmental data';

-- Column comments for farms table
COMMENT ON COLUMN farms.latitude IS 'Latitude in decimal degrees';
COMMENT ON COLUMN farms.longitude IS 'Longitude in decimal degrees';
COMMENT ON COLUMN farms.altitude IS 'Elevation in meters above sea level';
COMMENT ON COLUMN farms.boundary_coordinates IS 'GeoJSON polygon defining farm boundaries';
COMMENT ON COLUMN farms.soil_composition IS 'JSON object with detailed soil composition percentages';
COMMENT ON COLUMN farms.ndvi_data IS 'Normalized Difference Vegetation Index for crop health monitoring';
COMMENT ON COLUMN farms.climate_zone IS 'Köppen climate classification zone';

-- Column comments for crops table
COMMENT ON COLUMN crops.crop_type IS 'Type of crop (e.g., Maize, Rice, Tomato)';
COMMENT ON COLUMN crops.growth_stage IS 'Current growth stage: seedling, vegetative, flowering, fruiting, mature';
COMMENT ON COLUMN crops.health_status IS 'Crop health status: healthy, stressed, diseased';
COMMENT ON COLUMN crops.irrigation_method IS 'Irrigation method used: rain-fed, drip, sprinkler, flood, furrow, manual';
COMMENT ON COLUMN crops.fertilizer_used IS 'JSON array of fertilizers applied with dates and quantities';
COMMENT ON COLUMN crops.pesticides_used IS 'JSON array of pesticides/herbicides applied';

-- Column comments for animals table
COMMENT ON COLUMN animals.animal_type IS 'Type of animal: cattle, goat, sheep, pig, chicken, duck, turkey, etc.';
COMMENT ON COLUMN animals.gender_distribution IS 'JSON object with male/female count distribution';
COMMENT ON COLUMN animals.vaccination_records IS 'JSON array of vaccination history with dates and types';
COMMENT ON COLUMN animals.production_data IS 'JSON object tracking production metrics (milk, eggs, etc.)';
COMMENT ON COLUMN animals.feed_consumption IS 'JSON object with feed types and consumption rates';

-- Column comments for crop_recommendations table
COMMENT ON COLUMN crop_recommendations.confidence_score IS 'ML model confidence in recommendation (0-1)';
COMMENT ON COLUMN crop_recommendations.suitability_score IS 'Overall suitability score for the crop (0-100)';
COMMENT ON COLUMN crop_recommendations.model_version IS 'Version identifier of the ML model used (e.g., ml_v1.0)';
COMMENT ON COLUMN crop_recommendations.climate_factors IS 'JSON with temperature, rainfall, and climate zone data';
COMMENT ON COLUMN crop_recommendations.soil_factors IS 'JSON with pH, type, and composition data';
COMMENT ON COLUMN crop_recommendations.geographic_factors IS 'JSON with elevation, terrain, and location data';
COMMENT ON COLUMN crop_recommendations.market_factors IS 'JSON with demand, pricing, and profitability data';
COMMENT ON COLUMN crop_recommendations.benefits IS 'JSON array of crop benefits and advantages';
COMMENT ON COLUMN crop_recommendations.challenges IS 'JSON array of potential challenges and risks';
COMMENT ON COLUMN crop_recommendations.tips IS 'JSON array of cultivation tips and best practices';
COMMENT ON COLUMN crop_recommendations.alternative_crops IS 'JSON array of alternative crop suggestions';

-- ============================================================================
-- API ENDPOINTS AND FEATURES (Backend Integration)
-- ============================================================================
--
-- Farm Management:
-- - POST   /api/farms/                        Create farm with auto satellite data fetch
-- - GET    /api/farms/                        Get all user farms
-- - GET    /api/farms/{id}                    Get specific farm details
-- - PUT    /api/farms/{id}                    Update farm information
-- - DELETE /api/farms/{id}                    Delete farm
-- - GET    /api/farms/enrich-location         Get enriched location data (NDVI, climate, soil)
-- - POST   /api/farms/{id}/refresh-satellite  Refresh satellite/NDVI data for monitoring
-- - GET    /api/farms/{id}/weather            Get current weather and forecast
--
-- Crop Management:
-- - GET    /api/crops/types                   Get all crop types from master list
-- - POST   /api/crops/types                   Add new crop type (admin)
-- - GET    /api/farms/{id}/crops              Get all crops for a farm
-- - POST   /api/farms/{id}/crops              Add new crop (with smart predictions)
-- - GET    /api/crops/{id}                    Get specific crop details
-- - PUT    /api/crops/{id}                    Update crop information
-- - DELETE /api/crops/{id}                    Delete crop
--
-- Animal Management:
-- - GET    /api/farms/{id}/animals            Get all animals for a farm
-- - POST   /api/farms/{id}/animals            Add animals to farm
-- - GET    /api/animals/{id}                  Get specific animal group details
-- - PUT    /api/animals/{id}                  Update animal information
-- - DELETE /api/animals/{id}                  Delete animal record
--
-- AI Crop Recommendations:
-- - POST   /api/farms/{id}/crop-recommendations    Generate AI crop recommendations (ML + rules)
-- - GET    /api/farms/{id}/crop-recommendations    Get saved recommendations
--
-- Authentication:
-- - POST   /api/auth/register                 User registration
-- - POST   /api/auth/login                    User login (JWT token)
-- - POST   /api/auth/refresh                  Refresh access token
-- - POST   /api/auth/logout                   User logout
--
-- Auto-Enrichment Features:
-- - Satellite NDVI data automatically fetched and saved on farm creation
-- - Historical climate data (1 year) from NASA POWER API
-- - Soil composition from SoilGrids API
-- - Reverse geocoding for address, country, region
-- - Smart crop predictions (harvest dates, yields, irrigation)
--
-- Machine Learning:
-- - PyTorch crop recommendation model (6 crops: Maize, Rice, Cassava, Tomato, Soybean, Groundnut)
-- - Test accuracy: 49.44% (in training, improving)
-- - Fallback to rule-based recommendations if ML unavailable
-- - Auto-calculates harvest dates based on growth duration
-- - Predicts expected yields with confidence scores

-- ============================================================================
-- GRANT PERMISSIONS (adjust username as needed)
-- ============================================================================

-- Example: GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO agrifinsight_user;
-- Example: GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO agrifinsight_user;

-- ============================================================================
-- MAINTENANCE AND MONITORING
-- ============================================================================
--
-- Periodic Tasks (Recommended):
-- 1. Refresh satellite data weekly for active farms:
--    - Call POST /api/farms/{id}/refresh-satellite for each farm
-- 2. Update crop recommendations seasonally (every 3 months)
-- 3. Archive old crop_recommendations older than 1 year
-- 4. Backup ndvi_data for historical trend analysis
--
-- Database Optimization:
-- - Add indexes on frequently queried JSON fields
-- - Partition large tables by created_at for performance
-- - Regular VACUUM ANALYZE for PostgreSQL optimization
--
-- Monitoring Queries:
--
-- Check farms with outdated satellite data (>30 days):
-- SELECT id, name, last_satellite_image_date
-- FROM farms
-- WHERE last_satellite_image_date < NOW() - INTERVAL '30 days'
-- OR last_satellite_image_date IS NULL;
--
-- Get farms with low vegetation health (NDVI < 0.3):
-- SELECT id, name, ndvi_data->>'ndvi_value' as ndvi, ndvi_data->>'ndvi_interpretation' as health
-- FROM farms
-- WHERE (ndvi_data->>'ndvi_value')::float < 0.3;
--
-- Count active crops by type:
-- SELECT crop_type, COUNT(*) as count, SUM(expected_yield) as total_expected_yield
-- FROM crops
-- WHERE is_active = true
-- GROUP BY crop_type
-- ORDER BY count DESC;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- Run this command to verify all tables were created:
-- \dt

-- Run this to see table details:
-- \d+ farms

-- Run this to check data sources integration:
-- SELECT
--   COUNT(*) as total_farms,
--   COUNT(ndvi_data) as farms_with_satellite,
--   COUNT(avg_annual_rainfall) as farms_with_climate,
--   COUNT(soil_composition) as farms_with_soil
-- FROM farms;
