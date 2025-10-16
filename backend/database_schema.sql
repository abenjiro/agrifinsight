-- AgriFinSight Database Schema
-- PostgreSQL Database Schema for AgriFinSight Platform
-- Generated: 2025-10-16
-- Description: Complete database schema for agricultural analytics platform with geospatial capabilities

-- Drop existing tables (in reverse order of dependencies)
DROP TABLE IF EXISTS harvest_predictions CASCADE;
DROP TABLE IF EXISTS planting_recommendations CASCADE;
DROP TABLE IF EXISTS weather_data CASCADE;
DROP TABLE IF EXISTS analysis_results CASCADE;
DROP TABLE IF EXISTS crop_images CASCADE;
DROP TABLE IF EXISTS fields CASCADE;
DROP TABLE IF EXISTS farms CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS alembic_version CASCADE;

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

COMMENT ON TABLE users IS 'User accounts with role-based access (farmer, analyst, admin)';
COMMENT ON TABLE farms IS 'Farms with comprehensive geospatial, environmental, and climate data';
COMMENT ON TABLE fields IS 'Individual fields within farms for crop-specific tracking';
COMMENT ON TABLE crop_images IS 'Uploaded crop images for disease detection and analysis';
COMMENT ON TABLE analysis_results IS 'AI-powered analysis results for crop health and diseases';
COMMENT ON TABLE weather_data IS 'Historical and current weather data for locations';
COMMENT ON TABLE planting_recommendations IS 'AI-generated planting recommendations based on conditions';
COMMENT ON TABLE harvest_predictions IS 'Predictive analytics for harvest timing and yield';

-- Column comments for farms table
COMMENT ON COLUMN farms.latitude IS 'Latitude in decimal degrees';
COMMENT ON COLUMN farms.longitude IS 'Longitude in decimal degrees';
COMMENT ON COLUMN farms.altitude IS 'Elevation in meters above sea level';
COMMENT ON COLUMN farms.boundary_coordinates IS 'GeoJSON polygon defining farm boundaries';
COMMENT ON COLUMN farms.soil_composition IS 'JSON object with detailed soil composition percentages';
COMMENT ON COLUMN farms.ndvi_data IS 'Normalized Difference Vegetation Index for crop health monitoring';
COMMENT ON COLUMN farms.climate_zone IS 'Köppen climate classification zone';

-- ============================================================================
-- GRANT PERMISSIONS (adjust username as needed)
-- ============================================================================

-- Example: GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO agrifinsight_user;
-- Example: GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO agrifinsight_user;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- Run this command to verify all tables were created:
-- \dt

-- Run this to see table details:
-- \d+ farms
