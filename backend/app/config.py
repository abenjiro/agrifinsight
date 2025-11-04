"""
Configuration settings for AgriFinSight backend
Loads all settings from environment variables (.env file)
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings - all configurable via .env file"""

    # ============================================================================
    # Application Settings
    # ============================================================================
    app_name: str = "AgriFinSight API"
    app_version: str = "1.0.0"
    app_env: str = "development"  # development, staging, production
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS Origins (comma-separated in .env, converted to list)
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000"

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert CORS origins string to list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    # ============================================================================
    # Database Configuration
    # ============================================================================
    database_url: str = "postgresql://agrifinsight_user:your_secure_password@localhost:5432/agrifinsight"
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "agrifinsight"
    database_user: str = "agrifinsight_user"
    database_password: str = "your_secure_password"
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # ============================================================================
    # Redis Configuration
    # ============================================================================
    redis_url: str = "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    cache_ttl: int = 3600  # Cache time-to-live in seconds

    # ============================================================================
    # JWT Authentication
    # ============================================================================
    secret_key: str = "your-super-secret-key-change-in-production-minimum-32-chars"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # ============================================================================
    # File Storage
    # ============================================================================
    upload_dir: str = "uploads"
    max_file_size: int = 10485760  # 10MB in bytes
    allowed_file_types: str = '["image/jpeg", "image/png", "image/jpg", "image/webp"]'

    # AWS S3 (Optional)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_s3_bucket: Optional[str] = None
    aws_region: str = "us-east-1"

    # ============================================================================
    # External API Keys - Weather & Climate
    # ============================================================================
    openweather_api_key: Optional[str] = None
    openweather_api_url: str = "https://api.openweathermap.org/data/2.5"
    nasa_api_key: str = "DEMO_KEY"

    # ============================================================================
    # External API Keys - Satellite & Geospatial
    # ============================================================================
    sentinel_hub_client_id: Optional[str] = None
    sentinel_hub_client_secret: Optional[str] = None
    sentinel_hub_instance_id: Optional[str] = None
    gee_service_account: Optional[str] = None
    gee_private_key_path: Optional[str] = None
    planet_api_key: Optional[str] = None

    # ============================================================================
    # External API Keys - Soil & Elevation
    # ============================================================================
    soilgrids_api_url: str = "https://rest.isric.org/soilgrids/v2.0"
    elevation_api_url: str = "https://api.open-elevation.com/api/v1"

    # ============================================================================
    # External API Keys - Geocoding
    # ============================================================================
    nominatim_api_url: str = "https://nominatim.openstreetmap.org"
    nominatim_user_agent: str = "AgriFinSight/1.0"
    google_maps_api_key: Optional[str] = None

    # ============================================================================
    # AI/ML Configuration
    # ============================================================================
    model_path: str = "ai/models"
    crop_recommendation_model: str = "crop_recommendation_model.pth"
    disease_detection_model: str = "best_model.pth"
    confidence_threshold: float = 0.7
    use_gpu: bool = False
    model_batch_size: int = 32
    max_prediction_time: int = 30  # seconds

    # ============================================================================
    # Email Configuration
    # ============================================================================
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: str = "noreply@agrifinsight.com"
    smtp_tls: bool = True

    # ============================================================================
    # Monitoring & Logging
    # ============================================================================
    log_level: str = "INFO"
    log_file: str = "logs/agrifinsight.log"
    enable_sentry: bool = False
    sentry_dsn: Optional[str] = None

    # ============================================================================
    # API Rate Limiting
    # ============================================================================
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # ============================================================================
    # Feature Flags
    # ============================================================================
    enable_satellite_auto_fetch: bool = True
    enable_ml_recommendations: bool = True
    enable_email_notifications: bool = False
    enable_background_jobs: bool = False

    # ============================================================================
    # Development & Testing
    # ============================================================================
    testing: bool = False
    test_database_url: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = 'utf-8'

# Create settings instance
settings = Settings()
