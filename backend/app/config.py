"""
Configuration settings for AgriFinSight backend
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "AgriFinSight API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "postgresql://user:password@localhost/agrifinsight"
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "agrifinsight"
    database_user: str = "user"
    database_password: str = "password"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    
    # JWT
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # File Storage
    upload_dir: str = "uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list = ["image/jpeg", "image/png", "image/jpg"]
    
    # External APIs
    weather_api_key: Optional[str] = None
    weather_api_url: str = "https://api.openweathermap.org/data/2.5"
    
    # AI/ML
    model_path: str = "ai/models"
    confidence_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()
