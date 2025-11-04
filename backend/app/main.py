"""
AgriFinSight Backend API
Main FastAPI application entry point
Uses configuration from app.config for all settings
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Import configuration
from app.config import settings

# Import routes
from app.routes import auth, analysis, recommendations, farms, crops
from app.database import engine
from app.models.database import Base

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="AI-powered farming assistant API",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    debug=settings.debug
)

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        Base.metadata.create_all(bind=engine)
        print(f"✓ Database tables created successfully")
        print(f"✓ Environment: {settings.app_env}")
        print(f"✓ Debug mode: {settings.debug}")
        print(f"✓ CORS origins: {settings.cors_origins_list}")
    except Exception as e:
        print(f"✗ Error creating database tables: {e}")

# CORS middleware - uses settings from .env
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure via settings if needed in production
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AgriFinSight API",
        "version": settings.app_version,
        "environment": settings.app_env,
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agrifinsight-api",
        "version": settings.app_version,
        "environment": settings.app_env
    }

# Include routers
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(recommendations.router)
app.include_router(farms.router)
app.include_router(crops.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
