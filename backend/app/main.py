"""
AgriFinSight Backend API
Main FastAPI application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import os
from dotenv import load_dotenv

# Import routes
from app.routes import auth, analysis, recommendations, farms

# Load environment variables
load_dotenv()

# Create FastAPI application
app = FastAPI(
    title="AgriFinSight API",
    description="AI-powered farming assistant API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AgriFinSight API",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agrifinsight-api"
    }

# Include routers
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(recommendations.router)
app.include_router(farms.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
