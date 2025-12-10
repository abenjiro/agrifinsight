"""
Vercel Serverless Entry Point for AgriFinSight API
This file is required for Vercel to serve the FastAPI backend
"""

import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import the FastAPI app
from app.main import app

# Vercel expects the app to be named 'app' or exposed as a handler
handler = app
