"""
Authentication routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from passlib.context import CryptContext

from app.models.database import User
from app.config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def create_refresh_token(data: dict):
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

@router.post("/register")
async def register(
    email: str,
    password: str,
    phone: Optional[str] = None
):
    """Register a new user"""
    # TODO: Add database session dependency
    # TODO: Check if user already exists
    # TODO: Create new user
    # TODO: Return user data and tokens
    
    return {
        "message": "User registration endpoint - to be implemented",
        "email": email,
        "phone": phone
    }

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login user and return access token"""
    # TODO: Add database session dependency
    # TODO: Verify user credentials
    # TODO: Return access and refresh tokens
    
    return {
        "message": "User login endpoint - to be implemented",
        "username": form_data.username
    }

@router.post("/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token"""
    # TODO: Verify refresh token
    # TODO: Generate new access token
    
    return {
        "message": "Token refresh endpoint - to be implemented"
    }

@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """Logout user and invalidate token"""
    # TODO: Add token to blacklist
    # TODO: Return success message
    
    return {
        "message": "User logout endpoint - to be implemented"
    }

@router.get("/me")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user information"""
    # TODO: Verify token and return user data
    
    return {
        "message": "Current user endpoint - to be implemented"
    }
