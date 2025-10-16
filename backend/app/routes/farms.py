"""
Farm management routes
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.models.database import User, Farm, Field
from app.database import get_db
from app.routes.auth import get_current_user_from_token
from app.schemas import FarmCreate, FarmUpdate, Farm as FarmSchema, FieldCreate, Field as FieldSchema
from app.services.geospatial_service import geospatial_service

router = APIRouter(prefix="/farms", tags=["farm management"])

@router.get("/", response_model=List[FarmSchema])
async def get_farms(
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get all farms for current user"""
    farms = db.query(Farm).filter(Farm.user_id == current_user.id).all()
    return farms

@router.post("/", response_model=FarmSchema, status_code=status.HTTP_201_CREATED)
async def create_farm(
    farm_data: FarmCreate,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Create a new farm with comprehensive geospatial data"""
    new_farm = Farm(
        user_id=current_user.id,
        name=farm_data.name,
        address=farm_data.address,
        latitude=farm_data.latitude,
        longitude=farm_data.longitude,
        altitude=farm_data.altitude,
        boundary_coordinates=farm_data.boundary_coordinates,
        size=farm_data.size,
        size_unit=farm_data.size_unit or "acres",
        soil_type=farm_data.soil_type,
        soil_ph=farm_data.soil_ph,
        soil_composition=farm_data.soil_composition,
        terrain_type=farm_data.terrain_type,
        elevation_profile=farm_data.elevation_profile,
        climate_zone=farm_data.climate_zone,
        avg_annual_rainfall=farm_data.avg_annual_rainfall,
        avg_temperature=farm_data.avg_temperature,
        water_sources=farm_data.water_sources,
        timezone=farm_data.timezone,
        country=farm_data.country,
        region=farm_data.region,
        district=farm_data.district
    )

    db.add(new_farm)
    db.commit()
    db.refresh(new_farm)

    return new_farm

@router.get("/{farm_id}", response_model=FarmSchema)
async def get_farm(
    farm_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get specific farm details"""
    farm = db.query(Farm).filter(
        Farm.id == farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    return farm

@router.put("/{farm_id}", response_model=FarmSchema)
async def update_farm(
    farm_id: int,
    farm_data: FarmUpdate,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Update farm information"""
    farm = db.query(Farm).filter(
        Farm.id == farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    # Update fields if provided
    update_data = farm_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(farm, field, value)

    db.commit()
    db.refresh(farm)

    return farm

@router.delete("/{farm_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_farm(
    farm_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Delete a farm"""
    farm = db.query(Farm).filter(
        Farm.id == farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    db.delete(farm)
    db.commit()

    return None

@router.post("/enrich-location")
async def enrich_farm_location(
    latitude: float,
    longitude: float,
    current_user: User = Depends(get_current_user_from_token)
):
    """
    Enrich farm location with weather, soil, elevation, and geographic data
    """
    enriched_data = await geospatial_service.enrich_farm_location(latitude, longitude)
    return enriched_data

@router.get("/{farm_id}/weather")
async def get_farm_weather(
    farm_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get current weather and forecast for a farm"""
    farm = db.query(Farm).filter(
        Farm.id == farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    if not farm.latitude or not farm.longitude:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Farm coordinates not set"
        )

    weather_data = await geospatial_service.get_weather_data(farm.latitude, farm.longitude)
    return weather_data

@router.get("/{farm_id}/fields")
async def get_farm_fields(farm_id: int):
    """Get all fields for a farm"""
    
    # TODO: Add authentication dependency
    # TODO: Verify farm ownership
    # TODO: Query database for farm fields
    # TODO: Return field list
    
    return {
        "message": "Get farm fields endpoint - to be implemented",
        "farm_id": farm_id,
        "fields": []
    }

@router.post("/{farm_id}/fields")
async def create_field(
    farm_id: int,
    name: str,
    crop_type: Optional[str] = None,
    planting_date: Optional[datetime] = None,
    coordinates: Optional[str] = None
):
    """Create a new field in a farm"""
    
    # TODO: Add authentication dependency
    # TODO: Verify farm ownership
    # TODO: Create field in database
    # TODO: Return created field data
    
    return {
        "message": "Create field endpoint - to be implemented",
        "farm_id": farm_id,
        "field": {
            "name": name,
            "crop_type": crop_type,
            "planting_date": planting_date,
            "coordinates": coordinates
        }
    }

@router.get("/{farm_id}/fields/{field_id}")
async def get_field(farm_id: int, field_id: int):
    """Get specific field details"""
    
    # TODO: Add authentication dependency
    # TODO: Verify farm and field ownership
    # TODO: Query database for field details
    # TODO: Return field information
    
    return {
        "message": "Get field endpoint - to be implemented",
        "farm_id": farm_id,
        "field_id": field_id
    }

@router.put("/{farm_id}/fields/{field_id}")
async def update_field(
    farm_id: int,
    field_id: int,
    name: Optional[str] = None,
    crop_type: Optional[str] = None,
    planting_date: Optional[datetime] = None,
    coordinates: Optional[str] = None
):
    """Update field information"""
    
    # TODO: Add authentication dependency
    # TODO: Verify farm and field ownership
    # TODO: Update field in database
    # TODO: Return updated field data
    
    return {
        "message": "Update field endpoint - to be implemented",
        "farm_id": farm_id,
        "field_id": field_id
    }

@router.delete("/{farm_id}/fields/{field_id}")
async def delete_field(farm_id: int, field_id: int):
    """Delete a field"""
    
    # TODO: Add authentication dependency
    # TODO: Verify farm and field ownership
    # TODO: Delete field and related data
    # TODO: Return success message
    
    return {
        "message": "Delete field endpoint - to be implemented",
        "farm_id": farm_id,
        "field_id": field_id
    }
