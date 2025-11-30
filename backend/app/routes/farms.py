"""
Farm management routes
"""

from fastapi import APIRouter, HTTPException, Depends, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.models.database import User, Farm, Field
from app.database import get_db
from app.routes.auth import get_current_user_from_token
from app.schemas import FarmCreate, FarmUpdate, Farm as FarmSchema, FieldCreate, Field as FieldSchema
from app.services.geospatial_service import geospatial_service

router = APIRouter(prefix="/api/farms", tags=["farm management"])

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

    # Auto-enrich with environmental data if coordinates provided
    enrichment = {}
    if farm_data.latitude and farm_data.longitude:
        try:
            # Fetch comprehensive environmental data
            enrichment = await geospatial_service.enrich_farm_location(
                farm_data.latitude,
                farm_data.longitude,
                include_satellite=True
            )

            print(f"✅ Auto-enriched farm location with: {list(enrichment.keys())}")

        except Exception as e:
            # Don't fail farm creation if enrichment fails
            print(f"⚠️ Warning: Could not auto-enrich farm data: {e}")
            import traceback
            traceback.print_exc()

    # Prepare soil data from enrichment
    soil_data = enrichment.get("soil_data", {})
    soil_composition_data = soil_data.get("composition", {})

    # Determine soil type from composition (simplified logic)
    soil_type_auto = None
    if soil_composition_data:
        clay = soil_composition_data.get("clay", 0)
        sand = soil_composition_data.get("sand", 0)
        silt = soil_composition_data.get("silt", 0)

        if clay > 40:
            soil_type_auto = "Clay"
        elif sand > 60:
            soil_type_auto = "Sandy"
        elif silt > 40:
            soil_type_auto = "Silty"
        elif clay > 20 and sand > 40:
            soil_type_auto = "Sandy Loam"
        elif clay > 20 and silt > 40:
            soil_type_auto = "Silty Clay"
        else:
            soil_type_auto = "Loam"

    # Extract soil pH from composition data (phh2o)
    soil_ph_auto = soil_composition_data.get("phh2o")
    if soil_ph_auto:
        soil_ph_auto = soil_ph_auto / 10  # SoilGrids returns pH * 10

    new_farm = Farm(
        user_id=current_user.id,
        name=farm_data.name,
        # Location data - use provided or auto-enriched
        address=farm_data.address or enrichment.get("address"),
        latitude=farm_data.latitude,
        longitude=farm_data.longitude,
        altitude=farm_data.altitude or enrichment.get("altitude"),
        boundary_coordinates=farm_data.boundary_coordinates,
        size=farm_data.size,
        size_unit=farm_data.size_unit or "acres",
        # Soil data - use provided or auto-enriched
        soil_type=farm_data.soil_type or soil_type_auto,
        soil_ph=farm_data.soil_ph or soil_ph_auto,
        soil_composition=farm_data.soil_composition or soil_composition_data,
        terrain_type=farm_data.terrain_type,
        elevation_profile=farm_data.elevation_profile,
        # Climate data - use provided or auto-enriched
        climate_zone=farm_data.climate_zone or enrichment.get("climate_zone"),
        avg_annual_rainfall=farm_data.avg_annual_rainfall or enrichment.get("avg_annual_rainfall"),
        avg_temperature=farm_data.avg_temperature or enrichment.get("avg_temperature"),
        water_sources=farm_data.water_sources,
        # Geographic data - use provided or auto-enriched
        timezone=farm_data.timezone or enrichment.get("timezone"),
        country=farm_data.country or enrichment.get("country"),
        region=farm_data.region or enrichment.get("region"),
        district=farm_data.district or enrichment.get("district"),
        # Satellite data - from enrichment
        ndvi_data=enrichment.get("vegetation_health"),
        last_satellite_image_date=datetime.now() if enrichment.get("vegetation_health") else None
    )

    db.add(new_farm)
    db.commit()
    db.refresh(new_farm)

    return new_farm

@router.get("/enrich-location")
async def enrich_farm_location(
    current_user: User = Depends(get_current_user_from_token),
    latitude: float = Query(..., description="Latitude coordinate"),
    longitude: float = Query(..., description="Longitude coordinate")
):
    """
    Enrich farm location with weather, soil, elevation, and geographic data
    """
    enriched_data = await geospatial_service.enrich_farm_location(latitude, longitude)
    return enriched_data

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

@router.post("/{farm_id}/refresh-satellite")
async def refresh_farm_satellite_data(
    farm_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """
    Refresh satellite and NDVI data for a farm
    Useful for monitoring vegetation health over time
    """
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

    try:
        # Fetch latest satellite and NDVI data
        satellite_data = await geospatial_service.get_satellite_imagery(
            farm.latitude,
            farm.longitude
        )

        # Update farm with new satellite data
        if satellite_data.get("ndvi_data"):
            farm.ndvi_data = satellite_data["ndvi_data"]
            farm.last_satellite_image_date = datetime.now()

            db.commit()
            db.refresh(farm)

            return {
                "message": "Satellite data refreshed successfully",
                "ndvi_data": farm.ndvi_data,
                "last_update": farm.last_satellite_image_date,
                "recommendations": satellite_data.get("recommendations", [])
            }
        else:
            return {
                "message": "Satellite data unavailable",
                "error": "Could not fetch NDVI data"
            }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh satellite data: {str(e)}"
        )

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
