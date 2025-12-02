"""
Crop and Animal management routes
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.models.database import User, Farm, Crop, Animal, CropRecommendation, CropType
from app.database import get_db
from app.routes.auth import get_current_user_from_token
from app.schemas import (
    CropCreate, CropUpdate, Crop as CropSchema,
    AnimalCreate, AnimalUpdate, Animal as AnimalSchema,
    CropRecommendation as CropRecommendationSchema
)
from app.services.crop_recommendation_service import crop_recommendation_service
from app.services.ml_crop_recommendation_service import ml_crop_recommendation_service
from app.services.db_service import (
    crop_type_service, farm_service, crop_service as crop_db_service,
    animal_service as animal_db_service, crop_recommendation_service_db
)

router = APIRouter(prefix="/api", tags=["crops and animals"])


# ============= CROP ENDPOINTS =============

@router.get("/crops/types")
async def get_crop_types(
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get list of crop types with prediction data from the crop_types master table"""
    crop_types = crop_type_service.get_active_crop_types(db)
    return {
        "crop_types": [
            {
                "name": ct.name,
                "category": ct.category,
                "scientific_name": ct.scientific_name,
                "description": ct.description,
                "growth_duration_days": ct.growth_duration_days,
                "water_requirement": ct.water_requirement,
                "recommended_irrigation": ct.recommended_irrigation,
                "min_yield_per_acre": ct.min_yield_per_acre,
                "max_yield_per_acre": ct.max_yield_per_acre,
                "avg_yield_per_acre": ct.avg_yield_per_acre,
                "yield_unit": ct.yield_unit
            }
            for ct in crop_types
        ]
    }


@router.post("/crops/types")
async def add_crop_type(
    name: str,
    category: Optional[str] = None,
    description: Optional[str] = None,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Add a new crop type to the master list (admin only)"""
    # Check if crop type already exists
    existing = crop_type_service.find_by_name(db, name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Crop type already exists"
        )

    # Create new crop type
    new_crop_type = crop_type_service.create_crop_type(db, name, category, description)
    return {"message": "Crop type added successfully", "crop_type": new_crop_type.name}


@router.get("/farms/{farm_id}/crops", response_model=List[CropSchema])
async def get_farm_crops(
    farm_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get all crops for a specific farm"""
    # Verify farm ownership
    farm = farm_service.get_user_farm(db, farm_id, current_user.id)
    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    crops = crop_db_service.get_farm_crops(db, farm_id)
    return crops


@router.post("/farms/{farm_id}/crops", response_model=CropSchema, status_code=status.HTTP_201_CREATED)
async def create_crop(
    farm_id: int,
    crop_data: CropCreate,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Add a new crop to a farm"""
    # Verify farm ownership
    farm = farm_service.get_user_farm(db, farm_id, current_user.id)
    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    # Ensure farm_id matches
    if crop_data.farm_id != farm_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Farm ID mismatch"
        )

    new_crop = crop_db_service.create_crop(db, crop_data.model_dump())
    return new_crop


@router.get("/crops/{crop_id}", response_model=CropSchema)
async def get_crop(
    crop_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get specific crop details"""
    crop = crop_db_service.get_crop_by_id(db, crop_id)
    if not crop:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Crop not found"
        )

    # Verify ownership through farm
    farm = farm_service.get_user_farm(db, crop.farm_id, current_user.id)
    if not farm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return crop


@router.put("/crops/{crop_id}", response_model=CropSchema)
async def update_crop(
    crop_id: int,
    crop_data: CropUpdate,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Update crop information"""
    crop = db.query(Crop).filter(Crop.id == crop_id).first()

    if not crop:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Crop not found"
        )

    # Verify ownership
    farm = db.query(Farm).filter(
        Farm.id == crop.farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Update fields
    update_data = crop_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(crop, field, value)

    db.commit()
    db.refresh(crop)

    return crop


@router.delete("/crops/{crop_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_crop(
    crop_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Delete a crop"""
    crop = db.query(Crop).filter(Crop.id == crop_id).first()

    if not crop:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Crop not found"
        )

    # Verify ownership
    farm = db.query(Farm).filter(
        Farm.id == crop.farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    db.delete(crop)
    db.commit()

    return None


# ============= ANIMAL ENDPOINTS =============

@router.get("/farms/{farm_id}/animals", response_model=List[AnimalSchema])
async def get_farm_animals(
    farm_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get all animals for a specific farm"""
    # Verify farm ownership
    farm = db.query(Farm).filter(
        Farm.id == farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    animals = db.query(Animal).filter(Animal.farm_id == farm_id).all()
    return animals


@router.post("/farms/{farm_id}/animals", response_model=AnimalSchema, status_code=status.HTTP_201_CREATED)
async def create_animal(
    farm_id: int,
    animal_data: AnimalCreate,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Add animals to a farm"""
    # Verify farm ownership
    farm = db.query(Farm).filter(
        Farm.id == farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    # Ensure farm_id matches
    if animal_data.farm_id != farm_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Farm ID mismatch"
        )

    new_animal = Animal(**animal_data.model_dump())
    db.add(new_animal)
    db.commit()
    db.refresh(new_animal)

    return new_animal


@router.get("/animals/{animal_id}", response_model=AnimalSchema)
async def get_animal(
    animal_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get specific animal group details"""
    animal = db.query(Animal).filter(Animal.id == animal_id).first()

    if not animal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Animal record not found"
        )

    # Verify ownership through farm
    farm = db.query(Farm).filter(
        Farm.id == animal.farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return animal


@router.put("/animals/{animal_id}", response_model=AnimalSchema)
async def update_animal(
    animal_id: int,
    animal_data: AnimalUpdate,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Update animal information"""
    animal = db.query(Animal).filter(Animal.id == animal_id).first()

    if not animal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Animal record not found"
        )

    # Verify ownership
    farm = db.query(Farm).filter(
        Farm.id == animal.farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    # Update fields
    update_data = animal_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(animal, field, value)

    db.commit()
    db.refresh(animal)

    return animal


@router.delete("/animals/{animal_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_animal(
    animal_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Delete an animal record"""
    animal = db.query(Animal).filter(Animal.id == animal_id).first()

    if not animal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Animal record not found"
        )

    # Verify ownership
    farm = db.query(Farm).filter(
        Farm.id == animal.farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    db.delete(animal)
    db.commit()

    return None


# ============= CROP RECOMMENDATION ENDPOINTS =============

@router.post("/farms/{farm_id}/crop-recommendations", response_model=List[CropRecommendationSchema])
async def generate_crop_recommendations(
    farm_id: int,
    use_ml: bool = True,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """
    Generate AI-powered crop recommendations for a farm

    Parameters:
    - use_ml: If True, use ML model (default). If False, use rule-based system
    """
    # Get farm data
    farm = db.query(Farm).filter(
        Farm.id == farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    # Prepare farm data for recommendation
    farm_data = {
        "latitude": farm.latitude,
        "longitude": farm.longitude,
        "altitude": farm.altitude,
        "climate_zone": farm.climate_zone,
        "avg_temperature": farm.avg_temperature,
        "avg_annual_rainfall": farm.avg_annual_rainfall,
        "soil_type": farm.soil_type,
        "soil_ph": farm.soil_ph,
        "terrain_type": farm.terrain_type,
        "soil_composition": farm.soil_composition
    }

    # Generate recommendations using ML or rule-based approach
    if use_ml:
        recommendations = ml_crop_recommendation_service.generate_recommendations(farm_data)

        # Fallback to rule-based if ML fails
        if not recommendations:
            recommendations = crop_recommendation_service.generate_recommendations(farm_data)
    else:
        recommendations = crop_recommendation_service.generate_recommendations(farm_data)

    # Delete old recommendations for this farm to avoid duplicates
    db.query(CropRecommendation).filter(
        CropRecommendation.farm_id == farm_id
    ).delete()

    # Save new recommendations to database
    saved_recommendations = []

    # Valid fields for CropRecommendation model
    valid_fields = {
        'recommended_crop', 'confidence_score', 'suitability_score',
        'climate_factors', 'soil_factors', 'geographic_factors', 'market_factors',
        'planting_season', 'expected_yield_range', 'water_requirements',
        'care_difficulty', 'growth_duration_days', 'estimated_profit_margin',
        'market_demand', 'selling_price_range', 'benefits', 'challenges',
        'tips', 'alternative_crops', 'model_version'
    }

    for rec in recommendations:
        # Filter to only include valid database fields
        filtered_rec = {k: v for k, v in rec.items() if k in valid_fields}

        crop_rec = CropRecommendation(
            farm_id=farm_id,
            **filtered_rec
        )
        db.add(crop_rec)
        saved_recommendations.append(crop_rec)

    db.commit()

    # Refresh all saved recommendations
    for rec in saved_recommendations:
        db.refresh(rec)

    return saved_recommendations


@router.get("/farms/{farm_id}/crop-recommendations", response_model=List[CropRecommendationSchema])
async def get_crop_recommendations(
    farm_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get saved crop recommendations for a farm"""
    # Verify farm ownership
    farm = db.query(Farm).filter(
        Farm.id == farm_id,
        Farm.user_id == current_user.id
    ).first()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found"
        )

    # Get latest recommendations
    recommendations = db.query(CropRecommendation).filter(
        CropRecommendation.farm_id == farm_id,
        CropRecommendation.is_active == True
    ).order_by(CropRecommendation.created_at.desc()).all()

    return recommendations
