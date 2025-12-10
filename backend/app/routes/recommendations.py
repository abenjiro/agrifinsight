"""
Recommendations routes - Enhanced with Phase 3 Predictive Analytics
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.database import get_db
from app.models.database import Farm, Crop
from app.services.weather_service import weather_service
from app.services.planting_service import planting_service
from app.services.harvest_service import harvest_service
from app.services.ml_crop_recommendation_service import ml_crop_recommendation_service
from app.services.ai_planting_predictor import get_planting_predictor, predict_planting_time
from app.services.ai_harvest_predictor import get_harvest_predictor, predict_harvest_time

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


@router.get("/planting/{farm_id}")
async def get_planting_recommendations(
    farm_id: int,
    crop_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get intelligent planting time recommendations for a farm
    Includes weather analysis, seasonal timing, and soil suitability
    """
    # Get farm data
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")

    # Convert farm to dict
    farm_data = {
        'id': farm.id,
        'name': farm.name,
        'latitude': farm.latitude,
        'longitude': farm.longitude,
        'altitude': farm.altitude,
        'soil_type': farm.soil_type,
        'soil_ph': farm.soil_ph,
        'terrain_type': farm.terrain_type,
        'climate_zone': farm.climate_zone,
        'avg_temperature': farm.avg_temperature,
        'avg_annual_rainfall': farm.avg_annual_rainfall,
        'size': farm.size
    }

    if not farm_data['latitude'] or not farm_data['longitude']:
        raise HTTPException(
            status_code=400,
            detail="Farm must have GPS coordinates for recommendations"
        )

    try:
        if crop_type:
            # Single crop recommendation
            recommendation = await planting_service.get_planting_recommendation(
                farm_data, crop_type
            )
            return {
                "success": True,
                "farm_id": farm_id,
                "farm_name": farm.name,
                "recommendation": recommendation
            }
        else:
            # Multi-crop comparison
            default_crops = ['Maize', 'Rice', 'Cassava', 'Tomato', 'Soybean', 'Groundnut']
            comparison = await planting_service.get_multi_crop_comparison(
                farm_data, default_crops
            )
            return {
                "success": True,
                "farm_id": farm_id,
                "farm_name": farm.name,
                "comparison": comparison
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating planting recommendations: {str(e)}"
        )


@router.get("/harvest/{crop_id}")
async def get_harvest_recommendations(
    crop_id: int,
    db: Session = Depends(get_db)
):
    """
    Get harvest timing recommendations for a specific crop
    Includes maturity assessment, yield prediction, and weather forecast
    """
    # Get crop data
    crop = db.query(Crop).filter(Crop.id == crop_id).first()
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")

    # Get associated farm
    farm = db.query(Farm).filter(Farm.id == crop.farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found for this crop")

    # Convert farm to dict
    farm_data = {
        'latitude': farm.latitude,
        'longitude': farm.longitude,
        'altitude': farm.altitude,
        'soil_type': farm.soil_type,
        'soil_ph': farm.soil_ph,
        'climate_zone': farm.climate_zone,
        'avg_temperature': farm.avg_temperature,
        'avg_annual_rainfall': farm.avg_annual_rainfall,
        'size': farm.size
    }

    if not farm_data['latitude'] or not farm_data['longitude']:
        raise HTTPException(
            status_code=400,
            detail="Farm must have GPS coordinates for harvest predictions"
        )

    if not crop.planting_date:
        raise HTTPException(
            status_code=400,
            detail="Crop must have a planting date for harvest predictions"
        )

    try:
        prediction = await harvest_service.get_harvest_prediction(
            crop_type=crop.crop_type,
            planting_date=crop.planting_date.strftime('%Y-%m-%d'),
            farm_data=farm_data,
            current_growth_stage=crop.growth_stage
        )

        return {
            "success": True,
            "crop_id": crop_id,
            "crop_name": crop.crop_type,  # crop_type is the name of the crop
            "crop_type": crop.crop_type,
            "farm_name": farm.name,
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating harvest prediction: {str(e)}"
        )


@router.get("/weather/{farm_id}")
async def get_weather_forecast(
    farm_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get weather forecast for a farm location
    """
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")

    if not farm.latitude or not farm.longitude:
        raise HTTPException(
            status_code=400,
            detail="Farm must have GPS coordinates for weather forecast"
        )

    try:
        current_weather = await weather_service.get_current_weather(
            farm.latitude, farm.longitude
        )
        forecast = await weather_service.get_forecast(
            farm.latitude, farm.longitude, days=min(days, 14)
        )

        return {
            "success": True,
            "farm_id": farm_id,
            "farm_name": farm.name,
            "location": {
                "latitude": farm.latitude,
                "longitude": farm.longitude,
                "address": farm.address
            },
            "current_weather": current_weather,
            "forecast": forecast
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching weather data: {str(e)}"
        )


@router.get("/crops/{farm_id}")
async def get_crop_recommendations_ml(
    farm_id: int,
    top_n: int = 5,
    db: Session = Depends(get_db)
):
    """
    Get ML-based crop recommendations for a farm
    Uses trained PyTorch model for intelligent suggestions
    """
    farm = db.query(Farm).filter(Farm.id == farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")

    # Convert farm to dict for ML service
    farm_data = {
        'latitude': farm.latitude,
        'longitude': farm.longitude,
        'altitude': farm.altitude,
        'avg_temperature': farm.avg_temperature,
        'avg_annual_rainfall': farm.avg_annual_rainfall,
        'soil_ph': farm.soil_ph,
        'soil_type': farm.soil_type,
        'climate_zone': farm.climate_zone,
        'terrain_type': farm.terrain_type
    }

    try:
        recommendations = ml_crop_recommendation_service.generate_recommendations(
            farm_data, top_n=top_n
        )

        if not recommendations:
            raise HTTPException(
                status_code=503,
                detail="ML model not available. Please ensure the model is trained and loaded."
            )

        return {
            "success": True,
            "farm_id": farm_id,
            "farm_name": farm.name,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating crop recommendations: {str(e)}"
        )


@router.get("/care/{crop_id}")
async def get_crop_care_recommendations(
    crop_id: int,
    db: Session = Depends(get_db)
):
    """
    Get crop care recommendations based on current conditions
    """
    crop = db.query(Crop).filter(Crop.id == crop_id).first()
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")

    farm = db.query(Farm).filter(Farm.id == crop.farm_id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="Farm not found")

    try:
        # Get weather data
        current_weather = None
        if farm.latitude and farm.longitude:
            current_weather = await weather_service.get_current_weather(
                farm.latitude, farm.longitude
            )

        # Calculate crop age
        crop_age_days = 0
        if crop.planting_date:
            crop_age_days = (datetime.now().date() - crop.planting_date).days

        # Generate care recommendations based on growth stage
        care_recommendations = _generate_care_advice(
            crop.crop_type,
            crop.growth_stage,
            crop_age_days,
            current_weather
        )

        return {
            "success": True,
            "crop_id": crop_id,
            "crop_name": crop.crop_type,  # crop_type is the name of the crop
            "crop_type": crop.crop_type,
            "crop_age_days": crop_age_days,
            "growth_stage": crop.growth_stage,
            "current_weather": current_weather,
            "recommendations": care_recommendations
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating care recommendations: {str(e)}"
        )


def _generate_care_advice(
    crop_type: str,
    growth_stage: Optional[str],
    crop_age_days: int,
    weather: Optional[Dict]
) -> Dict:
    """
    Generate stage-specific care recommendations
    """
    recommendations = {
        "watering": [],
        "fertilization": [],
        "pest_control": [],
        "general_care": []
    }

    # Weather-based watering advice
    if weather:
        temp = weather.get('temperature', 25)
        humidity = weather.get('humidity', 65)

        if temp > 30:
            recommendations["watering"].append("High temperature - increase watering frequency")
        if humidity < 40:
            recommendations["watering"].append("Low humidity - ensure adequate soil moisture")

    # Stage-based recommendations
    if growth_stage == 'seedling' or crop_age_days < 21:
        recommendations["watering"].append("Keep soil consistently moist but not waterlogged")
        recommendations["fertilization"].append("Apply starter fertilizer if not done at planting")
        recommendations["pest_control"].append("Monitor for cutworms and damping off")
        recommendations["general_care"].append("Remove weeds carefully to avoid root damage")

    elif growth_stage == 'vegetative' or crop_age_days < 60:
        recommendations["watering"].append("Water deeply but less frequently to encourage root growth")
        recommendations["fertilization"].append("Apply nitrogen-rich fertilizer for leaf development")
        recommendations["pest_control"].append("Scout for leaf-eating insects weekly")
        recommendations["general_care"].append("Provide support stakes for tall plants")

    elif growth_stage == 'flowering' or crop_age_days < 90:
        recommendations["watering"].append("Critical period - maintain consistent moisture")
        recommendations["fertilization"].append("Switch to phosphorus-rich fertilizer for flower/fruit development")
        recommendations["pest_control"].append("Monitor for pollinators and avoid pesticides during flowering")
        recommendations["general_care"].append("Mulch around plants to conserve moisture")

    elif growth_stage == 'fruit_development':
        recommendations["watering"].append("Maintain steady moisture for fruit quality")
        recommendations["fertilization"].append("Apply potassium-rich fertilizer for fruit development")
        recommendations["pest_control"].append("Inspect fruits regularly for pests and diseases")
        recommendations["general_care"].append("Prune excess growth to improve air circulation")

    else:  # maturation or unknown
        recommendations["watering"].append("Reduce watering as harvest approaches")
        recommendations["fertilization"].append("Stop fertilization 2-3 weeks before harvest")
        recommendations["pest_control"].append("Continue monitoring for pests")
        recommendations["general_care"].append("Prepare for harvest - arrange equipment and storage")

    # Crop-specific advice
    crop_specific = {
        'maize': {
            "pest_control": ["Watch for fall armyworm, especially in vegetative stage"],
            "fertilization": ["Side-dress with nitrogen at knee-high stage"]
        },
        'tomato': {
            "pest_control": ["Monitor for early and late blight"],
            "general_care": ["Prune suckers for better fruit quality"]
        },
        'rice': {
            "watering": ["Maintain 2-5cm water level in flooded fields"],
            "pest_control": ["Scout for stem borers and leaf folders"]
        }
    }

    if crop_type.lower() in crop_specific:
        for category, advice in crop_specific[crop_type.lower()].items():
            recommendations[category].extend(advice)

    return recommendations


# Legacy endpoints for backward compatibility
@router.get("/planting")
async def get_planting_recommendations_legacy(
    farm_id: int,
    crop_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Legacy endpoint - redirects to new endpoint"""
    return await get_planting_recommendations(farm_id, crop_type, db)


@router.get("/harvest")
async def get_harvest_recommendations_legacy(
    crop_id: int,
    db: Session = Depends(get_db)
):
    """Legacy endpoint - redirects to new endpoint"""
    return await get_harvest_recommendations(crop_id, db)
