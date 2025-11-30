"""
Image analysis routes
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import shutil
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

from app.config import settings
from app.services.ai_service import AIService
from app.ml_models import get_disease_detector
from app.database import get_db
from app.models.database import CropImage, AnalysisResult, User
from app.routes.auth import get_current_user

router = APIRouter(prefix="/api/analysis", tags=["image analysis"])

# Initialize AI service (legacy)
ai_service = AIService()

# Get disease detector (new trained model)
try:
    disease_detector = get_disease_detector()
    print("✅ Disease detection model loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load disease detection model: {e}")
    disease_detector = None

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    farm_id: Optional[int] = None,
    field_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload crop image for analysis"""

    # Validate file type
    if file.content_type not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not allowed. Allowed types: {settings.allowed_file_types}"
        )

    # Validate file size
    if file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size {file.size} exceeds maximum allowed size of {settings.max_file_size} bytes"
        )

    # Create uploads directory if it doesn't exist
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(exist_ok=True)

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{timestamp}_{file.filename or 'image'}{file_extension}"
    file_path = upload_dir / filename

    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Analyze image with new trained model if available, otherwise fallback to legacy
        if disease_detector:
            try:
                detection_result = disease_detector.predict(str(file_path), top_k=3)

                if detection_result['success']:
                    prediction = detection_result['prediction']
                    disease_info = prediction['disease_info']

                    # Format for database storage
                    analysis_result = {
                        'disease_detected': prediction['condition'],
                        'disease_type': prediction['crop'],
                        'is_healthy': prediction['is_healthy'],
                        'confidence_score': prediction['confidence'],
                        'confidence_percentage': prediction['confidence_percentage'],
                        'severity': disease_info.get('severity', 'unknown'),
                        'description': disease_info.get('description', ''),
                        'treatments': disease_info.get('treatments', []),
                        'prevention': disease_info.get('prevention', []),
                        'recommendations': disease_info.get('treatments', []),
                        'treatment_advice': '. '.join(disease_info.get('treatments', [])),
                        'alternative_predictions': detection_result.get('alternative_predictions', []),
                        'model_version': 'efficientnet_b3_plantvillage_v1',
                        'is_confident': detection_result['is_confident']
                    }
                else:
                    raise Exception(detection_result.get('error', 'Prediction failed'))

            except Exception as e:
                print(f"⚠️ New model failed, falling back to legacy: {e}")
                analysis_result = ai_service.analyze_crop_health(str(file_path))
        else:
            # Use legacy AI service
            analysis_result = ai_service.analyze_crop_health(str(file_path))

        # Get user's first farm if farm_id not provided
        if not farm_id and current_user.farms:
            farm_id = current_user.farms[0].id

        # Save crop image to database
        crop_image = CropImage(
            farm_id=farm_id,
            field_id=field_id,
            user_id=current_user.id,
            image_url=str(file_path),
            filename=filename,
            file_size=file.size,
            analysis_status="completed"
        )
        db.add(crop_image)
        db.commit()
        db.refresh(crop_image)

        # Save analysis result to database
        analysis_record = AnalysisResult(
            image_id=crop_image.id,
            user_id=current_user.id,
            disease_detected=analysis_result.get("disease_detected") or analysis_result.get("disease_type", "Unknown"),
            confidence_score=analysis_result.get("confidence_score", 0.0),
            disease_type=analysis_result.get("disease_type"),
            severity=analysis_result.get("severity", "Unknown"),
            recommendations=str(analysis_result.get("recommendations", [])),
            treatment_advice=analysis_result.get("treatment_advice", ""),
            health_score=analysis_result.get("confidence_score", 0.0) * 100 if analysis_result.get("is_healthy") else 0.0
        )
        db.add(analysis_record)
        db.commit()
        db.refresh(analysis_record)

        return {
            "id": analysis_record.id,
            "image_id": crop_image.id,
            "message": "Image uploaded and analyzed successfully",
            "filename": filename,
            "file_path": str(file_path),
            "content_type": file.content_type,
            "size": file.size,
            "farm_id": farm_id,
            "field_id": field_id,
            "analysis_result": analysis_result
        }

    except Exception as e:
        # Clean up file if analysis fails
        if file_path.exists():
            file_path.unlink()
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get("/model-status")
async def model_status():
    """Return AI model load/status information"""
    try:
        status_info = {
            'new_model': {
                'loaded': disease_detector is not None,
                'model_name': 'EfficientNet-B3',
                'version': 'efficientnet_b3_plantvillage_v1',
                'num_classes': len(disease_detector.model.backbone.classifier) if disease_detector else 0,
                'device': str(disease_detector.device) if disease_detector else 'N/A',
                'trained_on': 'PlantVillage Dataset',
                'total_classes': 38
            },
            'legacy_model': ai_service.get_model_status()
        }
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model status: {str(e)}")

@router.get("/{image_id}/status")
async def get_analysis_status(image_id: int):
    """Get analysis status for uploaded image"""
    
    # TODO: Query database for image status
    # TODO: Return current processing status
    
    return {
        "message": "Analysis status endpoint - to be implemented",
        "image_id": image_id,
        "status": "pending"
    }

@router.get("/{image_id}/results")
async def get_analysis_results(image_id: int):
    """Get analysis results for processed image"""
    
    # TODO: Query database for analysis results
    # TODO: Return disease detection results and recommendations
    
    return {
        "message": "Analysis results endpoint - to be implemented",
        "image_id": image_id,
        "results": {
            "disease_detected": None,
            "confidence_score": 0.0,
            "recommendations": [],
            "treatment_advice": ""
        }
    }

@router.get("/history")
async def get_analysis_history(
    farm_id: Optional[int] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get analysis history for user or farm"""

    # Query analysis results for the current user
    query = db.query(AnalysisResult).filter(AnalysisResult.user_id == current_user.id)

    # Filter by farm_id if provided
    if farm_id:
        query = query.join(CropImage).filter(CropImage.farm_id == farm_id)

    # Order by created_at descending (most recent first)
    query = query.order_by(AnalysisResult.created_at.desc())

    # Apply pagination
    total = query.count()
    results = query.offset(offset).limit(limit).all()

    # Format response
    history = []
    for result in results:
        crop_image = db.query(CropImage).filter(CropImage.id == result.image_id).first()
        history.append({
            "id": result.id,
            "image_id": result.image_id,
            "disease_detected": result.disease_detected,
            "confidence_score": result.confidence_score,
            "disease_type": result.disease_type,
            "severity": result.severity,
            "recommendations": result.recommendations,
            "treatment_advice": result.treatment_advice,
            "health_score": result.health_score,
            "created_at": result.created_at.isoformat() if result.created_at else None,
            "image_url": crop_image.image_url if crop_image else None,
            "filename": crop_image.filename if crop_image else None,
            "farm_id": crop_image.farm_id if crop_image else None
        })

    return {
        "data": history,
        "total": total,
        "limit": limit,
        "offset": offset
    }

@router.post("/{analysis_id}/save")
async def save_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Save analysis to permanent storage"""

    # Verify the analysis exists and belongs to the current user
    analysis = db.query(AnalysisResult).filter(
        AnalysisResult.id == analysis_id,
        AnalysisResult.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Analysis is already saved in database, just confirm it exists
    return {
        "message": "Analysis saved successfully",
        "analysis_id": analysis_id,
        "saved": True
    }

@router.delete("/{image_id}")
async def delete_analysis(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete analysis and associated image"""

    # Get the crop image
    crop_image = db.query(CropImage).filter(
        CropImage.id == image_id,
        CropImage.user_id == current_user.id
    ).first()

    if not crop_image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Get all associated analysis results
    analysis_results = db.query(AnalysisResult).filter(
        AnalysisResult.image_id == image_id,
        AnalysisResult.user_id == current_user.id
    ).all()

    # Delete analysis results from database
    for analysis in analysis_results:
        db.delete(analysis)

    # Delete physical image file if it exists
    image_path = Path(crop_image.image_url)
    if image_path.exists():
        try:
            image_path.unlink()
        except Exception as e:
            print(f"Error deleting image file: {e}")

    # Delete crop image record from database
    db.delete(crop_image)
    db.commit()

    return {
        "message": "Analysis deleted successfully",
        "image_id": image_id,
        "deleted": True
    }
