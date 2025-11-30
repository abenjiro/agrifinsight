"""
Database service layer - centralizes database query logic
This service layer helps keep routes clean and makes testing easier
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.database import (
    CropType, User, Farm, Crop, Animal, CropRecommendation, TokenBlacklist
)


class CropTypeService:
    """Service for CropType operations"""

    @staticmethod
    def get_active_crop_types(db: Session) -> List[CropType]:
        """Get all active crop types ordered by name"""
        return db.query(CropType).filter(
            CropType.is_active == True
        ).order_by(CropType.name).all()

    @staticmethod
    def find_by_name(db: Session, name: str) -> Optional[CropType]:
        """Find crop type by name"""
        return db.query(CropType).filter(CropType.name == name).first()

    @staticmethod
    def create_crop_type(db: Session, name: str, category: Optional[str] = None,
                        description: Optional[str] = None) -> CropType:
        """Create a new crop type"""
        crop_type = CropType(name=name, category=category, description=description)
        db.add(crop_type)
        db.commit()
        db.refresh(crop_type)
        return crop_type


class FarmService:
    """Service for Farm operations"""

    @staticmethod
    def get_user_farm(db: Session, farm_id: int, user_id: int) -> Optional[Farm]:
        """Get a farm owned by specific user"""
        return db.query(Farm).filter(
            and_(Farm.id == farm_id, Farm.user_id == user_id)
        ).first()

    @staticmethod
    def get_user_farms(db: Session, user_id: int) -> List[Farm]:
        """Get all farms for a user"""
        return db.query(Farm).filter(Farm.user_id == user_id).all()


class CropService:
    """Service for Crop operations"""

    @staticmethod
    def get_farm_crops(db: Session, farm_id: int) -> List[Crop]:
        """Get all crops for a farm"""
        return db.query(Crop).filter(Crop.farm_id == farm_id).all()

    @staticmethod
    def get_crop_by_id(db: Session, crop_id: int) -> Optional[Crop]:
        """Get crop by ID"""
        return db.query(Crop).filter(Crop.id == crop_id).first()

    @staticmethod
    def create_crop(db: Session, crop_data: dict) -> Crop:
        """Create a new crop"""
        crop = Crop(**crop_data)
        db.add(crop)
        db.commit()
        db.refresh(crop)
        return crop

    @staticmethod
    def update_crop(db: Session, crop: Crop, update_data: dict) -> Crop:
        """Update crop with new data"""
        for field, value in update_data.items():
            setattr(crop, field, value)
        db.commit()
        db.refresh(crop)
        return crop

    @staticmethod
    def delete_crop(db: Session, crop: Crop) -> None:
        """Delete a crop"""
        db.delete(crop)
        db.commit()


class AnimalService:
    """Service for Animal operations"""

    @staticmethod
    def get_farm_animals(db: Session, farm_id: int) -> List[Animal]:
        """Get all animals for a farm"""
        return db.query(Animal).filter(Animal.farm_id == farm_id).all()

    @staticmethod
    def get_animal_by_id(db: Session, animal_id: int) -> Optional[Animal]:
        """Get animal by ID"""
        return db.query(Animal).filter(Animal.id == animal_id).first()

    @staticmethod
    def create_animal(db: Session, animal_data: dict) -> Animal:
        """Create a new animal record"""
        animal = Animal(**animal_data)
        db.add(animal)
        db.commit()
        db.refresh(animal)
        return animal

    @staticmethod
    def update_animal(db: Session, animal: Animal, update_data: dict) -> Animal:
        """Update animal with new data"""
        for field, value in update_data.items():
            setattr(animal, field, value)
        db.commit()
        db.refresh(animal)
        return animal

    @staticmethod
    def delete_animal(db: Session, animal: Animal) -> None:
        """Delete an animal record"""
        db.delete(animal)
        db.commit()


class CropRecommendationService:
    """Service for CropRecommendation operations"""

    @staticmethod
    def get_farm_recommendations(db: Session, farm_id: int, active_only: bool = True) -> List[CropRecommendation]:
        """Get crop recommendations for a farm"""
        query = db.query(CropRecommendation).filter(CropRecommendation.farm_id == farm_id)

        if active_only:
            query = query.filter(CropRecommendation.is_active == True)

        return query.order_by(CropRecommendation.created_at.desc()).all()

    @staticmethod
    def delete_user_farm_recommendations(db: Session, farm_id: int, user_id: int) -> None:
        """Delete all recommendations for a farm by user"""
        db.query(CropRecommendation).filter(
            and_(
                CropRecommendation.farm_id == farm_id,
                CropRecommendation.user_id == user_id
            )
        ).delete()
        db.commit()

    @staticmethod
    def create_recommendations(db: Session, recommendations: List[dict],
                              farm_id: int, user_id: int) -> List[CropRecommendation]:
        """Create multiple crop recommendations"""
        saved_recommendations = []

        for rec_data in recommendations:
            crop_rec = CropRecommendation(
                farm_id=farm_id,
                user_id=user_id,
                **rec_data
            )
            db.add(crop_rec)
            saved_recommendations.append(crop_rec)

        db.commit()

        # Refresh all
        for rec in saved_recommendations:
            db.refresh(rec)

        return saved_recommendations


class TokenBlacklistService:
    """Service for TokenBlacklist operations"""

    @staticmethod
    def is_blacklisted(db: Session, token: str) -> bool:
        """Check if a token is blacklisted"""
        return db.query(TokenBlacklist).filter(
            TokenBlacklist.token == token
        ).first() is not None

    @staticmethod
    def blacklist_token(db: Session, token: str, expires_at) -> TokenBlacklist:
        """Add token to blacklist"""
        blacklisted = TokenBlacklist(token=token, expires_at=expires_at)
        db.add(blacklisted)
        db.commit()
        db.refresh(blacklisted)
        return blacklisted


# Convenience instances for importing
crop_type_service = CropTypeService()
farm_service = FarmService()
crop_service = CropService()
animal_service = AnimalService()
crop_recommendation_service_db = CropRecommendationService()
token_blacklist_service = TokenBlacklistService()
