"""
Unit tests for database models
"""

import pytest
from datetime import datetime
from app.models.database import (
    User, Farm, Field, CropImage, AnalysisResult,
    WeatherData, PlantingRecommendation, HarvestPrediction,
    Crop, Animal, CropRecommendation, TokenBlacklist, CropType
)


class TestUserModel:
    """Test User model"""

    def test_create_user(self, db):
        """Test creating a user"""
        user = User(
            email="test@example.com",
            phone="+1234567890",
            password_hash="hashed_password",
            role="farmer"
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.role == "farmer"
        assert user.is_active is True
        assert user.created_at is not None

    def test_user_unique_email(self, db, test_user):
        """Test that email must be unique"""
        duplicate_user = User(
            email=test_user.email,
            password_hash="another_hash",
            role="farmer"
        )
        db.add(duplicate_user)
        with pytest.raises(Exception):
            db.commit()

    def test_user_relationships(self, db, test_user, test_farm):
        """Test user-farm relationship"""
        assert len(test_user.farms) > 0
        assert test_user.farms[0].id == test_farm.id


class TestFarmModel:
    """Test Farm model"""

    def test_create_farm(self, db, test_user):
        """Test creating a farm"""
        farm = Farm(
            user_id=test_user.id,
            name="New Test Farm",
            address="456 Farm Lane",
            latitude=6.0,
            longitude=-1.0,
            size=20.0,
            size_unit="acres",
            soil_type="Sandy"
        )
        db.add(farm)
        db.commit()
        db.refresh(farm)

        assert farm.id is not None
        assert farm.name == "New Test Farm"
        assert farm.user_id == test_user.id
        assert farm.latitude == 6.0
        assert farm.size == 20.0

    def test_farm_relationships(self, db, test_farm, test_field):
        """Test farm relationships"""
        assert test_farm.owner is not None
        assert len(test_farm.fields) > 0
        assert test_farm.fields[0].id == test_field.id

    def test_farm_json_fields(self, db, test_user):
        """Test JSON fields in farm model"""
        farm = Farm(
            user_id=test_user.id,
            name="JSON Test Farm",
            boundary_coordinates={"type": "Polygon", "coordinates": [[0, 0], [1, 1]]},
            soil_composition={"sand": 40, "clay": 30, "silt": 30},
            water_sources=[{"type": "well", "location": [0, 0]}]
        )
        db.add(farm)
        db.commit()
        db.refresh(farm)

        assert farm.boundary_coordinates["type"] == "Polygon"
        assert farm.soil_composition["sand"] == 40
        assert len(farm.water_sources) == 1


class TestFieldModel:
    """Test Field model"""

    def test_create_field(self, db, test_farm):
        """Test creating a field"""
        field = Field(
            farm_id=test_farm.id,
            name="New Field",
            crop_type="Rice",
            coordinates="5.0,-1.0"
        )
        db.add(field)
        db.commit()
        db.refresh(field)

        assert field.id is not None
        assert field.name == "New Field"
        assert field.crop_type == "Rice"
        assert field.farm_id == test_farm.id

    def test_field_farm_relationship(self, db, test_field, test_farm):
        """Test field-farm relationship"""
        assert test_field.farm.id == test_farm.id


class TestCropImageModel:
    """Test CropImage model"""

    def test_create_crop_image(self, db, test_farm, test_field):
        """Test creating a crop image"""
        image = CropImage(
            farm_id=test_farm.id,
            field_id=test_field.id,
            image_url="/uploads/crop.jpg",
            filename="crop.jpg",
            file_size=2048,
            analysis_status="pending"
        )
        db.add(image)
        db.commit()
        db.refresh(image)

        assert image.id is not None
        assert image.filename == "crop.jpg"
        assert image.analysis_status == "pending"
        assert image.uploaded_at is not None

    def test_crop_image_relationships(self, db, test_crop_image, test_farm, test_field):
        """Test crop image relationships"""
        assert test_crop_image.farm.id == test_farm.id
        assert test_crop_image.field.id == test_field.id


class TestAnalysisResultModel:
    """Test AnalysisResult model"""

    def test_create_analysis_result(self, db, test_crop_image):
        """Test creating an analysis result"""
        result = AnalysisResult(
            image_id=test_crop_image.id,
            disease_detected="Leaf Blight",
            confidence_score=0.85,
            disease_type="Fungal",
            severity="medium",
            recommendations="Apply fungicide",
            health_score=65.0
        )
        db.add(result)
        db.commit()
        db.refresh(result)

        assert result.id is not None
        assert result.disease_detected == "Leaf Blight"
        assert result.confidence_score == 0.85
        assert result.severity == "medium"


class TestCropModel:
    """Test Crop model"""

    def test_create_crop(self, db, test_farm, test_field):
        """Test creating a crop"""
        crop = Crop(
            farm_id=test_farm.id,
            field_id=test_field.id,
            crop_type="Tomato",
            variety="Roma",
            quantity=50.0,
            quantity_unit="kg",
            growth_stage="vegetative",
            health_status="healthy"
        )
        db.add(crop)
        db.commit()
        db.refresh(crop)

        assert crop.id is not None
        assert crop.crop_type == "Tomato"
        assert crop.variety == "Roma"
        assert crop.is_active is True
        assert crop.is_harvested is False


class TestAnimalModel:
    """Test Animal model"""

    def test_create_animal(self, db, test_farm):
        """Test creating an animal record"""
        animal = Animal(
            farm_id=test_farm.id,
            animal_type="cattle",
            breed="Holstein",
            quantity=10,
            health_status="healthy",
            purpose="dairy"
        )
        db.add(animal)
        db.commit()
        db.refresh(animal)

        assert animal.id is not None
        assert animal.animal_type == "cattle"
        assert animal.quantity == 10
        assert animal.is_active is True


class TestCropTypeModel:
    """Test CropType model"""

    def test_create_crop_type(self, db):
        """Test creating a crop type"""
        crop_type = CropType(
            name="Maize",
            category="grains",
            scientific_name="Zea mays",
            growth_duration_days=120,
            water_requirement="medium",
            avg_yield_per_acre=2500.0
        )
        db.add(crop_type)
        db.commit()
        db.refresh(crop_type)

        assert crop_type.id is not None
        assert crop_type.name == "Maize"
        assert crop_type.is_active is True


class TestTokenBlacklistModel:
    """Test TokenBlacklist model"""

    def test_create_blacklist_entry(self, db):
        """Test creating a token blacklist entry"""
        from datetime import datetime, timedelta

        token = "sample.jwt.token"
        expires_at = datetime.utcnow() + timedelta(hours=24)

        blacklist_entry = TokenBlacklist(
            token=token,
            expires_at=expires_at
        )
        db.add(blacklist_entry)
        db.commit()
        db.refresh(blacklist_entry)

        assert blacklist_entry.id is not None
        assert blacklist_entry.token == token
        assert blacklist_entry.blacklisted_at is not None
