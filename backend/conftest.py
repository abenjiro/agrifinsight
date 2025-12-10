"""
Pytest configuration and fixtures for AgriFinSight backend tests
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from datetime import datetime, timedelta

from app.main import app
from app.models.database import Base, User, Farm, Field, CropImage
from app.database import get_db
from app.routes.auth import get_password_hash, create_access_token

# Create in-memory SQLite database for testing
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db():
    """Create a fresh database for each test"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    """Create a test client with database dependency override"""
    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db):
    """Create a test user"""
    user = User(
        email="testuser@example.com",
        phone="+1234567890",
        password_hash=get_password_hash("testpassword123"),
        role="farmer",
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_user_token(test_user):
    """Create an access token for test user"""
    token = create_access_token(data={"sub": test_user.email})
    return token


@pytest.fixture
def auth_headers(test_user_token):
    """Create authorization headers with test user token"""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest.fixture
def test_farm(db, test_user):
    """Create a test farm"""
    farm = Farm(
        user_id=test_user.id,
        name="Test Farm",
        address="123 Farm Road, Test City",
        latitude=5.6037,
        longitude=-0.1870,
        size=10.5,
        size_unit="acres",
        soil_type="Loamy",
        soil_ph=6.5,
        climate_zone="Tropical"
    )
    db.add(farm)
    db.commit()
    db.refresh(farm)
    return farm


@pytest.fixture
def test_field(db, test_farm):
    """Create a test field"""
    field = Field(
        farm_id=test_farm.id,
        name="Test Field",
        crop_type="Maize",
        planting_date=datetime.utcnow(),
        expected_harvest_date=datetime.utcnow() + timedelta(days=120),
        coordinates="5.6037,-0.1870"
    )
    db.add(field)
    db.commit()
    db.refresh(field)
    return field


@pytest.fixture
def test_crop_image(db, test_farm, test_field):
    """Create a test crop image"""
    image = CropImage(
        farm_id=test_farm.id,
        field_id=test_field.id,
        image_url="/uploads/test_image.jpg",
        filename="test_image.jpg",
        file_size=1024,
        analysis_status="pending"
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


@pytest.fixture
def multiple_users(db):
    """Create multiple test users"""
    users = []
    for i in range(3):
        user = User(
            email=f"user{i}@example.com",
            phone=f"+123456789{i}",
            password_hash=get_password_hash(f"password{i}"),
            role="farmer",
            is_active=True
        )
        db.add(user)
        users.append(user)
    db.commit()
    for user in users:
        db.refresh(user)
    return users


@pytest.fixture
def sample_image_data():
    """Sample image upload data for testing"""
    return {
        "filename": "test_crop.jpg",
        "content_type": "image/jpeg",
        "size": 2048
    }
