# Backend Testing Documentation

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

## Test Organization

### Test Files
- `test_models.py` - Database model tests
- `test_auth.py` - Authentication endpoint tests
- `test_farms.py` - Farm management tests
- `test_analysis.py` - Crop analysis tests
- `test_services.py` - Service layer tests

### Test Markers
Use markers to organize and run specific test groups:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run auth tests
pytest -m auth

# Skip slow tests
pytest -m "not slow"
```

## Available Fixtures

### Database Fixtures
- `db` - Fresh database session
- `client` - FastAPI test client

### User Fixtures
- `test_user` - Pre-created user
- `test_user_token` - JWT token
- `auth_headers` - Authorization headers
- `multiple_users` - List of users

### Farm Fixtures
- `test_farm` - Pre-created farm
- `test_field` - Pre-created field
- `test_crop_image` - Pre-created image

## Writing New Tests

### Example: Testing an Endpoint

```python
def test_create_farm(client, auth_headers):
    """Test farm creation"""
    response = client.post(
        "/api/farms",
        headers=auth_headers,
        json={"name": "Test Farm", "size": 10}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Test Farm"
```

### Example: Testing a Model

```python
def test_user_model(db):
    """Test user model creation"""
    user = User(
        email="test@example.com",
        password_hash="hashed"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    assert user.id is not None
    assert user.email == "test@example.com"
```

## Coverage Requirements

Target coverage levels:
- Models: >90%
- API Endpoints: >85%
- Services: >80%
- Overall: >80%

## Continuous Integration

Tests run automatically on:
- Every commit
- Pull requests
- Pre-deployment

Ensure all tests pass before pushing code.
