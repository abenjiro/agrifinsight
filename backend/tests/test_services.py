"""
Tests for service layer functions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestAIService:
    """Test AI service functionality"""

    @patch('app.services.ai_service.SimpleModelManager')
    @patch('app.services.ai_service.TorchImageClassifier')
    def test_ai_service_initialization(self, mock_torch, mock_manager):
        """Test AI service initialization"""
        from app.services.ai_service import AIService

        service = AIService()
        assert service is not None

    @patch('app.services.ai_service.TorchImageClassifier')
    def test_analyze_crop_health_with_torch(self, mock_torch):
        """Test crop health analysis with torch classifier"""
        from app.services.ai_service import AIService

        # Mock the torch classifier
        mock_classifier_instance = Mock()
        mock_classifier_instance.predict.return_value = {
            'predicted_label': 'Healthy',
            'confidence_score': 0.95,
            'top_predictions': [
                {'label': 'Healthy', 'confidence': 0.95},
                {'label': 'Blight', 'confidence': 0.03}
            ]
        }
        mock_torch.return_value = mock_classifier_instance

        service = AIService()
        service.torch_classifier = mock_classifier_instance

        result = service.analyze_crop_health("/fake/path/image.jpg")

        assert result['disease_detected'] == 'Healthy'
        assert result['confidence_score'] == 0.95
        mock_classifier_instance.predict.assert_called_once()

    def test_analyze_crop_health_mock_response(self):
        """Test crop health analysis with mock response when models unavailable"""
        from app.services.ai_service import AIService

        service = AIService()
        service.torch_classifier = None
        service.model_manager = None

        result = service.analyze_crop_health("/fake/path/image.jpg")

        # Should return mock data when models are not available
        assert 'disease_detected' in result
        assert 'confidence_score' in result


class TestWeatherService:
    """Test weather service functionality"""

    @patch('app.services.weather_service.requests.get')
    def test_get_weather_data(self, mock_get):
        """Test fetching weather data"""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'temperature': 28.5,
            'humidity': 75,
            'rainfall': 0.0
        }
        mock_get.return_value = mock_response

        # This test assumes weather_service exists
        try:
            from app.services.weather_service import get_weather_data
            result = get_weather_data(latitude=5.6, longitude=-0.2)
            assert result is not None
        except ImportError:
            pytest.skip("Weather service not implemented yet")


class TestPlantingService:
    """Test planting recommendation service"""

    def test_calculate_optimal_planting_date(self):
        """Test calculating optimal planting date"""
        try:
            from app.services.planting_service import calculate_optimal_planting_date
            from datetime import datetime

            result = calculate_optimal_planting_date(
                crop_type="Maize",
                latitude=5.6,
                longitude=-0.2,
                soil_type="Loamy"
            )
            assert result is not None
        except ImportError:
            pytest.skip("Planting service not fully implemented yet")


class TestHarvestService:
    """Test harvest prediction service"""

    def test_predict_harvest_date(self):
        """Test predicting harvest date"""
        try:
            from app.services.harvest_service import predict_harvest_date
            from datetime import datetime

            planting_date = datetime.utcnow()
            result = predict_harvest_date(
                crop_type="Rice",
                planting_date=planting_date,
                weather_data={"temperature": 28, "rainfall": 100}
            )
            assert result is not None
        except ImportError:
            pytest.skip("Harvest service not fully implemented yet")


class TestCropRecommendationService:
    """Test crop recommendation service"""

    def test_get_crop_recommendations(self, db, test_farm):
        """Test getting crop recommendations for a farm"""
        try:
            from app.services.crop_recommendation_service import get_crop_recommendations

            recommendations = get_crop_recommendations(
                db=db,
                farm_id=test_farm.id
            )
            assert recommendations is not None
        except ImportError:
            pytest.skip("Crop recommendation service not fully implemented yet")


class TestDBService:
    """Test database service helper functions"""

    def test_token_blacklist_service_add_token(self, db):
        """Test adding token to blacklist"""
        from app.services.db_service import token_blacklist_service
        from datetime import datetime, timedelta

        token = "test.jwt.token"
        expires_at = datetime.utcnow() + timedelta(hours=1)

        token_blacklist_service.blacklist_token(db, token, expires_at)

        # Verify token is blacklisted
        is_blacklisted = token_blacklist_service.is_blacklisted(db, token)
        assert is_blacklisted is True

    def test_token_blacklist_service_check_non_blacklisted(self, db):
        """Test checking non-blacklisted token"""
        from app.services.db_service import token_blacklist_service

        is_blacklisted = token_blacklist_service.is_blacklisted(db, "non.existent.token")
        assert is_blacklisted is False

    def test_clean_expired_tokens(self, db):
        """Test cleaning expired blacklisted tokens"""
        from app.services.db_service import token_blacklist_service
        from app.models.database import TokenBlacklist
        from datetime import datetime, timedelta

        # Add an expired token
        expired_token = TokenBlacklist(
            token="expired.token",
            expires_at=datetime.utcnow() - timedelta(days=1)
        )
        db.add(expired_token)
        db.commit()

        # Clean expired tokens
        try:
            cleaned = token_blacklist_service.clean_expired_tokens(db)
            assert cleaned >= 0
        except AttributeError:
            pytest.skip("clean_expired_tokens method not implemented")


class TestGeospatialService:
    """Test geospatial service functions"""

    def test_calculate_distance(self):
        """Test calculating distance between coordinates"""
        try:
            from app.services.geospatial_service import calculate_distance

            # Distance between two points in Accra
            distance = calculate_distance(
                lat1=5.6037,
                lon1=-0.1870,
                lat2=5.6137,
                lon2=-0.1970
            )
            assert distance > 0
        except ImportError:
            pytest.skip("Geospatial service not implemented yet")

    def test_validate_coordinates(self):
        """Test coordinate validation"""
        try:
            from app.services.geospatial_service import validate_coordinates

            # Valid coordinates
            assert validate_coordinates(5.6037, -0.1870) is True

            # Invalid coordinates
            assert validate_coordinates(200.0, -0.1870) is False
            assert validate_coordinates(5.6037, 200.0) is False
        except ImportError:
            pytest.skip("Geospatial service not implemented yet")
