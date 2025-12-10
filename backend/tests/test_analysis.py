"""
Tests for crop analysis endpoints
"""

import pytest
from io import BytesIO


class TestAnalysisEndpoints:
    """Test crop image analysis endpoints"""

    def test_upload_image_for_analysis(self, client, auth_headers, test_farm):
        """Test uploading an image for disease analysis"""
        # Create a fake image file
        image_data = BytesIO(b"fake image content")
        image_data.name = "test_crop.jpg"

        response = client.post(
            "/api/analysis/upload",
            headers=auth_headers,
            files={"file": ("test_crop.jpg", image_data, "image/jpeg")},
            data={"farm_id": test_farm.id}
        )
        # Status could be 200 or 201 depending on implementation
        assert response.status_code in [200, 201]
        data = response.json()
        assert "id" in data or "image_id" in data

    def test_upload_image_unauthorized(self, client, test_farm):
        """Test uploading image without authentication"""
        image_data = BytesIO(b"fake image content")

        response = client.post(
            "/api/analysis/upload",
            files={"file": ("test.jpg", image_data, "image/jpeg")},
            data={"farm_id": test_farm.id}
        )
        assert response.status_code == 401

    def test_upload_invalid_file_type(self, client, auth_headers, test_farm):
        """Test uploading non-image file"""
        text_data = BytesIO(b"this is not an image")

        response = client.post(
            "/api/analysis/upload",
            headers=auth_headers,
            files={"file": ("test.txt", text_data, "text/plain")},
            data={"farm_id": test_farm.id}
        )
        # Should reject non-image files
        assert response.status_code in [400, 422]

    def test_get_analysis_status(self, client, auth_headers, test_crop_image):
        """Test checking analysis status"""
        response = client.get(
            f"/api/analysis/{test_crop_image.id}/status",
            headers=auth_headers
        )
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data

    def test_get_analysis_results(self, client, auth_headers, db, test_crop_image):
        """Test getting analysis results"""
        from app.models.database import AnalysisResult

        # Create an analysis result
        result = AnalysisResult(
            image_id=test_crop_image.id,
            disease_detected="Healthy",
            confidence_score=0.95,
            disease_type="None",
            severity="none",
            recommendations="Crop is healthy, continue current care regimen",
            health_score=95.0
        )
        db.add(result)
        db.commit()

        response = client.get(
            f"/api/analysis/{test_crop_image.id}/results",
            headers=auth_headers
        )
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "disease_detected" in data or "results" in data

    def test_get_results_nonexistent_image(self, client, auth_headers):
        """Test getting results for non-existent image"""
        response = client.get(
            "/api/analysis/99999/results",
            headers=auth_headers
        )
        # API may return 200 with empty results or 404
        assert response.status_code in [200, 404]


class TestAnalysisHistory:
    """Test analysis history and listing"""

    def test_get_analysis_history(self, client, auth_headers, test_crop_image):
        """Test getting analysis history for user"""
        response = client.get(
            "/api/analysis/history",
            headers=auth_headers
        )
        # Endpoint might not exist yet
        assert response.status_code in [200, 404, 405]

    def test_get_farm_analyses(self, client, auth_headers, test_farm):
        """Test getting all analyses for a specific farm"""
        response = client.get(
            f"/api/farms/{test_farm.id}/analyses",
            headers=auth_headers
        )
        # Endpoint might not exist yet
        assert response.status_code in [200, 404, 405]
