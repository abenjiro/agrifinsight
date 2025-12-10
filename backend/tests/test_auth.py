"""
Tests for authentication endpoints
"""

import pytest
from datetime import datetime, timedelta
from app.routes.auth import get_password_hash, verify_password


class TestAuthEndpoints:
    """Test authentication endpoints"""

    def test_register_new_user(self, client):
        """Test registering a new user"""
        response = client.post(
            "/api/auth/register",
            json={
                "email": "newuser@example.com",
                "phone": "+9876543210",
                "password": "securepassword123",
                "role": "farmer"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["email"] == "newuser@example.com"
        assert data["token_type"] == "bearer"

    def test_register_duplicate_email(self, client, test_user):
        """Test registering with duplicate email fails"""
        response = client.post(
            "/api/auth/register",
            json={
                "email": test_user.email,
                "password": "password123",
                "phone": "+1111111111"
            }
        )
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_register_duplicate_phone(self, client, test_user):
        """Test registering with duplicate phone fails"""
        response = client.post(
            "/api/auth/register",
            json={
                "email": "different@example.com",
                "password": "password123",
                "phone": test_user.phone
            }
        )
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"].lower()

    def test_login_success(self, client, test_user):
        """Test successful login"""
        response = client.post(
            "/api/auth/login",
            json={
                "email": test_user.email,
                "password": "testpassword123"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["email"] == test_user.email

    def test_login_wrong_password(self, client, test_user):
        """Test login with wrong password"""
        response = client.post(
            "/api/auth/login",
            json={
                "email": test_user.email,
                "password": "wrongpassword"
            }
        )
        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    def test_login_nonexistent_user(self, client):
        """Test login with non-existent email"""
        response = client.post(
            "/api/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123"
            }
        )
        assert response.status_code == 401

    def test_get_current_user(self, client, auth_headers):
        """Test getting current user information"""
        response = client.get(
            "/api/auth/me",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert "id" in data

    def test_get_current_user_unauthorized(self, client):
        """Test getting current user without token"""
        response = client.get("/api/auth/me")
        assert response.status_code == 401

    def test_verify_token(self, client, auth_headers):
        """Test token verification"""
        response = client.get(
            "/api/auth/verify",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["authenticated"] is True
        assert "user" in data

    def test_logout(self, client, auth_headers):
        """Test user logout"""
        response = client.post(
            "/api/auth/logout",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_refresh_token(self, client, test_user_token):
        """Test refreshing access token"""
        # First, get a refresh token by logging in
        login_response = client.post(
            "/api/auth/login",
            json={
                "email": "testuser@example.com",
                "password": "testpassword123"
            }
        )
        refresh_token = login_response.json()["refresh_token"]

        # Now use the refresh token to get a new access token
        response = client.post(
            "/api/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"


class TestPasswordHashing:
    """Test password hashing functions"""

    def test_password_hash_and_verify(self):
        """Test password hashing and verification"""
        password = "mysecretpassword"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False

    def test_same_password_different_hashes(self):
        """Test that same password produces different hashes (due to salt)"""
        password = "samepassword"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        assert hash1 != hash2
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True
