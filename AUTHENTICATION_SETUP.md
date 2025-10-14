# Authentication Setup Guide

## Overview

The authentication system has been fully implemented and integrated between the frontend and backend. Users can now register and login, with their credentials securely stored in the PostgreSQL database.

## What Was Fixed

### Backend Changes (`backend/app/routes/auth.py`)

1. **Registration Endpoint** (`POST /auth/register`)
   - Accepts email, password, and optional phone number
   - Validates that email and phone are unique
   - Hashes passwords using bcrypt
   - Creates user in database
   - Returns JWT access and refresh tokens
   - Returns user data

2. **Login Endpoint** (`POST /auth/login`)
   - Accepts email and password
   - Verifies credentials against database
   - Checks if user account is active
   - Returns JWT access and refresh tokens
   - Returns user data

3. **Token Refresh Endpoint** (`POST /auth/refresh`)
   - Validates refresh token
   - Issues new access token
   - Maintains user session

4. **Logout Endpoint** (`POST /auth/logout`)
   - Validates user token
   - Returns success message

5. **Get Current User** (`GET /auth/me`)
   - Returns authenticated user's information
   - Protected route requiring valid JWT token

### Frontend Changes

1. **RegisterPage.tsx**
   - Integrated with real API endpoint
   - Sends registration data to `POST /auth/register`
   - Stores tokens and user data in localStorage
   - Handles errors and displays messages
   - Redirects to dashboard on success

2. **LoginPage.tsx**
   - Integrated with real API endpoint
   - Sends credentials to `POST /auth/login`
   - Stores tokens and user data in localStorage
   - Handles errors and displays messages
   - Redirects to dashboard on success

### Database Initialization

1. **main.py Startup Event**
   - Automatically creates database tables on server startup
   - No manual database setup required

2. **init_db.py Script**
   - Optional manual database initialization script
   - Can be run separately if needed

## Setup Instructions

### Prerequisites

1. **PostgreSQL Database**
   - Ensure PostgreSQL is installed and running
   - Default configuration expects:
     - Host: `localhost`
     - Port: `5432`
     - Database: `agrifinsight`
     - User: `postgres`
     - Password: `root@123`

2. **Python Environment**
   - Python 3.9 or higher
   - Virtual environment activated

3. **Node.js Environment**
   - Node.js 18 or higher
   - npm installed

### Step 1: Database Setup

Create the PostgreSQL database:

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE agrifinsight;

# Exit psql
\q
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# The database tables will be created automatically when you start the server
# Or you can manually initialize with:
# python init_db.py

# Start the backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory
cd frontend/web

# Install dependencies (if not already installed)
npm install

# Start the development server
npm run dev
```

The frontend will be available at: `http://localhost:3000` (or the port shown in terminal)

## Testing the Authentication Flow

### Test Registration

1. Navigate to `http://localhost:3000/register`
2. Fill in the registration form:
   - Name: Your Name
   - Email: test@example.com
   - Password: Test123!
   - Confirm Password: Test123!
   - Role: Farmer
3. Check "I agree to the Terms of Service"
4. Click "Create account"
5. You should be redirected to the dashboard

### Test Login

1. Navigate to `http://localhost:3000/login`
2. Enter credentials:
   - Email: test@example.com
   - Password: Test123!
3. Click "Sign In"
4. You should be redirected to the dashboard

### Verify Database

Check that the user was created in the database:

```bash
# Connect to PostgreSQL
psql -U postgres -d agrifinsight

# Query users
SELECT id, email, phone, is_active, created_at FROM users;

# Exit
\q
```

## API Endpoints

### Authentication Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/auth/register` | Register new user | No |
| POST | `/auth/login` | Login user | No |
| POST | `/auth/refresh` | Refresh access token | No |
| POST | `/auth/logout` | Logout user | Yes |
| GET | `/auth/me` | Get current user | Yes |

### Request/Response Examples

#### Register User

**Request:**
```json
POST /auth/register
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "phone": "+1234567890"  // optional
}
```

**Response:**
```json
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "phone": "+1234567890",
    "is_active": true,
    "created_at": "2025-10-14T12:00:00"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### Login User

**Request:**
```json
POST /auth/login
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response:**
```json
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "phone": "+1234567890",
    "is_active": true,
    "created_at": "2025-10-14T12:00:00"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

## Security Features

1. **Password Hashing**: Passwords are hashed using bcrypt
2. **JWT Tokens**: Secure token-based authentication
3. **Token Expiration**:
   - Access tokens expire after 30 minutes
   - Refresh tokens expire after 7 days
4. **Protected Routes**: Endpoints require valid JWT tokens
5. **HTTPS Ready**: Configure SSL/TLS for production

## Troubleshooting

### Database Connection Issues

**Error: "could not connect to server"**
- Ensure PostgreSQL is running
- Check database credentials in `backend/app/config.py`
- Verify database exists: `psql -U postgres -l`

**Error: "database 'agrifinsight' does not exist"**
```bash
psql -U postgres
CREATE DATABASE agrifinsight;
\q
```

### Backend Issues

**Error: "No module named 'app'"**
- Ensure you're in the backend directory
- Activate virtual environment
- Install dependencies: `pip install -r requirements.txt`

**Error: "ModuleNotFoundError: No module named 'passlib'"**
```bash
pip install passlib[bcrypt]
```

### Frontend Issues

**Error: "Failed to fetch"**
- Ensure backend is running on port 8000
- Check CORS settings in backend
- Verify API URL in frontend

**Error: "Network request failed"**
- Check that backend server is accessible
- Verify firewall settings
- Check browser console for detailed errors

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=postgresql://postgres:root@123@localhost/agrifinsight

# JWT
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Application
DEBUG=True
```

### Frontend Environment

Create a `.env` file in the frontend/web directory:

```env
VITE_API_URL=http://localhost:8000
```

## Next Steps

1. **Protected Routes**: Implement route protection in frontend
2. **Password Reset**: Add forgot password functionality
3. **Email Verification**: Add email verification on registration
4. **User Profile**: Implement user profile management
5. **Token Blacklist**: Add token blacklisting for logout
6. **Rate Limiting**: Add rate limiting to prevent abuse
7. **2FA**: Implement two-factor authentication (optional)

## Production Considerations

1. **Change Secret Key**: Use a strong, random secret key
2. **Enable HTTPS**: Configure SSL/TLS certificates
3. **Secure Database**: Use strong database passwords
4. **Environment Variables**: Store sensitive data in environment variables
5. **CORS Configuration**: Restrict allowed origins
6. **Token Storage**: Consider using httpOnly cookies
7. **Monitoring**: Add logging and monitoring
8. **Backup**: Implement regular database backups

## Support

For issues or questions:
- Check API documentation at `http://localhost:8000/docs`
- Review error logs in terminal
- Check browser console for frontend errors
- Verify database connectivity with `psql`

---

**Last Updated**: October 14, 2025
**Status**: Authentication system fully functional
