# AgriFinSight Setup Instructions

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- PostgreSQL 14+
- Docker (optional but recommended)

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env

# Edit .env file with your configuration
# Set up your database URL, API keys, etc.

# Run the application
uvicorn app.main:app --reload
```

The backend will be available at: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 2. Database Setup

```bash
# Install PostgreSQL (if not already installed)
# macOS with Homebrew:
brew install postgresql
brew services start postgresql

# Create database
createdb agrifinsight

# Run migrations (when available)
# alembic upgrade head
```

### 3. Mobile App Setup

```bash
# Navigate to mobile directory
cd frontend/mobile

# Install dependencies
npm install

# Start the development server
npx expo start
```

### 4. Docker Setup (Alternative)

```bash
# Navigate to deployment directory
cd deployment/docker

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Development Workflow

### Backend Development
1. Make changes to Python files
2. The server will auto-reload with `--reload` flag
3. Test endpoints using Swagger UI at http://localhost:8000/docs

### Mobile Development
1. Make changes to React Native files
2. The app will hot-reload automatically
3. Use Expo Go app on your phone to test

### Database Changes
1. Modify models in `backend/app/models/database.py`
2. Create migration: `alembic revision --autogenerate -m "description"`
3. Apply migration: `alembic upgrade head`

## Project Structure

```
agrifinsight/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── models/         # Database models
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utility functions
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Backend container
├── frontend/
│   ├── mobile/            # React Native app
│   └── web/               # React web app
├── ai/                    # AI/ML models
├── data/                  # Data sources
├── deployment/            # Docker & K8s configs
└── docs/                  # Documentation
```

## Next Steps

1. **Set up database**: Create PostgreSQL database and run migrations
2. **Configure environment**: Update `.env` file with your settings
3. **Test API**: Use Swagger UI to test endpoints
4. **Develop features**: Start implementing core functionality
5. **Add AI models**: Integrate computer vision models
6. **Test with users**: Begin user testing with local farmers

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Kill process using port 8000
lsof -ti:8000 | xargs kill -9
```

**Database connection error:**
- Check PostgreSQL is running
- Verify database credentials in `.env`
- Ensure database exists

**Mobile app not loading:**
- Check Expo CLI is installed: `npm install -g @expo/cli`
- Clear cache: `npx expo start --clear`

**Docker issues:**
- Check Docker is running
- Rebuild containers: `docker-compose up --build`

## Support

For issues or questions:
1. Check the documentation in `/docs`
2. Review the implementation roadmap
3. Check GitHub issues (when available)
4. Contact the development team

## Environment Variables

Key environment variables to configure:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost/agrifinsight

# JWT
SECRET_KEY=your-secret-key-change-in-production

# External APIs
WEATHER_API_KEY=your-openweathermap-api-key

# File Storage
UPLOAD_DIR=uploads
MAX_FILE_SIZE=10485760
```

## Development Tips

1. **Use virtual environments** for Python dependencies
2. **Test frequently** with the Swagger UI
3. **Follow the API documentation** for endpoint specifications
4. **Use Git** for version control and collaboration
5. **Write tests** for new features
6. **Document changes** in commit messages

This setup provides a solid foundation for developing AgriFinSight with all the necessary tools and configurations in place.
