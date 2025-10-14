# AgriFinSight - Claude Code Context

## Project Overview

AgriFinSight is an AI-powered smart agriculture platform designed to help smallholder farmers make data-driven decisions about crop health monitoring, planting time predictions, and harvest optimization. The platform uses computer vision and machine learning to detect crop diseases, provide treatment recommendations, and offer planting and harvest timing advice.

### Core Value Proposition
- **Crop Health Monitoring**: AI-powered disease detection and treatment recommendations
- **Planting Time Predictions**: Weather-based recommendations for optimal planting
- **Harvest Optimization**: Timing predictions to maximize yield and reduce losses

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.9+)
- **Database**: PostgreSQL 14+
- **Cache**: Redis
- **Authentication**: JWT with refresh tokens
- **Image Processing**: Pillow, OpenCV
- **AI/ML**: TensorFlow 2.13+, PyTorch 2.0+, scikit-learn
- **Server**: Uvicorn (development), Gunicorn (production)

### Frontend
- **Web**: React 18 + TypeScript + Vite
- **Mobile**: React Native (being phased out)
- **Styling**: Tailwind CSS
- **UI Components**: Lucide React (icons)
- **State Management**: React Context API
- **HTTP Client**: Axios
- **Routing**: React Router v7
- **Charts/Maps**: amCharts 5 (including globe visualizations)

### Development Tools
- **Linting**: ESLint with TypeScript support
- **Type Checking**: TypeScript 5.9+
- **Build Tool**: Vite 7+
- **Package Manager**: npm

## Project Structure

```
agrifinsight/
├── backend/                 # FastAPI backend services
│   ├── app/
│   │   ├── models/         # Database models (SQLAlchemy)
│   │   │   └── database.py
│   │   ├── routes/         # API endpoints
│   │   │   ├── auth.py     # Authentication endpoints
│   │   │   ├── analysis.py # Image analysis endpoints
│   │   │   ├── farms.py    # Farm management endpoints
│   │   │   └── recommendations.py # AI recommendations
│   │   ├── services/       # Business logic
│   │   │   └── ai_service.py # AI model integration
│   │   ├── config.py       # Configuration management
│   │   ├── database.py     # Database connection
│   │   ├── schemas.py      # Pydantic schemas
│   │   └── main.py         # FastAPI app initialization
│   ├── uploads/            # Uploaded images (local storage)
│   ├── requirements.txt    # Python dependencies
│   └── venv/              # Python virtual environment
│
├── frontend/
│   └── web/               # React web application
│       ├── src/
│       │   ├── components/    # Reusable UI components
│       │   ├── pages/         # Page components
│       │   │   ├── HomePage.tsx      # Landing page with globe
│       │   │   ├── DashboardPage.tsx # Main dashboard
│       │   │   ├── LoginPage.tsx     # User login
│       │   │   └── RegisterPage.tsx  # User registration
│       │   ├── services/
│       │   │   └── api.ts    # API client and services
│       │   ├── types/
│       │   │   └── index.ts  # TypeScript type definitions
│       │   ├── utils/
│       │   │   └── cn.ts     # Class name utility (clsx + tailwind-merge)
│       │   ├── main.tsx      # App entry point
│       │   └── index.css     # Global styles
│       ├── package.json
│       ├── tsconfig.json
│       ├── vite.config.ts
│       ├── tailwind.config.js
│       ├── postcss.config.js
│       ├── .eslintrc.cjs
│       ├── env.example       # Environment variables template
│       └── README.md
│
├── ai/                     # AI/ML models and training
│   ├── models/            # Trained models
│   │   ├── best_model.pth           # PyTorch disease detection model
│   │   ├── generic_model.py         # Model architecture definition
│   │   └── torch_inference.py       # PyTorch inference logic
│   ├── training/          # Model training scripts
│   └── data/             # Training data
│
├── deployment/           # Deployment configurations
│   ├── docker/          # Docker configurations
│   └── kubernetes/      # K8s configurations
│
├── docs/                # Documentation
├── tests/               # Integration tests
│
├── README.md            # Project overview
├── TECHNICAL_ARCHITECTURE.md  # System architecture details
├── MVP_SPECIFICATION.md       # MVP scope and requirements
├── DEVELOPMENT_FLOW.md        # Development guidelines
├── IMPLEMENTATION_ROADMAP.md  # Feature roadmap
├── SETUP_INSTRUCTIONS.md      # Setup guide
└── DATABASE_AND_AI_SETUP_COMPLETE.md  # Setup completion notes
```

## API Endpoints

### Authentication (`/auth`)
- `POST /auth/register` - User registration
- `POST /auth/login` - User login (returns JWT token)
- `POST /auth/refresh` - Refresh access token
- `POST /auth/logout` - User logout

### Farms Management (`/farms`)
- `GET /farms` - Get all farms for authenticated user
- `POST /farms` - Create new farm
- `GET /farms/{id}` - Get specific farm details
- `PUT /farms/{id}` - Update farm
- `DELETE /farms/{id}` - Delete farm

### Analysis (`/analysis`)
- `POST /analysis/upload` - Upload crop image for disease detection
- `GET /analysis/{id}/status` - Check analysis status
- `GET /analysis/{id}/results` - Get analysis results

### Recommendations (`/recommendations`)
- `GET /recommendations/planting` - Get planting time recommendations
- `GET /recommendations/harvest` - Get harvest timing predictions
- `GET /recommendations/care` - Get crop care recommendations

## Database Schema

### Core Tables
```sql
-- Users and Authentication
users (id, email, phone, password_hash, created_at, updated_at)
user_profiles (user_id, name, location, language, timezone)

-- Farm Management
farms (id, user_id, name, location, size, soil_type, created_at)
fields (id, farm_id, name, coordinates, crop_type, planting_date)

-- Analysis and Results
crop_images (id, field_id, image_url, uploaded_at, status)
analysis_results (id, image_id, disease_detected, confidence, recommendations)
planting_recommendations (id, field_id, crop_type, optimal_date, confidence)
harvest_predictions (id, field_id, predicted_date, yield_estimate)

-- Historical Data
weather_data (id, location, date, temperature, humidity, rainfall, wind)
soil_data (id, location, ph_level, nutrients, moisture, test_date)
```

## AI/ML Models

### Disease Detection Model
- **Architecture**: PyTorch-based CNN (ResNet/EfficientNet)
- **Model File**: `ai/models/best_model.pth`
- **Inference**: `ai/models/torch_inference.py`
- **Task**: Multi-class crop disease classification
- **Output**: Disease type + confidence score + treatment recommendations

### Model Architecture
- Pre-trained models with transfer learning
- Fine-tuned on agricultural disease datasets
- Confidence scoring for reliability
- Multi-stage processing pipeline

## Development Setup

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd frontend/web
npm install
cp env.example .env
# Update VITE_API_URL in .env if needed
npm run dev  # Starts on http://localhost:3000
```

### Environment Variables
- **Backend**: Configure database URL, JWT secret, AWS credentials (if using S3)
- **Frontend**: `VITE_API_URL=http://localhost:8000/api`

## Git Status

### Current Branch
- `main` (no upstream tracking configured)

### Recent Changes
- Modified disease detection model integration
- Updated frontend with globe visualization on home page
- Enhanced UI/UX for dashboard and analysis pages
- Updated API routes for analysis and recommendations
- Improved authentication flow

### Deleted Files
- Mobile app files (React Native - being phased out)
- Old Keras model (`crop_disease_model_final.keras`)
- Legacy documentation (`AgriFinSight.docx`)

### Untracked Files
- PyTorch model files (`best_model.pth`)
- New inference scripts (`torch_inference.py`, `generic_model.py`)
- Backend uploads directory

## Development Phases

### Phase 1: Foundation ✅
- [x] Project structure setup
- [x] Git repository initialization
- [x] Basic FastAPI backend
- [x] User authentication
- [x] Database setup

### Phase 2: AI Features (In Progress)
- [x] Image processing pipeline
- [x] Disease detection model (PyTorch)
- [x] Analysis results API
- [ ] Model optimization and accuracy improvements

### Phase 3: Predictive Analytics (Planned)
- [ ] Weather API integration
- [ ] Planting recommendations
- [ ] Harvest predictions

### Phase 4: User Experience (Planned)
- [x] Web app development (React + TypeScript)
- [ ] Mobile app enhancements
- [ ] User interface polish
- [ ] Testing and validation

## Key Features Implemented

### Frontend
- ✅ User authentication (login/register)
- ✅ Responsive dashboard with modern UI
- ✅ Interactive globe visualization on home page (amCharts 5)
- ✅ Disease detection image upload
- ✅ Farm management interface
- ✅ Real-time analysis results display
- ✅ Tailwind CSS styling with custom components

### Backend
- ✅ JWT-based authentication
- ✅ RESTful API with FastAPI
- ✅ Image upload and storage
- ✅ PyTorch model integration for disease detection
- ✅ Database models with SQLAlchemy
- ✅ CORS configuration for frontend integration

### AI/ML
- ✅ PyTorch-based disease detection
- ✅ Image preprocessing pipeline
- ✅ Confidence scoring
- ✅ Treatment recommendations
- ⏳ Model training pipeline (in progress)

## Code Style and Conventions

### Python (Backend)
- PEP 8 style guide
- Type hints for function signatures
- Async/await for async operations
- Pydantic for data validation
- SQLAlchemy for ORM

### TypeScript/React (Frontend)
- Functional components with hooks
- TypeScript strict mode
- ESLint for code quality
- Tailwind CSS utility classes
- Custom utility functions (cn for class names)

## Common Commands

### Backend
```bash
# Run development server
uvicorn app.main:app --reload

# Run with custom host/port
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Install dependencies
pip install -r requirements.txt

# Format code
black app/

# Lint code
flake8 app/
```

### Frontend
```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Type check
tsc --noEmit
```

## API Documentation

When the backend is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Important Notes

### For AI Development
- Model files are stored in `ai/models/`
- Current model uses PyTorch (migrated from TensorFlow/Keras)
- Inference logic in `backend/app/services/ai_service.py`
- Images stored locally in `backend/uploads/` (consider S3 for production)

### For Frontend Development
- API base URL configured in `.env` file
- Authentication token stored in localStorage
- All API calls go through `src/services/api.ts`
- Tailwind CSS for styling - use utility classes
- amCharts 5 for advanced visualizations (globe on home page)

### For Backend Development
- FastAPI auto-generates OpenAPI docs
- Use Pydantic schemas for request/response validation
- Database migrations with Alembic (if configured)
- JWT tokens expire after configured time
- CORS enabled for frontend origin

## Testing Strategy

### Backend Testing
- Unit tests with pytest
- Async test support with pytest-asyncio
- API endpoint testing
- Model inference testing

### Frontend Testing
- Component testing (to be implemented)
- Integration testing
- E2E testing with user flows

## Deployment Considerations

### Production Checklist
- [ ] Configure production database (PostgreSQL)
- [ ] Set up Redis for caching
- [ ] Configure S3 or cloud storage for images
- [ ] Set environment variables
- [ ] Enable HTTPS
- [ ] Configure CORS properly
- [ ] Set up monitoring and logging
- [ ] Configure backup strategy
- [ ] Set up CI/CD pipeline

### Infrastructure
- Docker containers for backend and frontend
- Kubernetes for orchestration (optional)
- AWS/GCP for cloud hosting
- CDN for static assets

## Known Issues and Limitations

### Current Limitations
- Limited offline functionality
- Disease detection limited to trained crop types
- Local file storage (not scalable)
- No real-time notifications
- Limited language support

### Technical Debt
- Need to implement proper error handling
- Add comprehensive logging
- Implement rate limiting
- Add input validation
- Set up database migrations
- Add comprehensive test coverage

## Future Enhancements

### Short Term
- Improve model accuracy
- Add more crop disease types
- Implement weather API integration
- Enhanced error handling and validation
- Better image optimization

### Long Term
- IoT sensor integration
- Community features (farmer network)
- Market price predictions
- Supply chain management
- Mobile app improvements
- Multi-language support

## Contact and Resources

### Project Repository
- Git repository initialized locally
- Consider setting up GitHub/GitLab remote

### Documentation
- API docs: `/backend/app/main.py` (auto-generated)
- Architecture: `TECHNICAL_ARCHITECTURE.md`
- MVP Spec: `MVP_SPECIFICATION.md`
- Setup Guide: `SETUP_INSTRUCTIONS.md`

## Tips for Working with This Project

### When Adding New Features
1. Update relevant documentation files
2. Follow existing code structure and patterns
3. Add appropriate error handling
4. Update API documentation if adding endpoints
5. Test both frontend and backend integration
6. Commit with descriptive messages

### When Debugging
1. Check backend logs in terminal
2. Use FastAPI's `/docs` for API testing
3. Check browser console for frontend errors
4. Verify environment variables are set
5. Ensure database is running and accessible

### When Deploying
1. Test thoroughly in staging environment
2. Update environment variables for production
3. Ensure all dependencies are listed
4. Configure proper logging and monitoring
5. Set up backup and recovery procedures

---

**Last Updated**: October 2025
**Project Status**: Active Development - MVP Phase
**Primary Focus**: Disease detection, planting recommendations, and web interface
