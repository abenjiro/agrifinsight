# AgriFinSight

AI-powered farming assistant helping smallholder farmers with crop health monitoring, planting predictions, and harvest optimization.

## Project Overview

AgriFinSight is designed to help farmers make data-driven decisions about:
- **Crop Health Monitoring**: AI-powered disease detection and treatment recommendations
- **Planting Time Predictions**: Weather-based recommendations for optimal planting
- **Harvest Optimization**: Timing predictions to maximize yield and reduce losses

## Technology Stack

- **Backend**: Python + FastAPI + PostgreSQL
- **Frontend**: React Native (mobile-first) + React (web)
- **AI/ML**: TensorFlow/PyTorch + OpenCV
- **Cloud**: AWS (free tier to start)
- **Database**: PostgreSQL
- **Cache**: Redis

## Project Structure

```
agrifinsight/
├── backend/                 # FastAPI backend services
│   ├── app/
│   │   ├── models/         # Database models
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   └── utils/          # Utility functions
│   └── tests/              # Backend tests
├── frontend/
│   ├── mobile/             # React Native mobile app
│   └── web/                # React web dashboard
├── ai/                     # AI/ML models and training
│   ├── models/             # Trained models
│   ├── training/           # Model training scripts
│   ├── inference/          # Model inference services
│   └── data/               # Training data
├── data/                   # External data sources
│   ├── weather/            # Weather data
│   ├── soil/               # Soil data
│   └── crop_database/      # Crop information
├── docs/                   # Documentation
├── deployment/             # Deployment configurations
│   ├── docker/             # Docker configurations
│   └── kubernetes/         # K8s configurations
└── tests/                  # Integration tests
```

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+
- PostgreSQL 14+
- Docker (optional)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend/mobile
npm install
npx expo start
```

## Development Phases

### Phase 1: Foundation (Weeks 1-2)
- [x] Project structure setup
- [x] Git repository initialization
- [x] Basic FastAPI backend
- [x] User authentication
- [x] Database setup

### Phase 2: AI Features (Weeks 3-4)
- [x] Image processing pipeline
- [x] Disease detection model
- [x] Analysis results API

### Phase 3: Predictive Analytics (Weeks 5-6)
- [x] Weather API integration
- [x] Planting recommendations
- [x] Harvest predictions

### Phase 4: User Experience (Weeks 7-8)
- [ ] Mobile app development
- [x] User interface polish
- [x] Testing and validation

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please contact the development team.
