# ML-Based Crop Recommendation System

## Overview

This ML-based crop recommendation system uses a PyTorch neural network to predict the most suitable crops for a given farm based on its geospatial and environmental data.

## Features

### Input Features (9 total):
1. **Latitude** - Geographic location
2. **Longitude** - Geographic location
3. **Altitude/Elevation** - Height above sea level (meters)
4. **Average Temperature** - Annual average temperature (°C)
5. **Average Annual Rainfall** - Total annual rainfall (mm)
6. **Soil pH** - Soil acidity/alkalinity (4.0-8.5)
7. **Soil Type** (encoded) - Sandy, loam, clay, etc.
8. **Climate Zone** (encoded) - Tropical, subtropical, temperate, etc.
9. **Terrain Type** (encoded) - Flat, hilly, mountainous, etc.

### Output:
- **Crop Classification** - Probabilities for 6 crops:
  - Maize
  - Rice
  - Cassava
  - Tomato
  - Soybean
  - Groundnut

- **Suitability Score** - Predicted compatibility (0-100%)

## Model Architecture

```
Input (9 features)
    ↓
Dense Layer (128 neurons) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense Layer (64 neurons) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense Layer (32 neurons) + BatchNorm + ReLU + Dropout(0.3)
    ↓
    ├─→ Classification Head → 6 crop classes
    └─→ Regression Head → Suitability score (0-100)
```

## Training Pipeline

### Step 1: Generate Synthetic Dataset

```bash
cd ai/training
python generate_crop_dataset.py
```

This generates:
- `../data/crop_recommendation_dataset.csv` - 6,000 samples (1,000 per crop)
- `../data/encoders.json` - Feature encoding mappings
- `../data/dataset_stats.json` - Dataset statistics

**Dataset Composition:**
- 30% optimal conditions
- 30% good conditions
- 25% moderate conditions
- 15% poor conditions

### Step 2: Train the Model

```bash
python train_crop_model.py
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Loss: 70% Classification + 30% Regression (MSE)
- Batch Size: 64
- Epochs: 100 (with early stopping)
- Train/Val/Test Split: 70/15/15

**Output:**
- `../models/crop_recommendation_model.pth` - Trained model
- `../models/training_history.png` - Loss and accuracy curves
- `../models/confusion_matrix.png` - Classification performance

### Expected Performance

- **Training Accuracy**: ~95%+
- **Validation Accuracy**: ~92%+
- **Test Accuracy**: ~90%+

## Using the Trained Model

### Backend Integration

The model is automatically loaded in the backend service:

```python
from app.services.ml_crop_recommendation_service import ml_crop_recommendation_service

# Generate recommendations
farm_data = {
    "latitude": 6.5,
    "longitude": -1.5,
    "altitude": 300,
    "avg_temperature": 27,
    "avg_annual_rainfall": 1200,
    "soil_ph": 6.5,
    "soil_type": "loam",
    "climate_zone": "tropical",
    "terrain_type": "flat"
}

recommendations = ml_crop_recommendation_service.generate_recommendations(farm_data)
```

### API Endpoint

```bash
POST /api/farms/{farm_id}/crop-recommendations?use_ml=true
```

**Query Parameters:**
- `use_ml=true` - Use ML model (default)
- `use_ml=false` - Use rule-based system

## ML vs Rule-Based Comparison

| Feature | ML Model | Rule-Based |
|---------|----------|------------|
| **Learning** | Learns from data patterns | Fixed rules |
| **Adaptability** | Can improve with more data | Manual updates needed |
| **Precision** | Higher accuracy with training | Good for known scenarios |
| **Robustness** | Handles edge cases better | May miss unusual patterns |
| **Interpretability** | Black box (needs explanation) | Transparent logic |
| **Speed** | Very fast (GPU accelerated) | Fast |

## Dataset Details

### Synthetic Data Generation

The dataset is generated using agronomic knowledge and realistic parameter distributions:

```python
# Example: Maize optimal conditions
{
    "temp_range": (18, 32),
    "temp_optimal": 25,
    "rainfall_range": (500, 1000),
    "rainfall_optimal": 750,
    "ph_range": (5.5, 7.5),
    "ph_optimal": 6.5,
    "altitude_range": (0, 2500),
    "altitude_optimal": 1000,
    "soil_types": ["loam", "sandy loam", "clay loam"],
    "climate_zones": ["tropical", "subtropical", "temperate"]
}
```

### Suitability Score Calculation

The synthetic dataset includes calculated suitability scores based on:
- **Temperature match** (25% weight)
- **Rainfall match** (25% weight)
- **pH match** (20% weight)
- **Altitude match** (15% weight)
- **Soil type match** (10% weight)
- **Climate zone match** (5% weight)

## Extending the Model

### Add More Crops

1. Update `CROP_PROFILES` in `generate_crop_dataset.py`
2. Update `CROP_INFO` in `ml_crop_recommendation_service.py`
3. Regenerate dataset
4. Retrain model

### Use Real Data

Replace synthetic generation with real agricultural data:
```python
# Load from CSV/database
df = pd.read_csv('real_crop_yield_data.csv')

# Ensure same features
required_features = [
    'latitude', 'longitude', 'altitude',
    'avg_temperature', 'avg_annual_rainfall', 'soil_ph',
    'soil_type', 'climate_zone', 'terrain_type',
    'recommended_crop'
]
```

### Hyperparameter Tuning

Modify `train_crop_model.py`:
```python
train_model(
    epochs=150,              # More epochs
    batch_size=32,           # Smaller batches
    learning_rate=0.0005,    # Lower learning rate
)
```

## Troubleshooting

### Model Not Loading

If you see: `"ML recommendations will not be available"`

**Solution:**
```bash
# 1. Generate dataset
cd ai/training
python generate_crop_dataset.py

# 2. Train model
python train_crop_model.py

# 3. Verify model exists
ls -lh ../models/crop_recommendation_model.pth

# 4. Restart backend
cd ../../backend
uvicorn app.main:app --reload
```

### Low Accuracy

If test accuracy < 85%:

1. **Check data quality** - Ensure realistic feature ranges
2. **Increase dataset size** - Generate more samples per crop
3. **Adjust model architecture** - Try different hidden layer sizes
4. **Tune hyperparameters** - Experiment with learning rate, dropout

### Overfitting

If train accuracy >> validation accuracy:

1. **Increase dropout** - Change from 0.3 to 0.4 or 0.5
2. **Add regularization** - Increase weight_decay
3. **More training data** - Generate additional samples
4. **Early stopping** - Reduce patience parameter

## Performance Optimization

### GPU Training

If you have CUDA-capable GPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Training time: ~2-3 minutes on GPU vs ~10-15 minutes on CPU

### Inference Speed

Single prediction: ~5-10ms (CPU) / ~1-2ms (GPU)

## Future Enhancements

1. **Multi-task Learning** - Predict yield and profit simultaneously
2. **Ensemble Methods** - Combine multiple models
3. **Attention Mechanisms** - Focus on important features
4. **Transfer Learning** - Use pre-trained agricultural models
5. **Explainable AI** - Add SHAP/LIME for interpretability
6. **Active Learning** - Incorporate farmer feedback
7. **Time Series** - Add seasonal/temporal features
8. **Weather Integration** - Include forecast data
9. **Market Dynamics** - Add price predictions
10. **Regional Models** - Train separate models per region

## Citation

If using this ML system, please cite:

```
AgriFinSight Crop Recommendation System
Neural Network-based Crop Suitability Prediction
Based on Geospatial and Environmental Features
Version 1.0.0 (2025)
```

## License

Part of the AgriFinSight project - AI-powered farming assistant.
