"""
Train Harvest Time Prediction Model
Generates synthetic crop growth data and trains a Random Forest model
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seed for reproducibility
np.random.seed(42)

# Crop maturity parameters
CROP_MATURITY = {
    'maize': 120,
    'rice': 150,
    'cassava': 300,
    'tomato': 90,
    'soybean': 100,
    'groundnut': 120
}


def generate_harvest_data(n_samples=5000):
    """
    Generate synthetic harvest prediction data

    Features:
    - Crop growth: days since planting, height, leaf count, flowering stage
    - Weather: temperature, humidity, rainfall
    - Soil: moisture, nutrients (N/P/K)
    - Stress factors: disease pressure, pest pressure

    Target:
    - days_to_harvest: Days remaining until optimal harvest (0-60 days)
    """

    print(f"Generating {n_samples} synthetic training samples...")

    data = []
    crops = list(CROP_MATURITY.keys())

    for _ in range(n_samples):
        # Select random crop
        crop = np.random.choice(crops)
        maturity_days = CROP_MATURITY[crop]

        # Crop age (could be anywhere in growth cycle)
        days_since_planting = np.random.randint(1, maturity_days + 30)

        # Growth stage (0-1 scale, where 1 is mature)
        growth_progress = min(1.0, days_since_planting / maturity_days)

        # Plant metrics (correlate with growth stage)
        if crop in ['maize', 'rice', 'soybean', 'groundnut']:
            plant_height = growth_progress * np.random.uniform(80, 200)  # cm
            leaf_count = growth_progress * np.random.randint(15, 40)
        else:  # tomato, cassava
            plant_height = growth_progress * np.random.uniform(50, 150)
            leaf_count = growth_progress * np.random.randint(20, 60)

        # Flowering stage (0-1 scale)
        if growth_progress > 0.5:
            flowering_stage = min(1.0, (growth_progress - 0.5) * 2 + np.random.uniform(-0.1, 0.1))
        else:
            flowering_stage = 0

        # Fruit development (0-1 scale, starts after flowering)
        if flowering_stage > 0.5:
            fruit_development = min(1.0, (flowering_stage - 0.5) * 2 + np.random.uniform(-0.1, 0.1))
        else:
            fruit_development = 0

        # Weather conditions (recent averages)
        temperature_avg = np.random.uniform(15, 35)  # Celsius
        humidity_avg = np.random.uniform(40, 90)  # percentage
        rainfall_total = np.random.uniform(0, 200)  # mm over 14 days

        # Soil conditions
        soil_moisture = np.random.uniform(0.2, 0.8)  # fraction
        soil_nitrogen = np.random.uniform(20, 100)  # mg/kg
        soil_phosphorus = np.random.uniform(10, 80)  # mg/kg
        soil_potassium = np.random.uniform(20, 120)  # mg/kg

        # Stress factors (0-1 scale, higher is worse)
        disease_pressure = np.random.uniform(0, 0.6)
        pest_pressure = np.random.uniform(0, 0.5)

        # If crop is stressed, slow down maturity
        stress_factor = (disease_pressure + pest_pressure) / 2

        # Calculate days to harvest
        days_to_harvest = calculate_harvest_days(
            crop, days_since_planting, maturity_days, growth_progress,
            flowering_stage, fruit_development, temperature_avg,
            humidity_avg, rainfall_total, soil_moisture, soil_nitrogen,
            disease_pressure, pest_pressure, stress_factor
        )

        # Encode growth stage as numeric
        if growth_progress < 0.1:
            growth_stage = 0  # germination
        elif growth_progress < 0.25:
            growth_stage = 1  # seedling
        elif growth_progress < 0.5:
            growth_stage = 2  # vegetative
        elif growth_progress < 0.75:
            growth_stage = 3  # flowering
        elif growth_progress < 0.95:
            growth_stage = 4  # fruit_development
        else:
            growth_stage = 5  # maturation

        data.append({
            'days_since_planting': days_since_planting,
            'plant_height': plant_height,
            'leaf_count': leaf_count,
            'flowering_stage': flowering_stage,
            'fruit_development': fruit_development,
            'growth_stage': growth_stage,
            'temperature_avg': temperature_avg,
            'humidity_avg': humidity_avg,
            'rainfall_total': rainfall_total,
            'soil_moisture': soil_moisture,
            'soil_nitrogen': soil_nitrogen,
            'soil_phosphorus': soil_phosphorus,
            'soil_potassium': soil_potassium,
            'disease_pressure': disease_pressure,
            'pest_pressure': pest_pressure,
            'days_to_harvest': days_to_harvest
        })

    df = pd.DataFrame(data)

    print(f"✓ Generated {len(df)} samples")
    print(f"  Target range: {df['days_to_harvest'].min():.1f} - {df['days_to_harvest'].max():.1f} days")
    print(f"  Target mean: {df['days_to_harvest'].mean():.1f} days")

    return df


def calculate_harvest_days(crop, days_planted, maturity_days, growth_progress,
                           flowering, fruit_dev, temp, humidity, rainfall,
                           soil_moisture, soil_n, disease, pests, stress):
    """
    Calculate days remaining until optimal harvest

    Logic:
    - If crop is very immature: many days remaining
    - If crop is near maturity: few days remaining
    - If crop is overdue: 0 days (harvest immediately)
    - Stress factors slow maturity
    - Weather affects ripening speed
    """

    # Base calculation: remaining days to maturity
    base_days_remaining = max(0, maturity_days - days_planted)

    # Adjust for growth progress (more accurate than just days)
    if growth_progress >= 1.0:
        # Crop is mature
        days_remaining = 0
    elif growth_progress >= 0.9:
        # Very close to maturity
        days_remaining = np.random.uniform(0, 7)
    elif growth_progress >= 0.75:
        # Approaching maturity
        days_remaining = base_days_remaining * 0.3
    else:
        # Still developing
        days_remaining = base_days_remaining

    # Weather adjustments
    # Optimal temperature speeds up ripening
    if 20 <= temp <= 28:
        days_remaining *= 0.95  # 5% faster
    elif temp < 15 or temp > 35:
        days_remaining *= 1.15  # 15% slower

    # Good soil conditions speed up growth
    if soil_moisture > 0.5 and soil_n > 50:
        days_remaining *= 0.9
    elif soil_moisture < 0.3 or soil_n < 30:
        days_remaining *= 1.2

    # Stress factors slow down maturity
    stress_multiplier = 1 + (stress * 0.5)  # Up to 50% slower if very stressed
    days_remaining *= stress_multiplier

    # High disease/pest pressure may force early harvest
    if disease > 0.7 or pests > 0.6:
        days_remaining *= 0.7  # Harvest early to save crop

    # For crops with fruit (tomato), check fruit development
    if crop == 'tomato' and fruit_dev > 0.8:
        days_remaining = min(days_remaining, 5)  # Ready to harvest

    # Add some randomness
    noise = np.random.uniform(-3, 3)
    days_remaining = max(0, days_remaining + noise)

    # Cap at 60 days (we don't predict far into future)
    days_remaining = min(60, days_remaining)

    return round(days_remaining, 1)


def train_harvest_model(data_path=None, save_dir='../models'):
    """Train the harvest time prediction model"""

    print("=" * 70)
    print("HARVEST TIME PREDICTOR - MODEL TRAINING")
    print("=" * 70)

    # Generate or load data
    if data_path and os.path.exists(data_path):
        print(f"\n1. Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("\n1. Generating synthetic training data...")
        df = generate_harvest_data(n_samples=5000)

        # Save generated data
        os.makedirs('../data', exist_ok=True)
        data_save_path = '../data/harvest_training_data.csv'
        df.to_csv(data_save_path, index=False)
        print(f"   Saved training data to: {data_save_path}")

    print(f"   Dataset shape: {df.shape}")

    # Prepare features and target
    print("\n2. Preparing features and target...")
    feature_cols = [
        'days_since_planting', 'plant_height', 'leaf_count',
        'flowering_stage', 'fruit_development', 'growth_stage',
        'temperature_avg', 'humidity_avg', 'rainfall_total',
        'soil_moisture', 'soil_nitrogen', 'soil_phosphorus',
        'soil_potassium', 'disease_pressure', 'pest_pressure'
    ]

    X = df[feature_cols]
    y = df['days_to_harvest']

    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X)}")

    # Split data
    print("\n3. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")

    # Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\n5. Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    print("   Training in progress...")
    model.fit(X_train_scaled, y_train)
    print("   ✓ Training complete")

    # Cross-validation
    print("\n6. Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5,
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    cv_rmse = np.sqrt(-cv_scores)
    print(f"   CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f} days")

    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   RMSE:  {rmse:.2f} days")
    print(f"   MAE:   {mae:.2f} days")
    print(f"   R²:    {r2:.4f}")

    # Feature importance
    print("\n8. Feature importance:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:25s} {row['importance']:.4f}")

    # Save model
    print("\n9. Saving model...")
    os.makedirs(save_dir, exist_ok=True)

    model_save_path = os.path.join(save_dir, 'harvest_predictor.joblib')

    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': feature_cols,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std()
        },
        'feature_importance': feature_importance.to_dict('records'),
        'trained_at': datetime.now().isoformat()
    }, model_save_path)

    print(f"   ✓ Model saved to: {model_save_path}")

    # Plot results
    print("\n10. Generating visualizations...")
    plot_results(y_test, y_pred, feature_importance, save_dir)

    print("\n" + "=" * 70)
    print(f"✓ Training complete!")
    print(f"  Model accuracy: RMSE = {rmse:.2f} days, R² = {r2:.4f}")
    print(f"  Model saved to: {model_save_path}")
    print("=" * 70)

    return model, scaler, feature_importance


def plot_results(y_test, y_pred, feature_importance, save_dir):
    """Generate and save visualizations"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prediction vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Days to Harvest')
    axes[0].set_ylabel('Predicted Days to Harvest')
    axes[0].set_title('Prediction vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Feature importance
    top_features = feature_importance.head(10)
    axes[1].barh(range(len(top_features)), top_features['importance'])
    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features['feature'])
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Top 10 Feature Importance')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'harvest_model_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Plots saved to: {plot_path}")

    plt.close()


if __name__ == "__main__":
    # Train the model
    model, scaler, importance = train_harvest_model()

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. The model is ready to use in harvest_service.py")
    print("  2. Update harvest_predictor.py to load this trained model")
    print("  3. Test predictions with real crop data")
    print("=" * 70)
