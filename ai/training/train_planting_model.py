"""
Train Planting Time Prediction Model
Generates synthetic agricultural data and trains a Gradient Boosting model
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seed for reproducibility
np.random.seed(42)


def generate_planting_data(n_samples=5000):
    """
    Generate synthetic planting recommendation data

    Features:
    - Weather: temperature, rainfall, humidity, wind speed
    - Soil: temperature, moisture, pH, N/P/K levels
    - Location: elevation, latitude, longitude

    Target:
    - days_to_plant: Optimal days to wait before planting (0-90 days)
    """

    print(f"Generating {n_samples} synthetic training samples...")

    data = []

    for _ in range(n_samples):
        # Location (global agricultural regions)
        latitude = np.random.uniform(-60, 60)  # Avoid polar regions
        longitude = np.random.uniform(-180, 180)
        elevation = np.random.uniform(0, 1500)  # meters

        # Weather conditions (current 7-day average)
        temp_avg = np.random.uniform(10, 40)  # Celsius
        temp_min = temp_avg - np.random.uniform(2, 8)
        temp_max = temp_avg + np.random.uniform(2, 8)
        humidity_avg = np.random.uniform(30, 95)  # percentage
        rainfall_total = np.random.uniform(0, 300)  # mm over 7 days
        rainfall_days = np.random.randint(0, 8)  # rainy days in week
        wind_speed_avg = np.random.uniform(0, 20)  # km/h

        # Soil conditions
        soil_temperature = temp_avg + np.random.uniform(-3, 5)
        soil_moisture = np.random.uniform(0.1, 0.9)  # fraction
        soil_ph = np.random.uniform(4.5, 8.5)
        soil_nitrogen = np.random.uniform(20, 100)  # mg/kg
        soil_phosphorus = np.random.uniform(10, 80)  # mg/kg
        soil_potassium = np.random.uniform(20, 120)  # mg/kg

        # Calculate optimal planting time (days from now)
        days_to_plant = calculate_planting_days(
            temp_avg, temp_min, temp_max, humidity_avg, rainfall_total,
            rainfall_days, wind_speed_avg, soil_temperature, soil_moisture,
            soil_ph, soil_nitrogen, elevation, latitude
        )

        data.append({
            'temperature_avg': temp_avg,
            'temperature_min': temp_min,
            'temperature_max': temp_max,
            'humidity_avg': humidity_avg,
            'rainfall_total': rainfall_total,
            'rainfall_days': rainfall_days,
            'wind_speed_avg': wind_speed_avg,
            'soil_temperature': soil_temperature,
            'soil_moisture': soil_moisture,
            'soil_ph': soil_ph,
            'soil_nitrogen': soil_nitrogen,
            'soil_phosphorus': soil_phosphorus,
            'soil_potassium': soil_potassium,
            'elevation': elevation,
            'latitude': latitude,
            'longitude': longitude,
            'days_to_plant': days_to_plant
        })

    df = pd.DataFrame(data)

    print(f"✓ Generated {len(df)} samples")
    print(f"  Target range: {df['days_to_plant'].min():.1f} - {df['days_to_plant'].max():.1f} days")
    print(f"  Target mean: {df['days_to_plant'].mean():.1f} days")

    return df


def calculate_planting_days(temp_avg, temp_min, temp_max, humidity, rainfall,
                            rainy_days, wind_speed, soil_temp, soil_moisture,
                            soil_ph, soil_n, elevation, latitude):
    """
    Calculate optimal days to wait before planting based on conditions

    Logic:
    - If conditions are optimal now: 0-5 days
    - If minor issues: 5-20 days
    - If significant issues: 20-60 days
    - If very poor conditions: 60-90 days
    """

    penalty = 0

    # Temperature penalty (optimal: 18-30°C)
    if temp_avg < 15:
        penalty += (15 - temp_avg) * 3  # Too cold - wait for warmth
    elif temp_avg > 35:
        penalty += (temp_avg - 35) * 2  # Too hot - wait for cooler weather

    if temp_min < 10:
        penalty += (10 - temp_min) * 1.5  # Frost risk

    # Rainfall penalty
    if rainfall < 20:
        penalty += (20 - rainfall) * 0.5  # Too dry - wait for rain
    elif rainfall > 250:
        penalty += (rainfall - 250) * 0.3  # Too wet - wait for drier conditions

    if rainy_days > 6:
        penalty += (rainy_days - 6) * 3  # Waterlogging risk

    # Soil moisture penalty (optimal: 0.4-0.7)
    if soil_moisture < 0.3:
        penalty += (0.3 - soil_moisture) * 30  # Too dry
    elif soil_moisture > 0.8:
        penalty += (soil_moisture - 0.8) * 40  # Too wet

    # Soil pH penalty (optimal: 5.5-7.5)
    if soil_ph < 5.0:
        penalty += (5.0 - soil_ph) * 10  # Too acidic
    elif soil_ph > 8.0:
        penalty += (soil_ph - 8.0) * 10  # Too alkaline

    # Soil temperature penalty (optimal: 15-25°C)
    if soil_temp < 12:
        penalty += (12 - soil_temp) * 4  # Too cold for germination
    elif soil_temp > 30:
        penalty += (soil_temp - 30) * 2

    # Soil nitrogen bonus (higher N = faster planting)
    if soil_n > 60:
        penalty -= 5
    elif soil_n < 30:
        penalty += 10  # Low fertility - need amendment time

    # Wind speed penalty
    if wind_speed > 15:
        penalty += (wind_speed - 15) * 1.5  # High winds damage young plants

    # Seasonal adjustment (latitude-based)
    # Near equator: less seasonal variation
    # Higher latitudes: more seasonal constraints
    seasonal_factor = abs(latitude) / 60 * 10

    # Add some randomness for realism
    noise = np.random.uniform(-5, 5)

    days = max(0, min(90, penalty + seasonal_factor + noise))

    return round(days, 1)


def train_planting_model(data_path=None, save_dir='../models'):
    """Train the planting time prediction model"""

    print("=" * 70)
    print("PLANTING TIME PREDICTOR - MODEL TRAINING")
    print("=" * 70)

    # Generate or load data
    if data_path and os.path.exists(data_path):
        print(f"\n1. Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print("\n1. Generating synthetic training data...")
        df = generate_planting_data(n_samples=5000)

        # Save generated data
        os.makedirs('../data', exist_ok=True)
        data_save_path = '../data/planting_training_data.csv'
        df.to_csv(data_save_path, index=False)
        print(f"   Saved training data to: {data_save_path}")

    print(f"   Dataset shape: {df.shape}")

    # Prepare features and target
    print("\n2. Preparing features and target...")
    feature_cols = [
        'temperature_avg', 'temperature_min', 'temperature_max',
        'humidity_avg', 'rainfall_total', 'rainfall_days',
        'wind_speed_avg', 'soil_temperature', 'soil_moisture',
        'soil_ph', 'soil_nitrogen', 'soil_phosphorus', 'soil_potassium',
        'elevation', 'latitude', 'longitude'
    ]

    X = df[feature_cols]
    y = df['days_to_plant']

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
    print("\n5. Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
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

    model_save_path = os.path.join(save_dir, 'planting_predictor.joblib')

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
    axes[0].set_xlabel('Actual Days to Plant')
    axes[0].set_ylabel('Predicted Days to Plant')
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

    plot_path = os.path.join(save_dir, 'planting_model_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Plots saved to: {plot_path}")

    plt.close()


if __name__ == "__main__":
    # Train the model
    model, scaler, importance = train_planting_model()

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. The model is ready to use in planting_service.py")
    print("  2. Update planting_predictor.py to load this trained model")
    print("  3. Test predictions with real farm data")
    print("=" * 70)
