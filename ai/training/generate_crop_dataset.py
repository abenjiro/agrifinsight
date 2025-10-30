"""
Generate synthetic training dataset for crop recommendation ML model
Based on geospatial features and agronomic knowledge
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime

# Crop database with optimal conditions
CROP_PROFILES = {
    "Maize": {
        "temp_range": (18, 32),
        "temp_optimal": 25,
        "rainfall_range": (500, 1000),
        "rainfall_optimal": 750,
        "ph_range": (5.5, 7.5),
        "ph_optimal": 6.5,
        "altitude_range": (0, 2500),
        "altitude_optimal": 1000,
        "soil_types": ["loam", "sandy loam", "clay loam"],
        "climate_zones": ["tropical", "subtropical", "temperate"],
        "terrain_types": ["flat", "gently sloping", "rolling"]
    },
    "Rice": {
        "temp_range": (20, 38),
        "temp_optimal": 28,
        "rainfall_range": (1000, 3000),
        "rainfall_optimal": 2000,
        "ph_range": (5.0, 7.0),
        "ph_optimal": 6.0,
        "altitude_range": (0, 1500),
        "altitude_optimal": 500,
        "soil_types": ["clay", "clay loam", "silty clay"],
        "climate_zones": ["tropical", "subtropical"],
        "terrain_types": ["flat", "valley"]
    },
    "Cassava": {
        "temp_range": (20, 35),
        "temp_optimal": 27,
        "rainfall_range": (500, 1500),
        "rainfall_optimal": 1000,
        "ph_range": (4.5, 7.0),
        "ph_optimal": 6.0,
        "altitude_range": (0, 2000),
        "altitude_optimal": 800,
        "soil_types": ["sandy loam", "loam", "sandy"],
        "climate_zones": ["tropical", "subtropical"],
        "terrain_types": ["flat", "gently sloping", "rolling", "hilly"]
    },
    "Tomato": {
        "temp_range": (18, 27),
        "temp_optimal": 22,
        "rainfall_range": (600, 1300),
        "rainfall_optimal": 900,
        "ph_range": (6.0, 7.0),
        "ph_optimal": 6.5,
        "altitude_range": (0, 2000),
        "altitude_optimal": 1200,
        "soil_types": ["loam", "sandy loam"],
        "climate_zones": ["tropical", "subtropical", "temperate"],
        "terrain_types": ["flat", "gently sloping", "rolling"]
    },
    "Soybean": {
        "temp_range": (20, 30),
        "temp_optimal": 25,
        "rainfall_range": (500, 900),
        "rainfall_optimal": 700,
        "ph_range": (6.0, 7.5),
        "ph_optimal": 6.8,
        "altitude_range": (0, 2000),
        "altitude_optimal": 1000,
        "soil_types": ["loam", "sandy loam", "clay loam"],
        "climate_zones": ["tropical", "subtropical", "temperate"],
        "terrain_types": ["flat", "gently sloping"]
    },
    "Groundnut": {
        "temp_range": (20, 30),
        "temp_optimal": 25,
        "rainfall_range": (500, 1000),
        "rainfall_optimal": 750,
        "ph_range": (5.5, 6.5),
        "ph_optimal": 6.0,
        "altitude_range": (0, 1500),
        "altitude_optimal": 700,
        "soil_types": ["sandy loam", "loam", "sandy"],
        "climate_zones": ["tropical", "subtropical"],
        "terrain_types": ["flat", "gently sloping"]
    }
}

# Encoding mappings
SOIL_TYPE_MAPPING = {
    "sandy": 0, "sandy loam": 1, "loam": 2, "clay loam": 3,
    "clay": 4, "silty clay": 5, "silt": 6
}

CLIMATE_ZONE_MAPPING = {
    "tropical": 0, "subtropical": 1, "temperate": 2, "arid": 3, "semi-arid": 4
}

TERRAIN_TYPE_MAPPING = {
    "flat": 0, "gently sloping": 1, "rolling": 2, "hilly": 3, "mountainous": 4, "valley": 5
}


def calculate_suitability_score(features: dict, crop_profile: dict) -> float:
    """
    Calculate crop suitability score (0-100) based on how well features match crop profile
    """
    score = 0.0

    # Temperature score (25%)
    temp = features['avg_temperature']
    temp_min, temp_max = crop_profile['temp_range']
    temp_optimal = crop_profile['temp_optimal']

    if temp_min <= temp <= temp_max:
        temp_deviation = abs(temp - temp_optimal)
        temp_score = max(0, 25 - (temp_deviation / (temp_max - temp_min) * 25))
        score += temp_score

    # Rainfall score (25%)
    rainfall = features['avg_annual_rainfall']
    rain_min, rain_max = crop_profile['rainfall_range']
    rain_optimal = crop_profile['rainfall_optimal']

    if rain_min <= rainfall <= rain_max:
        rain_deviation = abs(rainfall - rain_optimal)
        rain_score = max(0, 25 - (rain_deviation / (rain_max - rain_min) * 25))
        score += rain_score

    # pH score (20%)
    ph = features['soil_ph']
    ph_min, ph_max = crop_profile['ph_range']
    ph_optimal = crop_profile['ph_optimal']

    if ph_min <= ph <= ph_max:
        ph_deviation = abs(ph - ph_optimal)
        ph_score = max(0, 20 - (ph_deviation / (ph_max - ph_min) * 20))
        score += ph_score

    # Altitude score (15%)
    altitude = features['altitude']
    alt_min, alt_max = crop_profile['altitude_range']
    alt_optimal = crop_profile['altitude_optimal']

    if alt_min <= altitude <= alt_max:
        alt_deviation = abs(altitude - alt_optimal)
        alt_score = max(0, 15 - (alt_deviation / (alt_max - alt_min) * 15))
        score += alt_score

    # Soil type match (10%)
    soil_type = features['soil_type']
    if soil_type in crop_profile['soil_types']:
        score += 10

    # Climate zone match (5%)
    climate_zone = features['climate_zone']
    if climate_zone in crop_profile['climate_zones']:
        score += 5

    return min(100, max(0, score))


def generate_sample(crop_name: str, crop_profile: dict, variation: str = 'optimal') -> dict:
    """
    Generate a single training sample for a crop
    variation: 'optimal', 'good', 'moderate', 'poor'
    """
    if variation == 'optimal':
        # Generate near-optimal conditions
        temp = np.random.normal(crop_profile['temp_optimal'], 1.5)
        rainfall = np.random.normal(crop_profile['rainfall_optimal'], 50)
        ph = np.random.normal(crop_profile['ph_optimal'], 0.2)
        altitude = np.random.normal(crop_profile['altitude_optimal'], 200)
        soil_type = np.random.choice(crop_profile['soil_types'])
        climate_zone = np.random.choice(crop_profile['climate_zones'])
        terrain_type = np.random.choice(crop_profile['terrain_types'])

    elif variation == 'good':
        # Generate good but not optimal conditions
        temp_range = crop_profile['temp_range']
        temp = np.random.uniform(
            temp_range[0] + 0.2 * (temp_range[1] - temp_range[0]),
            temp_range[1] - 0.2 * (temp_range[1] - temp_range[0])
        )

        rain_range = crop_profile['rainfall_range']
        rainfall = np.random.uniform(
            rain_range[0] + 0.2 * (rain_range[1] - rain_range[0]),
            rain_range[1] - 0.2 * (rain_range[1] - rain_range[0])
        )

        ph_range = crop_profile['ph_range']
        ph = np.random.uniform(
            ph_range[0] + 0.1 * (ph_range[1] - ph_range[0]),
            ph_range[1] - 0.1 * (ph_range[1] - ph_range[0])
        )

        altitude = np.random.uniform(crop_profile['altitude_range'][0], crop_profile['altitude_range'][1])
        soil_type = np.random.choice(crop_profile['soil_types'])
        climate_zone = np.random.choice(crop_profile['climate_zones'])
        terrain_type = np.random.choice(crop_profile['terrain_types'])

    elif variation == 'moderate':
        # Generate marginal conditions
        temp = np.random.uniform(crop_profile['temp_range'][0], crop_profile['temp_range'][1])
        rainfall = np.random.uniform(crop_profile['rainfall_range'][0], crop_profile['rainfall_range'][1])
        ph = np.random.uniform(crop_profile['ph_range'][0], crop_profile['ph_range'][1])
        altitude = np.random.uniform(crop_profile['altitude_range'][0], crop_profile['altitude_range'][1])

        # Mix of suitable and unsuitable soil types
        if np.random.random() > 0.3:
            soil_type = np.random.choice(crop_profile['soil_types'])
        else:
            all_soils = list(SOIL_TYPE_MAPPING.keys())
            soil_type = np.random.choice(all_soils)

        climate_zone = np.random.choice(crop_profile['climate_zones'])
        terrain_type = np.random.choice(list(TERRAIN_TYPE_MAPPING.keys()))

    else:  # poor
        # Generate unsuitable conditions
        temp_range = crop_profile['temp_range']
        if np.random.random() > 0.5:
            temp = np.random.uniform(temp_range[0] - 10, temp_range[0])
        else:
            temp = np.random.uniform(temp_range[1], temp_range[1] + 10)

        rain_range = crop_profile['rainfall_range']
        if np.random.random() > 0.5:
            rainfall = np.random.uniform(rain_range[0] - 300, rain_range[0])
        else:
            rainfall = np.random.uniform(rain_range[1], rain_range[1] + 500)

        ph = np.random.uniform(4.0, 8.5)
        altitude = np.random.uniform(0, 3000)

        all_soils = list(SOIL_TYPE_MAPPING.keys())
        soil_type = np.random.choice(all_soils)
        climate_zone = np.random.choice(list(CLIMATE_ZONE_MAPPING.keys()))
        terrain_type = np.random.choice(list(TERRAIN_TYPE_MAPPING.keys()))

    # Clip values to realistic ranges
    temp = np.clip(temp, 10, 45)
    rainfall = np.clip(rainfall, 200, 4000)
    ph = np.clip(ph, 4.0, 8.5)
    altitude = np.clip(altitude, 0, 3000)

    # Generate realistic lat/lon based on climate zone and altitude
    if climate_zone == 'tropical':
        latitude = np.random.uniform(-10, 10)
    elif climate_zone == 'subtropical':
        latitude = np.random.choice([1, -1]) * np.random.uniform(10, 30)
    else:  # temperate
        latitude = np.random.choice([1, -1]) * np.random.uniform(30, 50)

    longitude = np.random.uniform(-180, 180)

    features = {
        'latitude': latitude,
        'longitude': longitude,
        'altitude': altitude,
        'avg_temperature': temp,
        'avg_annual_rainfall': rainfall,
        'soil_ph': ph,
        'soil_type': soil_type,
        'climate_zone': climate_zone,
        'terrain_type': terrain_type
    }

    # Calculate suitability score
    suitability = calculate_suitability_score(features, crop_profile)

    return {
        **features,
        'recommended_crop': crop_name,
        'suitability_score': suitability
    }


def generate_dataset(samples_per_crop: int = 1000) -> pd.DataFrame:
    """
    Generate complete training dataset
    """
    print("Generating synthetic crop recommendation dataset...")

    all_samples = []

    for crop_name, crop_profile in CROP_PROFILES.items():
        print(f"Generating samples for {crop_name}...")

        # Distribution: 30% optimal, 30% good, 25% moderate, 15% poor
        optimal_count = int(samples_per_crop * 0.30)
        good_count = int(samples_per_crop * 0.30)
        moderate_count = int(samples_per_crop * 0.25)
        poor_count = samples_per_crop - optimal_count - good_count - moderate_count

        # Generate samples
        for _ in range(optimal_count):
            all_samples.append(generate_sample(crop_name, crop_profile, 'optimal'))

        for _ in range(good_count):
            all_samples.append(generate_sample(crop_name, crop_profile, 'good'))

        for _ in range(moderate_count):
            all_samples.append(generate_sample(crop_name, crop_profile, 'moderate'))

        for _ in range(poor_count):
            all_samples.append(generate_sample(crop_name, crop_profile, 'poor'))

    df = pd.DataFrame(all_samples)

    # Encode categorical features
    df['soil_type_encoded'] = df['soil_type'].map(SOIL_TYPE_MAPPING)
    df['climate_zone_encoded'] = df['climate_zone'].map(CLIMATE_ZONE_MAPPING)
    df['terrain_type_encoded'] = df['terrain_type'].map(TERRAIN_TYPE_MAPPING)

    # Encode target
    label_encoder = LabelEncoder()
    df['crop_label'] = label_encoder.fit_transform(df['recommended_crop'])

    print(f"\nDataset generated: {len(df)} samples")
    print(f"Crops: {list(CROP_PROFILES.keys())}")
    print(f"\nClass distribution:")
    print(df['recommended_crop'].value_counts())

    return df, label_encoder


def save_dataset(df: pd.DataFrame, label_encoder: LabelEncoder, output_dir: str = '../data'):
    """Save dataset and encoders"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    csv_path = f"{output_dir}/crop_recommendation_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDataset saved to: {csv_path}")

    # Save encoders
    encoders = {
        'crop_labels': list(label_encoder.classes_),
        'soil_types': SOIL_TYPE_MAPPING,
        'climate_zones': CLIMATE_ZONE_MAPPING,
        'terrain_types': TERRAIN_TYPE_MAPPING
    }

    encoders_path = f"{output_dir}/encoders.json"
    with open(encoders_path, 'w') as f:
        json.dump(encoders, f, indent=2)
    print(f"Encoders saved to: {encoders_path}")

    # Save statistics
    stats = {
        'total_samples': len(df),
        'num_crops': len(CROP_PROFILES),
        'features': list(df.columns),
        'feature_stats': {
            'latitude': {'min': float(df['latitude'].min()), 'max': float(df['latitude'].max()), 'mean': float(df['latitude'].mean())},
            'longitude': {'min': float(df['longitude'].min()), 'max': float(df['longitude'].max()), 'mean': float(df['longitude'].mean())},
            'altitude': {'min': float(df['altitude'].min()), 'max': float(df['altitude'].max()), 'mean': float(df['altitude'].mean())},
            'avg_temperature': {'min': float(df['avg_temperature'].min()), 'max': float(df['avg_temperature'].max()), 'mean': float(df['avg_temperature'].mean())},
            'avg_annual_rainfall': {'min': float(df['avg_annual_rainfall'].min()), 'max': float(df['avg_annual_rainfall'].max()), 'mean': float(df['avg_annual_rainfall'].mean())},
            'soil_ph': {'min': float(df['soil_ph'].min()), 'max': float(df['soil_ph'].max()), 'mean': float(df['soil_ph'].mean())},
        },
        'generated_at': datetime.now().isoformat()
    }

    stats_path = f"{output_dir}/dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_path}")


if __name__ == "__main__":
    # Generate dataset with 1000 samples per crop (6000 total)
    df, label_encoder = generate_dataset(samples_per_crop=1000)

    # Save to disk
    save_dataset(df, label_encoder)

    print("\n✓ Dataset generation complete!")
