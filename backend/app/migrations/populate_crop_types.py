"""
Populate crop_types table with comprehensive crop data
Run this script to populate the database with all crop types and their prediction data
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, backend_dir)

from app.database import get_db, engine
from app.models.database import CropType, Base

# Comprehensive crop data
CROP_DATA = [
    # Cereals & Grains
    {
        'name': 'Maize',
        'category': 'grains',
        'scientific_name': 'Zea mays',
        'description': 'Staple cereal grain crop widely grown in tropical and subtropical regions',
        'growth_duration_days': 120,
        'water_requirement': 'medium',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 2000,
        'max_yield_per_acre': 5000,
        'avg_yield_per_acre': 3500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Rice',
        'category': 'grains',
        'scientific_name': 'Oryza sativa',
        'description': 'Major cereal crop and staple food for over half of the world population',
        'growth_duration_days': 150,
        'water_requirement': 'high',
        'recommended_irrigation': 'flood',
        'min_yield_per_acre': 3000,
        'max_yield_per_acre': 6000,
        'avg_yield_per_acre': 4500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Wheat',
        'category': 'grains',
        'scientific_name': 'Triticum aestivum',
        'description': 'Important cereal grain used for making flour and bread',
        'growth_duration_days': 120,
        'water_requirement': 'medium',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 1500,
        'max_yield_per_acre': 3500,
        'avg_yield_per_acre': 2500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Barley',
        'category': 'grains',
        'scientific_name': 'Hordeum vulgare',
        'description': 'Versatile cereal grain used for food, animal feed, and brewing',
        'growth_duration_days': 90,
        'water_requirement': 'low',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 1200,
        'max_yield_per_acre': 2800,
        'avg_yield_per_acre': 2000,
        'yield_unit': 'kg'
    },
    {
        'name': 'Sorghum',
        'category': 'grains',
        'scientific_name': 'Sorghum bicolor',
        'description': 'Drought-resistant cereal grain ideal for arid regions',
        'growth_duration_days': 100,
        'water_requirement': 'low',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 1000,
        'max_yield_per_acre': 2500,
        'avg_yield_per_acre': 1750,
        'yield_unit': 'kg'
    },
    {
        'name': 'Millet',
        'category': 'grains',
        'scientific_name': 'Pennisetum glaucum',
        'description': 'Hardy grain crop highly tolerant to drought and poor soil',
        'growth_duration_days': 85,
        'water_requirement': 'low',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 800,
        'max_yield_per_acre': 2000,
        'avg_yield_per_acre': 1400,
        'yield_unit': 'kg'
    },

    # Legumes
    {
        'name': 'Soybean',
        'category': 'legumes',
        'scientific_name': 'Glycine max',
        'description': 'Protein-rich legume crop used for food, oil, and animal feed',
        'growth_duration_days': 100,
        'water_requirement': 'medium',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 800,
        'max_yield_per_acre': 1500,
        'avg_yield_per_acre': 1150,
        'yield_unit': 'kg'
    },
    {
        'name': 'Groundnut',
        'category': 'legumes',
        'scientific_name': 'Arachis hypogaea',
        'description': 'Oil-rich legume crop also known as peanut',
        'growth_duration_days': 120,
        'water_requirement': 'medium',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 1000,
        'max_yield_per_acre': 2500,
        'avg_yield_per_acre': 1750,
        'yield_unit': 'kg'
    },
    {
        'name': 'Beans',
        'category': 'legumes',
        'scientific_name': 'Phaseolus vulgaris',
        'description': 'Common bean crop rich in protein and fiber',
        'growth_duration_days': 75,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 600,
        'max_yield_per_acre': 1200,
        'avg_yield_per_acre': 900,
        'yield_unit': 'kg'
    },
    {
        'name': 'Peas',
        'category': 'legumes',
        'scientific_name': 'Pisum sativum',
        'description': 'Cool-season legume crop with high protein content',
        'growth_duration_days': 60,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 800,
        'max_yield_per_acre': 1500,
        'avg_yield_per_acre': 1150,
        'yield_unit': 'kg'
    },

    # Root & Tuber Crops
    {
        'name': 'Cassava',
        'category': 'tubers',
        'scientific_name': 'Manihot esculenta',
        'description': 'Starchy root crop that is a major source of carbohydrates',
        'growth_duration_days': 300,
        'water_requirement': 'low',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 8000,
        'max_yield_per_acre': 15000,
        'avg_yield_per_acre': 12000,
        'yield_unit': 'kg'
    },
    {
        'name': 'Yam',
        'category': 'tubers',
        'scientific_name': 'Dioscorea spp.',
        'description': 'Important tuber crop in tropical regions',
        'growth_duration_days': 240,
        'water_requirement': 'medium',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 6000,
        'max_yield_per_acre': 12000,
        'avg_yield_per_acre': 9000,
        'yield_unit': 'kg'
    },
    {
        'name': 'Sweet Potato',
        'category': 'tubers',
        'scientific_name': 'Ipomoea batatas',
        'description': 'Nutritious root crop rich in vitamins and minerals',
        'growth_duration_days': 120,
        'water_requirement': 'medium',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 5000,
        'max_yield_per_acre': 10000,
        'avg_yield_per_acre': 7500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Potato',
        'category': 'tubers',
        'scientific_name': 'Solanum tuberosum',
        'description': 'Widely cultivated tuber crop used as a staple food',
        'growth_duration_days': 90,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 8000,
        'max_yield_per_acre': 15000,
        'avg_yield_per_acre': 11500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Ginger',
        'category': 'tubers',
        'scientific_name': 'Zingiber officinale',
        'description': 'Spice crop with medicinal properties',
        'growth_duration_days': 240,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 3000,
        'max_yield_per_acre': 6000,
        'avg_yield_per_acre': 4500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Garlic',
        'category': 'vegetables',
        'scientific_name': 'Allium sativum',
        'description': 'Pungent bulb crop used as spice and medicine',
        'growth_duration_days': 150,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 2000,
        'max_yield_per_acre': 4000,
        'avg_yield_per_acre': 3000,
        'yield_unit': 'kg'
    },

    # Vegetables
    {
        'name': 'Tomato',
        'category': 'vegetables',
        'scientific_name': 'Solanum lycopersicum',
        'description': 'Popular fruit vegetable used in cooking worldwide',
        'growth_duration_days': 90,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 5000,
        'max_yield_per_acre': 12000,
        'avg_yield_per_acre': 8500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Pepper',
        'category': 'vegetables',
        'scientific_name': 'Capsicum annuum',
        'description': 'Spicy vegetable crop with various cultivars',
        'growth_duration_days': 90,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 2000,
        'max_yield_per_acre': 5000,
        'avg_yield_per_acre': 3500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Onion',
        'category': 'vegetables',
        'scientific_name': 'Allium cepa',
        'description': 'Bulb vegetable widely used as flavoring ingredient',
        'growth_duration_days': 110,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 8000,
        'max_yield_per_acre': 15000,
        'avg_yield_per_acre': 11500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Cabbage',
        'category': 'vegetables',
        'scientific_name': 'Brassica oleracea',
        'description': 'Leafy vegetable crop rich in vitamins',
        'growth_duration_days': 75,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 10000,
        'max_yield_per_acre': 20000,
        'avg_yield_per_acre': 15000,
        'yield_unit': 'kg'
    },
    {
        'name': 'Carrot',
        'category': 'vegetables',
        'scientific_name': 'Daucus carota',
        'description': 'Root vegetable rich in beta-carotene',
        'growth_duration_days': 70,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 6000,
        'max_yield_per_acre': 12000,
        'avg_yield_per_acre': 9000,
        'yield_unit': 'kg'
    },
    {
        'name': 'Okra',
        'category': 'vegetables',
        'scientific_name': 'Abelmoschus esculentus',
        'description': 'Warm-season vegetable also known as lady fingers',
        'growth_duration_days': 60,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 3000,
        'max_yield_per_acre': 6000,
        'avg_yield_per_acre': 4500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Garden Egg',
        'category': 'vegetables',
        'scientific_name': 'Solanum melongena',
        'description': 'African eggplant variety, also known as aubergine',
        'growth_duration_days': 85,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 4000,
        'max_yield_per_acre': 8000,
        'avg_yield_per_acre': 6000,
        'yield_unit': 'kg'
    },
    {
        'name': 'Cucumber',
        'category': 'vegetables',
        'scientific_name': 'Cucumis sativus',
        'description': 'Refreshing vine crop commonly used in salads',
        'growth_duration_days': 55,
        'water_requirement': 'high',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 5000,
        'max_yield_per_acre': 10000,
        'avg_yield_per_acre': 7500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Watermelon',
        'category': 'fruits',
        'scientific_name': 'Citrullus lanatus',
        'description': 'Large fruit crop with high water content',
        'growth_duration_days': 80,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 15000,
        'max_yield_per_acre': 30000,
        'avg_yield_per_acre': 22500,
        'yield_unit': 'kg'
    },

    # Fruits
    {
        'name': 'Plantain',
        'category': 'fruits',
        'scientific_name': 'Musa paradisiaca',
        'description': 'Starchy banana variety used as staple food',
        'growth_duration_days': 365,
        'water_requirement': 'high',
        'recommended_irrigation': 'manual',
        'min_yield_per_acre': 8000,
        'max_yield_per_acre': 15000,
        'avg_yield_per_acre': 11500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Banana',
        'category': 'fruits',
        'scientific_name': 'Musa acuminata',
        'description': 'Sweet fruit crop widely consumed fresh',
        'growth_duration_days': 300,
        'water_requirement': 'high',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 10000,
        'max_yield_per_acre': 20000,
        'avg_yield_per_acre': 15000,
        'yield_unit': 'kg'
    },
    {
        'name': 'Pineapple',
        'category': 'fruits',
        'scientific_name': 'Ananas comosus',
        'description': 'Tropical fruit with distinctive appearance and flavor',
        'growth_duration_days': 480,
        'water_requirement': 'low',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 12000,
        'max_yield_per_acre': 25000,
        'avg_yield_per_acre': 18500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Mango',
        'category': 'fruits',
        'scientific_name': 'Mangifera indica',
        'description': 'Popular tropical fruit tree',
        'growth_duration_days': 1460,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 3000,
        'max_yield_per_acre': 8000,
        'avg_yield_per_acre': 5500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Orange',
        'category': 'fruits',
        'scientific_name': 'Citrus sinensis',
        'description': 'Citrus fruit rich in vitamin C',
        'growth_duration_days': 1095,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 5000,
        'max_yield_per_acre': 12000,
        'avg_yield_per_acre': 8500,
        'yield_unit': 'kg'
    },

    # Cash Crops
    {
        'name': 'Cocoa',
        'category': 'cash_crops',
        'scientific_name': 'Theobroma cacao',
        'description': 'Tropical tree crop used for chocolate production',
        'growth_duration_days': 1095,
        'water_requirement': 'high',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 400,
        'max_yield_per_acre': 1000,
        'avg_yield_per_acre': 700,
        'yield_unit': 'kg'
    },
    {
        'name': 'Coffee',
        'category': 'cash_crops',
        'scientific_name': 'Coffea arabica',
        'description': 'Valuable cash crop used for beverage production',
        'growth_duration_days': 1095,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 600,
        'max_yield_per_acre': 1500,
        'avg_yield_per_acre': 1050,
        'yield_unit': 'kg'
    },
    {
        'name': 'Cotton',
        'category': 'cash_crops',
        'scientific_name': 'Gossypium hirsutum',
        'description': 'Fiber crop used in textile industry',
        'growth_duration_days': 150,
        'water_requirement': 'medium',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 800,
        'max_yield_per_acre': 1800,
        'avg_yield_per_acre': 1300,
        'yield_unit': 'kg'
    },
    {
        'name': 'Sugarcane',
        'category': 'cash_crops',
        'scientific_name': 'Saccharum officinarum',
        'description': 'Tall grass crop used for sugar production',
        'growth_duration_days': 365,
        'water_requirement': 'high',
        'recommended_irrigation': 'flood',
        'min_yield_per_acre': 30000,
        'max_yield_per_acre': 60000,
        'avg_yield_per_acre': 45000,
        'yield_unit': 'kg'
    },
    {
        'name': 'Tea',
        'category': 'cash_crops',
        'scientific_name': 'Camellia sinensis',
        'description': 'Evergreen shrub cultivated for leaf harvest',
        'growth_duration_days': 1095,
        'water_requirement': 'high',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 1000,
        'max_yield_per_acre': 2500,
        'avg_yield_per_acre': 1750,
        'yield_unit': 'kg'
    },
    {
        'name': 'Tobacco',
        'category': 'cash_crops',
        'scientific_name': 'Nicotiana tabacum',
        'description': 'Commercial crop used in tobacco industry',
        'growth_duration_days': 90,
        'water_requirement': 'medium',
        'recommended_irrigation': 'drip',
        'min_yield_per_acre': 1200,
        'max_yield_per_acre': 2500,
        'avg_yield_per_acre': 1850,
        'yield_unit': 'kg'
    },
    {
        'name': 'Coconut',
        'category': 'cash_crops',
        'scientific_name': 'Cocos nucifera',
        'description': 'Versatile palm tree crop with multiple uses',
        'growth_duration_days': 2555,
        'water_requirement': 'medium',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 5000,
        'max_yield_per_acre': 12000,
        'avg_yield_per_acre': 8500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Palm Oil',
        'category': 'cash_crops',
        'scientific_name': 'Elaeis guineensis',
        'description': 'Oil palm tree cultivated for edible oil production',
        'growth_duration_days': 1095,
        'water_requirement': 'high',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 3000,
        'max_yield_per_acre': 8000,
        'avg_yield_per_acre': 5500,
        'yield_unit': 'kg'
    },
    {
        'name': 'Rubber',
        'category': 'cash_crops',
        'scientific_name': 'Hevea brasiliensis',
        'description': 'Tree crop cultivated for latex production',
        'growth_duration_days': 2190,
        'water_requirement': 'high',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 1000,
        'max_yield_per_acre': 2500,
        'avg_yield_per_acre': 1750,
        'yield_unit': 'kg'
    },
    {
        'name': 'Cashew',
        'category': 'cash_crops',
        'scientific_name': 'Anacardium occidentale',
        'description': 'Tree crop producing edible nuts and cashew apple',
        'growth_duration_days': 1095,
        'water_requirement': 'low',
        'recommended_irrigation': 'rain-fed',
        'min_yield_per_acre': 400,
        'max_yield_per_acre': 1000,
        'avg_yield_per_acre': 700,
        'yield_unit': 'kg'
    },
]


def populate_crop_types():
    """Populate the crop_types table with comprehensive crop data"""

    # Create database session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        print("Starting crop types population...")

        # First, add the new columns to existing table if they don't exist
        print("\nAdding new columns to crop_types table...")
        try:
            with engine.connect() as conn:
                # Check if columns exist and add them if they don't
                columns_to_add = [
                    "ALTER TABLE crop_types ADD COLUMN IF NOT EXISTS growth_duration_days INTEGER",
                    "ALTER TABLE crop_types ADD COLUMN IF NOT EXISTS water_requirement VARCHAR(20)",
                    "ALTER TABLE crop_types ADD COLUMN IF NOT EXISTS recommended_irrigation VARCHAR(50)",
                    "ALTER TABLE crop_types ADD COLUMN IF NOT EXISTS min_yield_per_acre FLOAT",
                    "ALTER TABLE crop_types ADD COLUMN IF NOT EXISTS max_yield_per_acre FLOAT",
                    "ALTER TABLE crop_types ADD COLUMN IF NOT EXISTS avg_yield_per_acre FLOAT",
                    "ALTER TABLE crop_types ADD COLUMN IF NOT EXISTS yield_unit VARCHAR(20) DEFAULT 'kg'",
                ]

                for sql in columns_to_add:
                    try:
                        conn.execute(text(sql))
                        conn.commit()
                    except Exception as e:
                        print(f"Column might already exist or error: {e}")

        except Exception as e:
            print(f"Error adding columns (they might already exist): {e}")

        # Insert or update crop types
        added_count = 0
        updated_count = 0

        for crop_data in CROP_DATA:
            # Check if crop already exists
            existing_crop = db.query(CropType).filter(CropType.name == crop_data['name']).first()

            if existing_crop:
                # Update existing crop
                for key, value in crop_data.items():
                    setattr(existing_crop, key, value)
                updated_count += 1
                print(f"✓ Updated: {crop_data['name']}")
            else:
                # Create new crop
                new_crop = CropType(**crop_data)
                db.add(new_crop)
                added_count += 1
                print(f"✓ Added: {crop_data['name']}")

        # Commit all changes
        db.commit()

        print(f"\n{'='*60}")
        print(f"✅ SUCCESS!")
        print(f"{'='*60}")
        print(f"Crops added: {added_count}")
        print(f"Crops updated: {updated_count}")
        print(f"Total crops in database: {db.query(CropType).count()}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CROP TYPES DATABASE POPULATION")
    print("="*60 + "\n")
    populate_crop_types()
    print("\n✅ Database population complete!\n")
