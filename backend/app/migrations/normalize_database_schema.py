"""
Database Normalization Migration

Removes redundant user_id foreign keys from tables where user_id can be
derived through existing foreign key relationships.

Normalization Changes:
1. crop_images: Remove user_id (derive via farm_id → farms → user_id)
2. analysis_results: Remove user_id (derive via image_id → crop_images → farm_id → farms → user_id)
3. planting_recommendations: Remove user_id (derive via farm_id → farms → user_id)
4. harvest_predictions: Remove user_id (derive via farm_id → farms → user_id)
5. crop_recommendations: Remove user_id (derive via farm_id → farms → user_id)

Run this script to normalize an existing database.
For new installations, use the updated database_schema.sql file.

Usage:
    cd backend
    source venv/bin/activate
    python -m app.migrations.normalize_database_schema
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine, text, inspect
from app.config import get_settings

def main():
    """Run the normalization migration"""
    print("=" * 80)
    print("DATABASE NORMALIZATION MIGRATION")
    print("=" * 80)
    print("\nThis script will remove redundant user_id columns from:")
    print("  1. crop_images")
    print("  2. analysis_results")
    print("  3. planting_recommendations")
    print("  4. harvest_predictions")
    print("  5. crop_recommendations")
    print("\n⚠️  WARNING: This will modify your database schema!")
    print("⚠️  Make sure you have a backup before proceeding.")
    print("\n" + "=" * 80)

    response = input("\nDo you want to continue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Migration cancelled.")
        return

    # Get database connection
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)

    print("\n📊 Checking current database structure...")
    inspector = inspect(engine)

    with engine.connect() as conn:
        try:
            # 1. DROP user_id from crop_images
            print("\n1️⃣  Removing user_id from crop_images...")
            if 'user_id' in [col['name'] for col in inspector.get_columns('crop_images')]:
                # Drop index first
                conn.execute(text("DROP INDEX IF EXISTS idx_crop_images_user_id"))
                # Drop foreign key constraint
                conn.execute(text("ALTER TABLE crop_images DROP CONSTRAINT IF EXISTS crop_images_user_id_fkey"))
                # Drop column
                conn.execute(text("ALTER TABLE crop_images DROP COLUMN IF EXISTS user_id"))
                conn.commit()
                print("   ✅ Removed user_id from crop_images")
            else:
                print("   ⏭️  user_id column not found in crop_images (already removed)")

            # 2. DROP user_id from analysis_results
            print("\n2️⃣  Removing user_id from analysis_results...")
            if 'user_id' in [col['name'] for col in inspector.get_columns('analysis_results')]:
                # Drop index first
                conn.execute(text("DROP INDEX IF EXISTS idx_analysis_results_user_id"))
                # Drop foreign key constraint
                conn.execute(text("ALTER TABLE analysis_results DROP CONSTRAINT IF EXISTS analysis_results_user_id_fkey"))
                # Drop column
                conn.execute(text("ALTER TABLE analysis_results DROP COLUMN IF EXISTS user_id"))
                conn.commit()
                print("   ✅ Removed user_id from analysis_results")
            else:
                print("   ⏭️  user_id column not found in analysis_results (already removed)")

            # 3. DROP user_id from planting_recommendations
            print("\n3️⃣  Removing user_id from planting_recommendations...")
            if 'user_id' in [col['name'] for col in inspector.get_columns('planting_recommendations')]:
                # Drop index first
                conn.execute(text("DROP INDEX IF EXISTS idx_planting_recommendations_user_id"))
                # Drop foreign key constraint
                conn.execute(text("ALTER TABLE planting_recommendations DROP CONSTRAINT IF EXISTS planting_recommendations_user_id_fkey"))
                # Drop column
                conn.execute(text("ALTER TABLE planting_recommendations DROP COLUMN IF EXISTS user_id"))
                conn.commit()
                print("   ✅ Removed user_id from planting_recommendations")
            else:
                print("   ⏭️  user_id column not found in planting_recommendations (already removed)")

            # 4. DROP user_id from harvest_predictions
            print("\n4️⃣  Removing user_id from harvest_predictions...")
            if 'user_id' in [col['name'] for col in inspector.get_columns('harvest_predictions')]:
                # Drop index first
                conn.execute(text("DROP INDEX IF EXISTS idx_harvest_predictions_user_id"))
                # Drop foreign key constraint
                conn.execute(text("ALTER TABLE harvest_predictions DROP CONSTRAINT IF EXISTS harvest_predictions_user_id_fkey"))
                # Drop column
                conn.execute(text("ALTER TABLE harvest_predictions DROP COLUMN IF EXISTS user_id"))
                conn.commit()
                print("   ✅ Removed user_id from harvest_predictions")
            else:
                print("   ⏭️  user_id column not found in harvest_predictions (already removed)")

            # 5. DROP user_id from crop_recommendations
            print("\n5️⃣  Removing user_id from crop_recommendations...")
            if 'user_id' in [col['name'] for col in inspector.get_columns('crop_recommendations')]:
                # Drop index first
                conn.execute(text("DROP INDEX IF EXISTS idx_crop_recommendations_user_id"))
                # Drop foreign key constraint
                conn.execute(text("ALTER TABLE crop_recommendations DROP CONSTRAINT IF EXISTS crop_recommendations_user_id_fkey"))
                # Drop column
                conn.execute(text("ALTER TABLE crop_recommendations DROP COLUMN IF EXISTS user_id"))
                conn.commit()
                print("   ✅ Removed user_id from crop_recommendations")
            else:
                print("   ⏭️  user_id column not found in crop_recommendations (already removed)")

            print("\n" + "=" * 80)
            print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\n📊 Summary:")
            print("   • Removed redundant user_id columns from 5 tables")
            print("   • Dropped associated foreign key constraints")
            print("   • Dropped associated indexes")
            print("\n💡 Next Steps:")
            print("   1. Restart your backend server to use the updated models")
            print("   2. Test all API endpoints to ensure they work correctly")
            print("   3. Update any custom queries that referenced removed columns")
            print("\n📝 Note: You can now access user_id via joins:")
            print("   • crop_images → farm → user_id")
            print("   • analysis_results → image → farm → user_id")
            print("   • planting_recommendations → farm → user_id")
            print("   • harvest_predictions → farm → user_id")
            print("   • crop_recommendations → farm → user_id")
            print("\n" + "=" * 80)

        except Exception as e:
            print(f"\n❌ ERROR: Migration failed!")
            print(f"   {str(e)}")
            print("\n⚠️  Your database may be in an inconsistent state.")
            print("   Please restore from backup if necessary.")
            conn.rollback()
            raise

if __name__ == "__main__":
    main()
