#!/usr/bin/env python3
"""
Script to populate soil data for existing farms
Uses soil estimation service to add soil_type and soil_ph to farms that don't have it
"""

import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.database import SessionLocal
from app.models.database import Farm
from app.services.soil_estimation_service import soil_estimation_service


def populate_soil_data():
    """Populate soil data for all farms that don't have it"""
    db = SessionLocal()

    try:
        # Get all farms
        farms = db.query(Farm).all()

        print(f"\n{'='*80}")
        print(f"Populating Soil Data for Farms")
        print(f"{'='*80}\n")

        updated_count = 0
        skipped_count = 0
        error_count = 0

        for farm in farms:
            print(f"\nFarm: {farm.name} (ID: {farm.id})")
            print(f"  Location: {farm.latitude}, {farm.longitude}")

            # Skip if farm already has soil data
            if farm.soil_type and farm.soil_ph:
                print(f"  ✓ Already has soil data: {farm.soil_type}, pH {farm.soil_ph}")
                skipped_count += 1
                continue

            # Skip if farm doesn't have coordinates
            if not farm.latitude or not farm.longitude:
                print(f"  ✗ No GPS coordinates - skipping")
                error_count += 1
                continue

            try:
                # Estimate soil data
                estimation = soil_estimation_service.estimate_soil_data(
                    latitude=farm.latitude,
                    longitude=farm.longitude,
                    avg_rainfall=farm.avg_annual_rainfall,
                    climate_zone=farm.climate_zone
                )

                # Update farm
                farm.soil_type = estimation['soil_type']
                farm.soil_ph = estimation['soil_ph']

                print(f"  ✓ Estimated soil data:")
                print(f"    - Soil Type: {estimation['soil_type']}")
                print(f"    - Soil pH: {estimation['soil_ph']}")
                print(f"    - Confidence: {estimation['confidence']}")
                print(f"    - Region: {estimation['region']}")
                print(f"    - Notes: {estimation['notes']}")

                updated_count += 1

            except Exception as e:
                print(f"  ✗ Error estimating soil data: {e}")
                error_count += 1

        # Commit all changes
        if updated_count > 0:
            db.commit()
            print(f"\n{'='*80}")
            print(f"✓ Successfully updated {updated_count} farm(s)")

        print(f"\nSummary:")
        print(f"  - Updated: {updated_count}")
        print(f"  - Skipped (already has data): {skipped_count}")
        print(f"  - Errors: {error_count}")
        print(f"  - Total farms: {len(farms)}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        db.rollback()

    finally:
        db.close()


if __name__ == "__main__":
    populate_soil_data()
