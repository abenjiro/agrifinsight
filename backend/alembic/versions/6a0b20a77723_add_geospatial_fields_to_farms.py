"""add_geospatial_fields_to_farms

Revision ID: 6a0b20a77723
Revises: a673783b3d63
Create Date: 2025-10-16 01:42:27.735496

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6a0b20a77723'
down_revision = 'a673783b3d63'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename location to address
    op.alter_column('farms', 'location', new_column_name='address', existing_type=sa.String(500))

    # Add geospatial coordinates
    op.add_column('farms', sa.Column('latitude', sa.Float(), nullable=True))
    op.add_column('farms', sa.Column('longitude', sa.Float(), nullable=True))
    op.add_column('farms', sa.Column('altitude', sa.Float(), nullable=True))
    op.add_column('farms', sa.Column('boundary_coordinates', sa.JSON(), nullable=True))

    # Add farm properties
    op.add_column('farms', sa.Column('size_unit', sa.String(20), server_default='acres'))

    # Add soil and environmental data
    op.add_column('farms', sa.Column('soil_ph', sa.Float(), nullable=True))
    op.add_column('farms', sa.Column('soil_composition', sa.JSON(), nullable=True))
    op.add_column('farms', sa.Column('terrain_type', sa.String(100), nullable=True))
    op.add_column('farms', sa.Column('elevation_profile', sa.JSON(), nullable=True))

    # Add climate and weather
    op.add_column('farms', sa.Column('climate_zone', sa.String(100), nullable=True))
    op.add_column('farms', sa.Column('avg_annual_rainfall', sa.Float(), nullable=True))
    op.add_column('farms', sa.Column('avg_temperature', sa.Float(), nullable=True))
    op.add_column('farms', sa.Column('water_sources', sa.JSON(), nullable=True))

    # Add satellite and historical data
    op.add_column('farms', sa.Column('last_satellite_image_date', sa.DateTime(timezone=True), nullable=True))
    op.add_column('farms', sa.Column('satellite_image_url', sa.String(500), nullable=True))
    op.add_column('farms', sa.Column('ndvi_data', sa.JSON(), nullable=True))
    op.add_column('farms', sa.Column('land_use_history', sa.JSON(), nullable=True))

    # Add location metadata
    op.add_column('farms', sa.Column('timezone', sa.String(50), nullable=True))
    op.add_column('farms', sa.Column('country', sa.String(100), nullable=True))
    op.add_column('farms', sa.Column('region', sa.String(100), nullable=True))
    op.add_column('farms', sa.Column('district', sa.String(100), nullable=True))

    # Create indexes for coordinates
    op.create_index('idx_farms_latitude', 'farms', ['latitude'])
    op.create_index('idx_farms_longitude', 'farms', ['longitude'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_farms_longitude', 'farms')
    op.drop_index('idx_farms_latitude', 'farms')

    # Remove all new columns
    op.drop_column('farms', 'district')
    op.drop_column('farms', 'region')
    op.drop_column('farms', 'country')
    op.drop_column('farms', 'timezone')
    op.drop_column('farms', 'land_use_history')
    op.drop_column('farms', 'ndvi_data')
    op.drop_column('farms', 'satellite_image_url')
    op.drop_column('farms', 'last_satellite_image_date')
    op.drop_column('farms', 'water_sources')
    op.drop_column('farms', 'avg_temperature')
    op.drop_column('farms', 'avg_annual_rainfall')
    op.drop_column('farms', 'climate_zone')
    op.drop_column('farms', 'elevation_profile')
    op.drop_column('farms', 'terrain_type')
    op.drop_column('farms', 'soil_composition')
    op.drop_column('farms', 'soil_ph')
    op.drop_column('farms', 'size_unit')
    op.drop_column('farms', 'boundary_coordinates')
    op.drop_column('farms', 'altitude')
    op.drop_column('farms', 'longitude')
    op.drop_column('farms', 'latitude')

    # Rename address back to location
    op.alter_column('farms', 'address', new_column_name='location', existing_type=sa.String(255))
