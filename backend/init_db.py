"""
Database initialization script
Creates all database tables based on SQLAlchemy models
"""

from app.database import engine, create_tables
from app.models.database import Base

def init_database():
    """Initialize the database by creating all tables"""
    print("Initializing database...")
    print(f"Database URL: {engine.url}")

    try:
        # Drop all existing tables (optional - comment out for production)
        # Base.metadata.drop_all(bind=engine)
        # print("Dropped all existing tables")

        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("Successfully created all database tables!")

        # Print created tables
        print("\nCreated tables:")
        for table in Base.metadata.sorted_tables:
            print(f"  - {table.name}")

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

if __name__ == "__main__":
    init_database()
