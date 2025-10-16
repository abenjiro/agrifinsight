"""add_role_to_users

Revision ID: a673783b3d63
Revises: 5f6c033f0831
Create Date: 2025-10-16 01:27:35.013661

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a673783b3d63'
down_revision = '5f6c033f0831'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add role column to users table
    op.add_column('users', sa.Column('role', sa.String(length=50), nullable=True, server_default='farmer'))

    # Update existing users to have 'farmer' role if null
    op.execute("UPDATE users SET role = 'farmer' WHERE role IS NULL")


def downgrade() -> None:
    # Remove role column from users table
    op.drop_column('users', 'role')
