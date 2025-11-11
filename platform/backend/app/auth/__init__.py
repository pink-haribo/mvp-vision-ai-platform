"""
Authentication module

Handles JWT token generation, password hashing, and user authentication.
"""

from app.auth.jwt import create_access_token, verify_token
from app.auth.password import hash_password, verify_password
from app.auth.dependencies import get_current_user, get_current_active_user

__all__ = [
    "create_access_token",
    "verify_token",
    "hash_password",
    "verify_password",
    "get_current_user",
    "get_current_active_user",
]
