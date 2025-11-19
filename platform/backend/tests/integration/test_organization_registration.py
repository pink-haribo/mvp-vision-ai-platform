"""Test script for organization creation during user registration."""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

def test_user_registration():
    """Test user registration with organization creation."""
    print("=" * 60)
    print("Testing User Registration with Organization Creation")
    print("=" * 60)

    # Generate unique email for testing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_email = f"test_user_{timestamp}@example.com"

    # Test user data
    user_data = {
        "email": test_email,
        "password": "testpass123",
        "full_name": "Test User",
        "company": "삼성전자",  # Samsung Electronics
        "division": "MX",  # Mobile Experience
        "department": "AI Research Lab",
        "phone_number": "010-1234-5678",
        "bio": "Testing organization creation"
    }

    print(f"\n[1/4] Registering new user: {test_email}")
    print(f"Company: {user_data['company']}, Division: {user_data['division']}")

    try:
        response = requests.post(f"{BASE_URL}/auth/register", json=user_data)
        response.raise_for_status()

        user = response.json()
        print(f"[OK] User registered successfully")
        print(f"  User ID: {user['id']}")
        print(f"  Email: {user['email']}")
        print(f"  Full Name: {user['full_name']}")
        print(f"  System Role: {user['system_role']}")
        print(f"  Badge Color: {user.get('badge_color', 'N/A')}")
        print(f"  Active: {user['is_active']}")

        # Check if avatar_name is present (new field)
        if 'avatar_name' in user:
            print(f"  Avatar Name: {user['avatar_name']}")
        else:
            print(f"  [WARNING] avatar_name not in response")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Registration failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text}")
        return

    print(f"\n[2/4] Logging in with new user credentials")

    try:
        login_data = {
            "username": test_email,  # OAuth2 uses 'username' field
            "password": "testpass123"
        }

        response = requests.post(
            f"{BASE_URL}/auth/login",
            data=login_data,  # OAuth2PasswordRequestForm expects form data
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        response.raise_for_status()

        tokens = response.json()
        print(f"[OK] Login successful")
        print(f"  Access Token (first 50 chars): {tokens['access_token'][:50]}...")
        print(f"  Token Type: {tokens['token_type']}")

        # Decode JWT token to check payload
        import jwt
        decoded = jwt.decode(
            tokens['access_token'],
            options={"verify_signature": False}  # Skip signature verification for testing
        )

        print(f"\n[3/4] Checking JWT token payload")
        print(f"  User ID (sub): {decoded.get('sub')}")
        print(f"  Email: {decoded.get('email', '[NOT IN TOKEN]')}")
        print(f"  Role: {decoded.get('role', '[NOT IN TOKEN]')}")
        print(f"  Organization ID: {decoded.get('organization_id', '[NOT IN TOKEN]')}")
        print(f"  Token Type: {decoded.get('type')}")
        print(f"  Expires At: {datetime.fromtimestamp(decoded['exp'])}")

        # Get user info
        print(f"\n[4/4] Fetching user info via /auth/me")

        response = requests.get(
            f"{BASE_URL}/auth/me",
            headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        response.raise_for_status()

        user_info = response.json()
        print(f"[OK] User info retrieved")
        print(f"  User ID: {user_info['id']}")
        print(f"  Email: {user_info['email']}")
        print(f"  Full Name: {user_info['full_name']}")
        print(f"  System Role: {user_info['system_role']}")
        print(f"  Company: {user_info.get('company', 'N/A')}")
        print(f"  Division: {user_info.get('division', 'N/A')}")
        print(f"  Department: {user_info.get('department', 'N/A')}")

        # Check organization_id if available
        if 'organization_id' in user_info:
            print(f"  Organization ID: {user_info['organization_id']}")
        else:
            print(f"  [INFO] organization_id not in user response")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text}")
        return
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return

    print("\n" + "=" * 60)
    print("[OK] All tests completed successfully!")
    print("=" * 60)


def test_organization_query():
    """Query organizations table to verify creation."""
    print("\n" + "=" * 60)
    print("Checking Organizations in Database")
    print("=" * 60)

    import sqlite3
    from pathlib import Path

    db_path = Path(__file__).parent.parent / "data" / "db" / "platform_vision.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query organizations
        cursor.execute("""
            SELECT id, name, company, division, max_users, created_at
            FROM organizations
            ORDER BY created_at DESC
            LIMIT 5
        """)

        organizations = cursor.fetchall()
        print(f"\n[INFO] Found {len(organizations)} organizations (showing last 5):")

        for org in organizations:
            print(f"  ID: {org[0]}, Name: {org[1]}, Company: {org[2]}, Division: {org[3]}")

        # Query users with organization info
        cursor.execute("""
            SELECT u.id, u.email, u.full_name, u.system_role, u.organization_id, u.avatar_name, o.name as org_name
            FROM users u
            LEFT JOIN organizations o ON u.organization_id = o.id
            ORDER BY u.created_at DESC
            LIMIT 5
        """)

        users = cursor.fetchall()
        print(f"\n[INFO] Recent users with organization info:")

        for user in users:
            org_info = user[6] if user[6] else "No organization"
            avatar = user[5] if user[5] else "No avatar"
            print(f"  User: {user[1]}, Role: {user[3]}, Org: {org_info}, Avatar: {avatar}")

        conn.close()

    except sqlite3.Error as e:
        print(f"[ERROR] Database error: {e}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")


if __name__ == "__main__":
    # Test registration flow
    test_user_registration()

    # Check database
    test_organization_query()
