"""Quick test of API response to verify schema fields."""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

# Login as admin
login_response = requests.post(
    f"{BASE_URL}/auth/login",
    data={"username": "admin@example.com", "password": "admin123"}
)

token = login_response.json()["access_token"]

# Get user info
me_response = requests.get(
    f"{BASE_URL}/auth/me",
    headers={"Authorization": f"Bearer {token}"}
)

user_data = me_response.json()

print("API Response for /auth/me:")
print(json.dumps(user_data, indent=2))

# Check for expected fields
print("\nField Check:")
print(f"  avatar_name present: {'avatar_name' in user_data}")
print(f"  organization_id present: {'organization_id' in user_data}")

if 'avatar_name' in user_data:
    print(f"  avatar_name value: {user_data['avatar_name']}")

if 'organization_id' in user_data:
    print(f"  organization_id value: {user_data['organization_id']}")
