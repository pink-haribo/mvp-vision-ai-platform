#!/usr/bin/env python3
"""Test Dual Storage configuration."""

import json
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load .env
from dotenv import load_dotenv
load_dotenv(backend_dir / ".env")

# Import dual storage
from app.utils.dual_storage import dual_storage

def test_schema_retrieval():
    """Test retrieving schema from Internal Storage."""
    print("=" * 80)
    print("Testing Dual Storage - Schema Retrieval")
    print("=" * 80)

    framework = "ultralytics"
    print(f"\nRetrieving schema for framework: {framework}")

    schema_bytes = dual_storage.get_schema(framework)

    if schema_bytes:
        schema_dict = json.loads(schema_bytes.decode('utf-8'))
        print(f"\n[OK] Successfully retrieved schema!")
        print(f"  Framework: {schema_dict.get('framework')}")
        print(f"  Version: {schema_dict.get('version')}")
        print(f"  Description: {schema_dict.get('description')}")
        print(f"  Fields: {len(schema_dict.get('fields', []))}")
        print(f"  Presets: {list(schema_dict.get('presets', {}).keys())}")
        return True
    else:
        print(f"\n[ERROR] Failed to retrieve schema for {framework}")
        print("  Check:")
        print("  1. MinIO-Results is running on http://localhost:9002")
        print("  2. Schema was uploaded to config-schemas bucket")
        print("  3. Environment variables are set correctly")
        return False

if __name__ == "__main__":
    success = test_schema_retrieval()
    sys.exit(0 if success else 1)
