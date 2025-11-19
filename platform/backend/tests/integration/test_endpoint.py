#!/usr/bin/env python
"""Test the datasets/available endpoint to find the error."""
import sys
sys.path.insert(0, 'C:\\Users\\flyto\\Project\\Github\\mvp-vision-ai-platform\\mvp\\backend')

from app.db.database import SessionLocal
from app.db.models import Dataset
from typing import Optional

def test_endpoint():
    db = SessionLocal()
    try:
        # Query public datasets
        query = db.query(Dataset).filter(Dataset.visibility == 'public')
        datasets = query.all()

        # Convert to response format
        result = []
        for ds in datasets:
            try:
                item = {
                    "id": ds.id,
                    "name": ds.name,
                    "description": ds.description or f"Dataset - {ds.format} format",
                    "format": ds.format,
                    "labeled": ds.labeled,
                    "num_items": ds.num_images,
                    "size_mb": None,
                    "source": ds.storage_type,
                    "path": ds.id,
                }
                result.append(item)
                print(f"Success for {ds.name}: {item}")
            except Exception as e:
                print(f"Error processing {ds.name}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nTotal: {len(result)} datasets")
        return result

    except Exception as e:
        print(f"Database error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_endpoint()
