"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.db.database import Base, get_db
from app.main import app


# Test database
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test."""
    # Create tables
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Drop tables after test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with database dependency override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    # Clear overrides
    app.dependency_overrides.clear()


@pytest.fixture
def sample_training_config():
    """Sample training configuration for tests."""
    return {
        "framework": "ultralytics",
        "model_name": "yolo11n",
        "task_type": "object_detection",
        "dataset_path": "C:\\datasets\\det-coco8",
        "dataset_format": "yolo",
        "num_classes": 80,
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 0.01,
    }


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample test image."""
    from PIL import Image
    import numpy as np

    # Create a simple test image
    img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)

    return str(img_path)


@pytest.fixture
def sample_image_batch(tmp_path):
    """Create multiple test images for batch inference."""
    from PIL import Image
    import numpy as np

    image_paths = []
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img_path = tmp_path / f"test_image_{i}.jpg"
        img.save(img_path)
        image_paths.append(str(img_path))

    return image_paths
