"""Create a sample image dataset for testing."""

import os
from pathlib import Path
import numpy as np
from PIL import Image

# Dataset configuration
DATASET_ROOT = Path(__file__).parent.parent / "data" / "datasets" / "sample_dataset"
NUM_TRAIN_PER_CLASS = 20
NUM_VAL_PER_CLASS = 5
IMAGE_SIZE = (64, 64)
CLASSES = ["cats", "dogs"]


def create_random_image(size=(64, 64), color_range=(0, 255)):
    """Create a random colored image."""
    # Generate random RGB values
    img_array = np.random.randint(
        color_range[0], color_range[1], (*size, 3), dtype=np.uint8
    )
    return Image.fromarray(img_array, mode="RGB")


def create_dataset():
    """Create sample dataset directory structure with random images."""
    print(f"Creating sample dataset at: {DATASET_ROOT}")

    # Create directories
    for split in ["train", "val"]:
        for class_name in CLASSES:
            class_dir = DATASET_ROOT / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Created directory: {class_dir}")

    # Generate training images
    print("\nGenerating training images...")
    for class_name in CLASSES:
        class_dir = DATASET_ROOT / "train" / class_name
        for i in range(NUM_TRAIN_PER_CLASS):
            img = create_random_image(IMAGE_SIZE)
            img_path = class_dir / f"{class_name}_{i:03d}.jpg"
            img.save(img_path)
            if (i + 1) % 10 == 0:
                print(f"  {class_name}: {i + 1}/{NUM_TRAIN_PER_CLASS} images")

    # Generate validation images
    print("\nGenerating validation images...")
    for class_name in CLASSES:
        class_dir = DATASET_ROOT / "val" / class_name
        for i in range(NUM_VAL_PER_CLASS):
            img = create_random_image(IMAGE_SIZE)
            img_path = class_dir / f"{class_name}_val_{i:03d}.jpg"
            img.save(img_path)
        print(f"  {class_name}: {NUM_VAL_PER_CLASS} images")

    # Create README
    readme_path = DATASET_ROOT / "README.md"
    readme_content = f"""# Sample Dataset

This is a randomly generated dataset for testing the Vision AI Training Platform.

## Structure

```
sample_dataset/
├── train/
│   ├── {CLASSES[0]}/  ({NUM_TRAIN_PER_CLASS} images)
│   └── {CLASSES[1]}/  ({NUM_TRAIN_PER_CLASS} images)
└── val/
    ├── {CLASSES[0]}/  ({NUM_VAL_PER_CLASS} images)
    └── {CLASSES[1]}/  ({NUM_VAL_PER_CLASS} images)
```

## Details

- **Total classes**: {len(CLASSES)}
- **Training images**: {NUM_TRAIN_PER_CLASS * len(CLASSES)}
- **Validation images**: {NUM_VAL_PER_CLASS * len(CLASSES)}
- **Image size**: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels
- **Format**: JPEG

## Note

Images are randomly generated colored noise for testing purposes only.
This dataset is NOT suitable for actual model evaluation.
"""

    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"\n[SUCCESS] Sample dataset created successfully!")
    print(f"\nDataset path: {DATASET_ROOT}")
    print(f"Total images: {(NUM_TRAIN_PER_CLASS + NUM_VAL_PER_CLASS) * len(CLASSES)}")
    print(f"  - Training: {NUM_TRAIN_PER_CLASS * len(CLASSES)}")
    print(f"  - Validation: {NUM_VAL_PER_CLASS * len(CLASSES)}")
    print(f"\nUse this path in chat: ./data/datasets/sample_dataset")


if __name__ == "__main__":
    create_dataset()
