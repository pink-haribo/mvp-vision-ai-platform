#!/usr/bin/env python3
"""
Create zip files from DICE Format datasets for upload.

This script compresses each dataset in dice_format/ into a zip file
ready for upload to the platform.
"""

import os
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def create_dataset_zip(dataset_dir: Path, output_dir: Path):
    """
    Create a zip file from a DICE Format dataset.

    Args:
        dataset_dir: Path to dataset directory (contains annotations.json, meta.json, images/)
        output_dir: Output directory for zip files
    """
    dataset_name = dataset_dir.name
    zip_path = output_dir / f"{dataset_name}.zip"

    logger.info(f"Creating zip: {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all files in dataset directory
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = Path(root) / file
                # Calculate relative path from dataset root
                arcname = file_path.relative_to(dataset_dir)
                zip_file.write(file_path, arcname)
                logger.info(f"  Added: {arcname}")

    # Get zip file size
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Zip size: {zip_size_mb:.2f} MB")
    logger.info(f"  Output: {zip_path}\n")


def main():
    """Create zip files for all DICE Format datasets."""
    datasets_dir = Path("c:/datasets/dice_format")
    output_dir = Path("c:/datasets/dice_format_zips")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DICE Format Dataset Zip Creator")
    print("=" * 60)
    print(f"Source: {datasets_dir}")
    print(f"Output: {output_dir}\n")

    # Get all dataset directories
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]

    if not dataset_dirs:
        logger.error(f"No datasets found in {datasets_dir}")
        return

    logger.info(f"Found {len(dataset_dirs)} datasets\n")

    # Create zip for each dataset
    for dataset_dir in sorted(dataset_dirs):
        create_dataset_zip(dataset_dir, output_dir)

    print("=" * 60)
    print(f"[SUCCESS] Created {len(dataset_dirs)} zip files")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
