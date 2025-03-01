import os
import argparse
import pandas as pd
from tqdm import tqdm
import glob
from PIL import Image


def check_images(metadata_path, img_dir):
    """
    Check if image files in metadata exist and can be opened.

    Args:
        metadata_path: Path to metadata CSV file
        img_dir: Directory containing images
    """
    print(f"Checking image paths in {metadata_path}...")

    # Read metadata
    metadata = pd.read_csv(metadata_path)

    # Get image columns
    image_columns = []
    for col in metadata.columns:
        if "file" in col.lower() or "path" in col.lower() or "image" in col.lower():
            image_columns.append(col)

    if not image_columns:
        print("No image columns found in metadata.")
        return

    print(f"Found image columns: {', '.join(image_columns)}")

    # Check each image column
    for col in image_columns:
        print(f"\nChecking column: {col}")

        exists_count = 0
        missing_count = 0
        error_count = 0

        for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
            img_path = os.path.join(img_dir, row[col])

            if os.path.exists(img_path):
                exists_count += 1

                # Try to open the image
                try:
                    with Image.open(img_path) as img:
                        pass  # Just test if it can be opened
                except Exception as e:
                    error_count += 1
                    print(f"Error opening {img_path}: {e}")
            else:
                missing_count += 1
                if missing_count <= 10:  # Limit the number of missing files shown
                    print(f"Missing: {img_path}")

        print(
            f"Column {col}: {exists_count} files exist, {missing_count} missing, {error_count} have errors"
        )

    # Scan directory for image files
    print("\nScanning image directory for available image files...")
    all_image_files = []
    for ext in ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]:
        all_image_files.extend(
            glob.glob(os.path.join(img_dir, "**", ext), recursive=True)
        )

    print(f"Found {len(all_image_files)} image files in directory")

    if all_image_files:
        print("\nSample paths in image directory:")
        for path in all_image_files[:10]:
            print(f"  {os.path.relpath(path, img_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check image paths in metadata")
    parser.add_argument(
        "--metadata_path", type=str, required=True, help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--img_dir", type=str, required=True, help="Directory containing images"
    )

    args = parser.parse_args()
    check_images(args.metadata_path, args.img_dir)
