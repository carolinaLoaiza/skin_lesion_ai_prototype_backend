"""
Manual test script for Model B (ResNet-50 feature extractor).

This script allows you to test Model B feature extraction with real images.

Usage:
    python tests/manual/test_model_b_manual.py

Requirements:
    - Place test images in tests/manual/data/images/
    - Provide diameter value for each image
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.models import extract_features_with_model_b, get_feature_names
from app.utils import preprocess_image_from_path


def list_available_images():
    """
    List all images available in the data/images/ directory.

    Returns:
        List of image file paths
    """
    data_dir = Path(__file__).parent / "data" / "images"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return []

    # Supported image formats (case-insensitive patterns)
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']

    images = []
    for pattern in image_patterns:
        images.extend(data_dir.glob(pattern))

    # Remove duplicates (can happen on case-insensitive filesystems like Windows)
    unique_images = list(set(images))

    return sorted(unique_images)


def test_model_b_with_image(image_path: str, diameter: float):
    """
    Test Model B feature extraction with a single image.

    Args:
        image_path: Path to the image file
        diameter: Lesion diameter in millimeters
    """
    print("\n" + "=" * 70)
    print(f"TESTING MODEL B (ResNet-50 Feature Extractor)")
    print("=" * 70)
    print(f"Image:    {os.path.basename(image_path)}")
    print(f"Path:     {image_path}")
    print(f"Diameter: {diameter} mm")
    print("-" * 70)

    try:
        # Step 1: Load and preprocess image
        print("\n[1/3] Preprocessing image...")
        image_array = preprocess_image_from_path(image_path)
        print(f"      Shape: {image_array.shape}")
        print(f"      Dtype: {image_array.dtype}")
        print(f"      Range: [{image_array.min():.3f}, {image_array.max():.3f}]")

        # Step 2: Extract features with Model B
        print("\n[2/3] Extracting features with Model B...")
        features = extract_features_with_model_b(image_array, diameter)
        print(f"      Extracted {len(features)} features")
        print(f"      Feature range: [{features.min():.3f}, {features.max():.3f}]")

        # Step 3: Display features
        print("\n[3/3] Displaying extracted features...")
        feature_names = get_feature_names()

        # Visual result
        print("\n" + "=" * 70)
        print("EXTRACTED FEATURES (18 clinical features):")
        print("=" * 70)

        # Group features by type
        color_features = ["tbp_lv_A", "tbp_lv_B", "tbp_lv_C", "tbp_lv_H", "tbp_lv_L",
                         "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLB",
                         "tbp_lv_deltaLBnorm", "tbp_lv_color_std_mean", "tbp_lv_norm_color", "tbp_lv_stdL"]
        geometric_features = ["tbp_lv_areaMM2", "tbp_lv_area_perim_ratio", "tbp_lv_minorAxisMM",
                             "tbp_lv_perimeterMM", "tbp_lv_symm_2axis"]

        # Print color features
        print("\nCOLOR FEATURES:")
        for name in color_features:
            if name in feature_names:
                idx = feature_names.index(name)
                print(f"  {name:30s} = {features[idx]:>10.4f}")

        # Print geometric features
        print("\nGEOMETRIC FEATURES:")
        for name in geometric_features:
            if name in feature_names:
                idx = feature_names.index(name)
                print(f"  {name:30s} = {features[idx]:>10.4f}")

        print("=" * 70)

        # Feature statistics
        print("\nFEATURE STATISTICS:")
        print(f"  Mean:   {features.mean():>10.4f}")
        print(f"  Std:    {features.std():>10.4f}")
        print(f"  Min:    {features.min():>10.4f}")
        print(f"  Max:    {features.max():>10.4f}")
        print("=" * 70)

        return {
            "image": os.path.basename(image_path),
            "diameter": diameter,
            "features": features.tolist(),
            "feature_names": feature_names
        }

    except FileNotFoundError as e:
        print(f"\nERROR: Image file not found: {image_path}")
        print(f"       {str(e)}")
        return None
    except Exception as e:
        print(f"\nERROR: Feature extraction failed!")
        print(f"       {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_all_images(default_diameter: float = 6.5):
    """
    Test Model B with all images in the data directory.

    Args:
        default_diameter: Default diameter to use for all images (mm)
    """
    images = list_available_images()

    if not images:
        print("\nNo images found in tests/manual/data/images/")
        print("Please add test images to this directory.")
        return

    print(f"\nFound {len(images)} image(s) to test:")
    for i, img in enumerate(images, 1):
        print(f"  {i}. {img.name}")

    print(f"\nUsing default diameter: {default_diameter} mm for all images")

    results = []
    for image_path in images:
        result = test_model_b_with_image(str(image_path), default_diameter)
        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for i, result in enumerate(results, 1):
            mean_feature = sum(result['features']) / len(result['features'])
            print(f"{i}. {result['image']:30s} | Diameter: {result['diameter']:5.2f} mm | Mean Feature: {mean_feature:8.4f}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Model B with skin lesion images")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a specific image file (e.g., tests/manual/data/images/sample_1.jpg)"
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=6.5,
        help="Lesion diameter in millimeters (default: 6.5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all images in the data/images/ directory"
    )

    args = parser.parse_args()

    if args.image:
        # Test specific image with specified diameter
        test_model_b_with_image(args.image, args.diameter)
    elif args.all:
        # Test all images with default diameter
        test_all_images(args.diameter)
    else:
        # Interactive mode
        images = list_available_images()

        if not images:
            print("\nNo images found in tests/manual/data/images/")
            print("Please add test images to this directory.")
            print("\nUsage:")
            print("  python tests/manual/test_model_b_manual.py --image <path> --diameter <mm>")
            print("  python tests/manual/test_model_b_manual.py --all --diameter <mm>")
        else:
            print(f"\nFound {len(images)} image(s):")
            for i, img in enumerate(images, 1):
                print(f"  {i}. {img.name}")

            print("\nSelect an image to test (or 'all' for all images):")
            choice = input("> ").strip()

            if choice.lower() == 'all':
                diameter_input = input("Enter diameter in mm (default 6.5): ").strip()
                diameter = float(diameter_input) if diameter_input else 6.5
                test_all_images(diameter)
            elif choice.isdigit() and 1 <= int(choice) <= len(images):
                diameter_input = input("Enter diameter in mm (default 6.5): ").strip()
                diameter = float(diameter_input) if diameter_input else 6.5
                test_model_b_with_image(str(images[int(choice) - 1]), diameter)
            else:
                print("Invalid choice.")
