"""
Manual test script for Model A (DenseNet-121 image classifier).

This script allows you to test Model A with real skin lesion images.

Usage:
    python tests/manual/test_model_a_manual.py

Requirements:
    - Place test images in tests/manual/data/images/
    - Images should be skin lesion photos
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.models import predict_with_model_a, apply_threshold
from app.utils import preprocess_image_from_path
from app.core.config import settings


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


def test_model_a_with_image(image_path: str):
    """
    Test Model A with a single image.

    Args:
        image_path: Path to the image file
    """
    print("\n" + "=" * 70)
    print(f"TESTING MODEL A (DenseNet-121 Image Classifier)")
    print("=" * 70)
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Path:  {image_path}")
    print("-" * 70)

    try:
        # Step 1: Load and preprocess image
        print("\n[1/3] Preprocessing image...")
        image_array = preprocess_image_from_path(image_path)
        print(f"      Shape: {image_array.shape}")
        print(f"      Dtype: {image_array.dtype}")
        print(f"      Range: [{image_array.min():.3f}, {image_array.max():.3f}]")

        # Step 2: Run prediction
        print("\n[2/3] Running Model A prediction...")
        probability = predict_with_model_a(image_array)
        print(f"      Raw probability: {probability:.6f}")

        # Step 3: Apply threshold and interpret
        print("\n[3/3] Interpreting results...")
        threshold = settings.MODEL_A_THRESHOLD
        is_malignant = apply_threshold(probability, threshold)

        print(f"      Threshold: {threshold}")
        print(f"      Classification: {'MALIGNANT' if is_malignant else 'BENIGN'}")

        # Visual result
        print("\n" + "=" * 70)
        print("RESULTS:")
        print("=" * 70)
        print(f"  Malignancy Probability: {probability:.2%}")
        print(f"  Classification:         {('MALIGNANT' if is_malignant else 'BENIGN')}")
        print(f"  Confidence:             {abs(probability - 0.5) * 2:.2%}")

        # Risk interpretation
        if probability < 0.3:
            risk = "LOW"
        elif probability < 0.7:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        print(f"  Risk Level:             {risk}")
        print("=" * 70)

        return {
            "image": os.path.basename(image_path),
            "probability": probability,
            "is_malignant": is_malignant,
            "risk": risk
        }

    except FileNotFoundError as e:
        print(f"\nERROR: Image file not found: {image_path}")
        print(f"       {str(e)}")
        return None
    except Exception as e:
        print(f"\nERROR: Prediction failed!")
        print(f"       {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_all_images():
    """Test Model A with all images in the data directory."""
    images = list_available_images()

    if not images:
        print("\nNo images found in tests/manual/data/images/")
        print("Please add test images to this directory.")
        return

    print(f"\nFound {len(images)} image(s) to test:")
    for i, img in enumerate(images, 1):
        print(f"  {i}. {img.name}")

    results = []
    for image_path in images:
        result = test_model_a_with_image(str(image_path))
        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for i, result in enumerate(results, 1):
            status = "MALIGNANT" if result["is_malignant"] else "BENIGN"
            print(f"{i}. {result['image']:30s} | {result['probability']:.2%} | {status:9s} | {result['risk']:6s}")
        print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Model A with skin lesion images")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a specific image file (e.g., tests/manual/data/images/sample_1.jpg)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all images in the data/images/ directory"
    )

    args = parser.parse_args()

    if args.image:
        # Test specific image
        test_model_a_with_image(args.image)
    elif args.all:
        # Test all images
        test_all_images()
    else:
        # Interactive mode
        images = list_available_images()

        if not images:
            print("\nNo images found in tests/manual/data/images/")
            print("Please add test images to this directory.")
            print("\nUsage:")
            print("  python tests/manual/test_model_a_manual.py --image <path>")
            print("  python tests/manual/test_model_a_manual.py --all")
        else:
            print(f"\nFound {len(images)} image(s):")
            for i, img in enumerate(images, 1):
                print(f"  {i}. {img.name}")

            print("\nSelect an image to test (or 'all' for all images):")
            choice = input("> ").strip()

            if choice.lower() == 'all':
                test_all_images()
            elif choice.isdigit() and 1 <= int(choice) <= len(images):
                test_model_a_with_image(str(images[int(choice) - 1]))
            else:
                print("Invalid choice.")
