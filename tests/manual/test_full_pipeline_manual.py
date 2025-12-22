"""
Manual test script for the complete prediction pipeline.

This script tests the full pipeline using all 3 models:
- Model A: DenseNet-121 (image -> probability)
- Model B: ResNet-50 (image + diameter -> 18 features)
- Model C: XGBoost (features + metadata -> probability)
- Returns individual probabilities without combining

Usage:
    python tests/manual/test_full_pipeline_manual.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.models import (
    predict_with_model_a,
    extract_features_with_model_b,
    predict_with_model_c
)
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


def test_full_pipeline(
    image_path: str,
    age: int,
    sex: str,
    location: str,
    diameter: float
):
    """
    Test the complete prediction pipeline with all 3 models.

    Pipeline:
    1. Preprocess image
    2. Model A: image -> probability A
    3. Model B: image + diameter -> 18 features
    4. Model C: features + metadata -> probability C
    5. Return individual probabilities (no combining)

    Args:
        image_path: Path to the image file
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion anatomical location
        diameter: Lesion diameter in millimeters
    """
    print("\n" + "=" * 80)
    print("TESTING FULL PREDICTION PIPELINE (3 Models)")
    print("=" * 80)
    print(f"Image:    {os.path.basename(image_path)}")
    print(f"Path:     {image_path}")
    print(f"Metadata: age={age}, sex={sex}, location={location}, diameter={diameter} mm")
    print("=" * 80)

    try:
        # ========== STEP 1: Preprocess Image ==========
        print("\n[STEP 1/4] Preprocessing image...")
        image_array = preprocess_image_from_path(image_path)
        print(f"            Shape: {image_array.shape}")
        print(f"            Dtype: {image_array.dtype}")
        print(f"            Range: [{image_array.min():.3f}, {image_array.max():.3f}]")

        # ========== STEP 2: Model A Prediction ==========
        print("\n[STEP 2/4] Running Model A (DenseNet-121 Image Classifier)...")
        probability_a = predict_with_model_a(image_array)
        print(f"            Model A probability: {probability_a:.6f}")

        # ========== STEP 3: Model B Feature Extraction ==========
        print("\n[STEP 3/4] Running Model B (ResNet-50 Feature Extractor)...")
        features = extract_features_with_model_b(image_array, diameter)
        print(f"            Extracted {len(features)} features")
        print(f"            Feature range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"            Feature mean: {features.mean():.3f}")

        # ========== STEP 4: Model C Prediction ==========
        print("\n[STEP 4/4] Running Model C (XGBoost Tabular Classifier)...")
        probability_c = predict_with_model_c(features, age, sex, location, diameter)
        print(f"            Model C probability: {probability_c:.6f}")

        # ========== RESULTS SUMMARY ==========
        print("\n" + "=" * 80)
        print("FINAL RESULTS - COMPLETE PIPELINE")
        print("=" * 80)

        print("\nMODEL OUTPUTS:")
        print(f"  Model A (DenseNet-121):     {probability_a:.2%}")
        print(f"  Model C (XGBoost):          {probability_c:.2%}")

        print("\nFEATURES EXTRACTED (Model B):")
        print(f"  Number of features:         {len(features)}")
        print(f"  Feature mean:               {features.mean():.4f}")
        print(f"  Feature std:                {features.std():.4f}")
        print(f"  Feature min:                {features.min():.4f}")
        print(f"  Feature max:                {features.max():.4f}")

        print("\nMETADATA:")
        print(f"  Age:                        {age} years")
        print(f"  Sex:                        {sex}")
        print(f"  Location:                   {location}")
        print(f"  Diameter:                   {diameter} mm")

        print("=" * 80)

        return {
            "image": os.path.basename(image_path),
            "model_a_probability": probability_a,
            "model_c_probability": probability_c,
            "features": features.tolist(),
            "metadata": {
                "age": age,
                "sex": sex,
                "location": location,
                "diameter": diameter
            }
        }

    except FileNotFoundError as e:
        print(f"\nERROR: Image file not found: {image_path}")
        print(f"       {str(e)}")
        return None
    except Exception as e:
        print(f"\nERROR: Pipeline failed!")
        print(f"       {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_all_images_pipeline(
    age: int = 45,
    sex: str = "female",
    location: str = "back",
    diameter: float = 6.5
):
    """
    Test the full pipeline with all images using the same metadata.

    Args:
        age: Patient age (default: 45)
        sex: Patient sex (default: "female")
        location: Lesion location (default: "back")
        diameter: Lesion diameter (default: 6.5 mm)
    """
    images = list_available_images()

    if not images:
        print("\nNo images found in tests/manual/data/images/")
        print("Please add test images to this directory.")
        return

    print(f"\nFound {len(images)} image(s) to test:")
    for i, img in enumerate(images, 1):
        print(f"  {i}. {img.name}")

    print(f"\nUsing metadata for all images:")
    print(f"  Age:      {age} years")
    print(f"  Sex:      {sex}")
    print(f"  Location: {location}")
    print(f"  Diameter: {diameter} mm")

    results = []
    for image_path in images:
        result = test_full_pipeline(str(image_path), age, sex, location, diameter)
        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY - ALL IMAGES")
        print("=" * 80)
        print(f"{'#':<3} {'Image':<35} {'Model A':<10} {'Model C':<10}")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"{i:<3} {result['image']:<35} "
                  f"{result['model_a_probability']:.2%}     "
                  f"{result['model_c_probability']:.2%}")
        print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test full prediction pipeline")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file"
    )
    parser.add_argument(
        "--age",
        type=int,
        default=45,
        help="Patient age in years (default: 45)"
    )
    parser.add_argument(
        "--sex",
        type=str,
        default="female",
        help="Patient sex: male or female (default: female)"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="back",
        help="Lesion location (default: back)"
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=6.5,
        help="Lesion diameter in mm (default: 6.5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all images in data/images/ directory"
    )

    args = parser.parse_args()

    if args.image:
        # Test specific image
        test_full_pipeline(args.image, args.age, args.sex, args.location, args.diameter)
    elif args.all:
        # Test all images
        test_all_images_pipeline(args.age, args.sex, args.location, args.diameter)
    else:
        # Interactive mode
        images = list_available_images()

        if not images:
            print("\nNo images found in tests/manual/data/images/")
            print("Please add test images to this directory.")
            print("\nUsage:")
            print("  python tests/manual/test_full_pipeline_manual.py --image <path> --age 45 --sex female --location back --diameter 6.5")
            print("  python tests/manual/test_full_pipeline_manual.py --all --age 50 --sex male --location chest --diameter 7.0")
        else:
            print(f"\nFound {len(images)} image(s):")
            for i, img in enumerate(images, 1):
                print(f"  {i}. {img.name}")

            print("\nSelect an image to test (or 'all' for all images):")
            choice = input("> ").strip()

            if choice.lower() == 'all':
                # Ask for metadata for all images
                print("\nEnter metadata (will be used for all images):")
                try:
                    age_input = input("  Age (years, default 45): ").strip()
                    age = int(age_input) if age_input else 45

                    sex_input = input("  Sex (male/female, default female): ").strip()
                    sex = sex_input if sex_input else "female"

                    location_input = input("  Location (back/chest/arm/leg/..., default back): ").strip()
                    location = location_input if location_input else "back"

                    diameter_input = input("  Diameter (mm, default 6.5): ").strip()
                    diameter = float(diameter_input) if diameter_input else 6.5

                    test_all_images_pipeline(age, sex, location, diameter)
                except ValueError as e:
                    print(f"Invalid input: {e}")

            elif choice.isdigit() and 1 <= int(choice) <= len(images):
                # Ask for metadata for single image
                print("\nEnter metadata for this image:")
                try:
                    age_input = input("  Age (years, default 45): ").strip()
                    age = int(age_input) if age_input else 45

                    sex_input = input("  Sex (male/female, default female): ").strip()
                    sex = sex_input if sex_input else "female"

                    location_input = input("  Location (back/chest/arm/leg/..., default back): ").strip()
                    location = location_input if location_input else "back"

                    diameter_input = input("  Diameter (mm, default 6.5): ").strip()
                    diameter = float(diameter_input) if diameter_input else 6.5

                    test_full_pipeline(
                        str(images[int(choice) - 1]),
                        age,
                        sex,
                        location,
                        diameter
                    )
                except ValueError as e:
                    print(f"Invalid input: {e}")
            else:
                print("Invalid choice.")
