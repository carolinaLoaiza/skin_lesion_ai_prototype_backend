"""
Manual test script for Model C (XGBoost classifier).

This script allows you to test Model C with extracted features and metadata,
including SHAP explainability.

Usage:
    python tests/manual/test_model_c_manual.py

Requirements:
    - Provide 18 features from Model B (or use synthetic data)
    - Provide clinical metadata: age, sex, location, diameter
    - SHAP library for explainability
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from app.models import predict_with_model_c, extract_features_with_model_b, explain_prediction_with_shap
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


def test_model_c_with_features_only(
    features: np.ndarray,
    age: int,
    sex: str,
    location: str,
    diameter: float,
    test_shap: bool = True
):
    """
    Test Model C with pre-extracted features and metadata.

    Args:
        features: 18 features from Model B
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion anatomical location
        diameter: Lesion diameter in millimeters
        test_shap: Whether to test SHAP explainability (default: True)
    """
    print("\n" + "=" * 80)
    print("TESTING MODEL C (XGBoost Classifier) - Features Only Mode")
    print("=" * 80)
    print(f"Features: {len(features)} values")
    print(f"Metadata: age={age}, sex={sex}, location={location}, diameter={diameter} mm")
    print("-" * 80)

    try:
        # Display input features
        print("\n[1/3] Input features from Model B:")
        print(f"      Features shape: {features.shape}")
        print(f"      Features range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"      Features mean:  {features.mean():.3f}")

        # Run Model C prediction
        print("\n[2/3] Running Model C (XGBoost) prediction...")
        probability = predict_with_model_c(features, age, sex, location, diameter)
        print(f"      Raw probability: {probability:.6f}")

        # Test SHAP explainability
        explanation = None
        if test_shap:
            print("\n[3/3] Generating SHAP explanation...")
            try:
                explanation = explain_prediction_with_shap(features, age, sex, location, diameter)
                print(f"      Base value: {explanation['base_value']:.4f}")
                print(f"      Prediction: {explanation['prediction']:.4f}")
                print(f"      Features explained: {len(explanation['shap_values'])}")
            except Exception as e:
                print(f"      SHAP failed: {str(e)}")
        else:
            print("\n[3/3] Skipping SHAP explanation (test_shap=False)")

        # Interpret results
        print("\n" + "=" * 80)
        print("RESULTS:")
        print("=" * 80)
        print(f"  Malignancy Probability: {probability:.2%}")
        print(f"  Classification:         {'MALIGNANT' if probability >= 0.5 else 'BENIGN'}")
        print(f"  Confidence:             {abs(probability - 0.5) * 2:.2%}")

        # Risk interpretation
        if probability < 0.3:
            risk = "LOW"
        elif probability < 0.7:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        print(f"  Risk Level:             {risk}")
        print("=" * 80)

        # Display SHAP explanation if available
        if explanation:
            print("\nSHAP EXPLANATION - Top 10 Most Important Features:")
            print("=" * 80)

            # Create list of (name, shap_value, feature_value)
            features_impact = list(zip(
                explanation['feature_names'],
                explanation['shap_values'],
                explanation['feature_values']
            ))

            # Sort by absolute SHAP value
            features_impact_sorted = sorted(features_impact, key=lambda x: abs(x[1]), reverse=True)

            print(f"{'Feature Name':<40} {'SHAP Value':<15} {'Feature Value':<15} {'Impact'}")
            print("-" * 80)

            for name, shap_val, feat_val in features_impact_sorted[:10]:
                impact = "Increases risk" if shap_val > 0 else "Decreases risk"
                print(f"{name:<40} {shap_val:>14.6f} {feat_val:>14.3f} {impact}")

            print("=" * 80)

        return {
            "probability": probability,
            "risk": risk,
            "is_malignant": probability >= 0.5,
            "shap_explanation": explanation
        }

    except Exception as e:
        print(f"\nERROR: Prediction failed!")
        print(f"       {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_model_c_with_image(
    image_path: str,
    age: int,
    sex: str,
    location: str,
    diameter: float,
    test_shap: bool = True
):
    """
    Test complete pipeline: Image → Model B → Model C → SHAP.

    Args:
        image_path: Path to the image file
        age: Patient age in years
        sex: Patient sex ("male" or "female")
        location: Lesion anatomical location
        diameter: Lesion diameter in millimeters
        test_shap: Whether to test SHAP explainability (default: True)
    """
    print("\n" + "=" * 80)
    print("TESTING MODEL C (XGBoost Classifier) - Full Pipeline Mode")
    print("=" * 80)
    print(f"Image:    {os.path.basename(image_path)}")
    print(f"Path:     {image_path}")
    print(f"Metadata: age={age}, sex={sex}, location={location}, diameter={diameter} mm")
    print("-" * 80)

    try:
        # Step 1: Preprocess image
        print("\n[1/5] Preprocessing image...")
        image_array = preprocess_image_from_path(image_path)
        print(f"      Shape: {image_array.shape}")
        print(f"      Dtype: {image_array.dtype}")

        # Step 2: Extract features with Model B
        print("\n[2/5] Extracting features with Model B (ResNet-50)...")
        features = extract_features_with_model_b(image_array, diameter)
        print(f"      Extracted {len(features)} features")
        print(f"      Feature range: [{features.min():.3f}, {features.max():.3f}]")

        # Step 3: Run Model C prediction
        print("\n[3/5] Running Model C (XGBoost) prediction...")
        probability = predict_with_model_c(features, age, sex, location, diameter)
        print(f"      Raw probability: {probability:.6f}")

        # Step 4: Test SHAP explainability
        explanation = None
        if test_shap:
            print("\n[4/5] Generating SHAP explanation...")
            try:
                explanation = explain_prediction_with_shap(features, age, sex, location, diameter)
                print(f"      Base value: {explanation['base_value']:.4f}")
                print(f"      Prediction: {explanation['prediction']:.4f}")
                print(f"      Features explained: {len(explanation['shap_values'])}")
            except Exception as e:
                print(f"      SHAP failed: {str(e)}")
        else:
            print("\n[4/5] Skipping SHAP explanation (test_shap=False)")

        # Step 5: Interpret results
        print("\n[5/5] Interpreting results...")

        # Visual result
        print("\n" + "=" * 80)
        print("RESULTS:")
        print("=" * 80)
        print(f"  Malignancy Probability: {probability:.2%}")
        print(f"  Classification:         {'MALIGNANT' if probability >= 0.5 else 'BENIGN'}")
        print(f"  Confidence:             {abs(probability - 0.5) * 2:.2%}")

        # Risk interpretation
        if probability < 0.3:
            risk = "LOW"
        elif probability < 0.7:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        print(f"  Risk Level:             {risk}")
        print("=" * 80)

        # Feature summary
        print("\nFEATURE SUMMARY (Model B → Model C):")
        print(f"  Number of features extracted: {len(features)}")
        print(f"  Feature mean:                 {features.mean():.4f}")
        print(f"  Feature std:                  {features.std():.4f}")
        print("=" * 80)

        # Display SHAP explanation if available
        if explanation:
            print("\nSHAP EXPLANATION - Top 10 Most Important Features:")
            print("=" * 80)

            # Create list of (name, shap_value, feature_value)
            features_impact = list(zip(
                explanation['feature_names'],
                explanation['shap_values'],
                explanation['feature_values']
            ))

            # Sort by absolute SHAP value
            features_impact_sorted = sorted(features_impact, key=lambda x: abs(x[1]), reverse=True)

            print(f"{'Feature Name':<40} {'SHAP Value':<15} {'Feature Value':<15} {'Impact'}")
            print("-" * 80)

            for name, shap_val, feat_val in features_impact_sorted[:10]:
                impact = "Increases risk" if shap_val > 0 else "Decreases risk"
                print(f"{name:<40} {shap_val:>14.6f} {feat_val:>14.3f} {impact}")

            print("=" * 80)

        return {
            "image": os.path.basename(image_path),
            "probability": probability,
            "risk": risk,
            "is_malignant": probability >= 0.5,
            "features": features.tolist(),
            "shap_explanation": explanation,
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
        print(f"\nERROR: Prediction failed!")
        print(f"       {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_all_images_with_metadata(
    age: int = 45,
    sex: str = "female",
    location: str = "back",
    diameter: float = 6.5
):
    """
    Test Model C with all images using the same metadata.

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
        result = test_model_c_with_image(str(image_path), age, sex, location, diameter)
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

    parser = argparse.ArgumentParser(description="Test Model C with features and metadata")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "features-only"],
        default="full",
        help="Test mode: 'full' (image→B→C) or 'features-only' (synthetic features→C)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file (for full mode)"
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
        help="Test all images in data/images/ directory (full mode only)"
    )

    args = parser.parse_args()

    if args.mode == "features-only":
        # Test with synthetic features
        print("\nGenerating synthetic features for testing...")
        synthetic_features = np.random.rand(18).astype(np.float32) * 10  # Random features [0, 10]
        test_model_c_with_features_only(
            features=synthetic_features,
            age=args.age,
            sex=args.sex,
            location=args.location,
            diameter=args.diameter
        )

    elif args.mode == "full":
        if args.all:
            # Test all images
            test_all_images_with_metadata(
                age=args.age,
                sex=args.sex,
                location=args.location,
                diameter=args.diameter
            )
        elif args.image:
            # Test specific image
            test_model_c_with_image(
                image_path=args.image,
                age=args.age,
                sex=args.sex,
                location=args.location,
                diameter=args.diameter
            )
        else:
            # Interactive mode
            images = list_available_images()

            if not images:
                print("\nNo images found in tests/manual/data/images/")
                print("Please add test images to this directory.")
                print("\nUsage:")
                print("  python tests/manual/test_model_c_manual.py --mode full --image <path> --age 45 --sex female --location back --diameter 6.5")
                print("  python tests/manual/test_model_c_manual.py --mode features-only --age 45 --sex male --location chest --diameter 8.0")
                print("  python tests/manual/test_model_c_manual.py --all --age 50 --sex male --location arm --diameter 7.0")
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

                        test_all_images_with_metadata(age, sex, location, diameter)
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

                        test_model_c_with_image(
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
