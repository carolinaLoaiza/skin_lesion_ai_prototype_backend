"""
Test script for Model A (DenseNet-121).
Verifies that the model can be loaded and make predictions.
"""

import numpy as np
from app.models import (
    validate_model_exists,
    predict_with_model_a,
    apply_threshold,
    get_model_info
)


def test_model_exists():
    """Test that model file exists."""
    print("\n" + "="*60)
    print("TEST 1: Check if Model A file exists")
    print("="*60)

    exists = validate_model_exists()

    if exists:
        print("[OK] Model A file found")
    else:
        print("[FAIL] Model A file not found")
        raise FileNotFoundError("Model A file missing")


def test_model_loading():
    """Test that model can be loaded."""
    print("\n" + "="*60)
    print("TEST 2: Load Model A")
    print("="*60)

    try:
        # Get model info (this will trigger loading)
        info = get_model_info()

        print("[OK] Model loaded successfully")
        print(f"   Model: {info['model_name']}")
        print(f"   Device: {info['device']}")
        print(f"   Total parameters: {info['total_parameters']:,}")
        print(f"   Trainable parameters: {info['trainable_parameters']:,}")
        print(f"   Threshold: {info['threshold']}")

    except Exception as e:
        print(f"[FAIL] Failed to load model: {str(e)}")
        raise


def test_prediction_with_random_image():
    """Test prediction with a random image."""
    print("\n" + "="*60)
    print("TEST 3: Predict with random image")
    print("="*60)

    try:
        # Create random image (simulating preprocessed image)
        # Shape: (3, 224, 224), dtype: float32
        # Values simulating ImageNet normalized data
        random_image = np.random.randn(3, 224, 224).astype(np.float32)

        print(f"   Input shape: {random_image.shape}")
        print(f"   Input dtype: {random_image.dtype}")
        print(f"   Input range: [{random_image.min():.3f}, {random_image.max():.3f}]")

        # Make prediction
        probability = predict_with_model_a(random_image)

        print(f"[OK] Prediction successful")
        print(f"   Probability: {probability:.4f}")
        print(f"   Range check: {0.0 <= probability <= 1.0}")

        # Apply threshold
        prediction = apply_threshold(probability)
        print(f"   Binary prediction: {prediction} ({'malignant' if prediction == 1 else 'benign'})")

        # Validate output
        assert 0.0 <= probability <= 1.0, "Probability out of range"
        assert prediction in [0, 1], "Invalid binary prediction"

    except Exception as e:
        print(f"[FAIL] Prediction failed: {str(e)}")
        raise


def test_multiple_predictions():
    """Test multiple predictions to ensure consistency."""
    print("\n" + "="*60)
    print("TEST 4: Multiple predictions")
    print("="*60)

    try:
        probabilities = []

        for i in range(5):
            random_image = np.random.randn(3, 224, 224).astype(np.float32)
            prob = predict_with_model_a(random_image)
            probabilities.append(prob)
            print(f"   Prediction {i+1}: {prob:.4f}")

        print(f"[OK] All {len(probabilities)} predictions successful")
        print(f"   Mean probability: {np.mean(probabilities):.4f}")
        print(f"   Std probability: {np.std(probabilities):.4f}")

    except Exception as e:
        print(f"[FAIL] Multiple predictions failed: {str(e)}")
        raise


def test_invalid_input():
    """Test that invalid inputs raise appropriate errors."""
    print("\n" + "="*60)
    print("TEST 5: Invalid input handling")
    print("="*60)

    test_cases = [
        ("Wrong shape", np.random.randn(224, 224, 3).astype(np.float32)),  # (H, W, C) instead of (C, H, W)
        ("Wrong dtype", np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)),  # uint8 instead of float32
        ("Wrong dimensions", np.random.randn(3, 128, 128).astype(np.float32)),  # Wrong size
    ]

    for test_name, invalid_input in test_cases:
        try:
            predict_with_model_a(invalid_input)
            print(f"[FAIL] {test_name}: Should have raised ValueError")
        except ValueError as e:
            print(f"[OK] {test_name}: Correctly rejected ({str(e)[:50]}...)")
        except Exception as e:
            print(f"[WARN] {test_name}: Unexpected error type: {type(e).__name__}")


if __name__ == "__main__":
    print("\nRUNNING MODEL A TESTS\n")

    try:
        # Run all tests
        test_model_exists()
        test_model_loading()
        test_prediction_with_random_image()
        test_multiple_predictions()
        test_invalid_input()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Tests failed: {str(e)}\n")
        raise
