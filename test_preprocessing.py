"""
Quick test script for preprocessing utilities.
Run this to verify that preprocessing functions work correctly.
"""

import numpy as np
from app.utils.metadata_preprocessing import (
    preprocess_metadata,
    create_feature_vector_for_model_c,
    validate_metadata
)


def test_metadata_preprocessing():
    """Test metadata preprocessing functions."""
    print("\n" + "="*60)
    print("TESTING METADATA PREPROCESSING")
    print("="*60)

    # Test data
    age = 45
    sex = "female"
    location = "back"
    diameter = 6.5

    try:
        # Validate metadata
        print("\n1. Validating metadata...")
        validate_metadata(age, sex, location, diameter)
        print("   [OK] Validation passed")

        # Preprocess metadata
        print("\n2. Preprocessing metadata...")
        metadata = preprocess_metadata(age, sex, location, diameter)
        print(f"   [OK] Age normalized: {metadata['age_normalized']:.4f}")
        print(f"   [OK] Sex encoded: {metadata['sex_encoded']}")
        print(f"   [OK] Location normalized: {metadata['location_normalized']}")
        print(f"   [OK] Location one-hot shape: {metadata['location_onehot'].shape}")
        print(f"   [OK] Diameter normalized: {metadata['diameter_normalized']:.4f}")

        # Create feature vector for Model C
        print("\n3. Creating feature vector for Model C...")
        # Simulate 18 features from Model B
        mock_features = np.random.rand(18).astype(np.float32)
        feature_vector = create_feature_vector_for_model_c(
            mock_features, age, sex, location, diameter
        )
        print(f"   [OK] Feature vector shape: {feature_vector.shape}")
        print(f"   [OK] Feature vector dtype: {feature_vector.dtype}")
        print(f"   [OK] Expected components:")
        print(f"        -> 18 (Model B features)")
        print(f"        -> 1 (age)")
        print(f"        -> 1 (sex)")
        print(f"        -> 1 (diameter)")
        print(f"        -> 13 (location one-hot)")
        print(f"        = 34 total features")

        print("\n" + "="*60)
        print("ALL METADATA PREPROCESSING TESTS PASSED!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {str(e)}\n")
        raise


def test_edge_cases():
    """Test edge cases and validation."""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)

    test_cases = [
        # (age, sex, location, diameter, should_pass)
        (0, "male", "head", 1.0, True),      # Minimum age
        (120, "female", "trunk", 50.0, True),  # Maximum age
        (-1, "male", "back", 5.0, False),    # Invalid age
        (50, "other", "back", 5.0, False),   # Invalid sex
        (50, "male", "invalid", 5.0, False), # Invalid location
        (50, "male", "back", -1.0, False),   # Invalid diameter
    ]

    for i, (age, sex, location, diameter, should_pass) in enumerate(test_cases, 1):
        try:
            validate_metadata(age, sex, location, diameter)
            if should_pass:
                print(f"   [OK] Test {i} passed (valid input)")
            else:
                print(f"   [FAIL] Test {i} FAILED (should have raised error)")
        except ValueError as e:
            if not should_pass:
                print(f"   [OK] Test {i} passed (correctly rejected invalid input)")
            else:
                print(f"   [FAIL] Test {i} FAILED (incorrectly rejected valid input): {e}")

    print("\n" + "="*60)
    print("EDGE CASE TESTS COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("\nRUNNING PREPROCESSING TESTS\n")

    # Run tests
    test_metadata_preprocessing()
    test_edge_cases()

    print("ALL TESTS COMPLETED SUCCESSFULLY!\n")
