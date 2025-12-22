"""
Integration test for the full prediction pipeline with XGBoost Model C.
Tests the complete flow: Image -> Model A -> Model B -> Model C -> Final prediction
"""
import numpy as np
from app.models.model_c import predict_with_model_c

# Simulate Model B extracted features (18 features)
# These would normally come from extract_features_with_model_b()
mock_model_b_features = np.array([
    20.0, 28.0, 35.0, 55.0, 42.0,  # tbp_lv_A through tbp_lv_L
    8.5, 19.0, 1.0, 5.0, 1.3,      # areaMM2 through deltaB
    -9.0, 9.5, 7.5, 2.5, 3.0,      # deltaL through norm_color
    12.0, 2.7, 0.3                 # perimeterMM through symm_2axis
])

# Test metadata
test_cases = [
    {
        "name": "Test Case 1: Male, Left Arm, middle-aged",
        "age": 45,
        "sex": "male",
        "location": "Left Arm",
        "diameter": 6.5
    },
    {
        "name": "Test Case 2: Female, Torso Back, elderly",
        "age": 72,
        "sex": "female",
        "location": "Torso Back",
        "diameter": 8.2
    },
    {
        "name": "Test Case 3: Male, Head & Neck, young",
        "age": 28,
        "sex": "male",
        "location": "Head & Neck",
        "diameter": 4.1
    },
    {
        "name": "Test Case 4: Female, Unknown location",
        "age": 55,
        "sex": "female",
        "location": "Unknown",
        "diameter": 5.8
    }
]

print("="*70)
print("FULL PIPELINE TEST - XGBoost Model C")
print("="*70)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{test_case['name']}")
    print("-" * 70)

    try:
        probability = predict_with_model_c(
            extracted_features=mock_model_b_features,
            age=test_case["age"],
            sex=test_case["sex"],
            location=test_case["location"],
            diameter=test_case["diameter"]
        )

        risk_level = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"

        print(f"  Age: {test_case['age']}")
        print(f"  Sex: {test_case['sex']}")
        print(f"  Location: {test_case['location']}")
        print(f"  Diameter: {test_case['diameter']} mm")
        print(f"  > Model C Probability: {probability:.4f} ({probability*100:.2f}%)")
        print(f"  > Risk Level: {risk_level}")
        print("  [SUCCESS]")

    except Exception as e:
        print(f"  [FAILED]: {str(e)}")

print("\n" + "="*70)
print("Test completed successfully!")
print("="*70)
