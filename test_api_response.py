"""
Test script to verify the new API response structure.
Tests that the response contains separate probabilities without combining.
"""
import json

# Expected response structure
expected_response = {
    "model_a_probability": 0.72,
    "model_c_probability": 0.58,
    "extracted_features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                           1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
    "metadata": {
        "age": 45,
        "sex": "female",
        "location": "Torso Back",
        "diameter": 6.5
    }
}

print("Expected API Response Structure:")
print("=" * 60)
print(json.dumps(expected_response, indent=2))
print("=" * 60)

print("\nKey changes:")
print("  - REMOVED: final_probability (no longer combining A and C)")
print("  - REMOVED: risk_category (frontend will handle risk display)")
print("  - KEPT: model_a_probability (DenseNet-121 image prediction)")
print("  - KEPT: model_c_probability (XGBoost tabular prediction)")
print("  - KEPT: extracted_features (18 features from ResNet-50)")
print("  - KEPT: metadata (input parameters)")

print("\nFrontend will display:")
print("  - Model A probability separately")
print("  - Model C probability separately")
print("  - Allow user to compare both predictions")
