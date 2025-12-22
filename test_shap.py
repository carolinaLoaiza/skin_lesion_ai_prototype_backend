"""
Quick test to verify SHAP explainability works with Model C.
"""
import numpy as np
from app.models import explain_prediction_with_shap

# Create mock Model B features (18 features)
mock_features = np.array([
    20.0, 28.0, 35.0, 55.0, 42.0,  # tbp_lv_A through tbp_lv_L
    8.5, 19.0, 1.0, 5.0, 1.3,      # areaMM2 through deltaB
    -9.0, 9.5, 7.5, 2.5, 3.0,      # deltaL through norm_color
    12.0, 2.7, 0.3                 # perimeterMM through symm_2axis
])

# Test metadata
test_age = 45
test_sex = "male"
test_location = "Left Arm"
test_diameter = 6.5

print("="*80)
print("TESTING SHAP EXPLAINABILITY FOR MODEL C")
print("="*80)
print(f"Patient: {test_age} years, {test_sex}")
print(f"Lesion: {test_location}, {test_diameter} mm diameter")
print("="*80)

try:
    # Get SHAP explanation
    explanation = explain_prediction_with_shap(
        mock_features,
        test_age,
        test_sex,
        test_location,
        test_diameter
    )

    print("\nSHAP Explanation Generated Successfully!")
    print("="*80)

    print(f"\nBase value (expected output): {explanation['base_value']:.4f}")
    print(f"Prediction probability: {explanation['prediction']:.4f}")

    print(f"\nNumber of features explained: {len(explanation['shap_values'])}")

    # Show top 10 most important features
    print("\nTop 10 Features by Absolute SHAP Value:")
    print("-"*80)

    # Create list of (name, shap_value, feature_value)
    features_impact = list(zip(
        explanation['feature_names'],
        explanation['shap_values'],
        explanation['feature_values']
    ))

    # Sort by absolute SHAP value
    features_impact_sorted = sorted(features_impact, key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature Name':<40} {'SHAP Value':<15} {'Feature Value':<15} {'Impact'}")
    print("-"*80)

    for name, shap_val, feat_val in features_impact_sorted[:10]:
        impact = "Increases risk" if shap_val > 0 else "Decreases risk"
        print(f"{name:<40} {shap_val:>14.6f} {feat_val:>14.3f} {impact}")

    print("\n" + "="*80)
    print("SHAP Test Passed!")
    print("="*80)

except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
