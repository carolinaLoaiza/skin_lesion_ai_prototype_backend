"""
Quick test script to verify Model C (XGBoost) loads correctly.
"""
import numpy as np
from app.models.model_c import ModelCLoader, get_model_c_info

# Get Model C instance (singleton pattern with __new__)
model_c = ModelCLoader()

print(f"Model C loaded: {type(model_c._model).__name__}")
print(f"Number of features expected: {len(model_c._feature_names) if model_c._feature_names else 'Unknown'}")

if model_c._feature_names:
    print("\nFeature names:")
    for i, name in enumerate(model_c._feature_names):
        print(f"  {i+1}. {name}")

# Create dummy input (28 features)
dummy_features = np.random.rand(1, 28)
print(f"\nTest input shape: {dummy_features.shape}")

# Test prediction
try:
    model = model_c.get_model()
    proba = model.predict_proba(dummy_features)
    print(f"Prediction successful! Output shape: {proba.shape}")
    print(f"Probabilities: {proba[0]}")
except Exception as e:
    print(f"Prediction failed: {e}")

# Test model info function
print("\n" + "="*50)
print("Model C Info:")
print("="*50)
info = get_model_c_info()
for key, value in info.items():
    if key == "feature_names":
        print(f"{key}: {value[:5]}... (showing first 5)")
    else:
        print(f"{key}: {value}")
