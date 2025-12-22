"""Quick test to verify XGBoost model loads correctly."""
import joblib

model_path = "saved_models/model_c_xgb_4k.pkl"

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)

print(f"Model type: {type(model)}")
print(f"Model class: {model.__class__.__name__}")

# Check for feature names
if hasattr(model, 'feature_names_in_'):
    print(f"Features from feature_names_in_: {len(model.feature_names_in_)}")
    print(f"First 5 features: {list(model.feature_names_in_)[:5]}")
elif hasattr(model, 'get_booster'):
    try:
        booster = model.get_booster()
        feature_names = booster.feature_names
        print(f"Features from booster: {len(feature_names)}")
        print(f"First 5 features: {feature_names[:5]}")
    except Exception as e:
        print(f"Could not get features from booster: {e}")
else:
    print("No feature names found")

# Check if it has predict_proba
if hasattr(model, 'predict_proba'):
    print("✓ Model has predict_proba method")
else:
    print("✗ Model does NOT have predict_proba method")

print("\nModel loaded successfully!")
