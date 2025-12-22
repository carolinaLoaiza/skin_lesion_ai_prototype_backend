"""
Model loaders for ML models.
"""

from .model_a import (
    ModelALoader,
    predict_with_model_a,
    apply_threshold,
    validate_model_exists as validate_model_a_exists,
    get_model_info as get_model_a_info
)

from .model_b import (
    ModelBLoader,
    extract_features_with_model_b,
    get_feature_names,
    validate_model_b_exists,
    get_model_b_info
)

from .model_c import (
    ModelCLoader,
    predict_with_model_c,
    prepare_features_for_model_c,
    validate_model_c_exists,
    get_model_c_info,
    explain_prediction_with_shap
)

__all__ = [
    # Model A
    "ModelALoader",
    "predict_with_model_a",
    "apply_threshold",
    "validate_model_a_exists",
    "get_model_a_info",
    # Model B
    "ModelBLoader",
    "extract_features_with_model_b",
    "get_feature_names",
    "validate_model_b_exists",
    "get_model_b_info",
    # Model C
    "ModelCLoader",
    "predict_with_model_c",
    "prepare_features_for_model_c",
    "validate_model_c_exists",
    "get_model_c_info",
    "explain_prediction_with_shap",
]
