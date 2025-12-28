"""
Feature name mapping utilities.

Provides human-friendly names for technical feature identifiers
to improve frontend user experience.
"""

# Mapping from technical feature names to user-friendly display names
FEATURE_DISPLAY_NAMES = {
    # Color features (Lab color space - Inside Lesion)
    "tbp_lv_A": "Color Component A* (Inside Lesion)",
    "tbp_lv_B": "Color Component B* (Inside Lesion)",
    "tbp_lv_C": "Chroma (Inside Lesion)",
    "tbp_lv_H": "Hue Angle (Inside Lesion)",
    "tbp_lv_L": "Lightness L* (Inside Lesion)",

    # Geometric features
    "tbp_lv_areaMM2": "Lesion Area (mm²)",
    "tbp_lv_area_perim_ratio": "Border Jaggedness Index",
    "tbp_lv_minorAxisMM": "Minimum Lesion Diameter (mm)",
    "tbp_lv_perimeterMM": "Lesion Perimeter (mm)",
    "tbp_lv_symm_2axis": "Border Asymmetry Score",

    # Color variation features
    "tbp_lv_color_std_mean": "Color Irregularity",
    "tbp_lv_norm_color": "Normalized Color Variation",
    "tbp_lv_stdL": "Lightness Variability (Inside Lesion)",

    # Contrast features (Lesion vs Surrounding Skin)
    "tbp_lv_deltaA": "A* Color Contrast",
    "tbp_lv_deltaB": "B* Color Contrast",
    "tbp_lv_deltaL": "Lightness Contrast",
    "tbp_lv_deltaLB": "LB Color Contrast",
    "tbp_lv_deltaLBnorm": "Lesion-to-Skin Contrast",

    # Clinical metadata features
    "age_approx": "Patient Age",
    "sex": "Patient Sex",
    "anatom_site_general": "Lesion Location",
    "clin_size_long_diam_mm": "Lesion Diameter (mm)",
}


def get_friendly_name(technical_name: str) -> str:
    """
    Get user-friendly display name for a technical feature name.

    Args:
        technical_name: Technical feature identifier (e.g., "tbp_lv_A")

    Returns:
        User-friendly display name. Returns technical name if no mapping exists.

    Examples:
        >>> get_friendly_name("tbp_lv_A")
        "Color Component A* (Inside Lesion)"

        >>> get_friendly_name("tbp_lv_areaMM2")
        "Lesion Area (mm²)"
    """
    return FEATURE_DISPLAY_NAMES.get(technical_name, technical_name)


def get_all_feature_mappings() -> dict:
    """
    Get complete mapping of all technical to friendly feature names.

    Returns:
        Dictionary mapping technical names to friendly names
    """
    return FEATURE_DISPLAY_NAMES.copy()


def format_feature_for_display(technical_name: str, value: float, shap_value: float = None) -> dict:
    """
    Format a feature with its value for frontend display.

    Args:
        technical_name: Technical feature identifier
        value: Feature value
        shap_value: Optional SHAP value indicating feature importance

    Returns:
        Dictionary with formatted feature information

    Examples:
        >>> format_feature_for_display("tbp_lv_areaMM2", 45.2)
        {
            "technical_name": "tbp_lv_areaMM2",
            "display_name": "Lesion Area (mm²)",
            "value": 45.2
        }

        >>> format_feature_for_display("tbp_lv_deltaL", 0.5, shap_value=0.12)
        {
            "technical_name": "tbp_lv_deltaL",
            "display_name": "Lightness Contrast",
            "value": 0.5,
            "shap_value": 0.12,
            "impact": "increases"
        }
    """
    result = {
        "technical_name": technical_name,
        "display_name": get_friendly_name(technical_name),
        "value": value
    }

    if shap_value is not None:
        result["shap_value"] = shap_value
        result["impact"] = "increases" if shap_value > 0 else "decreases"

    return result
