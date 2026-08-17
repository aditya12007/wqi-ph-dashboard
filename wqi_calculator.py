"""
wqi_calculator.py

IMPORTANT: This dataset only contains pH readings (no DO, BOD, turbidity,
TDS, nitrate, or coliform). A true multi-parameter WQI cannot be computed
from pH alone. This module instead computes a pH-based water quality
SUB-INDEX (0-100), using the same style of quality-rating curve that the
pH term uses inside the standard NSF-WQI formula. It is clearly a partial
indicator, not a substitute for a full WQI.

If/when more parameters (DO, BOD, turbidity, etc.) become available, this
module is where you'd add their sub-indices and combine them with weights
into a true WQI, e.g.:

    WQI = sum(Wi * Qi) / sum(Wi)

where Qi is each parameter's 0-100 quality rating and Wi is its assigned
weight (e.g. NSF-WQI weights: DO 0.17, Fecal Coliform 0.16, pH 0.11,
BOD 0.11, Temperature 0.10, Turbidity 0.08, Total Phosphate 0.10,
Nitrates 0.10, Total Solids 0.07).
"""

import numpy as np
import pandas as pd

# Ideal pH range for most freshwater aquatic life / drinking water use
IDEAL_LOW, IDEAL_HIGH = 6.5, 8.5
IDEAL_CENTER = 7.5


def ph_sub_index(ph: float) -> float:
    """
    Converts a pH reading into a 0-100 quality rating (Qi).
    Score is 100 at the ideal center (7.5) and decays as pH moves away
    from the 6.5-8.5 ideal band, reaching 0 by pH 4 or pH 11.
    """
    if pd.isna(ph):
        return np.nan
    distance = abs(ph - IDEAL_CENTER)
    if distance <= 1.0:  # within 6.5-8.5
        score = 100 - (distance * 20)  # 100 at center, 80 at the edges
    else:
        # steeper falloff outside the ideal band
        score = 80 - (distance - 1.0) * 25
    return float(np.clip(score, 0, 100))


def classify(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 90:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Fair"
    if score >= 25:
        return "Poor"
    return "Very Poor"


CATEGORY_COLORS = {
    "Excellent": "#2E7D32",
    "Good": "#66BB6A",
    "Fair": "#FDD835",
    "Poor": "#FB8C00",
    "Very Poor": "#E53935",
    "Unknown": "#9E9E9E",
}


def add_wqi_columns(long_df: pd.DataFrame) -> pd.DataFrame:
    """Adds pH_sub_index and category columns to a long-format DataFrame."""
    out = long_df.copy()
    out["pH_sub_index"] = out["pH"].apply(ph_sub_index)
    out["category"] = out["pH_sub_index"].apply(classify)
    return out
