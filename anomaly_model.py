"""
anomaly_model.py

ML component of the dashboard: an Isolation Forest that learns each
sensor's normal pH behavior (level + typical hour-to-hour pattern) and
flags readings that look anomalous (sensor drift, spikes, stuck values).

Why Isolation Forest instead of a supervised classifier: we have no
labeled "good/bad" ground truth for individual readings, and only ~180
rows per sensor -- too little to reliably train a supervised model, but
enough for unsupervised anomaly detection, which is the appropriate ML
technique for this kind of sensor-monitoring problem.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def build_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a per-sensor feature set for anomaly detection:
    - pH value
    - hour of day (captures daily cycles)
    - rate of change from the previous reading (catches spikes/drift)
    """
    df = long_df.sort_values(["sensor", "datetime"]).copy()
    df["pH_diff"] = df.groupby("sensor")["pH"].diff().fillna(0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def detect_anomalies(long_df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Runs Isolation Forest per sensor (each sensor has its own normal
    range) and returns the input df with two new columns:
        anomaly_score : higher = more anomalous
        is_anomaly    : bool flag
    """
    df = build_features(long_df)
    results = []

    for sensor, group in df.groupby("sensor"):
        if len(group) < 10:
            group = group.copy()
            group["anomaly_score"] = 0.0
            group["is_anomaly"] = False
            results.append(group)
            continue

        features = group[["pH", "pH_diff", "hour_sin", "hour_cos"]]
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        model.fit(features)

        group = group.copy()
        # decision_function: lower = more anomalous -> flip sign so higher = more anomalous
        group["anomaly_score"] = -model.decision_function(features)
        group["is_anomaly"] = model.predict(features) == -1
        results.append(group)

    return pd.concat(results).sort_values(["sensor", "datetime"]).reset_index(drop=True)
