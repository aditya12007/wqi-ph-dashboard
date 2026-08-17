"""
app.py — pH Water Quality Monitoring Dashboard

Run with:
    streamlit run app.py

Features:
- Upload your own sensor export (or use the bundled sample data)
- pH-based water quality sub-index per sensor, per hour
- Trend charts per sensor with ideal-range shading
- ML anomaly detection (Isolation Forest) flagging unusual readings
- Summary table across all sensors
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data_loader import load_long_format
from wqi_calculator import add_wqi_columns, CATEGORY_COLORS, IDEAL_LOW, IDEAL_HIGH
from anomaly_model import detect_anomalies

st.set_page_config(page_title="Water Quality Dashboard", layout="wide")

st.title("💧 Water Quality Monitoring Dashboard")
st.caption(
    "Live pH monitoring across sensors, with a pH-based quality sub-index "
    "and ML-powered anomaly detection."
)

with st.expander("ℹ️ About this dashboard's WQI", expanded=False):
    st.markdown(
        """
This data source currently provides **pH only** (no dissolved oxygen, BOD,
turbidity, TDS, nitrate, or coliform readings). A true multi-parameter
Water Quality Index needs those additional parameters.

What's shown here is a **pH-based sub-index (0-100)**, computed the same
way the pH term is computed inside the standard NSF-WQI formula. Treat it
as a partial indicator, not a complete WQI. Add more parameter columns to
`wqi_calculator.py` and this becomes a full WQI automatically.
        """
    )

# ---------- Data loading ----------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "Upload sensor CSV (matches the pH燒機平台 export format)", type=["csv"]
)

DEFAULT_PATH = "data/water_quality.csv"

try:
    if uploaded is not None:
        long_df = load_long_format(uploaded)
        st.sidebar.success(f"Loaded {uploaded.name}")
    else:
        long_df = load_long_format(DEFAULT_PATH)
        st.sidebar.info("Using bundled sample data (1 week, 8 sensors)")
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

long_df = add_wqi_columns(long_df)

# ---------- Sidebar filters ----------
sensors = sorted(long_df["sensor"].unique())
selected_sensors = st.sidebar.multiselect("Sensors", sensors, default=sensors)

dates = sorted(long_df["date"].unique())
date_range = st.sidebar.select_slider(
    "Date range", options=dates, value=(dates[0], dates[-1])
)

contamination = st.sidebar.slider(
    "Anomaly sensitivity (higher = flags more readings)", 0.01, 0.15, 0.05, 0.01
)

filtered = long_df[
    (long_df["sensor"].isin(selected_sensors))
    & (long_df["date"] >= date_range[0])
    & (long_df["date"] <= date_range[1])
]

if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------- ML: anomaly detection ----------
with st.spinner("Running anomaly detection model..."):
    scored = detect_anomalies(filtered, contamination=contamination)

# ---------- Top-line metrics ----------
col1, col2, col3, col4 = st.columns(4)
avg_score = scored["pH_sub_index"].mean()
avg_ph = scored["pH"].mean()
n_anomalies = int(scored["is_anomaly"].sum())
n_sensors = scored["sensor"].nunique()

col1.metric("Avg. Quality Sub-Index", f"{avg_score:.1f} / 100")
col2.metric("Avg. pH", f"{avg_ph:.2f}")
col3.metric("Anomalous Readings", n_anomalies)
col4.metric("Sensors Monitored", n_sensors)

# ---------- Gauge ----------
gauge_col, cat_col = st.columns([1, 2])
with gauge_col:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_score,
            title={"text": "Overall pH Sub-Index"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1976D2"},
                "steps": [
                    {"range": [0, 25], "color": "#E53935"},
                    {"range": [25, 50], "color": "#FB8C00"},
                    {"range": [50, 70], "color": "#FDD835"},
                    {"range": [70, 90], "color": "#66BB6A"},
                    {"range": [90, 100], "color": "#2E7D32"},
                ],
            },
        )
    )
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with cat_col:
    cat_counts = scored["category"].value_counts().reindex(
        ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
    ).fillna(0)
    fig_bar = px.bar(
        x=cat_counts.index,
        y=cat_counts.values,
        color=cat_counts.index,
        color_discrete_map=CATEGORY_COLORS,
        labels={"x": "Category", "y": "Number of readings"},
        title="Readings by Quality Category",
    )
    fig_bar.update_layout(showlegend=False, height=280)
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------- Trend chart per sensor ----------
st.subheader("pH Trend by Sensor")
fig_trend = px.line(
    scored, x="datetime", y="pH", color="sensor",
    labels={"datetime": "Time", "pH": "pH"},
)
fig_trend.add_hrect(
    y0=IDEAL_LOW, y1=IDEAL_HIGH, fillcolor="green", opacity=0.08,
    line_width=0, annotation_text="Ideal range", annotation_position="top left",
)
anomaly_points = scored[scored["is_anomaly"]]
if not anomaly_points.empty:
    fig_trend.add_scatter(
        x=anomaly_points["datetime"], y=anomaly_points["pH"],
        mode="markers", marker=dict(color="red", size=9, symbol="x"),
        name="Anomaly (ML)",
    )
fig_trend.update_layout(height=450)
st.plotly_chart(fig_trend, use_container_width=True)

# ---------- Anomaly detail table ----------
st.subheader("🚨 Flagged Anomalies")
if anomaly_points.empty:
    st.success("No anomalies detected in the selected range.")
else:
    st.dataframe(
        anomaly_points[["datetime", "sensor", "pH", "pH_sub_index", "category", "anomaly_score"]]
        .sort_values("anomaly_score", ascending=False)
        .round(3),
        use_container_width=True,
    )

# ---------- Per-sensor summary table ----------
st.subheader("Per-Sensor Summary")
summary = (
    scored.groupby("sensor")
    .agg(
        avg_pH=("pH", "mean"),
        min_pH=("pH", "min"),
        max_pH=("pH", "max"),
        avg_sub_index=("pH_sub_index", "mean"),
        anomalies=("is_anomaly", "sum"),
        readings=("pH", "count"),
    )
    .round(2)
    .sort_values("avg_sub_index")
)
st.dataframe(summary, use_container_width=True)

st.caption(
    "Model: Isolation Forest (unsupervised), trained per-sensor on pH level, "
    "hour-of-day cycle, and reading-to-reading change."
)
