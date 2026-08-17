"""
data_loader.py
Loads the pH sensor CSV (as exported from the pH burn-in platform / pH燒機平台)
and reshapes it into a tidy long-format DataFrame:
    datetime | date | hour | sensor | pH

Handles the export quirks of this data source automatically:
- UTF-16LE encoding
- Tab-separated values
- Chinese column headers (日期 = Date, 小時 = Hour)
"""

import pandas as pd
import io


SENSOR_PREFIX = "pH燒機平台 - "


def _read_raw(file_obj_or_path):
    """Read the raw file, auto-handling UTF-16LE + tab separation."""
    if hasattr(file_obj_or_path, "read"):
        raw_bytes = file_obj_or_path.read()
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-16-le")
        buf = io.BytesIO(raw_bytes)
    else:
        buf = file_obj_or_path

    # Try UTF-16LE first (this data source's native export encoding),
    # then fall back to UTF-8 for hand-edited / re-saved files.
    for enc in ("utf-16-le", "utf-16", "utf-8-sig", "utf-8"):
        try:
            buf.seek(0) if hasattr(buf, "seek") else None
            df = pd.read_csv(buf, encoding=enc, sep="\t")
            if df.shape[1] > 1:
                return df
        except (UnicodeError, UnicodeDecodeError):
            continue
    raise ValueError("Could not parse file with any supported encoding.")


def load_long_format(file_obj_or_path):
    """
    Returns a tidy long-format DataFrame:
        datetime, date, hour, sensor, pH
    One row per (sensor, timestamp) reading.
    """
    df = _read_raw(file_obj_or_path)
    df = df.rename(columns={"日期": "date", "小時": "hour"})

    sensor_cols = [c for c in df.columns if c.startswith(SENSOR_PREFIX)]
    if not sensor_cols:
        raise ValueError(
            "No sensor columns found. Expected columns starting with "
            f"'{SENSOR_PREFIX}'."
        )

    long_df = df.melt(
        id_vars=["date", "hour"],
        value_vars=sensor_cols,
        var_name="sensor",
        value_name="pH",
    )
    long_df["sensor"] = long_df["sensor"].str.replace(SENSOR_PREFIX, "", regex=False)
    long_df["datetime"] = pd.to_datetime(
        long_df["date"] + " " + long_df["hour"].astype(int).astype(str) + ":00",
        format="%Y/%m/%d %H:%M",
    )
    long_df = long_df.sort_values(["sensor", "datetime"]).reset_index(drop=True)
    long_df["pH"] = pd.to_numeric(long_df["pH"], errors="coerce")
    long_df = long_df.dropna(subset=["pH"])
    return long_df[["datetime", "date", "hour", "sensor", "pH"]]
