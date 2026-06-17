"""
=============================================================================
  ML PREPROCESSING PIPELINE
  Dataset: Incident/Event Management Dataset
  Output:  ml_ready.csv — clean, encoded, target-engineered dataset
  Steps:
    1. Ruthless Purge (drop dead-weight columns + null start_datetime rows)
    2. MAD Timestamp Fix (engineer Target_Duration_Mins)
    3. Imputation & Feature Encoding
    4. NLP Rescue (S_risk from description)
=============================================================================
"""

import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — EDIT THIS
# =============================================================================
DATA_PATH   = "track 2.csv"       # <-- path to your raw CSV
OUTPUT_PATH = "ml_ready.csv"      # <-- where to save the processed file

# Per-cause caps (minutes) for the MAD clip — adjust as needed
CAUSE_CAPS = {
    "default":      180,   # fallback for any unrecognised cause
    "waterlogging": 360,
    "water logging":360,
    "flood":        360,
    "vehicle breakdown": 180,
    "breakdown":    180,
    "accident":     240,
    "fire":         300,
}

# =============================================================================
# STEP 0 — LOAD
# =============================================================================
print("Loading data …")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Raw shape: {df.shape}")

# =============================================================================
# STEP 1 — THE RUTHLESS PURGE
# =============================================================================
print("\n[Step 1] Ruthless Purge …")

# 1-A  Drop 95 %+ null columns
HIGH_NULL_COLS = [
    "comment", "meta_data", "map_file", "cargo_material",
    "age_of_truck", "reason_breakdown", "route_path",
    "assigned_to_police_id", "citizen_accident_id",
]
drop_high_null = [c for c in HIGH_NULL_COLS if c in df.columns]
df.drop(columns=drop_high_null, inplace=True)
print(f"  Dropped high-null cols  : {drop_high_null}")

# 1-B  Drop useless ID columns (keep police_station)
USELESS_IDS = [
    "id", "client_id", "created_by_id", "last_modified_by_id",
    "closed_by_id", "resolved_by_id", "kgid", "gba_identifier",
]
drop_ids = [c for c in USELESS_IDS if c in df.columns]
df.drop(columns=drop_ids, inplace=True)
print(f"  Dropped ID cols         : {drop_ids}")

# 1-C  Drop rows with null start_datetime
before = len(df)
df.dropna(subset=["start_datetime"], inplace=True)
print(f"  Dropped null start_datetime rows: {before - len(df)}")
print(f"  Shape after purge: {df.shape}")

# =============================================================================
# STEP 2 — THE MAD TIMESTAMP FIX  (engineer Target_Duration_Mins)
# =============================================================================
print("\n[Step 2] MAD Timestamp Fix …")

# 2-A  Parse datetimes (UTC-aware → tz-naive for arithmetic simplicity)
for col in ["start_datetime", "modified_datetime", "closed_datetime"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        df[col] = df[col].dt.tz_localize(None)   # strip tz → naive

# 2-B  Raw duration in minutes
def raw_duration(row):
    start = row["start_datetime"]
    if pd.isnull(start):
        return np.nan
    end = row["closed_datetime"] if pd.notnull(row.get("closed_datetime")) \
          else row.get("modified_datetime")
    if pd.isnull(end):
        return np.nan
    delta = (end - start).total_seconds() / 60
    return delta if delta >= 0 else np.nan   # negative = data error → NaN

df["Raw_Duration_Mins"] = df.apply(raw_duration, axis=1)
print(f"  Raw_Duration_Mins — median: {df['Raw_Duration_Mins'].median():.1f} min  "
      f"max: {df['Raw_Duration_Mins'].max():.1f} min  "
      f"nulls: {df['Raw_Duration_Mins'].isna().sum()}")

# 2-C  MAD clip per event_cause
def get_cap(cause):
    """Return the hard cap (minutes) for a given cause label."""
    if pd.isnull(cause):
        return CAUSE_CAPS["default"]
    cause_lower = str(cause).lower().strip()
    for key, cap in CAUSE_CAPS.items():
        if key in cause_lower:
            return cap
    return CAUSE_CAPS["default"]

def mad_clip(group):
    """Clip durations in a group using Median + 3*MAD threshold."""
    vals = group["Raw_Duration_Mins"].dropna()
    if len(vals) < 5:               # too few rows → just use the hard cap
        cap = get_cap(group.name)
        return group["Raw_Duration_Mins"].clip(upper=cap)

    median = vals.median()
    mad    = (vals - median).abs().median()
    threshold = median + 3 * mad

    cap = get_cap(group.name)
    final_cap = min(threshold, cap) if threshold < cap else cap   # never exceed hard cap

    return group["Raw_Duration_Mins"].clip(upper=final_cap)

if "event_cause" in df.columns:
    df["Target_Duration_Mins"] = (
        df.groupby("event_cause", group_keys=False)
          .apply(mad_clip)
    )
else:
    # Fallback: global MAD clip if event_cause is missing
    vals   = df["Raw_Duration_Mins"].dropna()
    median = vals.median()
    mad    = (vals - median).abs().median()
    df["Target_Duration_Mins"] = df["Raw_Duration_Mins"].clip(upper=median + 3 * mad)

print(f"  Target_Duration_Mins — median: {df['Target_Duration_Mins'].median():.1f} min  "
      f"max: {df['Target_Duration_Mins'].max():.1f} min  "
      f"nulls: {df['Target_Duration_Mins'].isna().sum()}")

# Drop rows where we couldn't compute a target (no end time at all)
before = len(df)
df.dropna(subset=["Target_Duration_Mins"], inplace=True)
print(f"  Dropped rows with no computable duration: {before - len(df)}")
df.drop(columns=["Raw_Duration_Mins"], inplace=True)

# =============================================================================
# STEP 3 — IMPUTATION & FEATURE ENCODING
# =============================================================================
print("\n[Step 3] Imputation & Encoding …")

# 3-A  veh_type: fill nulls with UNKNOWN (missing is a signal)
if "veh_type" in df.columns:
    null_veh = df["veh_type"].isna().sum()
    df["veh_type"] = df["veh_type"].fillna("UNKNOWN")
    print(f"  veh_type: filled {null_veh} nulls with 'UNKNOWN'")

# 3-B  priority: map to integers, fill nulls with 1 (Low)
PRIORITY_MAP = {"High": 3, "Medium": 2, "Low": 1,
                "high": 3, "medium": 2, "low": 1,
                "HIGH": 3, "MEDIUM": 2, "LOW": 1}
if "priority" in df.columns:
    df["priority"] = df["priority"].map(PRIORITY_MAP).fillna(1).astype(int)
    print(f"  priority: mapped to int (High=3, Medium=2, Low=1), nulls → 1")

# 3-C  One-hot encode event_cause, veh_type, zone
OHE_COLS = [c for c in ["event_cause", "veh_type", "zone"] if c in df.columns]
if OHE_COLS:
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=False, dtype=int)
    print(f"  One-hot encoded: {OHE_COLS}")

# =============================================================================
# STEP 4 — NLP RESCUE  (S_risk from description)
# =============================================================================
print("\n[Step 4] NLP Rescue — building S_risk …")

HIGH_RISK_KEYWORDS   = ["fire", "accident", "axle", "multi", "hazardous",
                         "water logging", "waterlogging", "chemical", "heavy"]
MEDIUM_RISK_KEYWORDS = ["puncture", "breakdown", "tyre", "starting problem",
                         "bussu", "car"]

def compute_s_risk(text):
    if pd.isnull(text) or str(text).strip() == "":
        return 1
    text_lower = str(text).lower()
    for kw in HIGH_RISK_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            return 3
    for kw in MEDIUM_RISK_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            return 2
    return 1

if "description" in df.columns:
    df["S_risk"] = df["description"].apply(compute_s_risk)
    risk_counts = df["S_risk"].value_counts().sort_index()
    print(f"  S_risk distribution:\n{risk_counts.to_string()}")
    df.drop(columns=["description"], inplace=True)
    print("  Dropped 'description' column")
else:
    print("  'description' column not found — S_risk set to 1")
    df["S_risk"] = 1

# =============================================================================
# STEP 5 — FINAL CLEANUP
# =============================================================================
print("\n[Step 5] Final cleanup …")

# Drop remaining raw datetime cols (not useful as ML features directly)
DATETIME_DROP = ["start_datetime", "modified_datetime", "closed_datetime",
                 "end_datetime", "created_date", "resolved_datetime"]
dt_drop = [c for c in DATETIME_DROP if c in df.columns]
df.drop(columns=dt_drop, inplace=True)
print(f"  Dropped raw datetime cols: {dt_drop}")

# Drop other text / geo cols that weren't handled above
OTHER_DROP = ["address", "end_address", "resolved_at_address",
              "veh_no", "junction", "corridor", "direction",
              "endlatitude", "endlongitude",
              "resolved_at_latitude", "resolved_at_longitude",
              "authenticated", "requires_road_closure",
              "event_type", "status",   # leakage risk — remove if needed
              ]
# NOTE: latitude, longitude, and police_station are intentionally kept.
# They are NOT ML features — drop them from X during train_test_split.
# They are preserved here for the downstream routing engine.
other_drop = [c for c in OTHER_DROP if c in df.columns]
df.drop(columns=other_drop, inplace=True)
print(f"  Dropped misc/leakage cols: {other_drop}")

# Convert all bool columns to int
bool_cols = df.select_dtypes(include="bool").columns.tolist()
df[bool_cols] = df[bool_cols].astype(int)

# =============================================================================
# SAVE
# =============================================================================
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n{'='*60}")
print(f"  ✅  ML-ready CSV saved → {OUTPUT_PATH}")
print(f"  Final shape : {df.shape}")
print(f"  Columns     : {list(df.columns)}")
print(f"{'='*60}")