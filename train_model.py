"""
=============================================================================
  XGBoost CLEARANCE TIME CLASSIFIER
  Input:  ml_ready.csv
  Output: xgb_clearance_classifier.pkl  +  feature_importances.csv
  Target: 3-class bin of Target_Duration_Mins
            0 = Quick  (<45 min)    → standard single-unit dispatch
            1 = Medium (45-120 min) → monitor, may need follow-up
            2 = Long   (>120 min)   → mobilise extra resources
=============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH        = "ml_ready.csv"
MODEL_PATH       = "xgb_clearance_classifier.pkl"
IMPORTANCES_PATH = "feature_importances.csv"

# Columns preserved for routing engine — excluded from the feature matrix
ROUTING_COLS  = ["latitude", "longitude", "police_station"]
TARGET_COL    = "Target_Duration_Mins"

RANDOM_STATE  = 42
TEST_SIZE     = 0.2

# =============================================================================
# 1. LOAD
# =============================================================================
print("=" * 60)
print("  XGBoost Clearance Time Classifier")
print("=" * 60)

print(f"\n[1] Loading {DATA_PATH} …")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"    Shape: {df.shape}")
print(f"    Raw target stats:\n"
      f"      mean   = {df[TARGET_COL].mean():.1f} min\n"
      f"      median = {df[TARGET_COL].median():.1f} min\n"
      f"      max    = {df[TARGET_COL].max():.1f} min")

# =============================================================================
# 2. DEFINE X AND y
# =============================================================================
print("\n[2] Defining features and target …")

# --- Features ---
drop_from_X = [TARGET_COL] + [c for c in ROUTING_COLS if c in df.columns]
X = df.drop(columns=drop_from_X)

# Sanitise column names for XGBoost (no special chars)
X.columns = [c.replace(" ", "_").replace("/", "_").replace("-", "_")
               .replace("(", "").replace(")", "")
             for c in X.columns]

# Ensure all columns are numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X.fillna(0, inplace=True)

# --- Target: bin into 3 operational classes ---
def bin_duration(mins):
    if mins < 45:  return 0   # Quick
    if mins < 120: return 1   # Medium
    return 2                   # Long

y = df[TARGET_COL].apply(bin_duration)

print(f"    Features : {X.shape[1]} columns")
print(f"    Samples  : {X.shape[0]} rows")
print(f"    Feature list:\n      {list(X.columns)}")
print(f"\n    Class distribution:")
print(f"      0 = Quick  (<45 min)    : {(y==0).sum():,} rows  ({100*(y==0).mean():.1f}%)")
print(f"      1 = Medium (45-120 min) : {(y==1).sum():,} rows  ({100*(y==1).mean():.1f}%)")
print(f"      2 = Long   (>120 min)   : {(y==2).sum():,} rows  ({100*(y==2).mean():.1f}%)")

# =============================================================================
# 3. TRAIN / TEST SPLIT  (stratified to preserve class ratios)
# =============================================================================
print(f"\n[3] Train/test split  (test={TEST_SIZE*100:.0f}%, seed={RANDOM_STATE}, stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"    Train: {X_train.shape[0]} rows   Test: {X_test.shape[0]} rows")

# =============================================================================
# 4. FIT XGBClassifier
# =============================================================================
print("\n[4] Training XGBClassifier …")

model = XGBClassifier(
    n_estimators     = 500,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    objective        = "multi:softprob",
    num_class        = 3,
    eval_metric      = "mlogloss",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    verbosity        = 0,
)

from sklearn.utils.class_weight import compute_sample_weight

# Custom weights: penalise Quick/Medium errors more, but don't destroy Long
WEIGHT_MAP = {0: 2.5, 1: 2.0, 2: 1.0}   # Quick=2.5x, Medium=2x, Long=1x
sample_weights = y_train.map(WEIGHT_MAP).values

model.fit(
    X_train, y_train,
    sample_weight = sample_weights,
    eval_set      = [(X_test, y_test)],
    verbose       = False,
)
print("    Done.")

# =============================================================================
# 5. EVALUATE
# =============================================================================
print("\n[5] Evaluation on held-out test set …")

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred) * 100

print(f"    Overall Accuracy: {acc:.1f}%  (random baseline = 33.3%)\n")
print("    Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Quick (<45 min)", "Medium (45-120 min)", "Long (>120 min)"]
))

cm = confusion_matrix(y_test, y_pred)
print("    Confusion Matrix (rows=actual, cols=predicted):")
print(f"                     Quick  Medium  Long")
labels = ["Quick ", "Medium", "Long  "]
for label, row in zip(labels, cm):
    print(f"      Actual {label} : {row}")

# =============================================================================
# 6. FEATURE IMPORTANCES
# =============================================================================
print("\n[6] Feature Importances (Top 20) …")

importances = pd.DataFrame({
    "feature"   : X.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\n{'Rank':<6}{'Feature':<45}{'Importance':>10}")
print("-" * 62)
for i, row in importances.head(20).iterrows():
    print(f"  {i+1:<4}{row['feature']:<45}{row['importance']:>10.4f}")

importances.to_csv(IMPORTANCES_PATH, index=False)
print(f"\n    Full importances saved → {IMPORTANCES_PATH}")

# =============================================================================
# 7. SAVE MODEL
# =============================================================================
print(f"\n[7] Saving model → {MODEL_PATH} …")
joblib.dump(model, MODEL_PATH)
print(f"    Saved. ({os.path.getsize(MODEL_PATH) / 1024:.1f} KB)")

# =============================================================================
# DONE
# =============================================================================
print(f"\n{'=' * 60}")
print(f"  ✅  Training complete.")
print(f"      Model   → {MODEL_PATH}")
print(f"      Reports → {IMPORTANCES_PATH}")
print(f"{'=' * 60}")