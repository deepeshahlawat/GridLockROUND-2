"""
=============================================================================
  END-TO-END EXPLORATORY DATA ANALYSIS (EDA) SCRIPT
  Dataset: Incident/Event Management Dataset (8,173 rows)
  Purpose: Full data profiling before ML model training
  Author: Senior Data Analyst Script
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import sys
import json
from datetime import datetime
from collections import Counter

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)

# =============================================================================
# CONFIGURATION — EDIT THIS
# =============================================================================
DATA_PATH = "track 2.csv"          # <-- Change to your file path
OUTPUT_DIR = "eda_outputs"              # All reports/plots saved here
DATETIME_COLS = ['start_datetime', 'end_datetime', 'modified_datetime',
                 'created_date', 'closed_datetime', 'resolved_datetime']
GEO_COLS = ['latitude', 'longitude', 'endlatitude', 'endlongitude',
            'resolved_at_latitude', 'resolved_at_longitude']
ID_COLS = ['id', 'client_id', 'created_by_id', 'last_modified_by_id',
           'assigned_to_police_id', 'citizen_accident_id', 'closed_by_id',
           'resolved_by_id', 'kgid', 'gba_identifier']
BOOL_COLS = ['requires_road_closure', 'authenticated']
CATEGORICAL_COLS = ['event_type', 'event_cause', 'status', 'direction',
                    'veh_type', 'corridor', 'priority', 'cargo_material',
                    'reason_breakdown', 'zone', 'junction', 'police_station']
TEXT_COLS = ['address', 'end_address', 'description', 'comment',
             'resolved_at_address', 'meta_data', 'route_path']
NUMERIC_COLS = ['veh_no', 'age_of_truck']

os.makedirs(OUTPUT_DIR, exist_ok=True)

DIVIDER = "=" * 80
SECTION  = "-" * 80

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def print_subsection(title):
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [SAVED] {path}")

def pct(n, total):
    if total == 0:
        return f"{n:,} (N/A)"
    return f"{n:,} ({100*n/total:.2f}%)"

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print_section("1. LOADING DATA")

try:
    if DATA_PATH.endswith('.csv'):
        df = pd.read_csv(DATA_PATH, low_memory=False)
    elif DATA_PATH.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  File loaded successfully: {DATA_PATH}")
except FileNotFoundError:
    print(f"  [ERROR] File not found: {DATA_PATH}")
    print("  Creating a DEMO run with synthetic data matching your schema...")
    # --- DEMO SYNTHETIC DATA (remove once you have real data) ---
    np.random.seed(42)
    n = 8173
    df = pd.DataFrame({
        'id': range(1, n+1),
        'event_type': np.random.choice(['Accident','Breakdown','Obstruction','Flooding', None], n, p=[0.35,0.30,0.20,0.10,0.05]),
        'latitude': np.random.uniform(12.8, 13.2, n),
        'longitude': np.random.uniform(77.4, 77.8, n),
        'endlatitude': np.where(np.random.rand(n) > 0.3, np.random.uniform(12.8, 13.2, n), np.nan),
        'endlongitude': np.where(np.random.rand(n) > 0.3, np.random.uniform(77.4, 77.8, n), np.nan),
        'address': np.random.choice(['MG Road', 'Outer Ring Road', 'NH48', None], n),
        'end_address': np.random.choice(['Silk Board', 'KR Puram', None, None], n),
        'event_cause': np.random.choice(['Vehicle Breakdown','Weather','Road Defect','Human Error', None], n, p=[0.30,0.20,0.15,0.25,0.10]),
        'requires_road_closure': np.random.choice([True, False, None], n, p=[0.20,0.70,0.10]),
        'start_datetime': pd.date_range('2022-01-01', periods=n, freq='1h'),
        'end_datetime': pd.date_range('2022-01-01 01:00', periods=n, freq='1h'),
        'status': np.random.choice(['Open','Closed','In Progress','Resolved', None], n, p=[0.25,0.35,0.20,0.15,0.05]),
        'authenticated': np.random.choice([True, False, None], n, p=[0.60,0.30,0.10]),
        'modified_datetime': pd.date_range('2022-01-01 00:30', periods=n, freq='1h'),
        'map_file': np.random.choice(['file1.png','file2.png', None, None, None], n),
        'direction': np.random.choice(['North','South','East','West','Both', None], n, p=[0.20,0.20,0.20,0.20,0.10,0.10]),
        'description': np.random.choice(['Minor accident', 'Major breakdown', None, None], n),
        'veh_type': np.random.choice(['Truck','Car','Bus','Two-Wheeler', None], n, p=[0.30,0.30,0.20,0.10,0.10]),
        'veh_no': np.where(np.random.rand(n) > 0.15, np.random.randint(1, 5, n), np.nan),
        'corridor': np.random.choice(['Corridor A','Corridor B','Corridor C', None], n, p=[0.30,0.30,0.20,0.20]),
        'priority': np.random.choice(['High','Medium','Low', None], n, p=[0.25,0.40,0.25,0.10]),
        'cargo_material': np.random.choice(['Hazardous','General','Perishable', None, None], n),
        'reason_breakdown': np.random.choice(['Engine Failure','Tyre Puncture','Fuel', None, None, None], n),
        'age_of_truck': np.where(np.random.rand(n) > 0.3, np.random.randint(1, 20, n), np.nan),
        'created_date': pd.date_range('2022-01-01', periods=n, freq='1h'),
        'route_path': np.random.choice(['path1', None, None, None], n),
        'client_id': np.random.choice(range(1, 50), n),
        'created_by_id': np.random.choice(range(1, 100), n),
        'last_modified_by_id': np.random.choice(range(1, 100), n),
        'assigned_to_police_id': np.where(np.random.rand(n) > 0.5, np.random.randint(1, 20, n), np.nan),
        'citizen_accident_id': np.where(np.random.rand(n) > 0.7, np.random.randint(1, 500, n), np.nan),
        'comment': np.random.choice(['Under review', 'Resolved quickly', None, None, None], n),
        'police_station': np.random.choice(['Station A','Station B', None, None], n),
        'meta_data': np.random.choice(['{}', '{"key":"val"}', None, None], n),
        'kgid': np.where(np.random.rand(n) > 0.4, np.random.randint(10000, 99999, n), np.nan),
        'resolved_at_address': np.random.choice(['MG Road', None, None, None], n),
        'resolved_at_latitude': np.where(np.random.rand(n) > 0.4, np.random.uniform(12.8, 13.2, n), np.nan),
        'resolved_at_longitude': np.where(np.random.rand(n) > 0.4, np.random.uniform(77.4, 77.8, n), np.nan),
        'closed_by_id': np.where(np.random.rand(n) > 0.35, np.random.randint(1, 100, n), np.nan),
        'closed_datetime': np.where(np.random.rand(n) > 0.35, pd.date_range('2022-01-02', periods=n, freq='1h').astype(str), None),
        'resolved_by_id': np.where(np.random.rand(n) > 0.4, np.random.randint(1, 100, n), np.nan),
        'resolved_datetime': np.where(np.random.rand(n) > 0.4, pd.date_range('2022-01-02 02:00', periods=n, freq='1h').astype(str), None),
        'gba_identifier': np.where(np.random.rand(n) > 0.6, np.random.randint(1000, 9999, n), np.nan),
        'zone': np.random.choice(['Zone 1','Zone 2','Zone 3','Zone 4', None], n, p=[0.25,0.25,0.25,0.15,0.10]),
        'junction': np.random.choice(['J1','J2','J3', None, None], n),
    })
    print("  [DEMO MODE] Synthetic data created.")

TOTAL_ROWS = len(df)
TOTAL_COLS = len(df.columns)
print(f"\n  Rows    : {TOTAL_ROWS:,}")
print(f"  Columns : {TOTAL_COLS}")

# =============================================================================
# 2. PARSE DATETIME COLUMNS
# =============================================================================

print_section("2. PARSING DATETIME COLUMNS")
for col in DATETIME_COLS:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        nulls = df[col].isna().sum()
        print(f"  {col:30s} → parsed | NaT count: {pct(nulls, TOTAL_ROWS)}")

# =============================================================================
# 3. SCHEMA OVERVIEW
# =============================================================================

print_section("3. SCHEMA & DATA TYPES")
schema_df = pd.DataFrame({
    'Column': df.columns,
    'Dtype': df.dtypes.values,
    'Non_Null': df.count().values,
    'Null_Count': df.isnull().sum().values,
    'Null_Pct': (df.isnull().sum().values / TOTAL_ROWS * 100).round(2),
    'Unique': [df[c].nunique(dropna=True) for c in df.columns],
    'Sample': [str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else 'ALL NULL' for c in df.columns]
})
print(schema_df.to_string(index=False))
schema_df.to_csv(os.path.join(OUTPUT_DIR, 'schema_overview.csv'), index=False)

# =============================================================================
# 4. NULLABILITY MATRIX
# =============================================================================

print_section("4. DETAILED NULL ANALYSIS")

null_counts = df.isnull().sum()
zero_counts = pd.Series({c: (df[c] == 0).sum() for c in df.columns})
empty_str_counts = pd.Series({c: (df[c].astype(str).str.strip() == '').sum() for c in df.columns})

null_df = pd.DataFrame({
    'Column': null_counts.index,
    'Null_Count': null_counts.values,
    'Null_Pct': (null_counts.values / TOTAL_ROWS * 100).round(2),
    'Zero_Count': zero_counts.values,
    'Zero_Pct': (zero_counts.values / TOTAL_ROWS * 100).round(2),
    'EmptyStr_Count': empty_str_counts.values,
}).sort_values('Null_Pct', ascending=False)

print("\n  Top columns by NULL %:")
print(null_df[null_df['Null_Count'] > 0].to_string(index=False))

null_df.to_csv(os.path.join(OUTPUT_DIR, 'null_analysis.csv'), index=False)

# Null heatmap
plt.figure(figsize=(22, 10))
null_matrix = df.isnull().astype(int)
# Sample for visual clarity
sample_idx = np.linspace(0, TOTAL_ROWS - 1, min(500, TOTAL_ROWS), dtype=int)
sns.heatmap(null_matrix.iloc[sample_idx].T, cmap='Reds', cbar=False,
            yticklabels=df.columns, xticklabels=False)
plt.title('NULL VALUE HEATMAP (Sampled 500 rows) — Red = Missing', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('null_heatmap.png')

# Null bar chart
fig, ax = plt.subplots(figsize=(18, 8))
cols_with_nulls = null_df[null_df['Null_Count'] > 0]
colors = ['#d62728' if p > 50 else '#ff7f0e' if p > 20 else '#1f77b4' for p in cols_with_nulls['Null_Pct']]
bars = ax.barh(cols_with_nulls['Column'], cols_with_nulls['Null_Pct'], color=colors)
ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
ax.axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='20% threshold')
for bar, val in zip(bars, cols_with_nulls['Null_Pct']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=8)
ax.set_xlabel('Null Percentage (%)')
ax.set_title('NULL % PER COLUMN — Red>50%, Orange>20%, Blue<20%', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
save_fig('null_percentage_bar.png')

# =============================================================================
# 5. CATEGORICAL COLUMN ANALYSIS
# =============================================================================

print_section("5. CATEGORICAL COLUMN ANALYSIS")

for col in CATEGORICAL_COLS:
    if col not in df.columns:
        continue
    print_subsection(f"COLUMN: {col}")
    s = df[col]
    null_n = s.isna().sum()
    unique_n = s.nunique(dropna=True)
    vc = s.value_counts(dropna=False)

    print(f"  Null count    : {pct(null_n, TOTAL_ROWS)}")
    print(f"  Unique values : {unique_n}")
    print(f"  Top 15 values :")
    print(vc.head(15).to_string())

    # Check for hidden nulls
    hidden_nulls = s.astype(str).str.lower().isin(['nan', 'none', 'null', 'na', '', 'n/a', 'unknown']).sum()
    if hidden_nulls > 0:
        print(f"  ⚠️  HIDDEN NULLS (string 'nan','none','null',etc.): {pct(hidden_nulls, TOTAL_ROWS)}")

    # Cardinality classification
    if unique_n == 1:
        print(f"  ⚠️  CONSTANT COLUMN — only 1 unique value. Drop for ML.")
    elif unique_n == TOTAL_ROWS:
        print(f"  ⚠️  HIGH CARDINALITY — every row unique (likely an ID).")
    elif unique_n > 50:
        print(f"  ⚠️  HIGH CARDINALITY ({unique_n} unique). Consider grouping/encoding carefully.")
    elif unique_n <= 10:
        print(f"  ✅  LOW CARDINALITY ({unique_n} unique). Good for one-hot or label encoding.")

# Categorical distribution plots
cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
n_cats = len(cat_cols_present)
if n_cats > 0:
    ncols = 3
    nrows = int(np.ceil(n_cats / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cat_cols_present):
        vc = df[col].value_counts(dropna=False).head(10)
        vc.plot(kind='barh', ax=axes[i], color='steelblue')
        axes[i].set_title(f'{col} (top 10)', fontweight='bold')
        axes[i].set_xlabel('Count')
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('CATEGORICAL COLUMN DISTRIBUTIONS', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig('categorical_distributions.png')

# =============================================================================
# 6. NUMERIC COLUMN ANALYSIS
# =============================================================================

print_section("6. NUMERIC COLUMN ANALYSIS")

num_cols_all = list(df.select_dtypes(include=[np.number]).columns)
print(f"\n  All numeric columns detected: {num_cols_all}\n")

numeric_stats = []
for col in num_cols_all:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    stats = {
        'Column': col,
        'Count': len(s),
        'Null_Count': df[col].isna().sum(),
        'Null_Pct': round(df[col].isna().sum() / TOTAL_ROWS * 100, 2),
        'Zero_Count': (s == 0).sum(),
        'Zero_Pct': round((s == 0).sum() / len(s) * 100, 2),
        'Negative_Count': (s < 0).sum(),
        'Min': s.min(),
        'Max': s.max(),
        'Mean': round(s.mean(), 4),
        'Median': s.median(),
        'Std': round(s.std(), 4),
        'Skewness': round(s.skew(), 4),
        'Kurtosis': round(s.kurtosis(), 4),
        'Q1': s.quantile(0.25),
        'Q3': s.quantile(0.75),
        'IQR': s.quantile(0.75) - s.quantile(0.25),
        'Unique': s.nunique(),
    }
    # Outlier detection (IQR method)
    IQR = stats['IQR']
    lower = stats['Q1'] - 1.5 * IQR
    upper = stats['Q3'] + 1.5 * IQR
    outliers = ((s < lower) | (s > upper)).sum()
    stats['Outliers_IQR'] = outliers
    stats['Outlier_Pct'] = round(outliers / len(s) * 100, 2)
    stats['Lower_Fence'] = round(lower, 4)
    stats['Upper_Fence'] = round(upper, 4)
    numeric_stats.append(stats)
    print_subsection(f"COLUMN: {col}")
    for k, v in stats.items():
        flag = ""
        if k == 'Null_Pct' and v > 30: flag = "  ⚠️  HIGH NULL"
        if k == 'Zero_Pct' and v > 30: flag = "  ⚠️  MANY ZEROS"
        if k == 'Skewness' and abs(v) > 1: flag = "  ⚠️  SKEWED — consider log transform"
        if k == 'Outlier_Pct' and v > 5: flag = "  ⚠️  MANY OUTLIERS"
        if k == 'Negative_Count' and v > 0: flag = "  ⚠️  NEGATIVE VALUES — check validity"
        print(f"  {k:25s}: {v}{flag}")

numeric_stats_df = pd.DataFrame(numeric_stats)
numeric_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'numeric_stats.csv'), index=False)

# Boxplots for numeric cols
if num_cols_all:
    plot_cols = [c for c in num_cols_all if c not in ID_COLS and df[c].dropna().shape[0] > 0]
    if plot_cols:
        fig, axes = plt.subplots(2, int(np.ceil(len(plot_cols)/2)), figsize=(20, 10))
        axes = axes.flatten()
        for i, col in enumerate(plot_cols):
            df[col].dropna().plot(kind='box', ax=axes[i], vert=True)
            axes[i].set_title(col, fontweight='bold')
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('BOXPLOTS — Numeric Columns (outlier detection)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_fig('numeric_boxplots.png')

    # Histograms
    fig, axes = plt.subplots(2, int(np.ceil(len(plot_cols)/2)), figsize=(20, 10))
    axes = axes.flatten()
    for i, col in enumerate(plot_cols):
        df[col].dropna().plot(kind='hist', bins=40, ax=axes[i], color='steelblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(col, fontweight='bold')
        axes[i].set_xlabel(col)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('HISTOGRAMS — Numeric Columns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('numeric_histograms.png')

# =============================================================================
# 7. GEO / COORDINATE ANALYSIS
# =============================================================================

print_section("7. GEOGRAPHIC COORDINATE ANALYSIS")

geo_pairs = [
    ('latitude', 'longitude', 'Start Location'),
    ('endlatitude', 'endlongitude', 'End Location'),
    ('resolved_at_latitude', 'resolved_at_longitude', 'Resolved Location'),
]

for lat_col, lon_col, label in geo_pairs:
    if lat_col not in df.columns or lon_col not in df.columns:
        continue
    print_subsection(f"{label} ({lat_col}, {lon_col})")
    lat = df[lat_col]
    lon = df[lon_col]
    both_null = (lat.isna() & lon.isna()).sum()
    one_null  = (lat.isna() ^ lon.isna()).sum()
    both_zero = ((lat == 0) & (lon == 0)).sum()
    valid     = lat.notna() & lon.notna() & (lat != 0) & (lon != 0)
    print(f"  Both null          : {pct(both_null, TOTAL_ROWS)}")
    print(f"  One null, one not  : {pct(one_null, TOTAL_ROWS)}  ⚠️  MISMATCHED PAIR")
    print(f"  Both zero (0,0)    : {pct(both_zero, TOTAL_ROWS)}  ⚠️  NULL ISLAND — invalid coords")
    print(f"  Valid pairs        : {pct(valid.sum(), TOTAL_ROWS)}")
    if valid.sum() > 0:
        print(f"  Lat range          : [{lat[valid].min():.4f}, {lat[valid].max():.4f}]")
        print(f"  Lon range          : [{lon[valid].min():.4f}, {lon[valid].max():.4f}]")
        # Out-of-range check (India bounding box as example)
        out_range = valid & ((lat < 6) | (lat > 38) | (lon < 68) | (lon > 98))
        if out_range.sum() > 0:
            print(f"  ⚠️  OUT-OF-RANGE (outside India bbox): {pct(out_range.sum(), TOTAL_ROWS)}")

    # Scatter plot
    if valid.sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        sample_valid = df[valid].sample(min(2000, valid.sum()), random_state=42)
        ax.scatter(sample_valid[lon_col], sample_valid[lat_col], alpha=0.3, s=5, c='steelblue')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{label} — Coordinate Scatter (sampled)', fontweight='bold')
        plt.tight_layout()
        save_fig(f'geo_scatter_{lat_col}.png')

# =============================================================================
# 8. DATETIME COLUMN ANALYSIS
# =============================================================================

print_section("8. DATETIME COLUMN ANALYSIS")

for col in DATETIME_COLS:
    if col not in df.columns:
        continue
    s = df[col].dropna()
    print_subsection(f"COLUMN: {col}")
    null_n = df[col].isna().sum()
    print(f"  Null count  : {pct(null_n, TOTAL_ROWS)}")
    if len(s) > 0:
        print(f"  Min date    : {s.min()}")
        print(f"  Max date    : {s.max()}")
        print(f"  Date range  : {(s.max() - s.min()).days} days")
        future = (s > pd.Timestamp.now(tz='UTC')).sum()
        if future > 0:
            print(f"  ⚠️  FUTURE DATES: {pct(future, TOTAL_ROWS)}")
        past_2000 = (s < pd.Timestamp('2000-01-01', tz='UTC')).sum()
        if past_2000 > 0:
            print(f"  ⚠️  SUSPICIOUSLY OLD DATES (<2000): {pct(past_2000, TOTAL_ROWS)}")
        # Distribution by year/month
        print(f"\n  By Year:")
        print(s.dt.year.value_counts().sort_index().to_string())
        print(f"\n  By Hour of Day:")
        print(s.dt.hour.value_counts().sort_index().head(24).to_string())

# Temporal logic checks
print_subsection("TEMPORAL LOGIC CHECKS")
checks = [
    ('start_datetime', 'end_datetime', 'start > end'),
    ('start_datetime', 'closed_datetime', 'start > closed'),
    ('start_datetime', 'resolved_datetime', 'start > resolved'),
    ('created_date', 'start_datetime', 'created > start'),
]
for c1, c2, label in checks:
    if c1 in df.columns and c2 in df.columns:
        both_valid = df[c1].notna() & df[c2].notna()
        violated = (df[c1][both_valid] > df[c2][both_valid]).sum()
        print(f"  {label:35s}: {pct(violated, both_valid.sum())} of valid pairs ⚠️" if violated > 0 else f"  {label:35s}: OK ✅")

# Duration analysis
if 'start_datetime' in df.columns and 'end_datetime' in df.columns:
    mask = df['start_datetime'].notna() & df['end_datetime'].notna()
    duration = (df.loc[mask, 'end_datetime'] - df.loc[mask, 'start_datetime']).dt.total_seconds() / 60
    duration_valid = duration[duration > 0]
    print(f"\n  Event Duration (minutes) — valid positive durations: {len(duration_valid):,}")
    if len(duration_valid) > 0:
        print(f"  Min: {duration_valid.min():.1f} | Max: {duration_valid.max():.1f} | "
              f"Mean: {duration_valid.mean():.1f} | Median: {duration_valid.median():.1f}")
        neg = (duration <= 0).sum()
        print(f"  ⚠️  Zero or negative durations: {pct(neg, mask.sum())}")

    # Plot duration distribution
    if len(duration_valid) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        clipped = duration_valid.clip(upper=duration_valid.quantile(0.99))
        clipped.plot(kind='hist', bins=60, ax=ax, color='teal', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Duration (minutes)')
        ax.set_title('EVENT DURATION DISTRIBUTION (99th pct clipped)', fontweight='bold')
        plt.tight_layout()
        save_fig('event_duration_histogram.png')

# Time series — events per day
if 'start_datetime' in df.columns:
    s = df['start_datetime'].dropna()
    if len(s) > 0:
        daily = s.dt.date.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(16, 5))
        daily.plot(ax=ax, color='steelblue', alpha=0.8)
        ax.set_title('EVENTS PER DAY OVER TIME', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Event Count')
        plt.tight_layout()
        save_fig('events_per_day.png')

# =============================================================================
# 9. ID / KEY COLUMN ANALYSIS
# =============================================================================

print_section("9. ID & KEY COLUMN ANALYSIS")

for col in ID_COLS:
    if col not in df.columns:
        continue
    s = df[col]
    null_n = s.isna().sum()
    unique_n = s.nunique(dropna=True)
    dupes = TOTAL_ROWS - null_n - unique_n
    print(f"\n  {col}:")
    print(f"    Null count    : {pct(null_n, TOTAL_ROWS)}")
    print(f"    Unique values : {unique_n:,}")
    if col == 'id':
        print(f"    Duplicates    : {dupes:,}  {'⚠️  DUPLICATE PRIMARY KEY!' if dupes > 0 else '✅ No duplicates'}")
    else:
        print(f"    Duplicates    : {dupes:,} (OK for FK columns)")
    if unique_n == TOTAL_ROWS - null_n:
        print(f"    ✅ All unique (likely primary/foreign key)")

# =============================================================================
# 10. BOOLEAN COLUMN ANALYSIS
# =============================================================================

print_section("10. BOOLEAN / FLAG COLUMN ANALYSIS")

for col in BOOL_COLS:
    if col not in df.columns:
        continue
    s = df[col]
    print_subsection(f"COLUMN: {col}")
    print(s.value_counts(dropna=False).to_string())
    null_n = s.isna().sum()
    print(f"\n  Null count: {pct(null_n, TOTAL_ROWS)}")
    unique_vals = s.dropna().unique()
    non_bool = [v for v in unique_vals if str(v).lower() not in ['true','false','1','0','yes','no']]
    if non_bool:
        print(f"  ⚠️  NON-BOOLEAN values found: {non_bool}")

# =============================================================================
# 11. TEXT COLUMN ANALYSIS
# =============================================================================

print_section("11. TEXT / FREE-TEXT COLUMN ANALYSIS")

for col in TEXT_COLS:
    if col not in df.columns:
        continue
    s = df[col].dropna().astype(str)
    null_n = df[col].isna().sum()
    empty_n = (s.str.strip() == '').sum()
    print_subsection(f"COLUMN: {col}")
    print(f"  Null count       : {pct(null_n, TOTAL_ROWS)}")
    print(f"  Empty strings    : {pct(empty_n, TOTAL_ROWS)}")
    print(f"  Unique values    : {s.nunique():,}")
    lengths = s.str.len()
    print(f"  Length — min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}")
    if col == 'meta_data':
        json_valid = s.apply(lambda x: True if x.strip().startswith('{') or x.strip().startswith('[') else False).sum()
        print(f"  JSON-like entries: {pct(json_valid, len(s))}")

# =============================================================================
# 12. CROSS-COLUMN CONSISTENCY CHECKS
# =============================================================================

print_section("12. CROSS-COLUMN CONSISTENCY CHECKS")

# Check: closed_datetime exists but status != Closed
if 'closed_datetime' in df.columns and 'status' in df.columns:
    closed_dt_not_null = df['closed_datetime'].notna()
    status_not_closed = df['status'].str.lower().fillna('') != 'closed'
    mismatch = (closed_dt_not_null & status_not_closed).sum()
    print(f"  closed_datetime set but status != Closed    : {pct(mismatch, TOTAL_ROWS)} {'⚠️' if mismatch > 0 else '✅'}")

# Check: resolved_datetime without resolved_by_id
if 'resolved_datetime' in df.columns and 'resolved_by_id' in df.columns:
    resolved_dt_not_null = df['resolved_datetime'].notna()
    resolver_null = df['resolved_by_id'].isna()
    mismatch2 = (resolved_dt_not_null & resolver_null).sum()
    print(f"  resolved_datetime set but resolved_by_id null: {pct(mismatch2, TOTAL_ROWS)} {'⚠️' if mismatch2 > 0 else '✅'}")

# Check: requires_road_closure=True but no direction
if 'requires_road_closure' in df.columns and 'direction' in df.columns:
    rrc = df['requires_road_closure'].astype(str).str.lower() == 'true'
    no_dir = df['direction'].isna()
    mismatch3 = (rrc & no_dir).sum()
    print(f"  road_closure=True but direction is null     : {pct(mismatch3, TOTAL_ROWS)} {'⚠️' if mismatch3 > 0 else '✅'}")

# Check: breakdown event but no reason_breakdown
if 'event_type' in df.columns and 'reason_breakdown' in df.columns:
    is_breakdown = df['event_type'].astype(str).str.lower().str.contains('breakdown', na=False)
    no_reason = df['reason_breakdown'].isna()
    mismatch4 = (is_breakdown & no_reason).sum()
    print(f"  event_type=Breakdown but reason_breakdown null: {pct(mismatch4, TOTAL_ROWS)} {'⚠️' if mismatch4 > 0 else '✅'}")

# Check: Geo start vs end mismatch
if 'endlatitude' in df.columns and 'endlongitude' in df.columns:
    end_lat_null = df['endlatitude'].isna()
    end_lon_null = df['endlongitude'].isna()
    one_end_null = (end_lat_null ^ end_lon_null).sum()
    print(f"  endlatitude/endlongitude mismatch (one null): {pct(one_end_null, TOTAL_ROWS)} {'⚠️' if one_end_null > 0 else '✅'}")

# =============================================================================
# 13. DUPLICATE ROW ANALYSIS
# =============================================================================

print_section("13. DUPLICATE ROW ANALYSIS")

total_dupes = df.duplicated().sum()
print(f"  Exact duplicate rows (all columns): {pct(total_dupes, TOTAL_ROWS)}")

# Subset duplicates (key business columns)
key_cols = [c for c in ['event_type', 'latitude', 'longitude', 'start_datetime'] if c in df.columns]
if key_cols:
    subset_dupes = df.duplicated(subset=key_cols).sum()
    print(f"  Duplicate on {key_cols}: {pct(subset_dupes, TOTAL_ROWS)}")

if 'id' in df.columns:
    id_dupes = df['id'].duplicated().sum()
    print(f"  Duplicate 'id' values: {pct(id_dupes, TOTAL_ROWS)} {'⚠️  PRIMARY KEY VIOLATION' if id_dupes > 0 else '✅'}")

# =============================================================================
# 14. CORRELATION ANALYSIS (numeric columns)
# =============================================================================

print_section("14. CORRELATION MATRIX (Numeric Columns)")

corr_cols = [c for c in num_cols_all if c not in ID_COLS and df[c].dropna().shape[0] > 10]
if len(corr_cols) >= 2:
    corr_matrix = df[corr_cols].corr()
    print(corr_matrix.round(3).to_string())
    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix.csv'))

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('CORRELATION HEATMAP — Numeric Columns', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig('correlation_heatmap.png')

    # High correlations
    print("\n  HIGH CORRELATIONS (|r| > 0.7):")
    found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                print(f"    {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {r:.3f}  ⚠️  Potential multicollinearity")
                found = True
    if not found:
        print("    None found above 0.7 threshold ✅")
else:
    print("  Not enough numeric columns for correlation analysis.")

# =============================================================================
# 15. ML READINESS REPORT
# =============================================================================

print_section("15. ML READINESS REPORT & RECOMMENDATIONS")

print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                    ML READINESS SUMMARY                             │
  └─────────────────────────────────────────────────────────────────────┘
""")

# Columns by null severity
severe_null = null_df[null_df['Null_Pct'] > 70]['Column'].tolist()
moderate_null = null_df[(null_df['Null_Pct'] > 30) & (null_df['Null_Pct'] <= 70)]['Column'].tolist()
low_null = null_df[(null_df['Null_Pct'] > 0) & (null_df['Null_Pct'] <= 30)]['Column'].tolist()
complete = null_df[null_df['Null_Pct'] == 0]['Column'].tolist()

print(f"  ✅  COMPLETE columns (0% null)          : {complete}")
print(f"\n  🟡  LOW NULL columns (1-30% null)       : {low_null}")
print(f"\n  🟠  MODERATE NULL columns (30-70% null) : {moderate_null}")
print(f"\n  🔴  SEVERE NULL columns (>70% null)     : {severe_null}")

print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │              DATA CLEANING RECOMMENDATIONS                          │
  └─────────────────────────────────────────────────────────────────────┘

  STEP 1 — DROP OR REVIEW SEVERE NULL COLUMNS (>70% null)
    → Consider dropping them entirely, or use them only as binary flags
      (e.g., is_<column>_present = 1/0)

  STEP 2 — HANDLE MODERATE NULL COLUMNS (30-70% null)
    → Categorical: fill with 'Unknown' or most-frequent value
    → Numeric: fill with median (not mean, due to skew/outliers)
    → Geo: fill from nearest neighbor or drop row

  STEP 3 — LOW NULL COLUMNS (1-30% null)
    → Numeric: median imputation or model-based imputation (KNN, MICE)
    → Categorical: mode imputation
    → Datetime: flag as 'missing' and possibly impute from related cols

  STEP 4 — GEO COORDINATE CLEANING
    → Drop rows where (lat==0 & lon==0) — NULL ISLAND
    → Drop rows where only one of lat/lon is present (mismatched pair)
    → Validate coordinate ranges for your operating region

  STEP 5 — DATETIME CLEANING
    → Drop rows with start_datetime > end_datetime (temporal paradox)
    → Convert all datetime to UTC if timezone-mixed
    → Extract features: hour, day_of_week, month, is_weekend, duration_mins

  STEP 6 — CATEGORICAL ENCODING
    → Low cardinality (≤10 unique): One-Hot Encoding
    → Medium cardinality (10-50): Target Encoding or Binary Encoding
    → High cardinality (>50): Embedding or Frequency Encoding
    → Boolean cols: ensure True/False → 1/0

  STEP 7 — FEATURE ENGINEERING IDEAS
    → duration_minutes = end_datetime - start_datetime
    → resolution_time = resolved_datetime - start_datetime
    → response_time = assigned_to_police_id fill time
    → geo_distance = haversine(start_lat/lon, end_lat/lon)
    → hour_of_day, day_of_week, is_weekend, month from start_datetime
    → is_resolved = (status == 'Resolved') → binary target
    → has_road_closure = requires_road_closure → binary feature

  STEP 8 — OUTLIER TREATMENT
    → Use IQR fencing for age_of_truck, veh_no, coordinates
    → Cap/floor at 1st and 99th percentile for skewed distributions
    → Log-transform highly skewed numeric columns (skewness > 2)

  STEP 9 — COLUMN DROPS FOR ML
    → Drop free-text columns (description, comment, meta_data) unless NLP
    → Drop ID columns (id, client_id, etc.) — no predictive value
    → Drop constant-value columns
    → Drop map_file, route_path (structural, not statistical)

  STEP 10 — TARGET VARIABLE
    → If predicting event outcome: use 'status' or derive 'is_resolved'
    → If predicting duration: use computed 'duration_minutes'
    → If predicting closure: use 'requires_road_closure'
    → Class imbalance check is critical for classification targets
""")

# =============================================================================
# 16. FINAL SUMMARY TABLE
# =============================================================================

print_section("16. FINAL COLUMN-BY-COLUMN SUMMARY TABLE")

summary_rows = []
for col in df.columns:
    s = df[col]
    null_pct = round(s.isna().sum() / TOTAL_ROWS * 100, 1)
    unique_n = s.nunique(dropna=True)
    dtype = str(s.dtype)
    if col in BOOL_COLS: ctype = 'Boolean'
    elif col in ID_COLS: ctype = 'ID/Key'
    elif col in DATETIME_COLS: ctype = 'DateTime'
    elif col in GEO_COLS: ctype = 'Geo'
    elif col in CATEGORICAL_COLS: ctype = 'Categorical'
    elif col in TEXT_COLS: ctype = 'Text'
    elif col in NUMERIC_COLS: ctype = 'Numeric'
    else: ctype = 'Unknown'
    rec = ''
    if null_pct > 70: rec = 'DROP or flag-encode'
    elif null_pct > 30: rec = 'Impute (median/mode/Unknown)'
    elif null_pct > 0: rec = 'Impute (KNN/median/mode)'
    else: rec = 'Ready / Encode'
    if col in ID_COLS: rec = 'DROP for ML'
    if col in TEXT_COLS: rec = 'DROP or NLP'
    summary_rows.append({
        'Column': col, 'ColType': ctype, 'Dtype': dtype,
        'Null_Pct': null_pct, 'Unique': unique_n, 'Recommendation': rec
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'final_column_summary.csv'), index=False)

# =============================================================================
# DONE
# =============================================================================

print_section("EDA COMPLETE")
print(f"""
  All outputs saved to: ./{OUTPUT_DIR}/
  ┌─────────────────────────────────────────────────┐
  │  Files generated:                               │
  │  • schema_overview.csv                          │
  │  • null_analysis.csv                            │
  │  • numeric_stats.csv                            │
  │  • correlation_matrix.csv                       │
  │  • final_column_summary.csv                     │
  │  • null_heatmap.png                             │
  │  • null_percentage_bar.png                      │
  │  • categorical_distributions.png               │
  │  • numeric_boxplots.png                         │
  │  • numeric_histograms.png                       │
  │  • geo_scatter_*.png                            │
  │  • event_duration_histogram.png                 │
  │  • events_per_day.png                           │
  │  • correlation_heatmap.png                      │
  └─────────────────────────────────────────────────┘
""")