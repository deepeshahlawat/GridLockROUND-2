import pandas as pd
import numpy as np

def run_diagnostic():
    print("="*60)
    print(" 🕵️‍♂️ SENIOR DATA ANALYST DIAGNOSTIC REPORT ")
    print("="*60)

    # Load Data
    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"\n[1] DATASET SHAPES")
    print(f"Train dataset: {train.shape[0]} rows, {train.shape[1]} columns")
    print(f"Test dataset:  {test.shape[0]} rows, {test.shape[1]} columns")

    print(f"\n[2] MISSING VALUES OVERVIEW")
    train_missing = train.isnull().sum()
    test_missing = test.isnull().sum()
    for col in train.columns:
        train_pct = (train_missing[col] / len(train)) * 100
        if col in test.columns:
            test_pct = (test_missing[col] / len(test)) * 100
            print(f" - {col.ljust(15)}: Train missing {train_pct:.2f}% | Test missing {test_pct:.2f}%")
        else:
            print(f" - {col.ljust(15)}: Train missing {train_pct:.2f}% | (Not in Test)")

    print(f"\n[3] TARGET VARIABLE ('demand') DEEP DIVE")
    demand = train['demand'].dropna()
    print(f" - Mean:   {demand.mean():.6f}")
    print(f" - Median: {demand.median():.6f}")
    print(f" - Min:    {demand.min():.6f}")
    print(f" - Max:    {demand.max():.6f}")
    print(f" - Zero Count: {(demand == 0).sum()} rows ({((demand == 0).sum() / len(demand))*100:.2f}%)")
    print(f" - Percentiles: 1%={demand.quantile(0.01):.4f}, 5%={demand.quantile(0.05):.4f}, 95%={demand.quantile(0.95):.4f}, 99%={demand.quantile(0.99):.4f}")

    print(f"\n[4] TEMPORAL SPLIT ANALYSIS ('day' & 'timestamp')")
    train_days = set(train['day'].dropna().unique())
    test_days = set(test['day'].dropna().unique())
    print(f" - Train 'day' range: {min(train_days)} to {max(train_days)} (Total unique days: {len(train_days)})")
    print(f" - Test 'day' range:  {min(test_days)} to {max(test_days)} (Total unique days: {len(test_days)})")
    
    shared_days = train_days.intersection(test_days)
    if len(shared_days) == 0:
        print(" -> SPLIT TYPE: Strict Future Time-Series (Test days are completely after Train days)")
    elif train_days == test_days:
        print(" -> SPLIT TYPE: Shuffled / Missing Hour Interpolation (Train and Test share the exact same days)")
    else:
        print(" -> SPLIT TYPE: Partial Overlap (Train and Test share some days, but not all)")

    print(f" - 'timestamp' Sample (Train): {train['timestamp'].dropna().head(3).tolist()}")
    print(f" - 'timestamp' Sample (Test):  {test['timestamp'].dropna().head(3).tolist()}")

    print(f"\n[5] SPATIAL DRIFT ANALYSIS ('geohash')")
    train_geo = set(train['geohash'].dropna().astype(str).unique())
    test_geo = set(test['geohash'].dropna().astype(str).unique())
    print(f" - Unique Train Geohashes: {len(train_geo)}")
    print(f" - Unique Test Geohashes:  {len(test_geo)}")
    
    unseen_geohashes = test_geo - train_geo
    print(f" - Unseen Geohashes in Test: {len(unseen_geohashes)} ({len(unseen_geohashes)/len(test_geo)*100:.2f}% of Test Set)")
    if len(unseen_geohashes) > 0:
        print("   -> WARNING: The model will have to predict locations it has never seen before!")

    print(f"\n[6] CATEGORICAL DRIFT ('RoadType', 'Weather', etc.)")
    cats = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for cat in cats:
        train_cat = set(train[cat].dropna().astype(str).unique())
        test_cat = set(test[cat].dropna().astype(str).unique())
        unseen_cat = test_cat - train_cat
        if len(unseen_cat) > 0:
            print(f" - {cat}: Test contains UNSEEN categories: {unseen_cat}")
        else:
            print(f" - {cat}: All test categories exist in train data. ({len(train_cat)} unique classes)")

    print(f"\n[7] NUMERIC OUTLIERS ('Temperature', 'NumberofLanes')")
    print(f" - Temperature Train range: {train['Temperature'].min()} to {train['Temperature'].max()}")
    print(f" - Temperature Test range:  {test['Temperature'].min()} to {test['Temperature'].max()}")
    print(f" - Lanes Train range: {train['NumberofLanes'].min()} to {train['NumberofLanes'].max()}")
    print(f" - Lanes Test range:  {test['NumberofLanes'].min()} to {test['NumberofLanes'].max()}")
    
    print("\n" + "="*60)
    print(" DIAGNOSTIC COMPLETE. PLEASE PASTE THIS OUTPUT. ")
    print("="*60)

if __name__ == '__main__':
    run_diagnostic()