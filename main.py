import pandas as pd
import numpy as np
import pygeohash as pgh
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

def engineer_features(df):
    print("-> Extracting Pure Spatio-Temporal Features...")
    df = df.copy()
    
    # 1. Spatial Tessellation
    df['latitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) else 0)
    df['longitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) else 0)
    
    # Prefix helps group nearby regions
    df['geo_prefix_5'] = df['geohash'].astype(str).str[:5]
    
    # 2. Perfect Time Parsing (The "H:M" Fix)
    # Split "H:M" into integers
    time_split = df['timestamp'].astype(str).str.split(':', expand=True)
    df['hour'] = time_split[0].astype(int)
    df['minute'] = time_split[1].astype(int)
    
    # Continuous Time Flow (Total minutes elapsed in the day)
    df['minutes_from_midnight'] = df['hour'] * 60 + df['minute']
    
    # Cyclical Time (Perfect circle mapping for a 24hr period)
    df['time_sin'] = np.sin(2 * np.pi * df['minutes_from_midnight'] / 1440.0)
    df['time_cos'] = np.cos(2 * np.pi * df['minutes_from_midnight'] / 1440.0)
    
    # Binary flag for the specific day (Since it's only Day 48 or 49)
    df['is_day_49'] = (df['day'] == 49).astype(int)
    
    # 3. Handle Missing Values / Formatting for CatBoost
    cat_cols = ['geohash', 'geo_prefix_5', 'RoadType', 'Weather']
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('missing')
        
    df['NumberofLanes'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())
    df['LargeVehicles'] = df['LargeVehicles'].map({'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0)
    df['Landmarks'] = df['Landmarks'].map({'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0)
    
    # Physics Capacity
    df['capacity'] = df['NumberofLanes'] * df['LargeVehicles'].apply(lambda x: 0.7 if x == 1 else 1.0)
    
    return df

def main():
    print("Loading datasets...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test_indices = test['Index'].values
    
    # Process Features
    train_df = engineer_features(train)
    test_df = engineer_features(test)
    
    # Feature Selection
    features = [
        'latitude', 'longitude', 'geohash', 'geo_prefix_5',
        'is_day_49', 'minutes_from_midnight', 'time_sin', 'time_cos', 
        'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 
        'Temperature', 'Weather', 'capacity'
    ]
    
    # CatBoost Native Categoricals
    cat_features = ['geohash', 'geo_prefix_5', 'RoadType', 'Weather']
    
    X = train_df[features]
    y = train_df['demand']
    X_test = test_df[features]
    
    print(f"-> Initiating 10-Fold CatBoost Ensemble (Optimized for [0, 1] Target)...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    test_predictions = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Training Fold {fold + 1} / 10 ---")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)
        
        # Optimized Model
        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.04, 
            depth=8,
            loss_function='RMSE',
            eval_metric='R2',
            random_seed=42 + fold,
            od_type='Iter',
            od_wait=200,
            verbose=500
        )
        
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        # Accumulate Predictions
        test_predictions += model.predict(test_pool) / kf.n_splits

    # ==========================================
    # 🚨 THE CRITICAL FIX: TARGET BOUNDING 🚨
    # ==========================================
    # Based on the EDA, demand is strictly between 0 and 1.
    print("\n-> Applying strict [0.0, 1.0] mathematical bounding...")
    test_predictions = np.clip(test_predictions, 0.0, 1.0)
    
    print("-> Saving Predictions...")
    submission = pd.DataFrame({
        'Index': test_indices,
        'demand': test_predictions
    })
    
    submission.to_csv('final_truth_submission.csv', index=False)
    print("✅ Success! 'final_truth_submission.csv' is ready for upload.")

if __name__ == '__main__':
    main()