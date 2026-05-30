import pandas as pd
import numpy as np
import pygeohash as pgh
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# PHASE 1: PRECISION FEATURE ENGINEERING
# ==========================================
def engineer_features(df):
    print("-> Extracting Micro-Patterns...")
    df = df.copy()
    
    # 1. Spatial Tessellation
    df['latitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) else 0)
    df['longitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) else 0)
    df['geo_prefix_4'] = df['geohash'].astype(str).str[:4]
    df['geo_prefix_5'] = df['geohash'].astype(str).str[:5]
    
    # 2. Temporal Dynamics
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour.fillna(0).astype(int)
    
    # The Modulo Trick: Extracting Weekly Seasonality
    df['day'] = df['day'].astype(int)
    df['day_of_week'] = df['day'] % 7  
    
    # Cyclical Time Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # 3. Clean Categorical types strictly for CatBoost
    cat_cols = ['geohash', 'geo_prefix_4', 'geo_prefix_5', 'RoadType', 'Weather', 'LargeVehicles', 'Landmarks']
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('missing')
    
    # 4. Physics interactions
    df['NumberofLanes'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())
    
    return df

# ==========================================
# PHASE 2: 10-FOLD ENSEMBLE EXECUTION
# ==========================================
def main():
    print("Loading datasets...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    test_indices = test['Index'].values
    
    # Engineer Features
    train_df = engineer_features(train)
    test_df = engineer_features(test)
    
    # Define features
    features = [
        'latitude', 'longitude', 'geohash', 'geo_prefix_4', 'geo_prefix_5',
        'hour', 'hour_sin', 'hour_cos', 'day', 'day_of_week', 
        'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 
        'Temperature', 'Weather'
    ]
    
    # Explicitly tell CatBoost which features are categorical (It will build crosses automatically!)
    cat_features = [
        'geohash', 'geo_prefix_4', 'geo_prefix_5', 
        'hour', 'day_of_week', 'RoadType', 
        'LargeVehicles', 'Landmarks', 'Weather'
    ]
    
    X = train_df[features]
    y = train_df['demand']
    X_test = test_df[features]
    
    print(f"-> Initiating 10-Fold CatBoost Ensemble...")
    
    # Initialize K-Fold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    test_predictions = np.zeros(len(X_test))
    oof_r2_scores = []
    
    # The 10-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n--- Training Fold {fold + 1} / 10 ---")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Create CatBoost Pools
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)
        
        # The Hyper-Tuned CatBoost Regressor
        model = CatBoostRegressor(
            iterations=3000,           # Massive tree count, stopped early by od_wait
            learning_rate=0.04,        # Optimal learning rate for tabular data
            depth=8,                   # Deep enough for complex combinations
            loss_function='RMSE',
            eval_metric='R2',          # We can track the hackathon metric directly!
            random_seed=42 + fold,     # Different seed per fold for maximum variance reduction
            od_type='Iter',
            od_wait=200,               # Stop if R2 doesn't improve for 200 trees
            verbose=500                # Print updates every 500 trees
        )
        
        # Train on 90%, Validate on 10%
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        # Track Validation R2
        best_score = model.get_best_score()['validation']['R2']
        oof_r2_scores.append(best_score)
        print(f"Fold {fold + 1} R2 Score: {best_score:.5f}")
        
        # Predict on the unseen test set and accumulate
        # We divide by 10 so by the end of the loop, we have the exact average.
        test_predictions += model.predict(test_pool) / kf.n_splits

    # Print out our local Cross-Validation Score
    print(f"\n======================================")
    print(f"✅ Local 10-Fold CV Average R2: {np.mean(oof_r2_scores):.5f}")
    print(f"======================================")

    # Post-processing: Demand physically cannot be less than 0
    test_predictions = np.clip(test_predictions, 0, None)
    
    # Format the Submission
    print("-> Saving Grandmaster Ensembled Predictions...")
    submission = pd.DataFrame({
        'Index': test_indices,
        'demand': test_predictions
    })
    
    submission.to_csv('catboost_ensemble_submission.csv', index=False)
    print("✅ Success! 'catboost_ensemble_submission.csv' is ready for upload.")

if __name__ == '__main__':
    main()