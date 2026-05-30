import pandas as pd
import numpy as np
import pygeohash as pgh
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

def engineer_features(df):
    print("-> Forging Ultimate Spatio-Temporal Matrix...")
    df = df.copy()
    
    # 1. Base Spatial
    df['latitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[0] if pd.notnull(x) else 0)
    df['longitude'] = df['geohash'].apply(lambda x: pgh.decode(x)[1] if pd.notnull(x) else 0)
    df['geo_prefix_5'] = df['geohash'].astype(str).str[:5]
    
    # 2. Perfect Time Parsing
    time_split = df['timestamp'].astype(str).str.split(':', expand=True)
    df['hour'] = time_split[0].astype(int)
    df['minute'] = time_split[1].astype(int)
    df['minutes_from_midnight'] = df['hour'] * 60 + df['minute']
    
    df['time_sin'] = np.sin(2 * np.pi * df['minutes_from_midnight'] / 1440.0)
    df['time_cos'] = np.cos(2 * np.pi * df['minutes_from_midnight'] / 1440.0)
    
    df['is_day_49'] = (df['day'] == 49).astype(int)
    
    # 3. New Interaction Features (Combining Environmental factors)
    df['Road_Weather'] = df['RoadType'].astype(str) + "_" + df['Weather'].astype(str)
    
    # 4. Handle Missing Values
    df['NumberofLanes'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())
    df['LargeVehicles'] = df['LargeVehicles'].map({'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0)
    df['Landmarks'] = df['Landmarks'].map({'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0)
    df['capacity'] = df['NumberofLanes'] * df['LargeVehicles'].apply(lambda x: 0.7 if x == 1 else 1.0)
    
    # 5. Pandas 'Category' Conversion (Mandatory for LGBM/XGB)
    cat_cols = ['geohash', 'geo_prefix_5', 'RoadType', 'Weather', 'Road_Weather']
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('missing').astype('category')
        
    return df

def main():
    print("Loading datasets...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test_indices = test['Index'].values
    
    # Process Features
    train_df = engineer_features(train)
    test_df = engineer_features(test)
    
    features = [
        'latitude', 'longitude', 'geohash', 'geo_prefix_5', 'Road_Weather',
        'is_day_49', 'minutes_from_midnight', 'time_sin', 'time_cos', 
        'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 
        'Temperature', 'Weather', 'capacity'
    ]
    
    cat_features = ['geohash', 'geo_prefix_5', 'RoadType', 'Weather', 'Road_Weather']
    
    X = train_df[features]
    y = train_df['demand']
    X_test = test_df[features]
    
    print(f"-> Initiating 'Holy Trinity' Blended Ensemble (5-Fold)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cb_preds = np.zeros(len(X_test))
    lgb_preds = np.zeros(len(X_test))
    xgb_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n================= FOLD {fold + 1} / 5 =================")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # -----------------------------------------------------
        # 1. CatBoost
        # -----------------------------------------------------
        print("-> Training CatBoost...")
        cb_model = CatBoostRegressor(
            iterations=2500, learning_rate=0.04, depth=8,
            loss_function='RMSE', random_seed=42 + fold, verbose=False
        )
        # CatBoost needs pure lists of categorical feature names
        cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), 
                     cat_features=cat_features, early_stopping_rounds=150)
        cb_preds += cb_model.predict(X_test) / kf.n_splits
        
        # -----------------------------------------------------
        # 2. LightGBM
        # -----------------------------------------------------
        print("-> Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2500, learning_rate=0.03, max_depth=9, num_leaves=63,
            objective='regression', metric='rmse', random_state=42 + fold, n_jobs=-1
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      categorical_feature=cat_features,
                      callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)])
        lgb_preds += lgb_model.predict(X_test) / kf.n_splits
        
        # -----------------------------------------------------
        # 3. XGBoost
        # -----------------------------------------------------
        print("-> Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=2500, learning_rate=0.03, max_depth=8,
            tree_method='hist', enable_categorical=True, 
            objective='reg:squarederror', random_state=42 + fold, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      verbose=False)
        xgb_preds += xgb_model.predict(X_test) / kf.n_splits

    # ==========================================
    # ⚖️ THE BLEND ⚖️
    # ==========================================
    print("\n-> Blending Models (CatBoost 40% | LightGBM 30% | XGBoost 30%)")
    final_predictions = (cb_preds * 0.40) + (lgb_preds * 0.30) + (xgb_preds * 0.30)

    # 🚨 STRICT TARGET BOUNDING 🚨
    print("-> Applying strict [0.0, 1.0] mathematical bounding...")
    final_predictions = np.clip(final_predictions, 0.0, 1.0)
    
    print("-> Saving Predictions...")
    submission = pd.DataFrame({
        'Index': test_indices,
        'demand': final_predictions
    })
    
    submission.to_csv('holy_trinity_submission.csv', index=False)
    print("✅ Success! 'holy_trinity_submission.csv' is ready for upload.")

if __name__ == '__main__':
    main()