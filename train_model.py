import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# 1. Load the weaponized dataset
df = pd.read_csv('ml_ready.csv')

# 2. Separate Features (X) and Target (y)
y = df['Target_Duration_Mins']

# DROP the geospatial/meta columns so the model only learns from the incident profile
X = df.drop(columns=['Target_Duration_Mins', 'latitude', 'longitude', 'police_station'])

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the XGBoost Regressor
# We use standard hyperparameters for speed. 
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate and flex
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"✅ Model Trained! Mean Absolute Error: {mae:.2f} minutes")

# Print Top 5 Feature Importances (Judges love this)
importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
print("\n🔥 Top 5 Causes of Delay:")
print(importances.sort_values(by='Importance', ascending=False).head(5))

# 6. Save the brain
joblib.dump(model, 'xgb_clearance_model.pkl')
print("\n💾 Model saved as 'xgb_clearance_model.pkl'")