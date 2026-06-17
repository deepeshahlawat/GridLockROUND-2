import pandas as pd
df = pd.read_csv("ml_ready.csv")
print(df["Target_Duration_Mins"].value_counts(bins=10).sort_index())
print(df.groupby("event_cause_vehicle_breakdown")["Target_Duration_Mins"].describe())