import pandas as pd
df = pd.read_csv("track 2.csv")
df['start_datetime'] = pd.to_datetime(df['start_datetime'], format='mixed', utc=True)
df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], format='mixed', utc=True)
df['raw_dur'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60
print(df.groupby('event_cause')['raw_dur'].describe(percentiles=[.25,.5,.75,.90,.95]))