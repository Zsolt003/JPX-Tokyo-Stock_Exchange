import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

start_date = datetime(2017, 1, 4)
n_days = 30
n_securities = 400
data = []

for sec_code in range(1301, 1301 + n_securities):
    base_open = 2734.0 + (sec_code - 1301) * 20
    trend_factor = 1 if sec_code % 2 == 1 else -1  # Odd: up, even: down
    for i in range(n_days):
        date = start_date + timedelta(days=i)
        open_price = base_open + i * 5 * trend_factor + (sec_code - 1301) * 0.1
        high_price = open_price + 20 + np.random.uniform(0, 5)
        low_price = open_price - 5 - np.random.uniform(0, 5)
        close_price = open_price + trend_factor * 3 + np.random.uniform(-2, 2)
        volume = 31400 + i * 500 + (sec_code - 1301) * 1000
        row_id = f"{date.strftime('%Y%m%d')}_{sec_code}"
        target = 0.0  # Placeholder
        data.append([row_id, date.strftime('%Y-%m-%d'), sec_code, open_price, high_price, low_price, close_price, volume, target])

df = pd.DataFrame(data, columns=["RowId", "Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume", "Target"])

# Compute Target
for sec_code in df['SecuritiesCode'].unique():
    sec_df = df[df['SecuritiesCode'] == sec_code].sort_values('Date').reset_index(drop=True)
    for i in range(len(sec_df) - 2):
        close_r1 = sec_df.loc[i + 1, 'Close']
        close_r2 = sec_df.loc[i + 2, 'Close']
        target = (close_r2 - close_r1) / close_r1
        df.loc[(df['SecuritiesCode'] == sec_code) & (df['Date'] == sec_df.loc[i, 'Date']), 'Target'] = target
    df.loc[(df['SecuritiesCode'] == sec_code) & (df['Date'].isin(sec_df['Date'].iloc[-2:])), 'Target'] = 0.0

# Ensure no missing or infinite values
df = df.replace([np.inf, -np.inf], np.nan)
if df.isnull().any().any():
    raise ValueError("Generated CSV contains missing values")

df.to_csv("input_data_12000.csv", index=False)
print("Generated input_data_12000.csv with", len(df), "rows")