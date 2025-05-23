import numpy as np
import pandas as pd

def calc_daily_spread_return(df_day, portfolio_size=200, toprank_weight_ratio=2):
    assert df_day['Rank'].min() == 0
    assert df_day['Rank'].max() == len(df_day) - 1
    weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
    purchase = (df_day.sort_values(by='Rank')['True_Target'][:portfolio_size] * weights).sum() / weights.mean()
    short = (df_day.sort_values(by='Rank', ascending=False)['True_Target'][:portfolio_size] * weights).sum() / weights.mean()
    return purchase - short

def hit_rate_at_k(df, k, mode="top"):
    if mode == "top":
        top_k_actual = set(df.nsmallest(k, 'Actual_Rank')['SecuritiesCode'])
        top_k_pred = set(df.nsmallest(k, 'Rank')['SecuritiesCode'])
    elif mode == "bottom":
        top_k_actual = set(df.nlargest(k, 'Actual_Rank')['SecuritiesCode'])
        top_k_pred = set(df.nlargest(k, 'Rank')['SecuritiesCode'])
    else:
        raise ValueError("mode should be 'top' or 'bottom'")
    return len(top_k_actual & top_k_pred) / k

def daily_metrics_df(df):
    metrics = []
    for date, group in df.groupby('Date'):
        if len(group) < 400:
            continue
        group = group.copy()
        daily_return = calc_daily_spread_return(group, 200, 2)
        group['Actual_Rank'] = group.groupby('Date')['True_Target'].rank(ascending=False, method='first') - 1
        hit_top = hit_rate_at_k(group, 200, "top")
        hit_bottom = hit_rate_at_k(group, 200, "bottom")
        metrics.append({
            'Date': date,
            'Daily_Spread_Return': daily_return,
            'HitRate@Top200': hit_top,
            'HitRate@Bottom200': hit_bottom,
            'Num_Stocks': len(group)
        })
    return pd.DataFrame(metrics).sort_values('Date')
