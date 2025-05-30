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

def compute_meta_df(data):
        models = ["LGB_Pred", "LSTM_Pred", "Ridge_Pred", "XGB_Pred"]
        all_model_dfs = []
        all_model_dailies = []

        # 1) DataFrame and daily metrics for each model
        for m in models:
            dfm = data[['RowId', 'Date', 'SecuritiesCode', 'Target']].copy()
            dfm['True_Target'] = dfm['Target']
            dfm['Predicted_Target'] = data[m]
            dfm['Rank'] = (
                dfm.groupby('Date')['Predicted_Target']
                .rank(ascending=False, method='first')
                - 1
            )
            dailym = daily_metrics_df(dfm)
            all_model_dfs.append(dfm)
            all_model_dailies.append(dailym)

        # 2) Rows of the model with the best Sharpe ratio per day
        meta_chunks = []
        meta_daily = []
        for d in sorted(data['Date'].unique()):
            best_sharpe = -1e9
            best_chunk = None
            best_daily = None
            for dfm, dailym in zip(all_model_dfs, all_model_dailies):
                day = dailym[dailym['Date'] == d]
                if day.empty:
                    continue
                sr = day['Daily_Spread_Return'].iloc[0]
                std = dailym['Daily_Spread_Return'].std()
                if std > 0 and sr / std > best_sharpe:
                    best_sharpe = sr / std
                    best_chunk = dfm[dfm['Date'] == d]
                    best_daily = day
            if best_chunk is not None:
                meta_chunks.append(best_chunk)
                meta_daily.append(best_daily)

        meta_df = pd.concat(meta_chunks, ignore_index=True)
        meta_daily_df = pd.concat(meta_daily, ignore_index=True)
        return meta_df, meta_daily_df