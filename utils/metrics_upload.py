import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
import os
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from .metrics import daily_metrics_df, compute_meta_df

warnings.filterwarnings("ignore", category=RuntimeWarning)

def _load_all_models():
    """Betölti a négy modellt, és visszaadja őket."""
    try:
        # Define models directory
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        print("Models directory:", models_dir)  # Debug

        # Ridge
        ridge_path = os.path.join(models_dir, "linreg_model.pkl")
        if not os.path.exists(ridge_path):
            raise FileNotFoundError(f"Ridge model not found at {ridge_path}")
        ridge_model = joblib.load(ridge_path)
        if not isinstance(ridge_model, Ridge):
            raise ValueError(f"Ridge model is {type(ridge_model)}, expected Ridge")
        print("Ridge model type:", type(ridge_model))  # Debug

        # XGBoost
        xgb_path = os.path.join(models_dir, "xgb_model.json")
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(xgb_path)
        print("XGBoost model type:", type(xgb_booster))  # Debug

        # LightGBM
        lgb_path = os.path.join(models_dir, "lgb_model.txt")
        if not os.path.exists(lgb_path):
            raise FileNotFoundError(f"LightGBM model not found at {lgb_path}")
        print("LightGBM file exists:", os.path.exists(lgb_path))  # Debug
        lgb_booster = lgb.Booster(model_file=lgb_path)
        print("LightGBM model type:", type(lgb_booster))  # Debug

        # LSTM
        lstm_path = os.path.join(models_dir, "lstm_model.h5")
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"LSTM model not found at {lstm_path}")
        lstm_model = load_model(lstm_path, compile=False)
        print("LSTM model type:", type(lstm_model))  # Debug

        return ridge_model, xgb_booster, lgb_booster, lstm_model
    except Exception as e:
        raise Exception(f"Failed to load model: {type(e)} - {str(e)}")

def _validate_csv(df):
    errors = []
    req = ["RowId", "Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume", "Target"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]

    if df[req].isnull().any().any():
        errors.append("NaNs in required columns")

    if len(df) < 500:
        errors.append("Need ≥500 rows")

    df['Date'] = pd.to_datetime(df['Date'])
    for code, g in df.groupby('SecuritiesCode'):
        if len(g) < 30:
            errors.append(f"{code}: fewer than 30 rows")
        ds = g.sort_values('Date')['Date']
        if (ds.diff().dropna() != timedelta(days=1)).any():
            errors.append(f"{code}: dates not consecutive")
        gg = g.sort_values('Date').reset_index(drop=True)
        if len(gg) >= 3:
            exp = (gg.loc[2, 'Close'] - gg.loc[1, 'Close'])/gg.loc[1, 'Close']
            if not np.isclose(gg.loc[0, 'Target'], exp, atol=1e-6):
                errors.append(f"{code}: first Target should be {exp:.6f}")

    return errors

def _generate_features(df, feature_cols):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['SecuritiesCode', 'Date']).reset_index(drop=True)

    df['Daily_Range']    = df['Close'] - df['Open']
    df['Relative_Range'] = df['Daily_Range'] / df['Open']
    df['Momentum_20']    = df.groupby('SecuritiesCode')['Close'].transform(lambda x: x - x.shift(20))

    def _rsi(x, w=14):
        d = x.diff()
        up = d.clip(lower=0)
        dn = -d.clip(upper=0)
        ru = up.rolling(w).mean()
        rd = dn.rolling(w).mean()
        return 100 - (100/(1 + ru/rd))
    df['RSI_14'] = df.groupby('SecuritiesCode')['Close'].transform(_rsi)

    def _macd(x, f=12, s=26):
        return x.ewm(span=f).mean() - x.ewm(span=s).mean()
    df['MACD'] = df.groupby('SecuritiesCode')['Close'].transform(_macd)

    df['DayOfWeek']    = df['Date'].dt.dayofweek
    df['Month']        = df['Date'].dt.month

    df['LogReturn']     = df.groupby('SecuritiesCode')['Close'].transform(lambda x: np.log(x/x.shift(1)))
    df['Volatility_10'] = df.groupby('SecuritiesCode')['LogReturn'].transform(lambda x: x.rolling(10).std())
    df['SharpeLike_10'] = df.groupby('SecuritiesCode')['LogReturn']\
                              .transform(lambda x: x.rolling(10).mean()/(x.rolling(10).std()+1e-6))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df[['RowId', 'Date', 'SecuritiesCode', 'Target'] + feature_cols]

def _compute_all_sharpe(df, window_size, feature_cols):
    ridge_m, xgb_m, lgb_m, lstm_m = _load_all_models()

    # Reset index to ensure alignment
    dfp = df.copy().reset_index(drop=True)
    dfp['True_Target'] = dfp['Target']
    print("dfp shape:", dfp.shape)  # Debug

    # Initialize results DataFrame
    results_df = dfp[['Date', 'SecuritiesCode', 'True_Target']].copy()

    # Ridge
    dfp['Ridge_Pred'] = ridge_m.predict(dfp[feature_cols])
    print("Ridge_Pred NaNs:", dfp['Ridge_Pred'].isna().sum())  # Debug
    sr_ridge = _sharpe_for(dfp, 'Ridge_Pred')
    results_df['Ridge_Pred'] = dfp['Ridge_Pred']
    results_df['Ridge_Rank'] = dfp.groupby('Date')['Ridge_Pred'].rank(ascending=False, method='first') - 1

    # XGBoost
    xg_input = dfp[feature_cols + ['Ridge_Pred']].copy()
    xg_input.rename(columns={'Ridge_Pred': 'Linear_Pred'}, inplace=True)
    dfp['XGB_Pred'] = xgb_m.predict(xgb.DMatrix(xg_input))
    print("XGB_Pred NaNs:", dfp['XGB_Pred'].isna().sum())  # Debug
    sr_xgb = _sharpe_for(dfp, 'XGB_Pred')
    results_df['XGB_Pred'] = dfp['XGB_Pred']
    results_df['XGB_Rank'] = dfp.groupby('Date')['XGB_Pred'].rank(ascending=False, method='first') - 1

    # LightGBM
    lg_input = dfp[feature_cols + ['Ridge_Pred', 'XGB_Pred']].copy()
    lg_input.rename(columns={'Ridge_Pred': 'Linear_Pred'}, inplace=True)
    dfp['LGBM_Pred'] = lgb_m.predict(lg_input)
    dfp['LGB_Pred'] = dfp['LGBM_Pred']  # For compute_meta_df
    print("LGBM_Pred NaNs:", dfp['LGBM_Pred'].isna().sum())  # Debug
    sr_lgb = _sharpe_for(dfp, 'LGBM_Pred')
    results_df['LGBM_Pred'] = dfp['LGBM_Pred']
    results_df['LGBM_Rank'] = dfp.groupby('Date')['LGBM_Pred'].rank(ascending=False, method='first') - 1

    # Drop NaNs before LSTM
    dfp.dropna(subset=['Ridge_Pred', 'XGB_Pred', 'LGBM_Pred'], inplace=True)
    dfp.reset_index(drop=True, inplace=True)
    print("dfp shape after dropna:", dfp.shape)  # Debug

    # LSTM
    scaler = MinMaxScaler()
    lstm_input = dfp[feature_cols + ['Ridge_Pred', 'XGB_Pred', 'LGBM_Pred']].copy()
    lstm_input.rename(columns={'Ridge_Pred': 'Linear_Pred'}, inplace=True)
    print("lstm_input shape:", lstm_input.shape)  # Debug
    Xt = scaler.fit_transform(lstm_input)
    print("Xt shape:", Xt.shape)  # Debug
    all_preds = np.full(len(dfp), np.nan)
    pred_count = 0  # Debug

    for code, sub in dfp.groupby('SecuritiesCode'):
        idx = sub.index.values
        print(f"SecuritiesCode {code} idx range:", idx.min(), idx.max())  # Debug
        if idx.max() >= Xt.shape[0]:
            print(f"Warning: idx {idx.max()} exceeds Xt rows {Xt.shape[0]} for {code}")
            continue
        seq = Xt[idx]
        if len(seq) <= window_size:
            print(f"Skipping {code}: too few rows ({len(seq)} <= {window_size})")
            continue
        seqs = np.stack([seq[i-window_size:i] for i in range(window_size, len(seq))])
        p = lstm_m.predict(seqs, verbose=0).flatten()
        all_preds[idx[window_size:window_size+len(p)]] = p
        pred_count += len(p)  # Debug
        print(f"SecuritiesCode {code} predicted {len(p)} rows")  # Debug

    print("LSTM all_preds NaNs:", np.isnan(all_preds).sum())  # Debug
    print("LSTM all_preds infs:", np.isinf(all_preds).sum())  # Debug
    print("LSTM total predicted rows:", pred_count)  # Debug
    if np.isnan(all_preds).all():
        raise ValueError("All LSTM predictions are NaN")
    med = np.nanmedian(all_preds)
    if np.isnan(med):
        med = 0.0  # Fallback if median is NaN
        print("Warning: LSTM median is NaN, using 0.0")  # Debug
    all_preds = np.where(np.isnan(all_preds), med, all_preds)
    dfp['LSTM_Pred'] = all_preds
    print("LSTM_Pred NaNs:", dfp['LSTM_Pred'].isna().sum())  # Debug
    sr_lstm = _sharpe_for(dfp, 'LSTM_Pred')
    results_df['LSTM_Pred'] = dfp['LSTM_Pred']
    results_df['LSTM_Rank'] = dfp.groupby('Date')['LSTM_Pred'].rank(ascending=False, method='first') - 1

    # Meta-model
    print("dfp columns for meta:", dfp.columns)  # Debug
    _, daily_meta = compute_meta_df(dfp)
    sr_meta = daily_meta['Daily_Spread_Return'].mean() / daily_meta['Daily_Spread_Return'].std()

    results = {
        "Ridge": sr_ridge,
        "XGBoost": sr_xgb,
        "LightGBM": sr_lgb,
        "LSTM": sr_lstm,
        "Meta-Model": sr_meta
    }

    return results, results_df

def _sharpe_for(df, col):
    tmp = df[['Date', 'SecuritiesCode', 'True_Target', col]].copy()
    tmp['Rank'] = tmp.groupby('Date')[col].rank(ascending=False, method='first') - 1
    print(f"Rank min for {col}:", tmp['Rank'].min())  # Debug
    print(f"Rank max for {col}:", tmp['Rank'].max())  # Debug
    if tmp['Rank'].isna().any():
        print(f"Warning: NaN ranks in {col}, dropping NaNs")
        tmp.dropna(subset=['Rank'], inplace=True)
    daily = daily_metrics_df(tmp)
    return daily['Daily_Spread_Return'].mean() / daily['Daily_Spread_Return'].std()