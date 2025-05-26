import streamlit as st
import pandas as pd
import os
from utils.metrics import daily_metrics_df

@st.cache_data
def load_data(fp):
    return pd.read_csv(fp)

def run():
    st.title("📈 Prediction Evaluation – Sharpe & Spread Return")

    # Session state inicializálása a kiválasztott modellek tárolására
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []

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

    # 1) Select dataset
    file_option = st.radio("Dataset:", ["Validation", "Test"])
    raw_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_preds.csv"
    meta_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_meta.csv"

    # 2) Load the original predictions
    data = load_data(raw_fp)

    # 3) Modellválasztás kezelése
    model_option = st.radio(
        "Model:",
        ["LGB_Pred", "LSTM_Pred", "Ridge_Pred", "XGB_Pred", "Meta_Model"],
        key="model_select"
    )

    # Gomb a modell hozzáadásához
    if st.button("Add Model"):
        if model_option not in st.session_state.selected_models:
            st.session_state.selected_models.append(model_option)

    # Gomb az összes modell törléséhez
    if st.button("Clear All Models"):
        st.session_state.selected_models = []

    # 4) Minden kiválasztott modellhez adatok és táblázat generálása
    for model in st.session_state.selected_models:
        st.subheader(f"Results for {model}")

        if model == "Meta_Model":
            # File paths
            meta_preds_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_meta_preds.csv"
            meta_daily_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_meta_daily.csv"

            # If files don't exist, compute and save them
            if not os.path.exists(meta_preds_fp) or not os.path.exists(meta_daily_fp):
                meta_df, meta_daily = compute_meta_df(data)
                meta_df_to_save = meta_df[[
                    'RowId', 'Date', 'SecuritiesCode', 'Target',
                    'Predicted_Target', 'Rank'
                ]]
                meta_df_to_save.to_csv(meta_preds_fp, index=False)
                meta_daily.to_csv(meta_daily_fp, index=False)

            # Load with all columns
            df = pd.read_csv(meta_preds_fp, parse_dates=['Date'])
            daily = pd.read_csv(meta_daily_fp, parse_dates=['Date'])

        else:
            # Standard model branch
            df = data[['Date', 'SecuritiesCode', 'Target']].copy()
            df['True_Target'] = df['Target']
            df['Predicted_Target'] = data[model]
            df['Rank'] = (
                df.groupby('Date')['Predicted_Target']
                .rank(ascending=False, method='first')
                - 1
            )
            daily = daily_metrics_df(df)

        # 5) Display results for the model
        overall_sharpe = daily['Daily_Spread_Return'].mean() / daily['Daily_Spread_Return'].std()
        st.metric(f"Aggregate Sharpe ({model})", f"{overall_sharpe:.4f}")

        st.subheader(f"Daily Metrics (Last 15) ({model})")
        st.dataframe(daily.tail(15), use_container_width=True)

    # 6) Kombinált grafikon az összes kiválasztott modellhez
    if st.session_state.selected_models:
        combined_data = pd.DataFrame()
        for model in st.session_state.selected_models:
            if model == "Meta_Model":
                meta_daily_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_meta_daily.csv"
                daily = pd.read_csv(meta_daily_fp, parse_dates=['Date'])
            else:
                df = data[['Date', 'SecuritiesCode', 'Target']].copy()
                df['True_Target'] = df['Target']
                df['Predicted_Target'] = data[model]
                df['Rank'] = (
                    df.groupby('Date')['Predicted_Target']
                    .rank(ascending=False, method='first')
                    - 1
                )
                daily = daily_metrics_df(df)
            combined_data[model] = daily.set_index('Date')['Daily_Spread_Return']

        st.subheader("Combined Daily Spread Return")
        st.line_chart(combined_data)