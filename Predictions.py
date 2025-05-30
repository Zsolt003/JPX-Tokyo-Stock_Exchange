import streamlit as st
import pandas as pd
import os
from utils.metrics import daily_metrics_df, compute_meta_df

@st.cache_data
def load_data(fp):
    return pd.read_csv(fp)

def run():
    st.title("📈 Prediction Evaluation – Sharpe & Spread Return")

    # Session state inicializálása a kiválasztott modellek tárolására
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []

    file_option = st.radio("Dataset:", ["Validation", "Test"])
    raw_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_preds.csv"
    meta_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_meta.csv"

    data = load_data(raw_fp)

    model_option = st.radio(
        "Model:",
        ["LGB_Pred", "LSTM_Pred", "Ridge_Pred", "XGB_Pred", "Meta_Model"],
        key="model_select"
    )

    if st.button("Add Model"):
        if model_option not in st.session_state.selected_models:
            st.session_state.selected_models.append(model_option)

    if st.button("Clear All Models"):
        st.session_state.selected_models = []

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

        # Display results for the model
        overall_sharpe = daily['Daily_Spread_Return'].mean() / daily['Daily_Spread_Return'].std()
        st.metric(f"Aggregate Sharpe ({model})", f"{overall_sharpe:.4f}")

        st.subheader(f"Daily Metrics (Last 15) ({model})")
        st.dataframe(daily.tail(15), use_container_width=True)

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