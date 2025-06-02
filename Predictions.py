# 1_Predictions.py

import streamlit as st
import pandas as pd
import os
from utils.metrics import daily_metrics_df, compute_meta_df
from utils.llm import ask_llm

@st.cache_data
def load_data(fp):
    return pd.read_csv(fp)

def run():
    st.title("📈 Prediction Evaluation – Sharpe & Spread Return")

    # Ensure we have a place to store which models are selected
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []

    # 1) Choose which dataset (Validation or Test)
    file_option = st.radio("Dataset:", ["Validation", "Test"], index=0)
    raw_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_preds.csv"
    meta_preds_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_meta_preds.csv"
    meta_daily_fp = f"data/{'val' if file_option == 'Validation' else 'test'}_meta_daily.csv"

    data = load_data(raw_fp)

    # 2) Pick one model to “add” to the session
    model_option = st.radio(
        "Model:",
        ["LGB_Pred", "LSTM_Pred", "Ridge_Pred", "XGB_Pred", "Meta_Model"],
        key="model_select"
    )

    # “Add Model” button appends to session_state.selected_models
    if st.button("➕ Add Model"):
        if model_option not in st.session_state.selected_models:
            st.session_state.selected_models.append(model_option)

    # “Clear All Models” resets the list
    if st.button("♻️ Clear All Models"):
        st.session_state.selected_models = []

    # 3) For each model in the list, compute & display its metrics
    for model in st.session_state.selected_models:
        st.subheader(f"Results for **{model}**")

        # If “Meta_Model” is chosen, ensure its CSVs exist (compute if not)
        if model == "Meta_Model":
            if not os.path.exists(meta_preds_fp) or not os.path.exists(meta_daily_fp):
                meta_df, meta_daily = compute_meta_df(data)

                # Save only the needed columns for predictions
                meta_df_to_save = meta_df[[
                    'RowId', 'Date', 'SecuritiesCode', 'Target',
                    'Predicted_Target', 'Rank'
                ]]
                meta_df_to_save.to_csv(meta_preds_fp, index=False)

                # Save the daily metrics for meta
                meta_daily.to_csv(meta_daily_fp, index=False)

            # Load the already‐saved meta outputs
            df = pd.read_csv(meta_preds_fp, parse_dates=['Date'])
            daily = pd.read_csv(meta_daily_fp, parse_dates=['Date'])

        else:
            # Standard single‐model branch (LGB, LSTM, Ridge, or XGB)
            df = data[['Date', 'SecuritiesCode', 'Target']].copy()
            df['True_Target'] = df['Target']
            df['Predicted_Target'] = data[model]
            df['Rank'] = (
                df.groupby('Date')['Predicted_Target']
                  .rank(ascending=False, method='first')
                  - 1
            )
            daily = daily_metrics_df(df)

        # Compute and show overall Sharpe for that model
        if daily['Daily_Spread_Return'].std() > 0:
            overall_sharpe = daily['Daily_Spread_Return'].mean() / daily['Daily_Spread_Return'].std()
        else:
            overall_sharpe = 0.0

        st.metric(f"Aggregate Sharpe ({model})", f"{overall_sharpe:.4f}")

        # Show last 15 rows of “daily” metrics
        st.subheader(f"Daily Metrics (Last 15) – {model}")
        st.dataframe(daily.tail(15), use_container_width=True)

    # 4) If at least one model is selected, plot their combined daily returns
    if st.session_state.selected_models:
        combined_data = pd.DataFrame()

        for model in st.session_state.selected_models:
            if model == "Meta_Model":
                # Load the precomputed “meta_daily_fp”
                daily_df = pd.read_csv(meta_daily_fp, parse_dates=['Date'])
            else:
                # Recompute daily for each “normal” model
                df_temp = data[['Date', 'SecuritiesCode', 'Target']].copy()
                df_temp['True_Target'] = df_temp['Target']
                df_temp['Predicted_Target'] = data[model]
                df_temp['Rank'] = (
                    df_temp.groupby('Date')['Predicted_Target']
                           .rank(ascending=False, method='first')
                    - 1
                )
                daily_df = daily_metrics_df(df_temp)

            # Set ‘Date’ as index and pull out “Daily_Spread_Return”
            series = daily_df.set_index('Date')['Daily_Spread_Return']
            combined_data[model] = series

        st.subheader("📊 Combined Daily Spread Return (Selected Models)")
        st.line_chart(combined_data)

        # 5) “Explain with LLM” – only visible to admin users
        # Check session_state.username
        is_admin = st.session_state.get("username", "") == "admin"

        if is_admin:
            st.markdown("**As an admin, you may ask the LLM to analyze these combined returns.**")
            if st.button("💡 Explain with LLM"):
                last_n_days = 150
                snippet = combined_data.tail(last_n_days).reset_index()
                snippet_csv = snippet.to_csv(index=False)

                prompt = (
                    "Below are the daily spread returns for several models in the JPX competition. "
                    "Columns correspond to models; rows correspond to dates (YYYY-MM-DD). "
                    "Values are the daily spread return for that day. Based on this data, please:\n\n"
                    "1. Explain why on certain dates the Sharpe ratio might have been unusually high or low. "
                    "(Hint: large positive/negative daily spread returns skew the mean/std.)\n"
                    "2. Identify any multi-day or seasonal trends (e.g., “Sharpe tends to rise around end-of-month,” "
                    "or “Sharpe dips at quarter boundaries,” etc.)\n"
                    "3. Summarize, in plain English, what patterns you observe (if multiple models are being chosen, which days which selected model performs better and a possible explanation).\n\n"
                    f"Here is the snippet (last {last_n_days} days):\n\n"
                    f"{snippet_csv}\n\n"
                    "Answer concisely but with enough detail so a user can understand why some days are better/worse and "
                    "what recurring trends exist."
                )

                with st.spinner("Contacting LLM, please wait…"):
                    try:
                        llm_reply = ask_llm(prompt)
                    except Exception as e:
                        llm_reply = f"❌ An error occurred when calling the LLM: {e}"

                st.subheader("🔍 LLM Explanation & Trend Analysis")
                st.write(llm_reply)

        else:
            # Non-admin users see a note explaining this feature is admin-only
            st.info("ℹ️ Only an admin can request LLM analysis of combined returns.")
