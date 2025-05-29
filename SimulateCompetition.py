import streamlit as st
import pandas as pd
import numpy as np
from utils.metrics import daily_metrics_df

@st.cache_data
def load_data(fp):
    return pd.read_csv(fp, parse_dates=['Date'])

def run():
    st.title("💱 Competition Simulation")
    model_sim = st.radio(
        "Model for simulation:",
        ["LGB_Pred", "LSTM_Pred", "Ridge_Pred", "XGB_Pred", "Meta_Model"]
    )

    if model_sim == "Meta_Model":
        val = load_data("data/val_meta_preds.csv")
        test = load_data("data/test_meta_preds.csv")
    else:
        val = load_data("data/val_preds.csv")
        test = load_data("data/test_preds.csv")

    # Concatenate
    sim = pd.concat([val, test], ignore_index=True)

    # Sample days
    dates = sorted(sim['Date'].dt.strftime("%Y-%m-%d").unique())
    n = max(1, int(0.1 * len(dates)))

    if st.button("🎲 Select New Days"):
        st.session_state.sampled_dates = list(np.random.choice(dates, size=n, replace=False))

    if 'sampled_dates' not in st.session_state:
        st.session_state.sampled_dates = list(np.random.choice(dates, size=n, replace=False))

    sampled_dates = st.session_state.sampled_dates
    st.write("📅 Sampled Days:", sampled_dates)

    # Filter and rank
    df_sim = sim[sim['Date'].dt.strftime("%Y-%m-%d").isin(sampled_dates)].copy()

    df_sim['True_Target'] = df_sim['Target']
    df_sim['Predicted_Target'] = df_sim[model_sim] if model_sim != "Meta_Model" else df_sim['Predicted_Target']
    df_sim['Rank'] = (
        df_sim.groupby('Date')['Predicted_Target']
            .rank(ascending=False, method='first')
        - 1
    )

    # Daily metrics
    daily_sim = daily_metrics_df(df_sim)
    overall   = daily_sim['Daily_Spread_Return'].mean() / daily_sim['Daily_Spread_Return'].std()
    hit_top   = daily_sim['HitRate@Top200'].mean()
    hit_bot   = daily_sim['HitRate@Bottom200'].mean()

    # Display
    st.metric("Simulated Sharpe", f"{overall:.4f}")

    st.subheader("Daily Spread Return")
    st.line_chart(daily_sim.set_index('Date')['Daily_Spread_Return'])

    st.subheader("Detailed Daily Metrics")
    st.dataframe(daily_sim, use_container_width=True)