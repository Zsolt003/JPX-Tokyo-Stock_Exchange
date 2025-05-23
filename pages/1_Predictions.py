import streamlit as st
import pandas as pd
from utils.metrics import daily_metrics_df

@st.cache_data
def load_data(fp): return pd.read_csv(fp)

st.title("Predikciók értékelése – Sharpe & Spread Return")

file_option = st.selectbox("Adathalmaz:", ["Validation", "Test"])
fp = "data/val_preds.csv" if file_option=="Validation" else "data/test_preds.csv"
data = load_data(fp)

model_option = st.selectbox("Modell:", ["LGB_Pred","LSTM_Pred"])

df = data[['Date','SecuritiesCode','Target']].copy()
df['True_Target']     = df['Target']
df['Predicted_Target']= data[model_option]
df['Rank'] = df.groupby('Date')['Predicted_Target']\
               .rank(ascending=False, method='first')-1

daily = daily_metrics_df(df)
overall_sharpe = daily['Daily_Spread_Return'].mean()/daily['Daily_Spread_Return'].std()
st.metric("Aggregált Sharpe", f"{overall_sharpe:.4f}")

st.subheader("Napi Spread Return")
st.line_chart(daily.set_index('Date')['Daily_Spread_Return'])

st.subheader("Napi metrikák (utolsó 15)")
st.dataframe(daily.tail(15), use_container_width=True)