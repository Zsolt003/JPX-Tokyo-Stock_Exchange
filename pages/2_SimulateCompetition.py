import streamlit as st
import pandas as pd
import numpy as np
from utils.metrics import daily_metrics_df

@st.cache_data
def load_data(fp):
    return pd.read_csv(fp)

st.title("💱 Simulate Competition")

# Modell kiválasztása
model_sim = st.selectbox("Modell a szimulációhoz:", ["LGB_Pred", "LSTM_Pred"])

# Adatok betöltése és összefűzése
val = load_data("data/val_preds.csv")
test = load_data("data/test_preds.csv")
sim = pd.concat([val, test], ignore_index=True)

# Egyedi dátumok és mintaméret (10%)
dates = sorted(sim['Date'].unique())
n = max(1, int(0.1 * len(dates)))

# Új véletlenszerű napok kiválasztása gombbal
if st.button("🎲 Új napok kiválasztása"):
    st.session_state.sampled_dates = list(np.random.choice(dates, size=n, replace=False))

# Ha még nincs session_state-ben naplista, alapértelmezésként létrehozzuk
if 'sampled_dates' not in st.session_state:
    st.session_state.sampled_dates = list(np.random.choice(dates, size=n, replace=False))

sampled_dates = st.session_state.sampled_dates

# Mintázott napok megjelenítése
st.write("📅 Mintavételezett napok:", sampled_dates)

# Adatok szűrése a kiválasztott napokra
df_sim = sim[sim['Date'].isin(sampled_dates)].copy()
df_sim['True_Target'] = df_sim['Target']
df_sim['Predicted_Target'] = df_sim[model_sim]
df_sim['Rank'] = df_sim.groupby('Date')['Predicted_Target'].rank(ascending=False, method='first') - 1

# Napi metrikák számítása
daily_sim = daily_metrics_df(df_sim)
overall = daily_sim['Daily_Spread_Return'].mean() / daily_sim['Daily_Spread_Return'].std()
hit_top = daily_sim['HitRate@Top200'].mean()
hit_bot = daily_sim['HitRate@Bottom200'].mean()

# Metrikák megjelenítése
st.metric("Szimulált Sharpe", f"{overall:.4f}")
st.metric("Avg HitRate Top200", f"{hit_top:.2%}")
st.metric("Avg HitRate Bottom200", f"{hit_bot:.2%}")

# Grafikon
st.subheader("Napi spread return")
st.line_chart(daily_sim.set_index('Date')['Daily_Spread_Return'])

# Táblázat
st.subheader("Részletes napi metrikák")
st.dataframe(daily_sim, use_container_width=True)