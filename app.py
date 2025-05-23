import streamlit as st

st.set_page_config(page_title="AI Tőzsdei App", layout="wide")

st.title("Üdvözöl a Saját AI Tőzsdei Webalkalmazás")
st.write("""
Ez az alkalmazás különböző gépi tanulási modellek (Ridge, XGBoost, LightGBM, LSTM)  
által generált előrejelzések összehasonlítását teszi lehetővé a JPX verseny adatai alapján.

Használd a bal oldali menüt a „Predictions” vagy „Simulate Competition” oldalakhoz navigáláshoz.
""")