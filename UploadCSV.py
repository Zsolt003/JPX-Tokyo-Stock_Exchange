import streamlit as st
import pandas as pd
import warnings
from utils.metrics_upload import _load_all_models, _validate_csv, _generate_features, _compute_all_sharpe

warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW_SIZE = 3
FEATURE_COLS = [
    'Open', 'MACD', 'DayOfWeek', 'SharpeLike_10', 'Relative_Range',
    'Momentum_20', 'RSI_14', 'Volatility_10', 'Month'
]

def run():
    st.title("📂 Upload CSV & Compute Sharpe")

    st.write("""
        Upload a CSV with columns  
        `RowId, Date, SecuritiesCode, Open, High, Low, Close, Volume, Target`  
        Validate it, generate features, then compute Sharpe Ratios  
        for Ridge, XGBoost, LightGBM, LSTM, and Meta-model.
    """)

    uploaded = st.file_uploader("Choose your CSV", type="csv")
    if not uploaded:
        return

    raw_df = pd.read_csv(uploaded)
    errs = _validate_csv(raw_df.copy())
    if errs:
        st.error("❌ Validation failed:")
        for e in errs:
            st.write(f"- {e}")
        return

    st.success("✅ CSV is valid")
    features_df = _generate_features(raw_df.copy(), FEATURE_COLS)
    st.subheader("Generated Features (head)")
    st.dataframe(features_df.head(), use_container_width=True)

    csv_bytes = features_df.to_csv(index=False).encode()
    st.download_button(
        "📥 Download Features CSV",
        data=csv_bytes,
        file_name="features_output.csv",
        mime="text/csv"
    )

    if st.button("▶️ Compute Sharpe Ratios"):
        try:
            results, results_df = _compute_all_sharpe(features_df.copy(), WINDOW_SIZE, FEATURE_COLS)
            st.subheader("Sharpe Ratios")
            for name, val in results.items():
                st.write(f"**{name}:** {val:.4f}")

            # Save results to CSV and provide download button
            results_csv = results_df.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Sharpe Results CSV",
                data=results_csv,
                file_name="sharpe_results.csv",
                mime="text/csv"
            )

            # Save Sharpe ratios to a separate CSV
            sharpe_df = pd.DataFrame(list(results.items()), columns=['Model', 'Sharpe_Ratio'])
            sharpe_csv = sharpe_df.to_csv(index=False).encode()
            st.download_button(
                "📥 Download Sharpe Ratios CSV",
                data=sharpe_csv,
                file_name="sharpe_ratios.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.exception(e)