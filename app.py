import streamlit as st

st.set_page_config(page_title="Stock Market App (JPX)", layout="wide")

# Sidebar spacing and font styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 2.0rem;
    }
    [data-testid="stSidebar"] label {
        font-size: 1.2rem;
    }
    .stMarkdown, .stText, .stDataFrame {
        font-size: 1.1rem;
    }
    h1, .stTitle {
        font-size: 2rem;
    }
    h2, .stHeader {
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation
selected_page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📈 Predictions", "💱 Competition Simulation", "ℹ️ About"]
)

if selected_page == "📈 Predictions":
    import Predictions
    Predictions.run()
elif selected_page == "💱 Competition Simulation":
    import SimulateCompetition
    SimulateCompetition.run()
elif selected_page == "ℹ️ About":
    st.title("ℹ️ About")
    st.write("""
        The purpose of the website is to analyze the Japanese financial market within the framework of developing and evaluating quantitative trading strategies. The research is based on data provided by the Japan Exchange Group, Inc. (JPX), which includes stock and derivative trading data from the Tokyo Stock Exchange (TSE), the Osaka Exchange (OSE), and the Tokyo Commodity Exchange (TOCOM). The JPX aimed to host a competition with this data on Kaggle, the world's largest data science community platform, where participants were tasked with developing advanced quantitative trading models that perform well in real-world scenarios.

        **Objectives to be achieved**

        1. **Analysis and Preparation of Stock Market Data**: Thoroughly examine the structure of Japanese stock market data, with particular attention to stock prices and historical data. Data cleaning and preparation are crucial for accurate model development.

        2. **Development and Testing of Quantitative Models**: The project involves developing algorithms and predictive models capable of forecasting stock returns. These models are used to rank stocks, aiming to select those with the highest expected returns and identify the least promising ones.

        3. **Measurement and Optimization of Strategy Performance**: The project employs the Sharpe ratio to evaluate trading strategies, which is a risk-adjusted measure of returns. The strategy's performance is determined by daily returns, enabling the construction of a profitable portfolio.
    """)
else:
    st.title("🏠 Home")
    st.write("""
        This application allows comparing machine learning model predictions 
        based on the JPX stock market competition data.
    """)