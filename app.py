import streamlit as st
import base64

# Set page config
st.set_page_config(page_title="Stock Market App (JPX)", layout="wide")

# Load and encode the background image
try:
    with open("exchange.png", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
except FileNotFoundError:
    st.error("Image not found!")
    encoded_image = ""

# Custom CSS styling for sidebar, content, and Home/About page styling
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] .stRadio > div {{
        display: flex;
        flex-direction: column;
        gap: 2.0rem;
    }}
    [data-testid="stSidebar"] label {{
        font-size: 1.2rem;
    }}
    .stMarkdown, .stText, .stDataFrame {{
        font-size: 1.1rem;
    }}
    h1, .stTitle {{
        font-size: 2rem;
    }}
    h2, .stHeader {{
        font-size: 1.5rem;
    }}
    /* Background image for Home page only */
    .home-container {{
        background-image: url('data:image/png;base64,{encoded_image}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding: 20px;
        border-radius: 10px;
        min-height: 400px;
    }}
    /* Optional: Ensure text is readable over the image */
    .home-container > div {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 5px;
    }}
    /* Style for the welcome text */
    .welcome-text {{
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        color: #FFD700; /* Gold color */
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3), 0 0 5px rgba(255, 215, 0, 0.5); /* Shadow and glow for golden effect */
        border-bottom: 2px solid #FFD700; /* Gold underline */
        padding-bottom: 10px;
        margin: 0 auto;
        max-width: 800px;
    }}
    /* Container and styling for About page */
    .about-container {{
        padding: 20px;
        border-radius: 10px;
        max-width: 800px;
        margin: 0 auto;
    }}
    .about-text {{
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        color: #FFD700; /* Gold color */
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3), 0 0 5px rgba(255, 215, 0, 0.5); /* Shadow and glow for golden effect */
        border-bottom: 2px solid #FFD700; /* Gold underline */
        padding-bottom: 10px;
        margin: 0 auto;
        max-width: 800px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# Show login page if not logged in
if not st.session_state.logged_in:
    import Login
    Login.run()
else:
    # Admin user has access to all pages including the AI Assistant
    if st.session_state.username == "admin":
        navigation_options = [
            "🏠 Home",
            "📈 Predictions",
            "💱 Simulate Competition",
            "📂 Upload CSV",
            "🤖 AI Assistant",  # Admin-only
            "ℹ️ About"
        ]
    else:
        navigation_options = [
            "🏠 Home",
            "📈 Predictions",
            "💱 Simulate Competition",
            "📂 Upload CSV",
            "ℹ️ About"
        ]

    # Sidebar navigation menu
    selected_page = st.sidebar.radio("Navigation", navigation_options)

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        # Clear user-specific session state
        for key in ['selected_models', 'sampled_dates']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Route to selected page
    if selected_page == "📈 Predictions":
        import Predictions
        Predictions.run()
    elif selected_page == "💱 Simulate Competition":
        import SimulateCompetition
        SimulateCompetition.run()
    elif selected_page == "📂 Upload CSV":
        import UploadCSV
        UploadCSV.run()
    elif selected_page == "🤖 AI Assistant" and st.session_state.username == "admin":
        import LLMAssistant
        LLMAssistant.run()
    elif selected_page == "ℹ️ About":
        st.title("ℹ️ About")
        # Wrap About page content in a container
        st.markdown('<div class="about-container">', unsafe_allow_html=True)
        # Style the about text
        st.markdown(
            """
            <div class="about-text">
            This platform analyzes the Japanese stock market using quantitative trading strategies 
            based on JPX (Japan Exchange Group) data from the Kaggle competition.<br><br>
            Main Goals:<br>
            1. Data Cleaning and Feature Engineering<br>
            2. Building Predictive Models for Return Forecasting<br>
            3. Evaluating Performance Using Sharpe Ratio
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.title("🏠 Home")
        # Wrap Home page content in a container with the background image
        st.markdown('<div class="home-container">', unsafe_allow_html=True)
        # Style the welcome text
        st.markdown(
            """
            <div class="welcome-text">
            Welcome to the JPX Stock Market App! Use the sidebar to navigate through the platform.<br>
            You can explore predictions, upload your own datasets, simulate the competition, or access the AI assistant (admin only).
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)