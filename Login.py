import streamlit as st
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Retrieve and parse VALID_CREDENTIALS from .env
try:
    VALID_CREDENTIALS = json.loads(os.getenv("VALID_CREDENTIALS", "{}"))
except json.JSONDecodeError as e:
    st.error(f"Error parsing VALID_CREDENTIALS from .env: {e}")
    VALID_CREDENTIALS = {}

def run():
    # Custom CSS for the login page
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #f9f9f9;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stTextInput > div > input {
            font-size: 1.1rem;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #ccc;
            width: 100%;
        }
        .stButton > button {
            font-size: 1.1rem;
            padding: 0.5rem 1rem;
            background-color: #007bff;
            color: white;
            border-radius: 5px;
            border: none;
            width: 100%;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .stError, .stSuccess {
            font-size: 1.1rem;
            padding: 0.5rem;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔒 Login")
    st.write("Please enter your username and password to access the application.")

    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        with st.form(key="login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome, {username}!")
                    # Redirect to the main app
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")
        st.markdown('</div>', unsafe_allow_html=True)