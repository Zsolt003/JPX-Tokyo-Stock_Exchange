import streamlit as st
from utils.llm import ask_llm  # Make sure you have this function defined

def run():
    # Restrict access to admin only
    if st.session_state.get("username") != "admin":
        st.error("You do not have access to this page.")
        return

    st.title("🤖 AI Assistant")
    st.markdown("Use this assistant to ask questions about financial modeling, machine learning, or the JPX competition.")

    # Input area for user prompt
    prompt = st.text_area("Enter your question:", placeholder="E.g. What is a Sharpe ratio?")

    if st.button("Submit"):
        if not prompt.strip():
            st.warning("Please enter a question before submitting.")
            return

        # Get response from OpenAI model
        with st.spinner("Generating response..."):
            try:
                response = ask_llm(prompt)
                st.success("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while contacting the language model: {e}")
