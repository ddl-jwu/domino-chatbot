import streamlit as st
import random

"""
TODO: Use Coveo to get top queries submitted by customers and populate `popular_questions` variable with the results
"""


def build_sidebar():
    popular_questions = [
        "Conda Virtual Environments in Compute Environments in Domino",
        "How to get workspace Logs And Support Bundle for 5.1 and Above",
        "Exotic IDE/VNC or Compute Environment Support - why is it not supported?",
        "How to set up Domino Model Monitoring (DMM) using the Domino API",
        "How to capture a HAR file from your Chrome web browser",
        "How to work with data in Domino",
        "How to sync files with MPI clusters in Domino",
        "How to exclude Project files from sync in Domino",
        "How can Domino help me build generative AI?",
        "How can I set a default Environment in Domino?",
        "Generate code for a custom metric to monitor Model toxicity",
        "What are the new features in Domino 5.10?",
    ]

    def insert_as_users_prompt(**kwargs):
        if prompt := kwargs.get("prompt"):
            st.session_state.messages.append({"role": "user", "content": prompt})

    def clear_chat_history():
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you today?"}
        ]

    # App sidebar
    st.image("/mnt/code/assets/pippy.png", width=50)
    st.title("Hi, I'm Pippy - your personal Domino Data Lab expert")
    st.sidebar.markdown("---")
    st.title("Ask me anything")

    # Pick any 4 questions randomly from popular_questions
    selected_questions = random.sample(popular_questions, 4)

    for question in selected_questions:
        st.sidebar.button(
            question,
            on_click=insert_as_users_prompt,
            kwargs={"prompt": question},
        )
    st.sidebar.markdown("---")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history, type="primary")
