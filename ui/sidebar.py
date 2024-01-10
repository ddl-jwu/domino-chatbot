import streamlit as st


def build_sidebar():
    POPULAR_QUESTION_1 = "How can Domino help me build generative AI?"
    POPULAR_QUESTION_2 = "How can I set a default Environment in Domino?"
    POPULAR_QUESTION_3 = "Generate code for a custom metric to monitor Model toxicity"
    POPULAR_QUESTION_4 = "What are the new features in Domino 5.10?"

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
    st.sidebar.button(
        POPULAR_QUESTION_1,
        on_click=insert_as_users_prompt,
        kwargs={"prompt": POPULAR_QUESTION_1},
    )
    st.sidebar.button(
        POPULAR_QUESTION_2,
        on_click=insert_as_users_prompt,
        kwargs={"prompt": POPULAR_QUESTION_2},
    )
    st.sidebar.button(
        POPULAR_QUESTION_3,
        on_click=insert_as_users_prompt,
        kwargs={"prompt": POPULAR_QUESTION_3},
    )
    st.sidebar.button(
        POPULAR_QUESTION_4,
        on_click=insert_as_users_prompt,
        kwargs={"prompt": POPULAR_QUESTION_4},
    )
    st.sidebar.markdown("---")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history, type="primary")
