import streamlit as st
import os
import mlflow 
import json
import requests
import pandas as pd
from mlflow.deployments import get_deploy_client
from ui.sidebar import build_sidebar
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatMlflow
from langchain.schema import HumanMessage, SystemMessage
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory

# Initialize conversation chain
if "conversation" not in st.session_state.keys():
    chat = ChatMlflow(
        target_uri=os.environ["DOMINO_MLFLOW_DEPLOYMENTS"],
        endpoint="chat",
    )
    st.session_state.conversation = ConversationChain(
        llm=chat,
        memory=ConversationSummaryMemory(llm=chat),
        verbose=True
    )

# Set MLflow experiment to use for logging
mlflow.set_experiment("chatbot-app")

# App title
st.set_page_config(page_title="Domino Pippy ChatAssist", layout="wide")

# App sidebar
with st.sidebar:
    build_sidebar()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# And display all stored chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Seek new input prompts from user
if prompt := st.chat_input("Chat with Pippy"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Get relevant docs through vector DB
def get_relevant_docs(user_input):
    relevant_docs = "Hi"
    return relevant_docs

# Query the Open AI Model
def queryOpenAIModel(user_input, past_user_inputs=None, generate_responses=None):

    relevant_docs = get_relevant_docs(user_input)

    template = """ If the user asks a question that is not related to Domino Data Labs, AI, or machine learning, respond with the following keyword: https://www.youtube.com/watch?v=dQw4w9WgXcQ. 
                    Otherwise, you are a virtual assistant for Domino Data Labs and your task is to answer questions related to Domino Data Labs which includes general AI/machine learning concepts.
                    When answering questions, only refer to the latest version of Domino. Do not use information from older versions of Domino. 
                    In your response, include a list of the references (with URL links) where you obtained the information from.
                    Here is some relevant context: {relevant_docs} """

    prompt_template = PromptTemplate(
        input_variables=["relevant_docs"],
        template=template
    )
    system_prompt = prompt_template.format(relevant_docs=relevant_docs)

    messages = [
        SystemMessage(
            content=system_prompt
        ),
        HumanMessage(
            content=user_input
        ),
    ]

    output = st.session_state.conversation.predict(input=messages)

    # Log results to MLflow
    with mlflow.start_run():
        mlflow.log_param("system_prompt", system_prompt)
        mlflow.log_param("user_input", user_input)
        mlflow.log_param("output", output)

    return output


# Function for generating LLM response
def generate_response(prompt):
    response_generated = queryOpenAIModel(prompt)
    return response_generated


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(st.session_state.messages[-1]["content"])
            st.write(response)

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
