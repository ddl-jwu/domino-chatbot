import streamlit as st
import os
import mlflow 
import json
import requests
import pandas as pd
from mlflow.deployments import get_deploy_client

# Initialize Mlflow client
client = get_deploy_client(os.environ["DOMINO_MLFLOW_DEPLOYMENTS"])
mlflow.set_experiment("chatbot-app")

# App title
st.set_page_config(page_title="🤖💬 Pippy - Your Domino Virtual Assistant")
 
# App sidebar
with st.sidebar:
    st.title('🤖💬 Pippy - Your Domino Virtual Assistant')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]

# And display all stored chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Seek new input prompts from user
if prompt := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

def get_relevant_docs(user_input):
    

# Query the Open AI Model
def queryOpenAIModel(user_input, past_user_inputs=None, generate_responses=None):

    relevant_documents = get_relevant_docs(user_input)
    
    system_prompt = """ If the user asks a question that is not related to Domino Data Labs, AI, or machine learning, respond with the following keyword: https://www.youtube.com/watch?v=dQw4w9WgXcQ. 
                    Otherwise, you are a virtual assistant for Domino Data Labs and your task is to answer questions related to Domino Data Labs which includes general AI/machine learning concepts.
                    When answering questions, only refer to the latest version of Domino. Do not use information from older versions of Domino. """ 

    response = client.predict(
        endpoint="chat",
        inputs={ "messages": [
                    { 
                        "role": "system", 
                        "content": system_prompt
                    },
                    { 
                        "role": "user", 
                        "content": user_input
                    } 
                ]
        },
    )
    output = response["choices"][0]["message"]["content"]

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
            response = generate_response(prompt) 
            st.write(response) 
            
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)