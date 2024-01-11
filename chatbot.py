import streamlit as st
import os
import mlflow 
import json
import requests
import pandas as pd
import pinecone
from mlflow.deployments import get_deploy_client
from ui.sidebar import build_sidebar
from langchain_community.embeddings import MlflowEmbeddings
from domino_data.vectordb import DominoPineconeConfiguration

# Number of texts to match (may be less if no suitable match)
NUM_TEXT_MATCHES = 3

# Initialize Mlflow client
client = get_deploy_client(os.environ["DOMINO_MLFLOW_DEPLOYMENTS"])
mlflow.set_experiment("chatbot-app")

# Initialize Pinecone index
datasource_name = "PineconeHackathon"
conf = DominoPineconeConfiguration(datasource=datasource_name)
api_key = os.environ.get("DOMINO_VECTOR_DB_METADATA", datasource_name)

pinecone.init(
    api_key=api_key,
    environment="domino",
    openapi_config=conf
)

# Choose appropriate index from Pinecone
# index_name = "hacktestlatestall"
index_name = "hacktest"
index = pinecone.Index(index_name)

# Create embeddings to embed queries
embeddings = MlflowEmbeddings(
    target_uri=os.environ["DOMINO_MLFLOW_DEPLOYMENTS"],
    endpoint="embeddings",
)

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
    embedded_query = embeddings.embed_query(user_input)
    return index.query(
        vector=embedded_query,
        top_k=NUM_TEXT_MATCHES,
        include_values=True,
        include_metadata=True
    )

# Query the Open AI Model
def queryOpenAIModel(user_input, past_user_inputs=None, generate_responses=None):
    relevant_docs = get_relevant_docs(user_input)
    # Get relevant URLs, filtering out repeated
    url_links = set([relevant_docs["matches"][i]["metadata"]["url"] for i in range(NUM_TEXT_MATCHES)])
    context = [relevant_docs["matches"][i]["metadata"]["text"] for i in range(NUM_TEXT_MATCHES)]
    
    system_prompt = """ If the user asks a question that is not related to Domino Data Lab, AI, or machine learning, respond with the following keyword: https://www.youtube.com/watch?v=dQw4w9WgXcQ. 
                    Otherwise, you are a virtual assistant for Domino Data Lab and your task is to answer questions related to Domino Data Lab which includes general AI/machine learning concepts.
                    When answering questions, only refer to the latest version of Domino. Do not use information from older versions of Domino. 
                    In your response, include the following url links at the end of your response {}.
                    Here is some relevant context: {}""".format(", ".join(url_links), ". ".join(context))

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
            response = generate_response(st.session_state.messages[-1]["content"])
            st.write(response)

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
