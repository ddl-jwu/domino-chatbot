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

# Mapping of release to versions to filter (patch releases and different formatting)
RELEASES_MAPPING = {
    "Latest (5.9)": ["latest", "5.9", "5-9-0"],
    "5.8": ["5.8", "5-8-0"],
    "5.7": ["5.7", "5-7-4", "5-7-3", "5-7-2", "5-7-1", "5-7-0"],
    "5.6": ["5.6", "5-6-2", "5-6-1", "5-6-0"], 
    "5.5": ["5.5", "5-5-4", "5-5-3", "5-5-2", "5-5-1", "5-5-0"], 
    "5.4": ["5.4", "5-4-1", "5-4-0"],
    "5.3": ["5.3", "5-3-3", "5-3-2", "5-3-1", "5-3-0"], 
    "5.2": ["5.2", "5-2-2", "5-2-1", "5-2-0"],
    "5.1": ["5.1", "5-1-4", "5-1-3", "5-1-2", "5-1-1", "5-1-0"],
    "5.0": ["5.0", "5-0-2", "5-0-1", "5-0-0"],
    "4.6": ["4.6", "4-6-4", "4-6-3", "4-6-2", "4-6-1", "4-6-0"], 
    "4.5": ["4.5", "4-5-2", "4-5-1", "4-5-0"], 
    "4.4": ["4.4", "4-4-2", "4-4-1", "4-4-0"], 
    "4.3": ["4.3", "4-3-3", "4-3-2", "4-3-1", "4-3-0"],
    "4.2": ["4.2", "4-2"], # "4-2" seems correct based on the data
    "4.1": ["4.1", "4-1"], 
    "3.6": ["3.6", "3-6"]
}

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
# index_name = "hacktestlatestall" # All 5.9 docs as of Jan 10, 2024
# index_name = "hacktest" # Sample of 5.9 docs relating to Data Sources and Datasets as of Jan 10, 2024
index_name = "hackdocslarge" # Very nearly all docs from all versions as of Jan 11, 2024
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
    domino_docs_version, doc_category = build_sidebar()

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

# Get optional query filter based on user input
def get_query_filter(user_input):    
    filter = dict()
    # Use version from user selection
    filter["version"] = {"$in": RELEASES_MAPPING[domino_docs_version]}

    # Use category from user selection
    if not "all" in doc_category.lower():
        filter["category"] = {"$eq": doc_category}

    return filter

# Get relevant docs through vector DB
def get_relevant_docs(user_input):
    filter = get_query_filter(user_input)
    embedded_query = embeddings.embed_query(user_input)
    
    return index.query(
        vector=embedded_query,
        top_k=NUM_TEXT_MATCHES,
        include_values=True,
        include_metadata=True,
        filter=filter
    )


# Query the Open AI Model
def queryOpenAIModel(user_input, past_user_inputs=None, generate_responses=None):
    relevant_docs = get_relevant_docs(user_input)
    actual_num_matches = len(relevant_docs["matches"])
    # Get relevant URLs, filtering out repeated
    url_links = set([relevant_docs["matches"][i]["metadata"]["url"] for i in range(actual_num_matches)])
    context = [relevant_docs["matches"][i]["metadata"]["text"] for i in range(actual_num_matches)]
    
    system_prompt = """ If the user asks a question that is not related to Domino Data Lab, AI, or machine learning, respond with the following keyword: https://www.youtube.com/watch?v=dQw4w9WgXcQ. 
                    Otherwise, you are a virtual assistant for Domino Data Lab and your task is to answer questions related to Domino Data Lab which includes general AI/machine learning concepts.
                    When answering questions, only refer to the {0} version of Domino. Do not use information from other versions of Domino. If you don't find an answer to the question the user asked in the {0} version of Domino, 
                    tell them that you looked into the {0} version of Domino but the feature or capability that they're looking for likely does not exist in that version. 
                    Do not hallucinate. If you don't find an answer, you can point user to the official version of the Domino Data Lab docs here: https://docs.dominodatalab.com/.
                    In your response, include the following url links at the end of your response {1}.
                    Also, at the end of your response, ask if your response was helpful and to please file a ticket with our support team at this link if further help is needed: 
                    https://tickets.dominodatalab.com/hc/en-us/requests/new#numberOfResults=5, embedded into the words "Support Ticket".
                    Here is some relevant context: {2}""".format(domino_docs_version, ", ".join(url_links), ". ".join(context))
                    
    response = client.predict(
        endpoint="chat",
        inputs={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
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
