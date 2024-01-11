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
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatMlflow
from langchain.schema import HumanMessage, SystemMessage
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryMemory

# Number of texts to match (may be less if no suitable match)
NUM_TEXT_MATCHES = 3

# Set MLflow experiment to use for logging
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
    domino_docs_version = build_sidebar()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Initialize or re-nitialize conversation chain
if "conversation" not in st.session_state.keys() or len(st.session_state.messages) <= 1:
    chat = ChatMlflow(
        target_uri=os.environ["DOMINO_MLFLOW_DEPLOYMENTS"],
        endpoint="chat",
    )
    st.session_state.conversation = ConversationChain(
        llm=chat,
        memory=ConversationSummaryMemory(llm=chat),
        verbose=True
    )

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
def get_query_filter(user_input, domino_docs_version):
    filter = dict()
    user_input_list = user_input.split()
    for i in range(len(user_input_list)):
        curr_word = user_input_list[i]
        # Find delimiter if it exists
        if curr_word.startswith('~'):
            curr_word = curr_word[1:]

            # In form "~key=value"
            assignment_idx = curr_word.find("=")
            if assignment_idx != -1 and assignment_idx + 1 < len(curr_word):
                metadata_key = curr_word[:assignment_idx]
                metadata_val = curr_word[assignment_idx + 1:]
                filter[metadata_key] = {"$eq": metadata_val}

            # Remove metadata filter command
            user_input_list[i] = ""
    
    # Rejoin with filter removed
    user_input = (" ").join(user_input_list)
    
    # Use version from user selection
    if "latest" in domino_docs_version.lower():
        filter["version"] = {"$eq": "latest"}
    else:
        filter["version"] = {"$eq": domino_docs_version}

    return filter

# Get relevant docs through vector DB
def get_relevant_docs(user_input, domino_docs_version):
    filter = get_query_filter(user_input, domino_docs_version)
    embedded_query = embeddings.embed_query(user_input)
    
    return index.query(
        vector=embedded_query,
        top_k=NUM_TEXT_MATCHES,
        include_values=True,
        include_metadata=True,
        filter=filter
    )

def build_system_prompt(user_input):

    # Retrieve context
    relevant_docs = get_relevant_docs(user_input, domino_docs_version)
    actual_num_matches = len(relevant_docs["matches"])
    url_links = set([relevant_docs["matches"][i]["metadata"]["url"] for i in range(actual_num_matches)])
    context = [relevant_docs["matches"][i]["metadata"]["text"] for i in range(actual_num_matches)]

    # Create prompt
    template = """ If the user asks a question that is not related to Domino Data Lab, AI, or machine learning, respond with the following keyword: https://www.youtube.com/watch?v=dQw4w9WgXcQ. 
                    Otherwise, you are a virtual assistant for Domino Data Lab and your task is to answer questions related to Domino Data Lab which includes general AI/machine learning concepts.
                    When answering questions, only refer to the {domino_docs_version} version of Domino. Do not use information from other versions of Domino.
                    In your response, include the following url links at the end of your response {url_links}.
                    Also, at the end of your response, ask if your response was helpful and to please file a ticket with our support team at this link if further help is needed: 
                    https://tickets.dominodatalab.com/hc/en-us/requests/new#numberOfResults=5, embedded into the words "Support Ticket".
                    Here is some relevant context: {context}"""

    prompt_template = PromptTemplate(
        input_variables=["domino_docs_version", "url_links", "context"],
        template=template
    )
    system_prompt = prompt_template.format(domino_docs_version=domino_docs_version, url_links=url_links, context=context)
    
    return system_prompt

# Query the Open AI Model
def queryOpenAIModel(user_input):

    system_prompt = build_system_prompt(user_input)            
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
