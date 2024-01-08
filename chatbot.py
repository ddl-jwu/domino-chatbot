import streamlit as st
import os
import json
import requests
import pandas as pd
 
# App title
st.set_page_config(page_title="🤖💬 Pippy - Your Domino Virtual Assistant")
 
# App sidebar
with st.sidebar:
    st.title('🤖💬 Pippy - Your Domino Virtual Assistant')
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = st.text_input('Enter Open AI API Token:', type='password')
        st.success('Proceed to entering your prompt message!', icon='👉')
    else:
        st.success('Open AI API Token provided!', icon='✅')

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

# Query the Open AI Model
def queryOpenAIModel(prompt_input, past_user_inputs=None, generate_responses=None):
    response = "Model training currently in progress. Please try again shortly ..."
    return response

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