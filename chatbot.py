import streamlit as st
import os
import json
import requests
import pandas as pd
 
API_TOKEN = os.environ['HUGGING_FACE_API_TOKEN']
API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"    
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# App title
st.set_page_config(page_title="🤖💬 Your Personal Chat Assistant")
 
# App sidebar
with st.sidebar:
    st.title('🤖💬 Your Personal Chat Assistant')
    if API_TOKEN ==  "":
        API_TOKEN = st.text_input('Enter Hugging Face API Token:', type='password')
        if not (API_TOKEN):
            st.warning('Please enter your Hugging Face API Token', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    else:
        st.success('Hugging Face API Token provided!', icon='✅')
    
    model = st.radio(
    "Select a model :point_down:",
    ('Microsoft DialoGPT-large', 'Your Domino Hosted Prediction API'))

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

# Query the microsoft DialoGPT
def query_DialoGPT_model(prompt_input, past_user_inputs = None, generated_responses = None):
    
    payload = {
        "inputs": {
            "past_user_inputs": past_user_inputs,
            "generated_responses": generated_responses,
            "text": prompt_input,
        },
    }
    
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    response = json.loads(response.content.decode("utf-8"))
    response.pop("warnings")
    return response.get('generated_text')        
        
# Query your own internal Model API
def query_model_api_in_Domino(prompt_input):
    response = requests.post("https://ragbot33570.cs.domino.tech:443/models/659c4c13d70be42e6cecfa75/latest/model",
        auth=(
            "RBI4FYX4AVzFgvSsc5PPcHzuZaLBGao8c6VarNAMEgnpYvLl7bqy1QT8sfbFYR21",
            "RBI4FYX4AVzFgvSsc5PPcHzuZaLBGao8c6VarNAMEgnpYvLl7bqy1QT8sfbFYR21"
        ),
        json={
          "data": {
            "input_string": prompt_input
          }
        }
    )
    return response.json().get('result')[0]

# Function for generating LLM response
def generate_response(prompt):
    if model == "Microsoft DialoGPT-large":
        response_generated = query_DialoGPT_model(prompt)
    elif model == "Your Domino Hosted Prediction API":
        response_generated = query_model_api_in_Domino(prompt)
    return response_generated


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
            
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)