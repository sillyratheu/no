import streamlit as st
import os
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

st.set_page_config(page_title="Shahajjo AI Backend", layout="centered")

# Hugging Face Repository details
REPO_ID = "vidclip/Shahajjo"

# Model files automatic download kora jodi local-e na thake
@st.cache_resource
def load_model_and_tokenizer():
    files = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
    for file in files:
        if not os.path.exists(file):
            try:
                hf_hub_download(repo_id=REPO_ID, filename=file, local_dir=".")
            except Exception as e:
                print(f"Download error for {file}: {e}")
                
    tokenizer = None
    if os.path.exists("tokenizer.json"):
        try:
            tokenizer = Tokenizer.from_file("tokenizer.json")
        except Exception as e:
            print(f"Tokenizer error: {e}")
            
    return tokenizer

tokenizer = load_model_and_tokenizer()

# Jodi Render gateway theke query parameter 'text' ashe, tahole shudhu JSON output dibe
query_params = st.query_params
if "text" in query_params:
    text_input = query_params["text"]
    
    if tokenizer is not None:
        encoded = tokenizer.encode(text_input)
        token_ids = encoded.ids
        decoded_response = tokenizer.decode(token_ids)
        
        st.json({
            "status": "success",
            "response": decoded_response,
            "reply": decoded_response,
            "message": decoded_response,
            "text": decoded_response
        })
    else:
        st.json({
            "status": "error",
            "response": "Tokenizer not loaded on Streamlit server."
        })
    st.stop() # Normal UI load howa bondho kore dibe API request er somoy

# Streamlit App er Normal UI (Jokhon keu browser theke direct visit korbe)
st.title("🇧🇩 Shahajjo AI Backend Server")
st.write("This Streamlit app runs 24/7 and acts as the heavy processing brain for Render API.")
st.info("Render API Endpoint link-ti use kore apnar app ba frontend theke kotha bolte parben.")

if tokenizer:
    st.success("Tokenizer & Model files loaded successfully!")
else:
    st.warning("Model files are downloading or missing.")
