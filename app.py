import streamlit as st
import os
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

st.set_page_config(page_title="Shahajjo AI Backend", layout="centered")

REPO_ID = "vidclip/Shahajjo"

@st.cache_resource
def load_model_and_tokenizer():
    files = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
    for file in files:
        if not os.path.exists(file):
            try:
                hf_hub_download(repo_id=REPO_ID, filename=file, local_dir=".")
            except Exception as e:
                print(f"Download error: {e}")
                
    tokenizer = None
    if os.path.exists("tokenizer.json"):
        try:
            tokenizer = Tokenizer.from_file("tokenizer.json")
        except Exception as e:
            print(f"Tokenizer error: {e}")
            
    return tokenizer

tokenizer = load_model_and_tokenizer()

# Safe query params check for Streamlit
query_params = st.query_params
text_input = query_params.get("text", None)

if text_input:
    if tokenizer is not None:
        try:
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
        except Exception as e:
            st.json({
                "status": "error",
                "response": f"Tokenizer error: {str(e)}"
            })
    else:
        st.json({
            "status": "error",
            "response": "Tokenizer files are still loading on Streamlit server."
        })
    st.stop()

# Normal UI (Jokhon keu direct browser-e visit korbe)
st.title("🇧🇩 Shahajjo AI Backend Server")
st.write("This Streamlit app runs 24/7 as the backend brain for Render API.")
if tokenizer:
    st.success("Tokenizer loaded successfully!")
else:
    st.warning("Loading tokenizer files from Hugging Face...")
