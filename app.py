import streamlit as st
import os
import json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

st.set_page_config(page_title="Shahajjo AI Backend", layout="centered")

REPO_ID = "vidclip/Shahajjo"

@st.cache_resource
def load_tokenizer():
    files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for file in files:
        if not os.path.exists(file):
            try:
                hf_hub_download(repo_id=REPO_ID, filename=file, local_dir=".")
            except Exception as e:
                print(f"Download error: {e}")
                
    if os.path.exists("tokenizer.json"):
        try:
            return Tokenizer.from_file("tokenizer.json")
        except Exception as e:
            print(f"Tokenizer error: {e}")
    return None

tokenizer = load_tokenizer()

# Render API Gateway theke asha query params check kora
query_params = st.query_params
text_input = query_params.get("text", None)

if text_input:
    if tokenizer:
        try:
            encoded = tokenizer.encode(text_input)
            decoded = tokenizer.decode(encoded.ids)
            result = {
                "status": "success",
                "response": decoded,
                "reply": decoded,
                "message": decoded,
                "text": decoded
            }
        except Exception as e:
            result = {"status": "error", "response": f"Tokenizer error: {str(e)}"}
    else:
        result = {"status": "error", "response": "Tokenizer files are still loading."}
    
    # Clean HTML div tag wrapping so Render can parse it easily
    st.markdown(f'<div id="api-json-response">{json.dumps(result)}</div>', unsafe_allow_html=True)
    st.stop()

# Normal Streamlit UI
st.title("🇧🇩 Shahajjo AI Backend Server")
st.write("24/7 Processing Brain for Render API Gateway")
if tokenizer:
    st.success("Tokenizer loaded successfully!")
else:
    st.warning("Loading tokenizer files...")