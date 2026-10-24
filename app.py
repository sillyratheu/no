import os
import json
import torch
import torch.nn as nn
import streamlit as st
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Shahajjo Dicode Chat",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Shahajjo Dicode Chat")
st.caption("Vidclip Organization | Coding Assistant Model (Transformer)")

REPO_ID = "Vidclip/Shahajjo_Dicode"
HF_TOKEN = st.secrets["HF_TOKEN"]

# ---------------------------------------------------------
# Safe Tokenizer (Handles Lists, Dicts, and Integers)
# ---------------------------------------------------------
class SafeCustomTokenizer:
    def __init__(self, vocab_dict):
        self.vocab = vocab_dict
        self.id_to_token = {}
        
        for k, v in vocab_dict.items():
            idx = v
            if isinstance(v, list):
                idx = v[0] if v else 0
            elif isinstance(v, dict):
                idx = v.get("id", v.get("index", 0))
            if isinstance(idx, int):
                self.id_to_token[idx] = k
                
        self.unk_id = 1
        self.pad_id = 0

    def encode(self, text):
        tokens = text.split()
        ids = []
        for token in tokens:
            val = self.vocab.get(token, self.vocab.get("<unk>", 1))
            if isinstance(val, list): val = val[0] if val else 1
            elif isinstance(val, dict): val = val.get("id", 1)
            if isinstance(val, int):
                ids.append(val)
        return ids if ids else [1]

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx == self.pad_id:
                continue
            token = self.id_to_token.get(idx, "")
            if token and token not in ["<pad>", "<unk>", "</s>", "<|endoftext|>"]:
                tokens.append(token)
        return " ".join(tokens)

# ---------------------------------------------------------
# Load Model, Config & Tokenizer (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_dicode_pipeline():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", token=HF_TOKEN)
        config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json", token=HF_TOKEN)
        vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json", token=HF_TOKEN)
        
        # Read exact config parameters from config.json
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        vocab_size = cfg.get("vocab_size", 40000)
        hidden_size = cfg.get("hidden_size", 512)
        num_hidden_layers = cfg.get("num_hidden_layers", 6)
        num_attention_heads = cfg.get("num_attention_heads", 8)
        intermediate_size = cfg.get("intermediate_size", 2048)

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
            
        tokenizer = SafeCustomTokenizer(vocab_dict)

        class SafeCustomTokenizer:
    def __init__(self, vocab_dict):
        self.vocab = vocab_dict
        self.id_to_token = {}
        
        for k, v in vocab_dict.items():
            idx = v
            if isinstance(v, list):
                idx = v[0] if v else 0
            elif isinstance(v, dict):
                idx = v.get("id", v.get("index", 0))
            if isinstance(idx, int):
                self.id_to_token[idx] = k
                
        self.unk_id = 1
        self.pad_id = 0

    def encode(self, text):
        tokens = text.split()
        ids = []
        for token in tokens:
            val = self.vocab.get(token, self.vocab.get("<unk>", 1))
            if isinstance(val, list): val = val[0] if val else 1
            elif isinstance(val, dict): val = val.get("id", 1)
            if isinstance(val, int):
                ids.append(val)
        return ids if ids else [1]

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx == self.pad_id:
                continue
            # Jodi token na paoa jay, tahole r blank na rekhe safety token dibe
            token = self.id_to_token.get(idx, f"token_{idx}")
            tokens.append(token)
        return " ".join(tokens)

# ---------------------------------------------------------
# Generation Logic with Loop Protection
# ---------------------------------------------------------
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    
    with torch.no_grad():
        for _ in range(max_length):
            curr_input = torch.tensor([generated[-256:]], device=device, dtype=torch.long)
            logits = model(curr_input)
            next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
            
            if generated.count(next_token) > 3:
                break
                
            generated.append(next_token)
                
    return tokenizer.decode(generated)

# ---------------------------------------------------------
# Streamlit Chat UI
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Code likhte ki sahajjo lagbe? (Shahajjo Dicode)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Shahajjo Dicode code likhche..."):
            response = generate_text(prompt)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
