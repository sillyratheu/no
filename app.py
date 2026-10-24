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
st.caption("Vidclip Organization | Coding Assistant Model (LSTM)")

# Repository Details & Token
REPO_ID = "Vidclip/Shahajjo_Dicode"
HF_TOKEN = st.secrets["HF_TOKEN"]

# ---------------------------------------------------------
# Exact LSTM Architecture for Dicode
# ---------------------------------------------------------
class ShahajjoDicodeConfig:
    def __init__(self, vocab_size=40000, embedding_dim=256, hidden_size=512, num_layers=2):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

class ShahajjoDicodeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

# ---------------------------------------------------------
# Ultra-Safe Tokenizer (Handles Lists, Dicts, and Integers)
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
        
        # Clean joining: check if tokens are words or characters
        return " ".join(tokens)

# ---------------------------------------------------------
# Load Model & Tokenizer (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_dicode_pipeline():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", token=HF_TOKEN)
        vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json", token=HF_TOKEN)
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)
            
        tokenizer = SafeCustomTokenizer(vocab_dict)
        config = ShahajjoDicodeConfig(vocab_size=len(vocab_dict) if len(vocab_dict) > 1000 else 40000)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ShahajjoDicodeModel(config)
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        return model, tokenizer, device, None
    except Exception as e:
        return None, None, None, str(e)

model, tokenizer, device, error_msg = load_dicode_pipeline()

if error_msg:
    st.error(f"❌ Model failed to load: {error_msg}")
    st.json({"status": "error", "response": "Model not loaded properly."})
    st.stop()

# ---------------------------------------------------------
# Generation Logic with Loop Protection
# ---------------------------------------------------------
def generate_text(prompt, max_length=60):
    input_ids = tokenizer.encode(prompt)
    generated = list(input_ids)
    
    with torch.no_grad():
        for _ in range(max_length):
            curr_input = torch.tensor([generated[-128:]], device=device, dtype=torch.long)
            logits = model(curr_input)
            next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
            
            # Prevent infinite repetition loop
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
