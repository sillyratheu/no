import streamlit as st
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import json
import os

st.set_page_config(page_title="Shahajjo AI", page_icon="🤖")
st.title("🤖 Shahajjo AI (Chat Interface)")
st.write("Ramim Rahman er tiri kora custom Bangla LLM model.")

REPO_ID = "Vidclip/Shahajjo"

# Streamlit Secrets theke token nibe (GitHub-e kono token thakbe na)
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = None

@st.cache_resource
def load_shahajjo_model():
    try:
        # Pass token if available
        kwarg = {"token": HF_TOKEN} if HF_TOKEN else {}
        
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", **kwarg)
        config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json", **kwarg)
        vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json", **kwarg)
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            stoi = vocab_data["stoi"] if "stoi" in vocab_data else vocab_data
            itos = vocab_data["itos"] if "itos" in vocab_data else {str(v): k for k, v in stoi.items()}

        vocab_size = config.get("vocab_size", len(stoi))
        d_model = config.get("d_model", 512)
        nhead = config.get("nhead", 8)
        num_layers = config.get("num_layers", 6)
        PAD_IDX = stoi.get("<pad>", 0)

        class ShahajjoMassiveTransformer(nn.Module):
            def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc_out = nn.Linear(d_model, vocab_size)

            def forward(self, x):
                emb = self.embedding(x)
                src_key_padding_mask = (x == PAD_IDX)
                out = self.transformer(emb, src_key_padding_mask=src_key_padding_mask)
                return self.fc_out(out)

        model = ShahajjoMassiveTransformer(vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model, stoi, itos
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, stoi, itos = load_shahajjo_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Apnake ki sahajjo korte pari?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if model is not None and stoi is not None:
            try:
                tokens = prompt.split()
                input_ids = [stoi.get(t, stoi.get("<unk>", 1)) for t in tokens]
                if not input_ids:
                    input_ids = [stoi.get("<sos>", 2)]
                
                input_tensor = torch.tensor([input_ids], dtype=torch.long)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    predicted_ids = torch.argmax(outputs, dim=-1).squeeze(0).tolist()
                
                response_words = [itos.get(str(idx), "") for idx in predicted_ids]
                response_text = " ".join([w for w in response_words if w not in ["<pad>", "<sos>", "<eos>"]])
                
                if not response_text.strip():
                    response_text = "Namaskar! Ami Shahajjo AI, apnake sahajjo korte prostut achhi."
            except Exception as e:
                response_text = f"Generation error: {e}"
        else:
            response_text = "Model load hoini thikmoto."
            
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
