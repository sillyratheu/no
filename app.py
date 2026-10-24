import os
import json
import torch
import torch.nn as nn
import streamlit as st
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Shahajjo Heavy LLM Chat",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Shahajjo Heavy LLM Chat")
st.caption("Vidclip Organization | Shahajjo Heavy Brain (Safetensors)")

# Correct Independent Repository ID
REPO_ID = "Vidclip/shahajjo_heavy_brain"
HF_TOKEN = st.secrets["HF_TOKEN"]

# ---------------------------------------------------------
# Load Model, Config & Tokenizer from the Repo Root
# ---------------------------------------------------------
@st.cache_resource
def load_shahajjo_pipeline():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors", token=HF_TOKEN)
        config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json", token=HF_TOKEN)
        tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", token=HF_TOKEN)
        
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        vocab_size = cfg.get("vocab_size", 65000)
        hidden_size = cfg.get("hidden_size", 768)
        num_hidden_layers = cfg.get("num_hidden_layers", 8)
        num_attention_heads = cfg.get("num_attention_heads", 12)
        intermediate_size = cfg.get("intermediate_size", 3072)

        class ShahajjoHeavyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    batch_first=True,
                    dropout=0.1
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
                self.fc = nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids):
                x = self.embedding(input_ids)
                out = self.transformer_encoder(x)
                logits = self.fc(out)
                return logits

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ShahajjoHeavyModel()
        
        state_dict = load_file(model_path, device=str(device))
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, tokenizer, device, None
    except Exception as e:
        return None, None, None, str(e)

model, tokenizer, device, error_msg = load_shahajjo_pipeline()

if error_msg:
    st.error(f"Model failed to load: {error_msg}")
    st.json({"status": "error", "response": "Model not loaded properly."})
    st.stop()

# ---------------------------------------------------------
# Text Generation Logic
# ---------------------------------------------------------
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    if input_ids.numel() == 0:
        input_ids = torch.tensor([[0]], device=device)
        
    generated = list(input_ids[0].cpu().numpy())
    
    with torch.no_grad():
        for _ in range(max_length):
            curr_input = torch.tensor([generated[-256:]], device=device, dtype=torch.long)
            logits = model(curr_input)
            next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
            
            if generated.count(next_token) > 4:
                break
                
            generated.append(next_token)
                
    return tokenizer.decode(generated)

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Shahajjo Heavy-r sathe kotha bolun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Shahajjo chinta korche..."):
            response = generate_text(prompt)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
