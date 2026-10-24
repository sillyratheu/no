import os
import json
import torch
import torch.nn as nn
import streamlit as st
from huggingface_hub import hf_hub_download
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Shahajjo LLM Ecosystem",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Shahajjo LLM Chat")
st.caption("Vidclip Organization | Local Resource Friendly Model")

# Repository Details & Token from Streamlit Secrets
REPO_ID = "Vidclip/Shahajjo"
HF_TOKEN = st.secrets["HF_TOKEN"]

# ---------------------------------------------------------
# Architecture Definition (Exact ShahajjoBrain Standard)
# ---------------------------------------------------------
class ShahajjoBrainConfig(PretrainedConfig):
    model_type = "shahajjo_brain"
    def __init__(self, vocab_size=40000, hidden_size=512, num_hidden_layers=6, num_attention_heads=8, intermediate_size=2048, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

class ShahajjoBrainModel(PreTrainedModel):
    config_class = ShahajjoBrainConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        out = self.transformer_encoder(x)
        logits = self.fc(out)
        return logits

# ---------------------------------------------------------
# Load Model & Tokenizer (Cached for performance)
# ---------------------------------------------------------
@st.cache_resource
def load_shahajjo_pipeline():
    try:
        # Download files from Hugging Face Hub using Streamlit Secret Token
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", token=HF_TOKEN)
        tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", token=HF_TOKEN)
        
        # Load Tokenizer
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.pad_token = "<pad>"
        
        # Initialize & Load Model Weights
        config = ShahajjoBrainConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ShahajjoBrainModel(config)
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, tokenizer, device, None
    except Exception as e:
        return None, None, None, str(e)

model, tokenizer, device, error_msg = load_shahajjo_pipeline()

# Error handling in UI
if error_msg:
    st.error(f"❌ Model failed to load: {error_msg}")
    st.json({"status": "error", "response": "Model not loaded properly."})
    st.stop()

# ---------------------------------------------------------
# Generation Logic
# ---------------------------------------------------------
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generated = input_ids[0].tolist()
    
    with torch.no_grad():
        for _ in range(max_length):
            curr_input = torch.tensor([generated[-256:]], device=device)
            logits = model(curr_input)
            next_token_logits = logits[0, -1, :]
            
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated.append(next_token)
            
            if tokenizer.decode([next_token]) in ["</s>", "<|endoftext|>"]:
                break
                
    return tokenizer.decode(generated, skip_special_tokens=True)

# ---------------------------------------------------------
# Streamlit Interface
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Shahajjo LLM-er sathe kotha bolun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Shahajjo chinta korche..."):
            response = generate_text(prompt)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
