import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO_ID = "vidclip/Shahajjo"
HIDDEN_SIZE = 512
NUM_LAYERS = 8
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048

# =============================================================
# Model Architecture
# =============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class ShahajjoBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ffn_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(self, x, mask=None):
        normed_x = self.attn_norm(x)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x, attn_mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.ffn_norm(x))
        return x

class ShahajjoCustomLLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, intermediate_size=INTERMEDIATE_SIZE):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            ShahajjoBlock(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# =============================================================
# Load Model & Tokenizer
# =============================================================
@st.cache_resource
def load_shahajjo_model():
    try:
        tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json")
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
        
        tokenizer = Tokenizer.from_file(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ShahajjoCustomLLM(vocab_size=vocab_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        return None, None, None

model, tokenizer, device = load_shahajjo_model()

def generate_response(prompt, max_tokens=50):
    if not model or not tokenizer:
        return "Model not loaded properly."
    
    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.token_to_id("</s>"):
                break
                
    return tokenizer.decode(input_ids[0].tolist())

# =============================================================
# API Endpoint Check via Query Parameters (?text=...)
# =============================================================
query_params = st.query_params
api_text = query_params.get("text", None)

if api_text:
    # Return JSON response like an API
    response_text = generate_response(api_text)
    api_output = {
        "status": "success",
        "prompt": api_text,
        "response": response_text
    }
    st.json(api_output)
else:
    # Normal Streamlit Chat UI Mode
    st.set_page_config(page_title="Shahajjo LLM", page_icon="🤖", layout="centered")
    st.title("🤖 Shahajjo AI Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask Shahajjo anything...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Shahajjo is thinking..."):
                response = generate_response(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
