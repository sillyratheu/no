import os
import json
import torch
import torch.nn as nn
import streamlit as st
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

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
# Exact Transformer Architecture from Your Training Script
# ---------------------------------------------------------
class ShahajjoBrainConfig:
    def __init__(self, vocab_size=40000, hidden_size=512, num_hidden_layers=6, num_attention_heads=8, intermediate_size=2048):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

class ShahajjoBrainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
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

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        out = self.transformer_encoder(x)
        logits = self.fc(out)
        return logits

# ---------------------------------------------------------
# Custom Vocab Tokenizer Fallback
# ---------------------------------------------------------
class CustomVocabTokenizer:
    def __init__(self, vocab_dict):
        self.vocab = vocab_dict
        self.id_to_token = {v: k for k, v in vocab_dict.items()}
        self.unk_id = self.vocab.get("<unk>", 0)
        self.pad_id = self.vocab.get("<pad>", 1)

    def encode(self, text):
        tokens = text.split()
        ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(idx, "<unk>") for idx in ids if idx != self.pad_id]
        return " ".join(tokens)

# ---------------------------------------------------------
# Load Model & Tokenizer (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_shahajjo_pipeline():
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", token=HF_TOKEN)
        vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json", token=HF_TOKEN)
        
        try:
            tok_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", token=HF_TOKEN)
            raw_tokenizer = Tokenizer.from_file(tok_path)
            
            class FastTokenizerWrapper:
                def __init__(self, tok):
                    self.tok = tok
                def encode(self, text):
                    return self.tok.encode(text).ids
                def decode(self, ids):
                    return self.tok.decode(ids)
                    
            tokenizer = FastTokenizerWrapper(raw_tokenizer)
        except Exception:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_dict = json.load(f)
            tokenizer = CustomVocabTokenizer(vocab_dict)

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

if error_msg:
    st.error(f"❌ Model failed to load: {error_msg}")
    st.json({"status": "error", "response": "Model not loaded properly."})
    st.stop()

# ---------------------------------------------------------
# Text Generation Logic
# ---------------------------------------------------------
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [0]
        
    generated = list(input_ids)
    
    with torch.no_grad():
        for _ in range(max_length):
            curr_input = torch.tensor([generated[-256:]], device=device, dtype=torch.long)
            logits = model(curr_input)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            generated.append(next_token)
            
            decoded_word = tokenizer.decode([next_token])
            if decoded_word in ["</s>", "<|endoftext|>"]:
                break
                
    return tokenizer.decode(generated)

# ---------------------------------------------------------
# Streamlit Chat UI
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
