import os
import json
import torch
import torch.nn as nn
import streamlit as st
from huggingface_hub import hf_hub_download, login

# Page configuration
st.set_page_config(page_title="Shahajjo AI | Chat Interface", page_icon="🤖", layout="centered")

st.title("🤖 Sahajjo AI - Indigenous LLM")
st.write("Bangla ebong English-er jonno toiri tomar nijsho custom Transformer model!")

REPO_ID = "Vidclip/Shahajjo"
HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN", None)

# Fixed Model Architecture Class
class ShahajjoMassiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        # Direct transformer forward pass without buggy padding mask tensor conversion
        out = self.transformer(emb)
        return self.fc_out(out)

# Cache resource to load model & tokenizer only once
@st.cache_resource
def load_model_and_tokenizer():
    try:
        if HF_TOKEN:
            login(token=HF_TOKEN)
        
        with st.spinner("📥 Downloading model files from Hugging Face..."):
            model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth")
            config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
            vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json")
        
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

        model = ShahajjoMassiveTransformer(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers, pad_idx=PAD_IDX
        )
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            
        return model, stoi, itos, PAD_IDX
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

model, stoi, itos, PAD_IDX = load_model_and_tokenizer()

if model is None:
    st.stop()

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Sahajjo AI ke kichu jiggesh koro..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate model response
    with st.chat_message("assistant"):
        with st.spinner("Chinta korche..."):
            try:
                tokens = prompt.strip().split()
                max_vocab_size = model.embedding.num_embeddings
                
                input_ids = []
                for t in tokens:
                    idx = stoi.get(t, stoi.get("<unk>", 1))
                    if not isinstance(idx, int) or idx < 0 or idx >= max_vocab_size:
                        idx = 1
                    input_ids.append(idx)
                
                if not input_ids:
                    input_ids = [stoi.get("<sos>", 2)]
                    
                generated_ids = list(input_ids)
                
                with torch.no_grad():
                    for _ in range(25):  # Max 25 tokens
                        input_tensor = torch.tensor([generated_ids], dtype=torch.long)
                        outputs = model(input_tensor)
                        next_token_logits = outputs[0, -1, :]
                        next_token_id = torch.argmax(next_token_logits).item()
                        
                        if next_token_id < 0 or next_token_id >= max_vocab_size:
                            next_token_id = 1
                            
                        if next_token_id == stoi.get("<eos>", 3):
                            break
                        generated_ids.append(next_token_id)
                    
                response_words = [itos.get(str(idx), "") for idx in generated_ids]
                response_text = " ".join([w for w in response_words if w not in ["<pad>", "<sos>", "<eos>"]])
                
                if not response_text.strip():
                    response_text = "Namaskar! Ami Shahajjo AI, apnake sahajjo korte prostut achhi."
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                err_msg = f"Generation error: {str(e)}"
                st.error(err_msg)
