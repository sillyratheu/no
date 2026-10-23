import streamlit as st
import torch
import torch.nn as nn
import json
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Streamlit secrets theke token nawa
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
REPO_ID = "vidclip/Shahajjo"

st.set_page_config(page_title="Shahajjo LLM Backend", layout="centered")

# ==========================================
# Transformer Architecture Class
# ==========================================
class ShahajjoBrain(nn.Module):
    def __init__(self, vocab_size=40000, embed_dim=512, num_layers=6, nhead=8):
        super(ShahajjoBrain, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out = self.transformer_encoder(embedded)
        logits = self.fc(out)
        return logits

@st.cache_resource
def load_all_from_hf():
    try:
        tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=HF_TOKEN)
        model_file_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", token=HF_TOKEN)
        
        vocab_size = len(tokenizer) if len(tokenizer) > 0 else 40000
        model = ShahajjoBrain(vocab_size=vocab_size, embed_dim=512, num_layers=6)
        
        state_dict = torch.load(model_file_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

tokenizer, model, error_msg = load_all_from_hf()

# Render / Web Request dhora (?text=hi)
query_params = st.query_params
text_input = query_params.get("text", None)

if text_input:
    if model and tokenizer:
        try:
            input_ids = tokenizer(text_input, return_tensors="pt")["input_ids"]
            max_new_tokens = 30  # Koto gulo new word/token generate korbe
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(input_ids)
                    # Shesh token-er logits niye next token predict kora
                    next_token_logits = outputs[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Input-er sathe notun token jure dewa
                    input_ids = torch.cat([input_ids, next_token_id], dim=1)
                    
                    # Jodi EOS (End of Sequence) pay, tahole theme jabe
                    if tokenizer.eos_token_id and next_token_id.item() == tokenizer.eos_token_id:
                        break
            
            decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            result = {
                "status": "success",
                "response": decoded,
                "reply": decoded,
                "message": decoded,
                "text": decoded
            }
        except Exception as e:
            result = {"status": "error", "response": f"Generation error: {str(e)}"}
    else:
        result = {"status": "error", "response": f"Failed to load: {error_msg}"}
    
    st.code(json.dumps(result), language="json")
    st.stop()

# Streamlit normal web page UI
st.title("🇧🇩 Shahajjo LLM Backend")
if model and tokenizer:
    st.success("Transformer Model & Tokenizer loaded successfully!")
else:
    st.error(f"Error loading model: {error_msg}")
