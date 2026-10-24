import streamlit as st
import torch
import torch.nn as nn
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

# Streamlit secrets theke token nawa
HF_TOKEN = st.secrets.get("HF_TOKEN", "")
REPO_ID = "vidclip/Shahajjo"

st.set_page_config(page_title="Shahajjo LLM Backend", layout="centered")

# ==========================================
# ShahajjoBrain Architecture
# ==========================================
class ShahajjoBrain(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, nhead=4):
        super(ShahajjoBrain, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len, x.device)
        embedded = self.embedding(x)
        out = self.transformer_encoder(embedded, mask=mask)
        logits = self.fc(out)
        return logits

@st.cache_resource
def load_all_from_hf():
    try:
        # 1. Native Tokenizer download & load (tokenizer.json)
        tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", token=HF_TOKEN)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 2. Model weights download using safetensors (model.safetensors)
        model_file_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors", token=HF_TOKEN)
        
        # 3. Model initialize kora vocab size diye
        vocab_size = tokenizer.get_vocab_size()
        model = ShahajjoBrain(vocab_size=vocab_size, embed_dim=256, num_layers=4, nhead=4)
        
        # Safetensors diye state_dict load kora
        state_dict = load_file(model_file_path)
        model.load_state_dict(state_dict)
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return tokenizer, model, device, None
    except Exception as e:
        return None, None, None, str(e)

tokenizer, model, device, error_msg = load_all_from_hf()

# Render / Web Request dhora (?text=hi)
query_params = st.query_params
text_input = query_params.get("text", None)

if text_input:
    if model and tokenizer:
        try:
            # Native tokenizer diye text encode kora
            encoded = tokenizer.encode(text_input)
            input_ids = encoded.ids
            if not input_ids:
                input_ids = [0]
            
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            max_new_tokens = 50  
            temperature = 0.7    
            repetition_penalty = 1.2  # বারবার একই শব্দ বা লুপ আটকাতে পেনাল্টি
            top_k = 40                # সেরা K টি টোকেন ফিল্টার করতে
            
            pad_id = tokenizer.token_to_id("<pad>")
            eos_id = tokenizer.token_to_id("</s>")
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # সিকোয়েন্স অনেক বড় হয়ে গেলে ক্র্যাশ এড়াতে চেক
                    if input_tensor.size(1) > 512:
                        break
                        
                    outputs = model(input_tensor)
                    next_token_logits = outputs[:, -1, :] / temperature
                    
                    # --- Repetition Penalty লজিক ---
                    for token_id in set(input_tensor[0].tolist()):
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty
                            
                    # --- Top-K Sampling ফিল্টারিং ---
                    v, idx = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    
                    # Pad বা EOS টোকেন পেলে থেমে যাবে
                    token_val = next_token_id.item()
                    if (pad_id is not None and token_val == pad_id) or (eos_id is not None and token_val == eos_id):
                        break
                        
                    input_tensor = torch.cat([input_tensor, next_token_id], dim=1)
            
            # Output token ids ke decode kore clean kora
            output_ids = input_tensor[0].tolist()
            decoded = tokenizer.decode(output_ids).strip()
            
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
    
    # Render-er BeautifulSoup / API Gateway-er jonno JSON output
    st.code(json.dumps(result), language="json")
    st.stop()

# Streamlit normal web page UI
st.title("🇧🇩 Shahajjo LLM Backend")
if model and tokenizer:
    st.success("ShahajjoBrain & Native BPE Tokenizer loaded successfully via safetensors!")
else:
    st.error(f"Error loading model from Hugging Face: {error_msg}")
