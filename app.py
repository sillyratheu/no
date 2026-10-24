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
        tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", token=HF_TOKEN)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        model_file_path = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors", token=HF_TOKEN)
        
        vocab_size = tokenizer.get_vocab_size()
        model = ShahajjoBrain(vocab_size=vocab_size, embed_dim=256, num_layers=4, nhead=4)
        
        state_dict = load_file(model_file_path)
        model.load_state_dict(state_dict)
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return tokenizer, model, device, None
    except Exception as e:
        return None, None, None, str(e)

tokenizer, model, device, error_msg = load_all_from_hf()

# URL Query Parameters ধরা
query_params = st.query_params
text_input = query_params.get("text", None)
output_format = query_params.get("format", None) # format=json চেক করার জন্য

# ১. যদি 외부 App বা API থেকে রিকোয়েস্ট আসে (text এবং format=json থাকলে)
if text_input and output_format == "json":
    if model and tokenizer:
        try:
            encoded = tokenizer.encode(text_input)
            input_ids = encoded.ids if encoded.ids else [0]
            
            prompt_length = len(input_ids)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            max_new_tokens = 40  
            temperature = 0.7    
            repetition_penalty = 1.35  
            
            pad_id = tokenizer.token_to_id("<pad>")
            eos_id = tokenizer.token_to_id("</s>")
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if input_tensor.size(1) > 256:
                        break
                        
                    outputs = model(input_tensor)
                    next_token_logits = outputs[:, -1, :] / temperature
                    
                    for token_id in set(input_tensor[0].tolist()):
                        if next_token_logits[0, token_id] < 0:
                            next_token_logits[0, token_id] *= repetition_penalty
                        else:
                            next_token_logits[0, token_id] /= repetition_penalty
                    
                    top_k = 30
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    
                    token_val = next_token_id.item()
                    if (pad_id is not None and token_val == pad_id) or (eos_id is not None and token_val == eos_id):
                        break
                        
                    input_tensor = torch.cat([input_tensor, next_token_id], dim=1)
            
            generated_ids = input_tensor[0][prompt_length:].tolist()
            decoded_reply = tokenizer.decode(generated_ids).strip()
            
            if not decoded_reply:
                decoded_reply = tokenizer.decode(input_tensor[0].tolist()).strip()
            
            result = {
                "status": "success",
                "response": decoded_reply
            }
        except Exception as e:
            result = {"status": "error", "response": f"Generation error: {str(e)}"}
    else:
        result = {"status": "error", "response": f"Failed to load: {error_msg}"}
    
    # সংক্ষেপে শুধু JSON প্রিন্ট করে পড়া সহজ করা
    st.code(json.dumps(result), language="json")
    st.stop()

# ==========================================
# ২. সাধারণ ইউজারের জন্য Streamlit UI
# ==========================================
st.title("🇧🇩 Shahajjo LLM Backend")

if model and tokenizer:
    st.success("ShahajjoBrain Model Ready!")
    
    # ইন্টারঅ্যাক্টিভ চ্যাট বাক্সে টেস্ট করার অপশন
    user_prompt = st.text_input("মেসেজ লিখুন:", value=text_input if text_input else "")
    if st.button("Generate") and user_prompt:
        with st.spinner("চিন্তা করছে..."):
            encoded = tokenizer.encode(user_prompt)
            input_ids = encoded.ids if encoded.ids else [0]
            prompt_length = len(input_ids)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                for _ in range(40):
                    if input_tensor.size(1) > 256:
                        break
                    outputs = model(input_tensor)
                    next_token_logits = outputs[:, -1, :] / 0.7
                    
                    top_k = 30
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    
                    token_val = next_token_id.item()
                    if token_val == tokenizer.token_to_id("</s>"):
                        break
                    input_tensor = torch.cat([input_tensor, next_token_id], dim=1)
            
            generated_ids = input_tensor[0][prompt_length:].tolist()
            reply = tokenizer.decode(generated_ids).strip()
            st.write("**উত্তর:**", reply)
else:
    st.error(f"Error loading model: {error_msg}")
