import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import os

st.set_page_config(page_title="Shahajjo LLM", page_icon="🤖")
st.title("🇧🇩 Shahajjo - Bengali Transformer LLM")
st.write("A self-trained lightweight generative LLM running on Hugging Face & Streamlit.")

# Load Tokenizer & Model from HF Repo safely
@st.cache_resource
def load_model_and_tokenizer():
    REPO_ID = "Vidclip/Shahajjo"
    
    try:
        # Download necessary model files if not present locally
        if not os.path.exists("tokenizer.json"):
            hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", repo_type="model", local_dir=".")
        if not os.path.exists("model.pth"):
            hf_hub_download(repo_id=REPO_ID, filename="model.pth", repo_type="model", local_dir=".")
    except Exception as e:
        st.error(f"File download error: {e}")
        return None, None, None
        
    tokenizer = Tokenizer.from_file("tokenizer.json")
    
    # Architecture config (matching your stable model setup)
    VOCAB_SIZE = tokenizer.get_vocab_size()
    HIDDEN_SIZE = 256
    NUM_LAYERS = 6
    NUM_HEADS = 8
    INTERMEDIATE_SIZE = 1024
    MAX_POS = 128
    
    class ShahajjoTransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
            self.pos_embedding = nn.Embedding(MAX_POS, HIDDEN_SIZE)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=HIDDEN_SIZE, 
                nhead=NUM_HEADS, 
                dim_feedforward=INTERMEDIATE_SIZE, 
                batch_first=True, 
                dropout=0.1
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
            self.fc_out = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

        def forward(self, input_ids):
            input_ids = torch.clamp(input_ids, min=0, max=VOCAB_SIZE - 1)
            seq_len = input_ids.size(1)
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = self.embedding(input_ids) + self.pos_embedding(positions[:, :seq_len])
            x = self.transformer(x)
            return self.fc_out(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShahajjoTransformerModel().to(device)
    
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"Model weights load error: {e}")
        
    return model, tokenizer, device

with st.spinner("Model & Tokenizer load hocche... ⏳"):
    model, tokenizer, device = load_model_and_tokenizer()

if model and tokenizer:
    # Chat UI Interface
    user_prompt = st.text_input("Tomar ki jiggesh korar ache?", "<|user|> হাই <|assistant|>")

    if st.button("Generate Response 🚀"):
        if user_prompt:
            inputs = tokenizer.encode(user_prompt)
            input_ids = [min(tok_id, tokenizer.get_vocab_size() - 1) for tok_id in inputs.ids]
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
            
            generated_ids = input_ids.copy()
            temperature = 0.7
            MAX_POS = 128
            
            with torch.no_grad():
                for _ in range(35):
                    context = input_tensor[:, -MAX_POS:]
                    logits = model(context)
                    next_token_logits = logits[0, -1, :] / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    generated_ids.append(next_token)
                    next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    input_tensor = torch.cat([input_tensor, next_tensor], dim=1)
                    
                    if next_token == 3: # EOS token
                        break
                        
            response_text = tokenizer.decode(generated_ids)
            st.success("Shahajjo-er Uttar:")
            st.write(response_text)
else:
    st.warning("Model load kora sombhob hoyni. File gulo check koro.")
