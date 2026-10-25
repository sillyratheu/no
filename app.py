import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Shahajjo LLM", page_icon="🇧🇩")
st.title("🇧🇩 Shahajjo - Heavy Bengali Transformer LLM")
st.write("A self-trained heavy generative LLM (Hidden Size: 512, Vocab: 8000) running on Hugging Face & Streamlit.")

# Load Tokenizer & Model from HF Repo
@st.cache_resource
def load_model_and_tokenizer():
    REPO_ID = "Vidclip/Shahajjo"
    
    # Download files if needed
    hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", repo_type="model", local_dir=".")
    hf_hub_download(repo_id=REPO_ID, filename="model.pth", repo_type="model", local_dir=".")
    
    tokenizer = Tokenizer.from_file("tokenizer.json")
    
    # Updated Heavy Architecture config
    VOCAB_SIZE = 8000
    HIDDEN_SIZE = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    INTERMEDIATE_SIZE = 2048
    MAX_POS = 128
    
    class ShahajjoHeavyTransformer(nn.Module):
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
    model = ShahajjoHeavyTransformer().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    
    return model, tokenizer, device, VOCAB_SIZE, MAX_POS

with st.spinner("Heavy Model & Pro Tokenizer load hocche... ⏳"):
    model, tokenizer, device, VOCAB_SIZE, MAX_POS = load_model_and_tokenizer()

# Chat UI
user_prompt = st.text_input("Tomar ki jiggesh korar ache?", "হাই")

if st.button("Generate Response 🚀"):
    if user_prompt:
        inputs = tokenizer.encode(user_prompt)
        input_ids = [min(tok_id, VOCAB_SIZE - 1) for tok_id in inputs.ids]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        generated_ids = input_ids.copy()
        temperature = 0.4
        repetition_penalty = 1.6  # রিপিটেশন এবং লুপ এড়ানোর জন্য স্ট্রং পেনাল্টি
        top_k = 50                # গিব্রিশ বা আবলতাবল টোকেন ফিল্টার করার জন্য
        
        with torch.no_grad():
            for _ in range(35):
                context = input_tensor[:, -MAX_POS:]
                logits = model(context)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply Repetition Penalty
                for token_id in set(generated_ids):
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty
                
                # Apply Top-K Filtering
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k)
                    min_allowed = values[-1]
                    next_token_logits[next_token_logits < min_allowed] = -float('Inf')
                
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