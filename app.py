import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import os

# Page Config
st.set_page_config(
    page_title="Shahajjo LLM - Bengali AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 সাহায্য (Shahajjo) - Bengali Causal AI")
st.markdown("আপনার বিশ্বস্ত বাংলা এআই অ্যাসিস্ট্যান্ট (768-Hidden, Vocab 8000)। Hugging Face ও Streamlit-এর মাধ্যমে পরিচালিত।")

# Hugging Face Config
REPO_ID = "Vidclip/Shahajjo"

@st.cache_resource
def load_model_and_tokenizer():
    print("📥 Downloading model and tokenizer from Hugging Face...", flush=True)
    
    # Download model.pth and tokenizer.json from HF if not present
    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", repo_type="model")
        tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", repo_type="model")
    except Exception as e:
        st.error(f"Failed to download files from Hugging Face: {e}")
        return None, None, None

    # Load Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 768-Hidden Architecture Specifications (Must match the trained model!)
    VOCAB_SIZE = 8000
    HIDDEN_SIZE = 768  
    NUM_LAYERS = 6      
    NUM_HEADS = 12      
    INTERMEDIATE_SIZE = 3072
    MAX_POS = 128

    class ShahajjoBalancedCausalTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
            self.pos_embedding = nn.Embedding(MAX_POS, HIDDEN_SIZE)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=HIDDEN_SIZE, nhead=NUM_HEADS, dim_feedforward=INTERMEDIATE_SIZE,
                batch_first=True, dropout=0.1
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
            self.fc_out = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

        def forward(self, input_ids):
            input_ids = torch.clamp(input_ids, min=0, max=VOCAB_SIZE - 1)
            seq_len = input_ids.size(1)
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = self.embedding(input_ids) + self.pos_embedding(positions[:, :seq_len])
            
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
            x = self.transformer(x, mask=mask, is_causal=True)
            
            logits = self.fc_out(x)
            return logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShahajjoBalancedCausalTransformer().to(device)
    
    # Load trained weights securely
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, tokenizer, device

with st.spinner("🔄 সাহায্য (Shahajjo) মডেল লোড হচ্ছে, একটু অপেক্ষা করুন..."):
    model, tokenizer, device = load_model_and_tokenizer()

if model is not None and tokenizer is not None:
    st.success("✅ সাহায্য মডেল সফলভাবে লোড হয়েছে!")

    # Chat Interface State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    prompt = st.chat_input("সাহায্য-কে কিছু জিজ্ঞেস করুন...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generation Logic
        with st.chat_message("assistant"):
            with st.spinner("চিস্তা করা হচ্ছে..."):
                formatted_prompt = f"<|user|> {prompt} <|assistant|>"
                encoded = tokenizer.encode(formatted_prompt)
                input_ids = [min(t, 8000 - 1) for t in encoded.ids]
                
                # Generate tokens autoregressively
                generated_ids = input_ids.copy()
                max_new_tokens = 40
                
                for _ in range(max_new_tokens):
                    context_tensor = torch.tensor([generated_ids[-128:]], dtype=torch.long, device=device)
                    with torch.no_grad():
                        logits = model(context_tensor)
                        next_token_logits = logits[0, -1, :]
                        next_token_id = torch.argmax(next_token_logits).item()
                        
                    if next_token_id == 2:  # EOS token
                        break
                    generated_ids.append(next_token_id)
                
                # Decode response
                response_ids = generated_ids[len(input_ids):]
                response_text = tokenizer.decode(response_ids)
                
                if not response_text.strip():
                    response_text = "আমি আপনার কথাটি বুঝতে পেরেছি। বলুন, আর কী সাহায্য করতে পারি?"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.error("⚠️ মডেল লোড করতে সমস্যা হয়েছে। অনুগ্রহ করে Hugging Face-এ ফাইলগুলো চেক করুন।")
