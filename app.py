import streamlit as st
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from safetensors.torch import load_file
import os

st.title("🤖 ShahajjoBrain - Custom Multilingual LLM")
st.write("Pure native tokenizer, safetensors ebong URL Query Parameter support-er sathe toiri Streamlit app!")

@st.cache_resource
def load_resources():
    tokenizer_path = "tokenizer_dir/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        return None, None
        
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    class ShahajjoBrain(nn.Module):
        def __init__(self, vocab_size, embed_dim=256, num_layers=4, nhead=4):
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

    vocab_size = tokenizer.get_vocab_size()
    model = ShahajjoBrain(vocab_size=vocab_size, embed_dim=256, num_layers=4, nhead=4)
    
    model_path = "model.safetensors"
    if os.path.exists(model_path):
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict)
    else:
        return None, None
        
    model.eval()
    return model, tokenizer

with st.spinner("Model ebong tokenizer load hocche..."):
    model, tokenizer = load_resources()

if model and tokenizer:
    # URL theke query parameter nawa (jehetu bolecho: /?text=prompt)
    query_params = st.query_params
    url_prompt = query_params.get("text", "")

    # Jodi URL-e text thake, tahole oita auto input-e bose jabe
    prompt = st.text_input("Apnar query ba prompt likhun:", value=url_prompt)

    # Jodi URL-e prompt thake ba user button e click kore
    if url_prompt or st.button("Generate Response"):
        target_prompt = url_prompt if url_prompt else prompt
        if target_prompt.strip():
            # Pure native tokenizer encoding
            encoded = tokenizer.encode(target_prompt)
            input_ids = encoded.ids
            generated_ids = list(input_ids)
            
            # Generation loop
            with torch.no_grad():
                for _ in range(25): # Max 25 tokens generate korbe
                    input_tensor = torch.tensor([generated_ids]).long()
                    outputs = model(input_tensor)
                    next_token_logits = outputs[0, -1, :]
                    next_token_id = torch.argmax(next_token_logits).item()
                    generated_ids.append(next_token_id)
                    
            # Decode generated tokens back to text
            response_text = tokenizer.decode(generated_ids)
            st.subheader("ShahajjoBrain Response:")
            st.write(response_text)
        else:
            st.warning("Prothom e ekti prompt ba query likhun!")
else:
    st.error("Model ba Tokenizer file pawa jayni! Prothomay training script run kore file gulo toiri kore nao.")
