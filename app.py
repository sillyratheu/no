import os
import json
import torch
import re
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import uvicorn

app = FastAPI(title="Shahajjo AI Permanent API")

REPO_ID = "Vidclip/Shahajjo"
HF_TOKEN = os.environ.get("HF_TOKEN")

print("📥 Downloading model configuration and weights from Hugging Face...")
try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", token=HF_TOKEN)
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json", token=HF_TOKEN)
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.json", token=HF_TOKEN)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    
    stoi = vocab_data["stoi"] if "stoi" in vocab_data else vocab_data
    itos = vocab_data["itos"] if "itos" in vocab_data else {str(v): k for k, v in stoi.items()}
    PAD_IDX = stoi.get("<pad>", 0)
    print("✅ Config and Vocabulary loaded successfully!")
except Exception as e:
    print(f"⚠️ Error downloading files: {e}")

class ShahajjoMassiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer(emb, src_key_padding_mask=(x == PAD_IDX))
        return self.fc_out(out)

# Model Initialization
vocab_size = config.get("vocab_size", len(stoi))
model = ShahajjoMassiveTransformer(
    vocab_size=vocab_size,
    d_model=config.get("d_model", 512),
    nhead=config.get("nhead", 8),
    num_layers=config.get("num_layers", 6)
)

if 'model_path' in locals() and os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ PyTorch Model loaded successfully on CPU memory!")

class QueryRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "Active", "message": "Shahajjo AI Permanent API is running smoothly!"}

@app.post("/generate")
def generate_text(request: QueryRequest):
    try:
        tokens = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z0-9]+|[^\s\w]', request.prompt.strip())
        input_ids = [stoi.get(t, stoi.get("<unk>", 1)) for t in tokens]
        if not input_ids:
            input_ids = [stoi.get("<sos>", 2)]
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_ids = torch.argmax(outputs, dim=-1).squeeze(0).tolist()
            
        response_words = [itos.get(str(idx), "") for idx in predicted_ids if idx not in [PAD_IDX, stoi.get("<sos>", 2), stoi.get("<eos>", 3)]]
        response_text = " ".join(response_words)
        
        return {"response": response_text if response_text else "Namaskar! Ami Shahajjo AI."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)