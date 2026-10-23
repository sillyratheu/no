import streamlit as st
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "vidclip/Shahajjo"

st.set_page_config(page_title="Shahajjo LLM Backend", layout="centered")

@st.cache_resource
def load_llm():
    try:
        tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            REPO_ID, 
            token=HF_TOKEN, 
            torch_dtype=torch.float32, 
            low_cpu_mem_usage=True
        )
        return tokenizer, model
    except Exception as e:
        print(f"Model load error: {e}")
        return None, None

tokenizer, model = load_llm()

# Streamlit link-e query parameter dhora (jemon: https://shahajjo.streamlit.app/?text=hi)
query_params = st.query_params
text_input = query_params.get("text", None)

if text_input:
    if model and tokenizer:
        try:
            inputs = tokenizer(text_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7
                )
            
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
        result = {"status": "error", "response": "Model or Tokenizer failed to load."}
    
    # Render-er BeautifulSoup/gateway er jonno pure JSON output <code` block-e
    st.code(json.dumps(result), language="json")
    st.stop()

# Normal Streamlit Web UI (jodi keu direct browser-e link open kore)
st.title("🇧🇩 Shahajjo LLM Backend")
if model and tokenizer:
    st.success("LLM Model & PyTorch loaded successfully in Streamlit!")
else:
    st.warning("Loading LLM model from Hugging Face...")
