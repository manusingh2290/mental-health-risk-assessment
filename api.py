from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
import re
import emoji
from cryptography.fernet import Fernet
import os
from dotenv import load_dotenv

app = FastAPI()

# Allow the frontend to communicate with this API
app.add_middleware(
    CORSMiddleware, 
    # Add your local development port, and eventually your live frontend URL
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500", "https://your-future-domain.com"], 
    allow_methods=["GET", "POST"], # Restrict to only what you use
    allow_headers=["*"]
)

# ==========================================
# 1. ENCRYPTION SETUP (AES-128)
# ==========================================
# In a real production app, save this key in a secure .env file!
load_dotenv() # Loads variables from a .env file
MASTER_KEY = os.getenv("MASTER_KEY").encode()
cipher_suite = Fernet(MASTER_KEY)

# ==========================================
# 2. LOAD AI MODELS (NLP & BERT)
# ==========================================
# SpaCy for Privacy/NER
nlp = spacy.load("en_core_web_sm")

# HuggingFace Transformer for Mental Health Classification
model_path = "./final_mental_health_model_1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# ==========================================
# 3. DATA MODELS & UTILS
# ==========================================
class UserInput(BaseModel):
    text: str

class EncryptedPost(BaseModel):
    encrypted_text: str

def clean_text(text: str):
    """
    Privacy Layer: Scrubs PII (Emails, Phones, Names, Locations) 
    before the AI processes it or the Faculty reads it.
    """
    text = re.sub(r'\S+@\S+', '[EMAIL_REDACTED]', text)
    text = re.sub(r'\b\d{10}\b', '[PHONE_REDACTED]', text)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "LOC"]:
            text = text.replace(ent.text, f"[{ent.label_}_REDACTED]")
    return emoji.demojize(text, delimiters=(" [", "] "))

# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.post("/analyze")
async def analyze_text(request: UserInput):
    """
    Used by the Student App. 
    Scrubs text, generates AI prediction, and encrypts BOTH versions for Supabase.
    """
    raw_text = request.text
    sanitized_text = clean_text(raw_text)
    
    # Run the text through the fine-tuned RoBERTa model
    inputs = tokenizer(sanitized_text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Calculate probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
    top_idx = probs.index(max(probs))
    
    # Encrypt Data at Rest (End-to-End Encryption strategy)
    enc_raw = cipher_suite.encrypt(raw_text.encode()).decode()
    enc_sanitized = cipher_suite.encrypt(sanitized_text.encode()).decode()
    
    return {
        "encrypted_original": enc_raw,      
        "encrypted_sanitized": enc_sanitized,
        "top_prediction": model.config.id2label[top_idx],
        "top_confidence": round(max(probs) * 100, 2)
    }

@app.post("/decrypt")
async def decrypt_data(request: UserInput):
    """
    Helper endpoint for the frontend to read the encrypted text chunks.
    """
    try:
        decrypted = cipher_suite.decrypt(request.text.encode()).decode()
        return {"decrypted_text": decrypted}
    except:
        return {"decrypted_text": "[DECRYPTION_ERROR: Invalid or Corrupted Data]"}

@app.post("/single_analysis")
async def analyze_scrubbed_post(request: EncryptedPost):
    """
    Used by the Faculty Dashboard.
    Decrypts the sanitized text on-the-fly and generates the exact 
    probability distribution array for the Bar Chart.
    """
    try:
        # Decrypt the ALREADY-SCRUBBED text (so PII never touches the admin frontend)
        scrubbed_text = cipher_suite.decrypt(request.encrypted_text.encode()).decode()

        # Multi-Task AI Prediction
        inputs = tokenizer(scrubbed_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

        # Map precise labels to their probabilities to prevent index mismatch bugs
        distribution = []
        for idx, prob in enumerate(probs):
            label_name = model.config.id2label[idx]
            distribution.append({
                "label": label_name,
                "probability": prob
            })
            
        # Auto-sort so the highest probability is always the top bar in the graph
        distribution = sorted(distribution, key=lambda x: x["probability"], reverse=True)

        return {
            "distribution": distribution, 
            "decrypted_scrubbed_text": scrubbed_text 
        }
    except Exception as e:
        return {"error": f"Secured Analysis Failure: {str(e)}"}