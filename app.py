import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import altair as alt
import emoji
import re
import spacy

# --- 1. Page Configuration ---
st.set_page_config(page_title="Mental Health NLP Dashboard", page_icon="🧠", layout="wide")
st.title("🧠 Privacy-Aware Mental Health Risk Detection")
st.markdown("""
This system analyzes text for multiple mental health risks using a fine-tuned DistilRoBERTa model. 
**Privacy Feature:** All input is processed locally. Structural PII (Emails/Phones) and Contextual PII (Names/Locations) are scrubbed prior to inference using Regex and SpaCy NER.
""")

# --- 2. Load Models (Cached for speed) ---
@st.cache_resource
def load_nlp_models():
    # Load SpaCy for Named Entity Recognition
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
        st.stop()
        
    # Load Hugging Face Model (Ensure this folder exists in the same directory)
    model_path = "./final_mental_health_model_1" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    return nlp, tokenizer, model

nlp, tokenizer, model = load_nlp_models()

# --- 3. Privacy & Preprocessing Functions ---
def scrub_pii(text):
    text = str(text)
    
    # Phase 1: Regex for strict patterns (Emails & Phones)
    text = re.sub(r'\S+@\S+', '[EMAIL_REDACTED]', text)
    text = re.sub(r'\b\d{10}\b', '[PHONE_REDACTED]', text)
    
    # Phase 2: SpaCy NER for Contextual PII (Names & Locations)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            text = text.replace(ent.text, '[NAME_REDACTED]')
        elif ent.label_ in ["GPE", "LOC"]: # GPE = Geopolitical Entity (Cities/Countries)
            text = text.replace(ent.text, '[LOCATION_REDACTED]')
            
    return text

def clean_text(text):
    text = scrub_pii(text)
    text = emoji.demojize(text, delimiters=(" [", "] "))
    return text

# --- 4. User Interface ---
st.sidebar.header("🛡️ System Logs (Privacy Monitor)")
st.sidebar.info("Watch this panel to see the automated data anonymization in real-time.")

user_input = st.text_area("Enter Patient/User Text:", height=150, placeholder="Type a message here...")

if st.button("Analyze Risk Phase", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Scrubbing PII and analyzing context..."):
            
            # Step A: Clean Input
            safe_text = clean_text(user_input)
            
            # Update Sidebar Privacy Log
            st.sidebar.write("**Original Input:**", f"*{user_input}*")
            st.sidebar.write("**Sanitized Input (Sent to AI):**", f"*{safe_text}*")
            st.sidebar.success("✅ PII Scrubbing Complete. Data is anonymized.")

            # Step B: Inference
            inputs = tokenizer(safe_text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
            
            # Step C: Get Results dynamically based on model config
            id2label = model.config.id2label
            results_df = pd.DataFrame({
                "Disorder": [id2label[i] for i in range(len(probabilities))],
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=False)

            top_prediction = results_df.iloc[0]["Disorder"]
            top_confidence = results_df.iloc[0]["Probability"] * 100

            # --- 5. Display Results ---
            st.markdown("---")
            st.markdown("### 📊 Diagnostic Analysis")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label="Primary Risk Detected", value=top_prediction, delta=f"{top_confidence:.1f}% Confidence", delta_color="off")
                
                # Clinical Escalation Logic
                if top_prediction in ["Suicidal", "Depression"]:
                    st.error("⚠️ **HIGH RISK IDENTIFIED**\n\nImmediate clinical escalation recommended.")
                elif top_prediction in ["Normal"]:
                    st.success("✅ **BASELINE STATE**\n\nNo immediate acute risk detected.")
                else:
                    st.warning("🟡 **MODERATE DISTRESS**\n\nElevated stress or anxiety patterns detected. Monitor closely.")

            with col2:
                # Explainability Chart
                st.markdown("**Probability Distribution (All Classes)**")
                chart = alt.Chart(results_df).mark_bar().encode(
                    x=alt.X('Probability:Q', axis=alt.Axis(format='%')),
                    y=alt.Y('Disorder:N', sort='-x'),
                    color=alt.Color('Probability:Q', scale=alt.Scale(scheme='reds')),
                    tooltip=['Disorder', alt.Tooltip('Probability:Q', format='.2%')]
                ).properties(height=250)
                st.altair_chart(chart, use_container_width=True)