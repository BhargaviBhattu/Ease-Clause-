# pages/summarizer.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

# ---------------------------
# Page CSS styling
# ---------------------------
st.markdown(
    """
    <style>
    .title {font-size: 32px; font-weight: bold; margin-bottom: 20px;}
    .subtitle {font-size: 20px; margin-bottom: 15px; color: #555;}
    .card {background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-bottom: 15px;}
    textarea {width: 100%;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Load abstractive model (BART)
# ---------------------------
@st.cache_resource
def load_abstractive_model():
    """
    Load and cache the BART model for abstractive summarization.
    Returns the tokenizer and model from Hugging Face's Transformers library."""
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()  # ensure model is in inference mode for efficiency
    return tokenizer, model

abst_tokenizer, abst_model = load_abstractive_model()

# ---------------------------
# Abstractive summarization (Purvesh's contribution)
# ---------------------------
def abstractive_summarize(text: str, max_length: int = 130, min_length: int = 30) -> str:
    """
    Generate a human-like abstractive summary using the BART model.
    Follows a 4-step pipeline:
        1. Preprocess input text
        2. Pass text through pre-trained BART CNN model
        3. Generate and decode summary
        4. Postprocess for readability
    """
    
    if not text.strip():
        return " ⚠️ Please provide text to summarize."
    
    # Preprocessing
    inputs = abst_tokenizer(
        text,
        return_tensors="pt", 
        max_length=1024, 
        truncation=True,
        )

    # Summary generation
    summary_ids = abst_model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return abst_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ---------------------------
# Enhanced Hybrid Extractive Summarization
# ---------------------------
def hybrid_summarize(text: str, compression_ratio: float = 0.4) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) <= 2:
        return " ".join(sentences)

    # 1. Calculate TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    
    # Normalize scores to a 0-1 range
    if sentence_scores.max() > 0:
        sentence_scores = (sentence_scores - sentence_scores.min()) / (sentence_scores.max() - sentence_scores.min())
    
    # 2. Calculate position scores (higher for earlier sentences)
    position_scores = np.array([1 / (i + 1) for i in range(len(sentences))])
    # Normalize position scores
    position_scores = (position_scores - position_scores.min()) / (position_scores.max() - position_scores.min())

    # 3. Calculate length scores (closer to average length is better)
    sentence_lengths = np.array([len(s.split()) for s in sentences])
    avg_length = np.mean(sentence_lengths)
    length_scores = np.exp(-np.abs(sentence_lengths - avg_length) / avg_length)

    # 4. Combine scores with adjustable weights
    alpha = 0.5  # Weight for TF-IDF
    beta = 0.3   # Weight for Position
    gamma = 0.2  # Weight for Length
    
    total_scores = (alpha * sentence_scores + beta * position_scores + gamma * length_scores)
    
    # Debugging: Print scores to see the ranking
    # for i, (s, score) in enumerate(zip(sentences, total_scores)):
    #     st.write(f"Sentence {i+1} (Score: {score:.4f}): {s}")
    
    # 5. Select top sentences
    n = max(1, int(len(sentences) * compression_ratio))
    top_idx = np.argsort(total_scores)[-n:]
    top_idx_sorted = sorted(top_idx)
    summary_sentences = [sentences[i] for i in top_idx_sorted]

    return " ".join(summary_sentences)

# ---------------------------
# Streamlit UI
# ---------------------------
st.markdown('<div class="title">📝 Text Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Choose summarization type and provide text</div>', unsafe_allow_html=True)

# Summarization type
method = st.radio("Select Method", ["Abstractive (BART)", "Hybrid Extractive"])

# Text input area
text_input = st.text_area("Paste text here...", height=200)

# File upload
uploaded_file = st.file_uploader("Or upload a TXT/DOCX/PDF file", type=["txt", "docx", "pdf"])
extracted_text = ""
if uploaded_file:
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".txt":
            extracted_text = uploaded_file.read().decode("utf-8")
        elif ext == ".docx":
            from docx import Document
            doc = Document(uploaded_file)
            extracted_text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(uploaded_file)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            extracted_text = "\n".join(pages)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

# Combine pasted and uploaded text
final_text = (text_input.strip() or extracted_text.strip())

# Compression ratio for hybrid
compression_ratio = st.slider(
    "Hybrid compression ratio (only for Hybrid Extractive)", 0.1, 1.0, 0.4, 0.05
)

# Generate summary
if st.button("Generate Summary"):
    if final_text:
        with st.spinner("Generating summary..."):
            if method == "Abstractive (BART)":
                summary = abstractive_summarize(final_text)
            else:
                summary = hybrid_summarize(final_text, compression_ratio=compression_ratio)

        st.markdown('<div class="subtitle">Original Text</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{final_text}</div>', unsafe_allow_html=True)

        st.markdown('<div class="subtitle">Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">{summary}</div>', unsafe_allow_html=True)
    else:
        st.warning("Please provide text or upload a file to summarize.")

