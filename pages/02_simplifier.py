import streamlit as st
from utils.simplifier import simplify_text
from utils.glossary_manager import load_glossary, highlight_terms, inject_glossary_styles
import os

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Text Simplifier",
    page_icon="✨",
    layout="wide"
)

# ---------------------------
# Inject glossary tooltip CSS
# ---------------------------
inject_glossary_styles()

# Load glossary
glossary = load_glossary("data/glossary.json")

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
    <style>
        .main {background-color: #f8fafc; padding: 2rem;}
        .stTextArea textarea {font-size: 1rem !important; line-height: 1.6;}
        .output-box {background-color: #ffffff; border-radius: 12px; padding: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); min-height: 400px;}
        .title {font-size: 1.8rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;}
        .subheader {color: #475569; font-size: 1rem;}
        .stButton>button {background-color: #2563eb; color: white; border-radius: 8px; padding: 0.5rem 1rem; border: none; transition: all 0.3s;}
        .stButton>button:hover {background-color: #1d4ed8;}
        mark.tooltip {background-color: #fffa91; font-weight: bold; border-radius: 3px; cursor: help; padding: 0 2px;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Menu
# ---------------------------
st.sidebar.title("🧠 Simplification Menu")
level = st.sidebar.radio(
    "Select Simplification Level",
    ["Basic", "Intermediate", "Advanced"],
    index=1
)
st.sidebar.markdown("---")
st.sidebar.info("Choose a level and click **Simplify Text** to view results.")

# ---------------------------
# Main Layout
# ---------------------------
st.title("✨ Text Simplifier")
st.markdown("Simplify complex legal or professional text into more readable forms. Choose a simplification level to control how deep the rewriting goes.")

# ---------------------------
# Input Options
# ---------------------------
st.markdown("### 📝 Input Options")

# Text input area
text_input = st.text_area(
    "Paste or type your text below:",
    placeholder="Enter your text here...",
    height=200
)

# File upload
uploaded_file = st.file_uploader("Or upload a TXT / DOCX / PDF document", type=["txt", "docx", "pdf"])
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
        st.error(f"⚠️ Failed to read file: {e}")

# Combine input sources
final_text = (text_input.strip() or extracted_text.strip())

# ---------------------------
# Simplify Button
# ---------------------------
if st.button("🔍 Simplify Text"):
    if final_text:
        with st.spinner("Simplifying... Please wait ⏳"):
            simplified_output = simplify_text(final_text, level)

        # Highlight glossary terms in original text
        highlighted_text = highlight_terms(final_text, glossary)

        # Display Output Columns — Contribution by Purvesh Patel
        # This section shows original and simplified text side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='title'>📄 Original Text</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='output-box'>{highlighted_text}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='title'>🔹 Simplified Text ({level} Level)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='output-box'>{simplified_output}</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter text or upload a document before simplifying.")

