# utils/simplifier.py
"""
Robust text simplifier utility.

Usage:
    from utils.simplifier import simplify_text
    out = simplify_text(long_legal_text, level="Intermediate", debug=True)

Notes:
- Default model is google/flan-t5-large (instruction tuned). If not available or
  too heavy for your environment, set MODEL_NAME to "google/flan-t5-base" or use BART fallback.
- This file includes simple rule-based preprocessing and then uses the model to rewrite.
- Chunking is applied automatically for long inputs to avoid token overflow.
"""

from functools import lru_cache
import re
import html
import math
import os
import logging
from typing import Callable, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Make sure NLTK punkt & stopwords are available (quiet)
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt", quiet=True)

# Ensure punkt_tab (new tokenizer dependency)
try:
    nltk.data.find("tokenizers/punkt_tab")
except Exception:
    nltk.download("punkt_tab", quiet=True)

# Ensure stopwords
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

# ---------------------------
# Configuration
# ---------------------------
# Default model choice — change this if you prefer a different model.
# NOTE: flan-t5-large is best for instruction-following but heavy.
MODEL_NAME = os.getenv("SIMPLIFIER_MODEL", "google/flan-t5-large")

# Device: prefer GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic replacements dictionary (legal -> plain)
COMMON_REPLACEMENTS = {
    "utilize": "use",
    "commence": "start",
    "terminate": "end",
    "endeavor": "try",
    "assistance": "help",
    "individuals": "people",
    "approximately": "about",
    "purchase": "buy",
    "objective": "goal",
    "requirement": "need",
    "consequently": "so",
    "therefore": "so",
    "subsequently": "after",
    "nevertheless": "but",
    "furthermore": "also",
    "in addition": "also",
    "in order to": "to",
    "in the event that": "if",
    "in accordance with": "under",
    "hereinafter": "from now on",
    "aforementioned": "mentioned earlier",
    "pursuant to": "under",
    "in witness whereof": "to confirm this",
}

# T5-style prefixes (kept short; flan understands natural instructions too)
PREFIXES = {
    "Basic": "Simplify for a beginner reader: ",
    "Intermediate": "Simplify and clarify: ",
    "Advanced": "Simplify, compress and shorten: "
}

# Setup simple logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------
# Model loading (cached)
# ---------------------------
@lru_cache(maxsize=2)
def load_seq2seq_model(model_name: str = MODEL_NAME):
    """
    Load tokenizer and model, move model to DEVICE.
    Cached to avoid repeated downloads/loads.
    """
    try:
        logger.info(f"Loading model: {model_name} on device {DEVICE}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.warning(f"Failed to load {model_name}: {e}")
        # Fallback to a more common summarization model (BART) if FLAN isn't available
        fallback = "facebook/bart-large-cnn"
        logger.info(f"Falling back to {fallback}")
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForSeq2SeqLM.from_pretrained(fallback)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model


# ---------------------------
# Utility helpers
# ---------------------------
def clean_and_join(sentences: List[str]) -> str:
    """Capitalize, strip, and join sentences with periods."""
    sentences = [s.strip().capitalize() for s in sentences if s and s.strip()]
    return ". ".join(sentences).strip()


def apply_replacements(text: str, replacements: dict) -> str:
    """Apply whole-word, case-insensitive replacements from dictionary."""
    for k, v in replacements.items():
        text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.IGNORECASE)
    return text


# ---------------------------
# Rule-based simplification levels
# ---------------------------
def basic_simplify(text: str) -> str:
    """Remove common stopwords to make sentences more direct (aggressive)."""
    sentences = sent_tokenize(text)
    out = []
    for s in sentences:
        words = word_tokenize(s)
        filtered = [w for w in words if w.lower() not in STOPWORDS]
        out.append(" ".join(filtered))
    return clean_and_join(out)


def intermediate_simplify(text: str) -> str:
    """Replace complex legal words with simpler synonyms using replacements."""
    return apply_replacements(text, COMMON_REPLACEMENTS)


def advanced_simplify(text: str) -> str:
    """More aggressive replacements + sentence compression for long sentences."""
    text = intermediate_simplify(text)
    deep = {
        r"\bthe party of the first part\b": "first person",
        r"\bthe party of the second part\b": "second person",
        r"\bshall\b": "will",
        r"\bmust\b": "has to",
        r"\bprior to\b": "before",
        r"\bat this point in time\b": "now",
    }
    # apply deep replacements using regex patterns
    for k, v in deep.items():
        text = re.sub(k, v, text, flags=re.IGNORECASE)

    sentences = sent_tokenize(text)
    compressed = []
    for s in sentences:
        # If a sentence is very long, keep the first n words and append ellipsis
        if len(s.split()) > 22:
            s = " ".join(s.split()[:18]) + "..."
        compressed.append(s)
    return clean_and_join(compressed)


# ---------------------------
# Chunking helper (sentence based)
# ---------------------------
def chunk_text_by_words(text: str, max_words: int = 700) -> List[str]:
    """
    Split text into chunks by sentences without exceeding max_words per chunk.
    Returns a list of chunk strings.
    """
    sentences = sent_tokenize(text)
    chunks = []
    cur = []
    cur_count = 0
    for s in sentences:
        s_count = len(s.split())
        if cur_count + s_count > max_words and cur:
            chunks.append(" ".join(cur))
            cur = []
            cur_count = 0
        cur.append(s)
        cur_count += s_count
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ---------------------------
# Model call helpers
# ---------------------------
def _call_model(tokenizer, model, prompt: str, max_new_tokens: int = 200, num_beams: int = 4) -> str:
    """
    Tokenize, move to device, run generate, decode. Returns model string.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    # move tensors to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "early_stopping": True,
        "do_sample": False,
    }
    outputs = model.generate(**inputs, **generate_kwargs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


# ---------------------------
# Top-level simplify function
# ---------------------------
def simplify_text(text: str, level: str = "Intermediate", debug: bool = False) -> str:
    """
    Simplify `text` at the chosen `level` ("Basic"|"Intermediate"|"Advanced").
    If debug=True, the raw model output is returned prefixed for visibility.
    """
    if not text or not text.strip():
        return "Please enter text to simplify."

    # Step 1: rule-based pre-processing
    if level == "Basic":
        rule_text = basic_simplify(text)
    elif level == "Intermediate":
        rule_text = intermediate_simplify(text)
    elif level == "Advanced":
        rule_text = advanced_simplify(text)
    else:
        rule_text = text

    # Small safeguard: if rule_text is empty (very aggressive basic), fall back to original
    if not rule_text.strip():
        rule_text = text

    # Load model & tokenizer (cached)
    tokenizer, model = load_seq2seq_model(MODEL_NAME)

    # Prepare a clear instruction prompt that T5/Flan prefers
    # Keep the instruction short and explicit
    def make_prompt(chunk: str) -> str:
        return (
            "You are a helpful assistant that rewrites legal or technical text into plain English.\n"
            "Rewrite the text in short, clear sentences. Keep the legal meaning unchanged.\n\n"
            f"Text: {chunk}\n\nSimplified:"
        )

    # Step 2: chunk the input to avoid token overflow
    chunks = chunk_text_by_words(rule_text, max_words=650)

    # For each chunk, call the model to produce a simplified chunk
    simplified_chunks = []
    raw_outputs = []
    for c in chunks:
        prompt = make_prompt(c)
        try:
            out = _call_model(tokenizer, model, prompt, max_new_tokens=220, num_beams=4)
        except Exception as e:
            # If model call fails for any reason, fallback to returning rule_text
            logger.exception(f"Model generation failed: {e}")
            out = c  # fallback: use chunk itself
        simplified_chunks.append(out)
        raw_outputs.append(out)

    # If there were multiple chunks, we will combine partial simplifications and summarize again
    if len(simplified_chunks) > 1:
        combined = " ".join(simplified_chunks)
        # A second-pass simplification to merge and smooth partial outputs
        try:
            second_pass = _call_model(tokenizer, model,
                "Combine and shorten these simplified pieces into a coherent simplified paragraph:\n\n" + combined,
                max_new_tokens=220, num_beams=4)
            final = second_pass
        except Exception:
            final = combined
    else:
        final = simplified_chunks[0]

    # Post-processing: minimal cleanup and HTML escape to be safe when rendering
    final = final.strip()
    final = re.sub(r"\s+\.", ".", final)  # fix spacing before periods
    final = html.escape(final)  # escape so UI can safely insert HTML around it

    # If debug, return a visible combined debug string (raw returned text before escaping)
    if debug:
        dbg = {
            "model_name": MODEL_NAME,
            "device": str(DEVICE),
            "rule_text_preview": rule_text[:600],
            "raw_model_outputs_preview": " ||| ".join([o[:300] for o in raw_outputs]),
            "final_simplified_preview": final[:800],
        }
        dbg_str = (
            f"DEBUG SUMMARY\nModel: {dbg['model_name']}\nDevice: {dbg['device']}\n\n"
            f"Rule text preview:\n{dbg['rule_text_preview']}\n\n"
            f"Raw model outputs (per chunk):\n{dbg['raw_model_outputs_preview']}\n\n"
            f"Final simplified (escaped for UI):\n{dbg['final_simplified_preview']}"
        )
        # Return debug text unescaped to the caller so they can see it. Caller (Streamlit) should show in st.code()
        return dbg_str

    return final
