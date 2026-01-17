# utils/simplifier.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from functools import lru_cache


# -----------------------------
# Model configuration
# -----------------------------
MODEL_NAME = "google/flan-t5-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Instruction prompts
# -----------------------------
PROMPTS = {
    "Basic": (
        "Rewrite the following text in very simple English. "
        "Use very short sentences and basic words.\n\n"
    ),
    "Intermediate": (
    "Rewrite the following legal text for a non-lawyer. "
    "Remove legal phrasing and formal tone. "
    "It is okay if some details are shortened, as long as the main meaning remains.\n\n"
    ),

    
    "Advanced": (
        "Rewrite the following text in a much shorter and simpler form. "
        "Keep only the main ideas.\n\n"
    ),
}


# -----------------------------
# Load model (cached)
# -----------------------------
@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# -----------------------------
# Simplification function
# -----------------------------
def simplify_text(text: str, level: str = "Intermediate") -> str:
    if not text or not text.strip():
        return ""

    tokenizer, model = load_model()

    prompt = PROMPTS.get(level, PROMPTS["Intermediate"]) + text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=1,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()
