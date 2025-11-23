# app.py
import os
import re
import io
from typing import List, Tuple

import numpy as np
import gradio as gr

# Transformers / HF
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# NLP / text stats
import nltk
from nltk import word_tokenize, sent_tokenize
from textblob import TextBlob
import textstat

# --- Ensure necessary NLTK data is available ---
nltk_data_needed = ["punkt", "averaged_perceptron_tagger"]
for pkg in nltk_data_needed:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg=="punkt" else f"taggers/{pkg}")
    except Exception:
        nltk.download(pkg)

# -------------------------
# Configuration: Model repo
# -------------------------
MODEL_REPO = os.getenv("MODEL_REPO", "pkim62/CEFR-classification-model")
HF_TOKEN = os.getenv("HF_TOKEN", None)
device = "cpu"  # Render free tier

# -------------------------
# Authenticate & load model
# -------------------------
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print("HF login failed:", e)

print(f"Loading model from {MODEL_REPO} on device={device} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)
cefr_model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO, trust_remote_code=False)
cefr_model.to(device)
cefr_model.eval()

# Build id2label safely
id2label = getattr(cefr_model.config, "id2label", {i: str(i) for i in range(cefr_model.config.num_labels)})

# -------------------------
# Utility functions
# -------------------------
def predict_cefr(text: str) -> Tuple[str, float, dict]:
    if not text.strip():
        return "N/A", 0.0, {}
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = cefr_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
    labels = [id2label.get(i, str(i)) for i in range(len(probs))]
    best_idx = int(np.argmax(probs))
    best_label = labels[best_idx]
    best_conf = float(probs[best_idx])
    probs_dict = {labels[i]: float(probs[i]) for i in range(len(probs))}
    return best_label, best_conf, probs_dict

def compute_ttr(text: str) -> float:
    toks = [t.lower() for t in word_tokenize(text)]
    return len(set(toks)) / len(toks) if toks else 0.0

def avg_sentence_length(text: str) -> float:
    sents = sent_tokenize(text)
    lengths = [len(word_tokenize(s)) for s in sents]
    return float(sum(lengths)/len(lengths)) if sents else 0.0

def readability_flesch(text: str) -> float:
    try:
        return float(textstat.flesch_reading_ease(text))
    except Exception:
        return 0.0

GRAMMAR_PATTERNS = [
    (re.compile(r"\b([A-Za-z]+) \1\b", re.IGNORECASE), "Repeated word"),
    (re.compile(r"\s{2,}"), "Multiple spaces"),
    (re.compile(r"\.([A-Za-z])"), "Missing space after period"),
    (re.compile(r"\b([i])\b"), "Lowercase 'i' used as pronoun"),
]

def find_grammar_issues(text: str) -> List[Tuple[int,int,str]]:
    issues = []
    for pat, msg in GRAMMAR_PATTERNS:
        for m in pat.finditer(text):
            start, end = m.span()
            issues.append((start,end,msg))
    sents = sent_tokenize(text)
    running = 0
    for s in sents:
        stripped = s.lstrip()
        if stripped and stripped[0].islower():
            idx = text.find(stripped, running)
            if idx >= 0: issues.append((idx, idx+1, "Sentence starts with lowercase letter"))
        running = text.find(s, running)+len(s) if text.find(s,running)>=0 else running
    return sorted(issues, key=lambda x: x[0])

def escape_html(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;").replace("'","&#x27;")

def highlight_text_with_issues(text: str, issues: List[Tuple[int,int,str]]) -> str:
    if not issues: return "<div>" + text.replace("\n","<br>") + "</div>"
    out, last = [], 0
    for (s,e,msg) in issues:
        if s>last: out.append(escape_html(text[last:s]))
        out.append(f'<span class="grammar-issue" title="{escape_html(msg)}">{escape_html(text[s:e])}</span>')
        last = e
    if last < len(text): out.append(escape_html(text[last:]))
    return "<div>" + "".join(out).replace("\n","<br>") + "</div>"

# --- TextBlob sentiment
def sentiment_score(text: str) -> float:
    return TextBlob(text).sentiment.polarity if text else 0.0

# --- Readability/TTR interpretation
def interpret_readability(score):
    if score >= 90: return "Very easy to read — ideal for general audiences."
    if score >= 70: return "Easy to read — clear for most readers."
    if score >= 50: return "Moderately difficult — suitable for intermediate users."
    if score >= 30: return "Fairly difficult — may require advanced readers."
    return "Very difficult — academic or technical in tone."

def interpret_ttr(ttr):
    if ttr > 0.6: return "High lexical diversity — rich and varied vocabulary."
    if ttr > 0.4: return "Moderate lexical diversity — balanced word choice."
    return "Low lexical diversity — vocabulary repetition observed."

cefr_map = {
        'C': "Advanced proficiency — The writing demonstrates advanced competence. It produces clear, well-structured texts of complex subjects, expanding and supporting points of view with examples, reasons, and appropriate conclusions. It effectively varies tone, style, and register according to addressee and theme.",
        'B': "Intermediate proficiency — The writing reflects intermediate proficiency. It produces straightforward connected texts on familiar subjects, linking shorter elements into coherent sequences.",
        'A': "Basic proficiency — The writing shows basic ability to communicate about personal or familiar matters using simple words and short, isolated phrases."
    }

def generate_narrative(predicted_cefr, grammar_count, avg_sentence_length, sentiment, ttr, readability, confidence):
    feedback = []
    if avg_sentence_length < 10: feedback.append("Sentences are quite short; consider combining ideas for more complexity.")
    elif avg_sentence_length > 25: feedback.append("Sentences are rather long; consider breaking them for clarity.")
    else: feedback.append("Sentence length is well-balanced.")
    if sentiment < -0.2: feedback.append("The tone leans negative.")
    elif sentiment > 0.2: feedback.append("The tone leans positive.")
    else: feedback.append("Tone is neutral and balanced.")
    feedback.append(f"**Readability (Flesch Score {readability}):** {interpret_readability(readability)}")
    feedback.append(f"**Lexical Diversity (TTR {ttr:.2f}):** {interpret_ttr(ttr)}")
    feedback.append(f"**Overall CEFR Level:** {predicted_cefr}\n{cefr_map.get(predicted_cefr,'')}")
    if sentiment < -0.2: sentiment_interpretation = "The text conveys a negative or critical tone."
    elif sentiment > 0.2: sentiment_interpretation = "The overall tone of your writing is positive and engaging."
    else: sentiment_interpretation = "The tone appears neutral and balanced."
    summary = (
        f"This text has {grammar_count} grammar issue(s), "
        f"a lexical diversity (TTR) of {ttr:.2f}, "
        f"and a readability score of {readability:.1f}. "
        f"The average sentence length is {avg_sentence_length:.1f} words. "
        f"The predicted CEFR level is **{predicted_cefr}**, with a confidence of **{confidence:.2f}**. "
        f"{sentiment_interpretation}"
    )
    return "\n".join(feedback), summary

def make_probability_text(probs_dict: dict) -> str:
    if not probs_dict: return "N/A"
    lines = [f"{k}: {v:.2f}" for k,v in probs_dict.items()]
    return "\n".join(lines)

# -------------------------
# Gradio callback
# -------------------------
def cefr_feedback_app(text: str):
    text = (text or "").strip()[:5000]

    label, conf, probs = predict_cefr(text)
    ttr = compute_ttr(text)
    flesch = readability_flesch(text)
    avg_sent_len = avg_sentence_length(text)
    sentiment = sentiment_score(text)

    issues = find_grammar_issues(text)
    grammar_count = len(issues)
    grammar_list_str = "\n".join([f"{i+1}. [{start}:{end}] {msg} — \"{escape_html(text[start:end])}\""
                                  for i,(start,end,msg) in enumerate(issues)]) or "No obvious issues detected."
    highlighted_html = highlight_text_with_issues(text, issues)

    narrative, summary = generate_narrative(label, grammar_count, avg_sent_len, sentiment, ttr, flesch, conf)
    chart_text = make_probability_text(probs)

    return (
        label,
        float(conf),
        grammar_list_str,
        highlighted_html,
        float(ttr),
        float(flesch),
        float(avg_sent_len),
        f"{sentiment:.2f}",
        narrative,
        summary,
        chart_text
    )

# -------------------------------
# Gradio UI
# -------------------------------
custom_css = """
<style>
textarea:focus, input:focus { outline: none !important; box-shadow: none !important; border: 1px solid #ccc !important; }
.grammar-issue { background-color: #ffdddd; border-bottom: 1px dotted red; cursor: help; }
.grammar-issue:hover { background-color: #ffecec; }
.custom-disclaimer { font-size:0.85em; color:#555; background-color:#f9f9f9; padding:10px; border-radius:8px; }
</style>
"""

with gr.Blocks(theme="gradio/soft", css=custom_css, analytics_enabled=False) as app:
    gr.Markdown("## ✍️ Automated Writing Feedback (with CEFR Classification)")
    gr.Markdown("Paste up to 500 words. Grammar issues are highlighted with hover tooltips.")
    text_input = gr.Textbox(lines=10, placeholder="Paste your text here...", label="Your Text")
    analyze_btn = gr.Button("🔍 Analyze Text")

    with gr.Row():
        cefr_output = gr.Textbox(label="Predicted CEFR Level")
        confidence_output = gr.Number(label="Confidence")
    with gr.Row():
        grammar_output = gr.Textbox(label="Grammar Issues & Suggestions", lines=10)
        grammar_highlight_html = gr.HTML(label="Highlighted Text (Hover for Suggestions)")
    with gr.Row():
        ttr_output = gr.Number(label="Lexical Diversity (TTR)")
        readability_output = gr.Number(label="Readability (Flesch Score)")
    with gr.Row():
        sentence_length_output = gr.Number(label="Average Sentence Length")
        sentiment_output = gr.Textbox(label="Sentiment Polarity")

    narrative_output = gr.Textbox(label="Detailed Narrative Feedback", lines=6)
    summary_output = gr.Textbox(label="Summary", lines=3)
    chart_output = gr.Textbox(label="CEFR Probabilities")  # Text-based probabilities

    gr.Markdown("""
    <div class="custom-disclaimer">
    ⚠️ <b>Disclaimer:</b><br>
    This tool is for <b>demonstration purposes only</b> and should not be used for commercial or official scoring.
    </div>
    """)

    analyze_btn.click(
        cefr_feedback_app,
        inputs=text_input,
        outputs=[
            cefr_output, confidence_output, grammar_output, grammar_highlight_html,
            ttr_output, readability_output, sentence_length_output, sentiment_output,
            narrative_output, summary_output, chart_output
        ]
    )

# -------------------------------
# Launch
# -------------------------------
if __name__ == "__main__":
    import os
    
    # Render provides PORT — REQUIRED
    port = int(os.environ.get("PORT", "10000"))
    
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        enable_queue=True,
        share=False
    )



