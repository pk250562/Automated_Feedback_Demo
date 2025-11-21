# optimized app.py for Render free tier (HF Inference API + lightweight processing)
import os
import re
import html
import time
import requests
from functools import lru_cache

import gradio as gr
import textstat

# -------------------------------
# Environment & config
# -------------------------------
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # disable analytics

# Public CEFR model on Hugging Face
HF_API_URL = "https://router.huggingface.co/models/pkim62/CEFR-classification-model"

LANGUAGETOOL_API_URL = "https://api.languagetool.org/v2/check"

# -------------------------------
# Tiny sentiment lexicons (very small, lightweight)
# -------------------------------
POS_WORDS = {"good","great","excellent","positive","happy","enjoy","love","nice","well","improve","success","benefit"}
NEG_WORDS = {"bad","poor","terrible","negative","sad","hate","worse","problem","issue","difficult","hard","fail"}

def simple_sentiment(text):
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    score = (pos - neg) / max(1, len(words))
    return round(score * 5, 2)

# -------------------------------
# LanguageTool grammar check
# -------------------------------
def check_grammar_api(text):
    data = {"text": text, "language": "en-US"}
    try:
        resp = requests.post(LANGUAGETOOL_API_URL, data=data, timeout=8)
        j = resp.json()
        return j.get("matches", [])
    except Exception as e:
        print("LanguageTool error:", e)
        return []

def build_grammar_summary(text, matches, max_items=10):
    grammar_count = len(matches)
    detailed = []
    for m in matches[:max_items]:
        start = m.get("offset", 0)
        length = m.get("length", 0)
        error_text = text[start:start+length] if length>0 else ""
        raw = m.get("replacements", [])
        repls = []
        for r in raw[:3]:
            if isinstance(r, dict):
                repls.append(r.get("value",""))
            else:
                repls.append(str(r))
        replacements = ", ".join(repls) if repls else "No suggestions"
        detailed.append(f"• Issue: {m.get('message','')}\n    Text: \"{error_text}\"\n    Suggestion(s): {replacements}")
    summary = f"Total Grammar Issues: {grammar_count}\n\n" + "\n\n".join(detailed) if detailed else "✅ No major grammar issues detected."

    # highlighted html
    highlighted = ""
    last_index = 0
    for m in matches:
        start = m.get("offset", 0)
        end = start + m.get("length", 0)
        msg = html.escape(m.get("message","")).replace("'", "&#39;").replace('"', "&quot;")
        highlighted += html.escape(text[last_index:start])
        if end > start:
            highlighted += f"<span class='grammar-issue' title=\"{msg}\">{html.escape(text[start:end])}</span>"
        last_index = end
    highlighted += html.escape(text[last_index:])
    return summary, highlighted, grammar_count

# -------------------------------
# CEFR prediction via HF Inference API
# -------------------------------
@lru_cache(maxsize=256)
def predict_cefr(text):
    payload = {"inputs": text}
    try:
        r = requests.post(HF_API_URL, json=payload, timeout=30)
        print("API RAW:", r.text, flush=True)
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            print("HF API error:", data.get("error"))
            return "N/A", 0.0, {}
        if isinstance(data, list) and len(data) > 0:
            prob_dict = {d.get("label", f"lbl_{i}"): float(d.get("score", 0.0)) for i,d in enumerate(data)}
            top = max(data, key=lambda x: x.get("score",0))
            label = top.get("label","N/A")
            confidence = float(top.get("score",0.0))
            return label, confidence, prob_dict
        return "N/A", 0.0, {}
    except Exception as e:
        print("HF Inference API request failed:", e)
        return "N/A", 0.0, {}

# -------------------------------
# Lightweight text metrics
# -------------------------------
def truncate_text(text, max_words=500):
    words = text.split()
    return (" ".join(words[:max_words]), True) if len(words) > max_words else (text, False)

def compute_ttr(text):
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 2)

def avg_sentence_len(text):
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    words = re.findall(r"\w+", text)
    return (len(words) / len(sentences)) if sentences else 0.0

# -------------------------------
# Interpretations & narrative
# -------------------------------
cefr_map = {
    'C': "Advanced proficiency — well structured, uses varied register.",
    'B': "Intermediate proficiency — coherent, suitable for familiar topics.",
    'A': "Basic proficiency — simple sentences and limited range."
}

def interpret_readability(score):
    if score >= 90: return "Very easy to read"
    if score >= 70: return "Easy to read"
    if score >= 50: return "Moderately difficult"
    if score >= 30: return "Fairly difficult"
    return "Very difficult"

def generate_narrative(predicted_cefr, grammar_count, avg_sentence_length, sentiment, ttr, readability, confidence):
    feedback = []
    if avg_sentence_length < 10:
        feedback.append("Sentences are short; consider combining ideas.")
    elif avg_sentence_length > 25:
        feedback.append("Sentences are long; consider breaking them for clarity.")
    else:
        feedback.append("Sentence length is balanced.")

    if sentiment < -0.2:
        feedback.append("Tone leans negative.")
    elif sentiment > 0.2:
        feedback.append("Tone leans positive.")
    else:
        feedback.append("Tone is neutral.")

    feedback.append(f"Readability (Flesch {readability}): {interpret_readability(readability)}")
    feedback.append(f"Lexical Diversity (TTR {ttr}): {'High' if ttr>0.6 else 'Moderate' if ttr>0.4 else 'Low'}")
    feedback.append(f"Predicted CEFR level: {predicted_cefr} — {cefr_map.get(predicted_cefr,'')}")

    summary = (f"This text has {grammar_count} grammar issue(s), a TTR of {ttr:.2f}, "
               f"readability {readability:.1f}, average sentence length {avg_sentence_length:.1f}. "
               f"CEFR: {predicted_cefr} (confidence {confidence:.2f}).")
    return "\n\n".join(feedback), summary

# -------------------------------
# Main app logic
# -------------------------------
def cefr_feedback_app(text):
    start = time.time()
    text, truncated = truncate_text(text)
    if not text.strip():
        return ["N/A"]*8 + ["No text provided.", None]

    matches = check_grammar_api(text)
    grammar_summary, highlighted_text, grammar_count = build_grammar_summary(text, matches)

    ttr = compute_ttr(text)
    readability = round(textstat.flesch_reading_ease(text), 2)
    avg_len = avg_sentence_len(text)
    sentiment = simple_sentiment(text)

    predicted_cefr, confidence, prob_dict = predict_cefr(text)
    narrative_feedback, summary = generate_narrative(predicted_cefr, grammar_count, avg_len, sentiment, ttr, readability, confidence)

    if truncated:
        summary += "\n\n⚠️ Note: Only the first 500 words were analyzed."

    elapsed = time.time() - start
    print(f"Processed request in {elapsed:.2f}s")
    return (
        predicted_cefr, confidence, grammar_summary, highlighted_text,
        ttr, readability, round(avg_len,1), f"{sentiment:.2f}",
        narrative_feedback, summary, None
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

with gr.Blocks(theme="gradio/default", css=custom_css) as app:
    gr.Markdown("## ✍️ Automated Writing Feedback (with CEFR Classification)")
    gr.Markdown("Paste up to 500 words. Grammar issues are highlighted with hover tooltips.")
    text_input = gr.Textbox(lines=10, placeholder="Paste your text here (max 500 words)...", label="Your Text")
    analyze_btn = gr.Button("🔍 Analyze Text")

    with gr.Row():
        cefr_output = gr.Textbox(label="Predicted CEFR Level")
        confidence_output = gr.Number(label="Confidence")
    with gr.Row():
        grammar_output = gr.Textbox(label="Grammar Issues & Suggestions", lines=10, max_lines=15)
        grammar_highlight_html = gr.HTML(label="Highlighted Text (Hover for Suggestions)")
    with gr.Row():
        ttr_output = gr.Number(label="Lexical Diversity (TTR)")
        readability_output = gr.Number(label="Readability (Flesch Score)")
    with gr.Row():
        sentence_length_output = gr.Number(label="Average Sentence Length")
        sentiment_output = gr.Textbox(label="Sentiment Score")

    narrative_output = gr.Textbox(label="Detailed Narrative Feedback", lines=6, max_lines=10)
    summary_output = gr.Textbox(label="Summary", lines=3, max_lines=6)
    chart_output = gr.Image(label="CEFR Probability Distribution")  # will be None

    gr.Markdown("""
    <div class="custom-disclaimer">
    ⚠️ <b>Disclaimer:</b><br>
    This tool is for <b>demonstration purposes only</b> and should not be used for high-stakes scoring.
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
# Warmup function to reduce cold start latency
# -------------------------------
def warmup_cefr_model():
    dummy_text = "This is a warm-up text to initialize the CEFR model."
    try:
        label, conf, _ = predict_cefr(dummy_text)
        print(f"Warmup complete: label={label}, confidence={conf:.2f}")
    except Exception as e:
        print("Warmup failed:", e)

# -------------------------------
# Launch app
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    warmup_cefr_model()
    app.launch(server_name="0.0.0.0", server_port=port, share=False, show_error=True)

