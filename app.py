# ============================================================
# 🚀 Optimized HF Spaces Automated Writing Feedback App
# ============================================================
import os
import re
import html
import torch
import gradio as gr
import numpy as np
from functools import lru_cache
from PIL import Image
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------------------
# 🔧 Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🔧 LanguageTool API
# -------------------------------
LANGUAGETOOL_API_URL = "https://api.languagetool.org/v2/check"

def check_grammar_api(text):
    data = {"text": text, "language": "en-US"}
    try:
        response = requests.post(LANGUAGETOOL_API_URL, data=data, timeout=10)
        return response.json().get("matches", [])
    except Exception as e:
        print("LanguageTool API error:", e)
        return []

# -------------------------------
# 🔧 Load INT8 CEFR model from local repo (Render)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the folder containing your model files (model.safetensors, tokenizer.json, etc.)
MODEL_LOCAL_PATH = "./"  # repo root

# Check if files exist
required_files = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json"
]
missing_files = [f for f in required_files if not os.path.isfile(os.path.join(MODEL_LOCAL_PATH, f))]
if missing_files:
    raise FileNotFoundError(f"Missing model files in {MODEL_LOCAL_PATH}: {missing_files}")

# Load tokenizer from local files
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH, use_fast=False)

# Load INT8 quantized model from local files
cefr_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_LOCAL_PATH,
    device_map="auto",
    load_in_8bit=True
)

# Move model to the selected device
cefr_model.to(device)
cefr_model.eval()

# Mapping from prediction index to label
id2label = cefr_model.config.id2label
print(f"✅ Loaded INT8 CEFR model from {MODEL_LOCAL_PATH} on device {device}")


# -------------------------------
# 🔹 CEFR feedback mapping
# -------------------------------
cefr_map = {
    'C': "Advanced proficiency — The writing demonstrates advanced competence. "
         "It produces clear, well-structured texts of complex subjects...",
    'B': "Intermediate proficiency — The writing reflects intermediate proficiency...",
    'A': "Basic proficiency — The writing shows basic ability to communicate..."
}

# -------------------------------
# 🔍 Core Analysis (lazy imports for speed)
# -------------------------------
def truncate_text(text, max_words=500):
    words = text.split()
    return (" ".join(words[:max_words]), True) if len(words) > max_words else (text, False)

def analyze_text(text):
    from lexicalrichness import LexicalRichness
    import textstat
    from textblob import TextBlob

    matches = check_grammar_api(text)
    grammar_count = len(matches)

    # Detailed issues (max 10)
    detailed_issues = []
    for m in matches[:10]:
        error_text = text[m["offset"]: m["offset"] + m["length"]]
        raw_repl = m.get("replacements", [])
        replacements = ", ".join([r.get("value","") if isinstance(r, dict) else str(r) for r in raw_repl[:3]]) \
                       if raw_repl else "No suggestions"
        issue = f"• Issue: {m['message']}\n    Text: \"{error_text}\"\n    Suggestion(s): {replacements}"
        detailed_issues.append(issue)

    grammar_summary = f"Total Grammar Issues: {grammar_count}\n\n" + "\n\n".join(detailed_issues) \
        if detailed_issues else "✅ No major grammar issues detected."

    # Highlight text with tooltips
    highlighted_text = ""
    last_index = 0
    for m in matches:
        start, end = m["offset"], m["offset"] + m["length"]
        msg = html.escape(m["message"]).replace("'", "&#39;").replace('"', "&quot;")
        highlighted_text += html.escape(text[last_index:start])
        highlighted_text += f"<span class='grammar-issue' title=\"{msg}\">{html.escape(text[start:end])}</span>"
        last_index = end
    highlighted_text += html.escape(text[last_index:])

    # Lexical & readability
    lex = LexicalRichness(text)
    ttr = round(lex.ttr, 2)
    readability = round(textstat.flesch_reading_ease(text), 2)
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    words = text.split()
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    sentiment = TextBlob(text).sentiment.polarity

    return grammar_summary, highlighted_text, ttr, readability, avg_sentence_length, sentiment, grammar_count

# -------------------------------
# 🧠 CEFR prediction (cached)
# -------------------------------
@lru_cache(maxsize=128)
def predict_cefr(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = cefr_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    label = id2label.get(pred_idx, f"Label {pred_idx}")
    confidence = float(np.max(probs))
    prob_dict = {id2label[i]: float(p) for i, p in enumerate(probs)}
    return label, confidence, prob_dict

# -------------------------------
# 🔹 CEFR probability chart
# -------------------------------
def plot_cefr_probs(prob_dict):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4,3), dpi=100)
    ax.bar(prob_dict.keys(), prob_dict.values(), color='cornflowerblue')
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    ax.set_title("CEFR Probability Distribution")
    plt.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(img)

# -------------------------------
# 🎨 Gradio App
# -------------------------------
def cefr_feedback_app(text):
    text, truncated = truncate_text(text)
    if not text.strip():
        return ["N/A"] * 8 + ["No text provided.", None]

    grammar_summary, highlighted_text, ttr, readability, avg_sentence_length, sentiment, grammar_count = analyze_text(text)
    predicted_cefr, confidence, prob_dict = predict_cefr(text)
    narrative_feedback = f"Predicted CEFR: {predicted_cefr}, Confidence: {confidence:.2f}"
    cefr_chart = plot_cefr_probs(prob_dict)

    if truncated:
        narrative_feedback += "\n⚠️ Only first 500 words analyzed."

    return predicted_cefr, confidence, grammar_summary, highlighted_text, ttr, readability, round(avg_sentence_length,1), f"{sentiment:.2f}", narrative_feedback, narrative_feedback, cefr_chart

custom_css = """
<style>
textarea:focus, input:focus { outline: none !important; box-shadow: none !important; border: 1px solid #ccc !important; }
.grammar-issue { background-color: #ffdddd; border-bottom: 1px dotted red; cursor: help; }
.grammar-issue:hover { background-color: #ffecec; }
</style>
"""

with gr.Blocks(theme="gradio/soft", css=custom_css) as app:
    gr.Markdown("## ✍️ Automated Writing Feedback (INT8 CEFR Model)")
    text_input = gr.Textbox(lines=10, placeholder="Paste your text...", label="Your Text")
    analyze_btn = gr.Button("🔍 Analyze Text")
    cefr_output = gr.Textbox(label="CEFR Level")
    confidence_output = gr.Number(label="Confidence")
    grammar_output = gr.Textbox(label="Grammar Issues", lines=10)
    grammar_highlight_html = gr.HTML(label="Highlighted Text")
    analyze_btn.click(cefr_feedback_app, inputs=text_input,
                      outputs=[cefr_output, confidence_output, grammar_output, grammar_highlight_html,
                               cefr_output, confidence_output, cefr_output, cefr_output, cefr_output, cefr_output])

# -------------------------------
# 🚀 Launch
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.launch(server_name="0.0.0.0", server_port=port, share=False)

