# ============================================================
# 🚀 Optimized HF Spaces Automated Writing Feedback App
# ============================================================
import matplotlib
matplotlib.use("Agg")

import os
import re
import torch
import textstat
import gradio as gr
import numpy as np
from lexicalrichness import LexicalRichness
from textblob import TextBlob
import matplotlib.pyplot as plt
from PIL import Image
import html
import requests
from functools import lru_cache

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
        result = response.json()
        return result.get("matches", [])
    except Exception as e:
        print("LanguageTool API error:", e)
        return []

# -------------------------------
# 🔧 Load CEFR model once
# -------------------------------

MODEL_REPO = "pkim62/CEFR-classification-model"
HF_TOKEN = os.environ.get("HF_TOKEN")  # will set in Render

# -------------------------------
# 🔹 CEFR feedback mapping
# -------------------------------
cefr_map = {
    'C': "Advanced proficiency — The writing demonstrates advanced competence. "
         "It produces clear, well-structured texts of complex subjects, expanding and supporting points of view "
         "with examples, reasons, and appropriate conclusions. It effectively varies tone, style, and register "
         "according to addressee and theme.",
    'B': "Intermediate proficiency — The writing reflects intermediate proficiency. "
         "It produces straightforward connected texts on familiar subjects, linking shorter elements into coherent sequences.",
    'A': "Basic proficiency — The writing shows basic ability to communicate about personal or familiar matters "
         "using simple words and short, isolated phrases."
}

# -------------------------------
# 🔍 Core Analysis
# -------------------------------
def truncate_text(text, max_words=500):
    words = text.split()
    return (" ".join(words[:max_words]), True) if len(words) > max_words else (text, False)

def analyze_text(text):
    matches = check_grammar_api(text)
    grammar_count = len(matches)

    # Build detailed issues (limit 10)
    detailed_issues = []
    for m in matches[:10]:
        error_text = text[m["offset"]: m["offset"] + m["length"]]
        raw_repl = m.get("replacements", [])
        if raw_repl:
            # extract string values from dicts
            repl_strings = []
            for r in raw_repl[:3]:
                if isinstance(r, dict):
                    repl_strings.append(r.get("value", ""))
                else:
                    repl_strings.append(str(r))
            replacements = ", ".join(repl_strings)
        else:
            replacements = "No suggestions"

        
        issue = f"• Issue: {m['message']}\n    Text: \"{error_text}\"\n    Suggestion(s): {replacements}"
        detailed_issues.append(issue)

    grammar_summary = f"Total Grammar Issues: {grammar_count}\n\n" + "\n\n".join(detailed_issues) \
        if detailed_issues else "✅ No major grammar issues detected."

    # Highlight text with tooltips
    highlighted_text = ""
    last_index = 0

    for m in matches:
        start, end = m["offset"], m["offset"] + m["length"]

        # Safely escape the tooltip text
        msg = html.escape(m["message"]).replace("'", "&#39;").replace('"', "&quot;")

        # Add clean text before the error
        highlighted_text += html.escape(text[last_index:start])

        # Add highlighted span with tooltip
        highlighted_text += (
            f"<span class='grammar-issue' title=\"{msg}\">{html.escape(text[start:end])}</span>"
        )

        last_index = end

    # Add any remaining text
    highlighted_text += html.escape(text[last_index:])

    # Lexical and readability metrics
    lex = LexicalRichness(text)
    ttr = round(lex.ttr, 2)
    readability = round(textstat.flesch_reading_ease(text), 2)

    # Sentence length
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    words = text.split()
    avg_sentence_length = len(words) / len(sentences) if sentences else 0

    # Sentiment
    sentiment = TextBlob(text).sentiment.polarity

    return grammar_summary, highlighted_text, ttr, readability, avg_sentence_length, sentiment, grammar_count

# -------------------------------
# 🧠 CEFR prediction (cached for speed if same text repeats)
# -------------------------------
@lru_cache(maxsize=128)
def predict_cefr(text):
    """
    Predict CEFR level using Hugging Face Inference API.
    Returns: label, confidence, probability dict
    """
    HF_TOKEN = os.environ["HF_TOKEN"]
    API_URL = f"https://api-inference.huggingface.co/models/pkim62/CEFR-classification-model"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        data = response.json()

        # Extract predicted label & scores
        if isinstance(data, dict) and "error" in data:
            # model not ready or error
            return "N/A", 0.0, {}
        
        # HF API returns a list of dicts like [{"label": "C", "score": 0.85}, ...]
        pred = data[0]
        label = pred["label"]
        confidence = float(pred["score"])
        # Build probability dict
        prob_dict = {d["label"]: float(d["score"]) for d in data}
        return label, confidence, prob_dict

    except Exception as e:
        print("HF Inference API error:", e)
        return "N/A", 0.0, {}


# -------------------------------
# 🧠 Interpretations
# -------------------------------
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

def generate_narrative(predicted_cefr, grammar_count, avg_sentence_length, sentiment, ttr, readability, confidence):
    feedback = []

    # Sentence structure
    if avg_sentence_length < 10: feedback.append("Sentences are quite short; consider combining ideas for more complexity.")
    elif avg_sentence_length > 25: feedback.append("Sentences are rather long; consider breaking them for clarity.")
    else: feedback.append("Sentence length is well-balanced.")

    # Sentiment
    if sentiment < -0.2: feedback.append("The tone leans negative.")
    elif sentiment > 0.2: feedback.append("The tone leans positive.")
    else: feedback.append("Tone is neutral and balanced.")

    feedback.append(f"**Readability (Flesch Score {readability}):** {interpret_readability(readability)}")
    feedback.append(f"**Lexical Diversity (TTR {ttr}):** {interpret_ttr(ttr)}")
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

    return "\n\n".join(feedback), summary

# -------------------------------
# 🔹 CEFR probability chart
# -------------------------------
def plot_cefr_probs(prob_dict):
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
    narrative_feedback, summary = generate_narrative(predicted_cefr, grammar_count, avg_sentence_length, sentiment, ttr, readability, confidence)
    cefr_chart = plot_cefr_probs(prob_dict)

    if truncated:
        summary += "\n\n⚠️ Note: Only the first 500 words were analyzed."

    return (
        predicted_cefr, confidence, grammar_summary, highlighted_text,
        ttr, readability, round(avg_sentence_length,1), f"{sentiment:.2f}",
        narrative_feedback, summary, cefr_chart
    )

# -------------------------------
# 🔹 Gradio interface
# -------------------------------
custom_css = """
<style>
textarea:focus, input:focus { outline: none !important; box-shadow: none !important; border: 1px solid #ccc !important; }
.grammar-issue { background-color: #ffdddd; border-bottom: 1px dotted red; cursor: help; }
.grammar-issue:hover { background-color: #ffecec; }
.custom-disclaimer { font-size:0.85em; color:#555; background-color:#f9f9f9; padding:10px; border-radius:8px; }
</style>
"""

with gr.Blocks(theme="gradio/soft", css=custom_css) as app:
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
    chart_output = gr.Image(label="CEFR Probability Distribution")

    gr.Markdown("""
    <div class="custom-disclaimer">
    ⚠️ <b>Disclaimer:</b><br>
    This tool is for <b>demonstration purposes only</b> and should not be used for commercial, official, or
    high-stakes essay scoring. Feedback is automatically generated and may contain inaccuracies.
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
# 🚀 Launch
# -------------------------------
if __name__ == "__main__":
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    port = int(os.environ.get("PORT", 7860))

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )



