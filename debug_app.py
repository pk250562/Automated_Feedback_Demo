import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# 🔧 Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🔧 Debug function
# -------------------------------
def debug_cefr_model():
    result = {}

    # List all files in the repo
    result["files_in_repo"] = {f: os.path.getsize(f) for f in os.listdir(".")}

    # Check model.safetensors
    safetensors_path = "model.safetensors"
    if os.path.isfile(safetensors_path):
        with open(safetensors_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
            if "version https://git-lfs.github.com/spec/v1" in first_line:
                result["model_safetensors"] = "❌ LFS pointer, not real weights"
            else:
                result["model_safetensors"] = "✅ real weights"
    else:
        result["model_safetensors"] = "❌ not found"

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(".", use_fast=False, local_files_only=True)
        result["tokenizer"] = "✅ loaded"
    except Exception as e:
        result["tokenizer"] = f"❌ failed: {e}"

    # Load model (INT8)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            ".", device_map="auto", load_in_8bit=True, local_files_only=True
        )
        model.to(device)
        model.eval()
        result["model"] = "✅ loaded"
    except Exception as e:
        result["model"] = f"❌ failed: {e}"

    return result

# -------------------------------
# 🔹 Gradio UI
# -------------------------------
with gr.Blocks() as app:
    gr.Markdown("## ⚡ CEFR INT8 Model Debug")
    gr.Markdown("This will check if your INT8 model and tokenizer are loaded correctly without calling HF API.")
    debug_btn = gr.Button("Run Debug")
    debug_output = gr.Textbox(lines=15)
    debug_btn.click(fn=debug_cefr_model, inputs=[], outputs=debug_output)

# -------------------------------
# 🚀 Launch
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.launch(server_name="0.0.0.0", server_port=port)
