import os
import threading
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = os.getenv("MODEL_NAME", "google-t5/t5-small")
LANGUAGES = [
    "English", "French", "German", "Spanish", "Romanian", "Hindi", "Tamil"
]

# Load once when the app starts; keep generation serialized for small CPU instances.
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
generation_lock = threading.Lock()
print("Model loaded.")

def translate(text, source_language, target_language, max_new_tokens):
    if not text or not text.strip():
        return "Please enter some text to translate."
    if source_language == target_language:
        return "Choose two different languages."
    if len(text) > 5000:
        return "Please keep your text under 5,000 characters for this demo."

    prompt = f"translate {source_language} to {target_language}: {text.strip()}"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with generation_lock, torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            num_beams=4,
            early_stopping=True
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return result or "The model did not return a translation. Try a shorter sentence."

def swap_languages(source, target):
    return target, source

custom_css = """
.gradio-container {max-width: 1000px !important; margin: 0 auto !important;}
#hero {text-align: center; padding: 1.25rem 0 .5rem;}
#hero h1 {font-size: 2.25rem; margin-bottom: .35rem;}
#hero p {color: var(--body-text-color-subdued); font-size: 1.05rem;}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
    css=custom_css,
    title="AI Translator"
) as demo:
    with gr.Column(elem_id="hero"):
        gr.Markdown("# 🌐 AI Translator")
        gr.Markdown("Translate text between languages with a simple AI-powered demo.")

    with gr.Row():
        with gr.Column(scale=1):
            source = gr.Dropdown(
                choices=LANGUAGES, value="English", label="Translate from"
            )
        with gr.Column(scale=0, min_width=80):
            gr.Markdown(" ")
            swap = gr.Button("⇄ Swap", variant="secondary")
        with gr.Column(scale=1):
            target = gr.Dropdown(
                choices=LANGUAGES, value="French", label="Translate to"
            )

    with gr.Row():
        input_text = gr.Textbox(
            label="Your text",
            placeholder="Type or paste the text you want to translate…",
            lines=7,
            max_lines=14
        )
        output_text = gr.Textbox(
            label="Translation",
            placeholder="Your translation will appear here…",
            lines=7,
            max_lines=14,
            interactive=False
        )

    with gr.Row():
        max_tokens = gr.Slider(
            minimum=32, maximum=256, value=128, step=32,
            label="Maximum output length (tokens)"
        )
    with gr.Row():
        translate_btn = gr.Button("Translate", variant="primary", size="lg")
        clear_btn = gr.ClearButton(
            [input_text, output_text], value="Clear"
        )

    translate_btn.click(
        fn=translate,
        inputs=[input_text, source, target, max_tokens],
        outputs=output_text
    )
    input_text.submit(
        fn=translate,
        inputs=[input_text, source, target, max_tokens],
        outputs=output_text
    )
    swap.click(
        fn=swap_languages,
        inputs=[source, target],
        outputs=[source, target]
    )

    gr.Markdown(
        """
        ---
        **Model note:** This demo uses `google-t5/t5-small`, a general-purpose
        pretrained T5 model rather than a translation-specialized model. Results
        may be inaccurate, especially for Tamil and some language pairs. Avoid
        using it for legal, medical, or other high-stakes translations.
        """
    )

if __name__ == "__main__":
    # Hosted platforms provide their own public URL; share=True is not needed.
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
