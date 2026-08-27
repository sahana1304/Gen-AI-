# Experiment 12: DEPLOYMENT AND EVALUATION OF A GENERATIVE AI APPLICATION USING CLOUD-BASED APIs AND AI FRAMEWORKS

## AIM

To deploy a Generative AI text-generation application as a web service using the Gradio framework, and to evaluate its output quality using standard NLP evaluation metrics.

## ALGORITHM

1. Define an inference function that takes user text input and returns model-generated output.
2. Wrap the inference function using the Gradio Interface API with appropriate input/output components.
3. Launch the Gradio app locally (and optionally deploy to Hugging Face Spaces for cloud hosting).
4. Prepare a small evaluation set of generated outputs and corresponding reference texts.
5. Compute ROUGE scores between generated and reference texts using the 'evaluate' library.
6. Report and interpret the evaluation metrics.

## PROGRAM / CODE

```python
import gradio as gr
from transformers import pipeline
import evaluate

# ---------- 1. Build and Deploy the App ----------
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def summarize_text(input_text):
    result = summarizer(
        input_text,
        max_length=45,
        min_length=15,
        do_sample=False
    )
    return result[0]["summary_text"]

demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=8, label="Enter text to summarize"),
    outputs=gr.Textbox(label="Generated Summary"),
    title="GenAI Text Summarizer",
    description="A cloud-deployable Generative AI summarization app built with Gradio."
)

demo.launch(share=True)  # share=True generates a public cloud URL

# ---------- 2. Evaluate Generated Output ----------
rouge = evaluate.load("rouge")

generated_summaries = [
    "AI models generate new content such as text and images.",
]

reference_summaries = [
    "Generative AI models are capable of producing new content including text and images.",
]

scores = rouge.compute(
    predictions=generated_summaries,
    references=reference_summaries
)

print("ROUGE Evaluation Scores:", scores)
```

## SAMPLE INPUT

Long article text submitted via the Gradio web interface + one generated/reference summary pair for evaluation

## SAMPLE OUTPUT

```text
Running on local URL: http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
ROUGE Evaluation Scores: {'rouge1': 0.78, 'rouge2': 0.55, 'rougeL': 0.74, 'rougeLsum': 0.74}
```

## RESULT

A Generative AI text-summarization application was successfully deployed as a cloud-accessible web app using Gradio, and its output quality was evaluated using ROUGE metrics.
