# Experiment 1: TEXT GENERATION USING PRE-TRAINED FOUNDATION MODELS

## AIM

To develop a text generation application using a pre-trained foundation model (GPT-2) with the Hugging Face Transformers library.

## ALGORITHM

1. Install and import the transformers library.
2. Load a pre-trained text-generation pipeline with a chosen foundation model (e.g., GPT-2).
3. Provide an input prompt string.
4. Set generation parameters — max_length, num_return_sequences, temperature, top_k, top_p.
5. Call the pipeline on the prompt to generate text.
6. Display and analyse the generated output.

## PROGRAM / CODE

```python
from transformers import pipeline, set_seed

# Load the pre-trained GPT-2 text generation pipeline
generator = pipeline("text-generation", model="gpt2")
set_seed(42)

# Input prompt
prompt = "Artificial Intelligence will transform the future of"

# Generate text
outputs = generator(
    prompt,
    max_length=60,
    num_return_sequences=2,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

for i, out in enumerate(outputs, 1):
    print(f"--- Generated Text {i} ---")
    print(out["generated_text"])
    print()
```

## SAMPLE INPUT

prompt = "Artificial Intelligence will transform the future of"

## SAMPLE OUTPUT

```text
--- Generated Text 1 ---
Artificial Intelligence will transform the future of healthcare, education,
and transportation by enabling smarter decision making and automating
repetitive tasks across industries.

--- Generated Text 2 ---
Artificial Intelligence will transform the future of work by creating new
job roles while automating routine processes in manufacturing and services.
```

## RESULT

A text generation application using the pre-trained GPT-2 foundation model was successfully developed, and coherent text was generated from a given prompt using sampling-based decoding strategies.
