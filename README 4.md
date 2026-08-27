# Experiment 4: TEXT SUMMARIZATION AND QUESTION-ANSWERING SYSTEM USING LARGE LANGUAGE MODELS

## AIM

To develop a text summarization system and a question-answering system using pre-trained Large Language Models (BART and DistilBERT).

## ALGORITHM

1. Load the summarization pipeline with a pre-trained BART model.
2. Provide a long passage of text and generate a summary with defined min/max length.
3. Load the question-answering pipeline with a pre-trained DistilBERT-SQuAD model.
4. Provide a context passage and a natural-language question.
5. Run the QA pipeline to extract the answer span along with a confidence score.
6. Display the summary and the answer.

## PROGRAM / CODE

```python
from transformers import pipeline

# ---------- Text Summarization ----------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """Generative AI refers to a class of artificial intelligence models capable of
producing new content such as text, images, audio, and video. Large Language Models (LLMs)
such as GPT and LLaMA are trained on massive text corpora and can perform a wide range of
natural language tasks including translation, summarization, and question answering. These
models are increasingly being deployed in industry applications ranging from customer support
to software development, transforming how humans interact with machines."""

summary = summarizer(article, max_length=45, min_length=20, do_sample=False)
print("Summary:
", summary[0]["summary_text"])

# ---------- Question Answering ----------
qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)
context = article
question = "What are Large Language Models trained on?"
answer = qa(question=question, context=context)

print("
Question:", question)
print("Answer:", answer["answer"], "| Confidence:", round(answer["score"], 3))
```

## SAMPLE INPUT

Article about Generative AI and LLMs (see code) + Question: 'What are Large Language Models trained on?'

## SAMPLE OUTPUT

```text
Summary:
Generative AI models produce new content such as text, images, audio and video.
Large Language Models are trained on massive text corpora and perform many NLP tasks.
Question: What are Large Language Models trained on?
Answer: massive text corpora | Confidence: 0.87
```

## RESULT

A text summarization system using BART and a question-answering system using DistilBERT-SQuAD were successfully developed and tested on sample text.
