# Experiment 5: SENTIMENT ANALYSIS AND DOCUMENT CLASSIFICATION USING FOUNDATION MODELS

## AIM

To perform sentiment analysis and multi-class document classification using pre-trained foundation models.

## ALGORITHM

1. Load the sentiment-analysis pipeline with a fine-tuned foundation model.
2. Pass sample review sentences to the pipeline and record sentiment + score.
3. Load the zero-shot-classification pipeline with BART-large-MNLI.
4. Define a document and a list of candidate category labels.
5. Run zero-shot classification to obtain the probability for each label.
6. Display sentiment results and the top predicted document category.

## PROGRAM / CODE

```python
from transformers import pipeline

# ---------- Sentiment Analysis ----------
sentiment_analyzer = pipeline("sentiment-analysis")

reviews = [
    "The new smartphone has an amazing camera and battery life!",
    "The delivery was late and the packaging was damaged."
]

for review in reviews:
    result = sentiment_analyzer(review)[0]
    print(f"Review: {review}\n -> {result['label']} ({round(result['score'],3)})\n")

# ---------- Document Classification (Zero-Shot) ----------
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

document = "The central bank raised interest rates to control rising inflation."
candidate_labels = ["Politics", "Economy", "Sports", "Technology"]

classification = classifier(document, candidate_labels)

print("Document:", document)
for label, score in zip(classification["labels"], classification["scores"]):
    print(f"{label}: {round(score,3)}")
```

## SAMPLE INPUT

Two product reviews (sentiment) + one news sentence with candidate labels [Politics, Economy, Sports, Technology]

## SAMPLE OUTPUT

```text
Review: The new smartphone has an amazing camera and battery life!
-> POSITIVE (0.999)

Review: The delivery was late and the packaging was damaged.
-> NEGATIVE (0.998)

Document: The central bank raised interest rates to control rising inflation.
Economy: 0.94
Politics: 0.04
Technology: 0.01
Sports: 0.01
```

## RESULT

Sentiment analysis and zero-shot document classification were successfully performed using pre-trained foundation models, correctly identifying sentiment polarity and document category.
