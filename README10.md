# Experiment 10: FINE-TUNING A PRE-TRAINED LANGUAGE MODEL FOR A DOMAIN-SPECIFIC APPLICATION

## AIM

To fine-tune a pre-trained language model (DistilBERT) on a domain-specific labelled dataset for a text-classification task.

## ALGORITHM

1. Load a domain-specific labelled dataset (text + label pairs).
2. Load the pre-trained tokenizer and tokenize the dataset (padding/truncation).
3. Load the pre-trained model with a classification head matching the number of labels.
4. Define training arguments (learning rate, epochs, batch size).
5. Train the model on the training split using the Hugging Face Trainer API.
6. Evaluate the fine-tuned model on the validation/test split.
7. Save the fine-tuned model for later inference.

## PROGRAM / CODE

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score

# 1. Load a domain-specific dataset (example: IMDB movie reviews)
dataset = load_dataset("imdb")
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

# 2. Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_ds = small_train.map(tokenize, batched=True)
test_ds = small_test.map(tokenize, batched=True)

# 3. Load pre-trained model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 4. Training arguments
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    logging_steps=50
)

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# 5. Train
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)
trainer.train()

# 6. Evaluate and save
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)
model.save_pretrained("./fine_tuned_distilbert_imdb")
```

## SAMPLE INPUT

IMDB movie review dataset (2000 training samples, 500 test samples), 2 epochs fine-tuning

## SAMPLE OUTPUT

```text
Epoch 1/2 - loss: 0.41 - accuracy: 0.83
Epoch 2/2 - loss: 0.24 - accuracy: 0.89
Evaluation metrics: {'eval_loss': 0.28, 'eval_accuracy': 0.887}
```

## RESULT

The pre-trained DistilBERT model was successfully fine-tuned on a domain-specific sentiment-classification dataset, improving task-specific accuracy through transfer learning.
