# Experiment 2: PROMPT ENGINEERING TECHNIQUES FOR CONTENT GENERATION, REASONING AND TASK AUTOMATION

## AIM

To implement zero-shot, one-shot, few-shot, and chain-of-thought prompting techniques for content generation, reasoning, and task automation using a Large Language Model.

## ALGORITHM

1. Define the task to be solved (e.g., arithmetic reasoning, content generation, classification).
2. Construct a zero-shot prompt containing only the instruction and query.
3. Construct a few-shot prompt containing 2–3 example input-output pairs followed by the query.
4. Construct a chain-of-thought prompt that asks the model to 'think step by step'.
5. Send each prompt to the LLM and record the response.
6. Compare the outputs of the three techniques for correctness and quality.

## PROGRAM / CODE

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

# 1. Zero-shot prompt
zero_shot_prompt = "Classify the sentiment of this review as Positive or Negative: 'The product quality is excellent!'
Sentiment:"

# 2. Few-shot prompt
few_shot_prompt = """Review: 'I loved this movie, it was fantastic.'
Sentiment: Positive
Review: 'The service was slow and disappointing.'
Sentiment: Negative
Review: 'The product quality is excellent!'
Sentiment:"""

# 3. Chain-of-Thought prompt
cot_prompt = """Q: A shop had 15 apples. It sold 6 and then received 10 more. How many apples now?
A: Let's think step by step. 15 - 6 = 9. 9 + 10 = 19. The answer is 19.
Q: A library had 120 books. It lent out 45 and bought 30 new books. How many books now?
A: Let's think step by step."""

for name, p in [("Zero-shot", zero_shot_prompt),
                ("Few-shot", few_shot_prompt),
                ("Chain-of-Thought", cot_prompt)]:
    out = generator(p, max_length=len(p.split())+40, num_return_sequences=1, do_sample=False)
    print(f"=== {name} ===")
    print(out[0]["generated_text"])
    print()
```

## SAMPLE INPUT

Sentiment classification query and a two-step arithmetic word problem (see prompts above)

## SAMPLE OUTPUT

```text
=== Zero-shot ===
Sentiment: Positive
=== Few-shot ===
Review: 'The product quality is excellent!'
Sentiment: Positive
=== Chain-of-Thought ===
A: Let's think step by step. 120 - 45 = 75. 75 + 30 = 105. The answer is 105.
```

## RESULT

Zero-shot, few-shot, and chain-of-thought prompting techniques were successfully implemented, and their impact on content generation and reasoning accuracy was analysed and compared.
