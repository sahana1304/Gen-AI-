# Experiment 7: AI-POWERED CODE GENERATION AND DEBUGGING ASSISTANT

## AIM

To develop an AI-powered assistant that generates code from natural-language descriptions and helps identify/fix bugs in existing code using a pre-trained code-generation model.

## ALGORITHM

1. Load a pre-trained code-generation model and its tokenizer.
2. Provide a natural-language instruction describing the required function.
3. Generate code using the model and display the output.
4. Provide a snippet of buggy code with an explicit 'fix the bug' instruction.
5. Generate the corrected code using the model.
6. Compare the buggy and fixed versions to verify correctness.

## PROGRAM / CODE

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

def generate_code(prompt, max_new_tokens=80):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 1. Code generation from a natural-language instruction
prompt1 = "# Write a Python function to check if a number is prime
def is_prime(n):"
print("Generated Function:
", generate_code(prompt1))

# 2. Debugging a faulty snippet
buggy_code = """# The following function should return the factorial of n, but has a bug. Fix it.
def factorial(n):
    result = 0
    for i in range(1, n+1):
        result = result * i
    return result
# Corrected function:
def factorial_fixed(n):"""

print("
Debug Suggestion:
", generate_code(buggy_code, max_new_tokens=60))
```

## SAMPLE INPUT

Instruction: 'check if a number is prime' + buggy factorial() function initialised with result = 0

## SAMPLE OUTPUT

```text
Generated Function:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

Debug Suggestion:
def factorial_fixed(n):
    result = 1
    for i in range(1, n+1):
        result = result * i
    return result
```

## RESULT

An AI-powered assistant was successfully developed that generates correct Python code from natural-language instructions and identifies/fixes bugs in existing code using a pre-trained code LLM.
