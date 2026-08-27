# Experiment 3: CONVERSATIONAL AI CHATBOT USING TRANSFORMER-BASED LANGUAGE MODELS

## AIM

To build a conversational AI chatbot capable of holding a multi-turn dialogue using a transformer-based language model (DialoGPT).

## ALGORITHM

1. Load the DialoGPT tokenizer and model.
2. Initialise an empty chat-history tensor.
3. In a loop, accept user input and encode it, appending an end-of-sentence token.
4. Concatenate the new input with the existing chat history.
5. Generate a response from the model using the combined history as context.
6. Decode and display only the newly generated tokens as the bot's reply.
7. Update the chat history and repeat for the next turn.

## PROGRAM / CODE

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
chat_history_ids = None

print("Chatbot ready! Type 'quit' to exit.")
for step in range(5):
    user_input = input(">> User: ")
    if user_input.lower() == "quit":
        break

    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, return_tensors="pt"
    )
    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None else new_input_ids
    )

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )
    print(f"Bot: {response}")
```

## SAMPLE INPUT

>> User: Hi, how are you? >> User: What can you help me with?

## SAMPLE OUTPUT

```text
Bot: I'm doing great, thanks for asking! How about you?
Bot: I can chat with you about almost anything - just ask away!
```

## RESULT

A multi-turn conversational chatbot was successfully built using the transformer-based DialoGPT model, capable of retaining dialogue context across turns.
