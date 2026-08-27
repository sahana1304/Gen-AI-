# Experiment 11: AI-BASED CONTENT GENERATION SYSTEM FOR TEXT, IMAGE AND MULTIMEDIA APPLICATIONS

## AIM

To develop an integrated AI-based content generation system that produces text, image and audio content from a single high-level user input.

## ALGORITHM

1. Accept a high-level topic/theme from the user.
2. Use a text-generation LLM to produce a short article/caption on the topic.
3. Derive an image prompt from the generated text.
4. Use a diffusion model to generate an accompanying image from the image prompt.
5. Convert the generated text into speech using a text-to-speech model.
6. Save/display the generated text, image, and audio as the final multimedia content package.

## PROGRAM / CODE

```python
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import torch

topic = "The benefits of renewable energy"

# 1. Text generation
text_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)
text_prompt = f"Write a short, engaging paragraph about: {topic}"
generated_text = text_generator(
    text_prompt, max_length=80
)[0]["generated_text"]

print("Generated Text:
", generated_text)

# 2. Image generation (derived prompt)
image_prompt = f"An illustration representing {topic}, digital art"
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = sd_pipe(
    image_prompt,
    num_inference_steps=25
).images[0]

image.save("content_image.png")
print("Image saved as content_image.png")

# 3. Audio generation (text-to-speech)
tts = gTTS(text=generated_text, lang="en")
tts.save("content_audio.mp3")
print("Audio saved as content_audio.mp3")
```

## SAMPLE INPUT

topic = "The benefits of renewable energy"

## SAMPLE OUTPUT

```text
Generated Text:
Renewable energy sources like solar and wind reduce carbon emissions,
lower energy costs over time, and help create a sustainable future for
generations to come.
Image saved as content_image.png
Audio saved as content_audio.mp3
```

## RESULT

An integrated AI-based content generation system producing text, image, and audio outputs from a single topic input was successfully developed.
