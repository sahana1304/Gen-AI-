# Experiment 9: MULTIMODAL AI APPLICATION INTEGRATING TEXT AND IMAGE INPUTS

## AIM

To develop a multimodal AI application that processes both text and image inputs using a pre-trained vision-language model (BLIP) to perform image captioning and visual question answering.

## ALGORITHM

1. Load a pre-trained BLIP model and processor for image captioning.
2. Load an input image using PIL.
3. Pass the image through the processor and model to generate a caption.
4. Load a BLIP-VQA model and processor.
5. Provide the same image along with a natural-language question.
6. Generate and display the answer produced by the model.

## PROGRAM / CODE

```python
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)
from PIL import Image
import requests

image_url = "https://images.unsplash.com/photo-1519125323398-675f0ddb6308"
raw_image = Image.open(
    requests.get(image_url, stream=True).raw
).convert("RGB")

# ---------- Image Captioning ----------
cap_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
cap_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

inputs = cap_processor(raw_image, return_tensors="pt")
caption_ids = cap_model.generate(**inputs, max_new_tokens=30)
caption = cap_processor.decode(
    caption_ids[0], skip_special_tokens=True
)
print("Generated Caption:", caption)

# ---------- Visual Question Answering ----------
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

question = "What animal is in the picture?"
vqa_inputs = vqa_processor(raw_image, question, return_tensors="pt")
answer_ids = vqa_model.generate(**vqa_inputs)
answer = vqa_processor.decode(
    answer_ids[0], skip_special_tokens=True
)

print("Question:", question)
print("Answer:", answer)
```

## SAMPLE INPUT

Image of a dog in a field (URL) + Question: 'What animal is in the picture?'

## SAMPLE OUTPUT

```text
Generated Caption: a dog running through a grassy field
Question: What animal is in the picture?
Answer: dog
```

## RESULT

A multimodal AI application integrating text and image inputs was successfully developed using the BLIP model, performing both image captioning and visual question answering.
