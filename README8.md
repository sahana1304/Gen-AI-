# Experiment 8: IMAGE GENERATION APPLICATION USING DIFFUSION MODELS

## AIM

To implement an image generation application using a pre-trained Diffusion Model (Stable Diffusion) that synthesises images from text prompts.

## ALGORITHM

1. Load the pre-trained Stable Diffusion pipeline onto the GPU.
2. Define a descriptive text prompt for the desired image.
3. Set the number of inference (denoising) steps and guidance scale.
4. Run the pipeline: start from random noise and iteratively denoise conditioned on the prompt.
5. Retrieve and save/display the final generated image.
6. Experiment with different prompts and guidance-scale values to observe quality changes.

## PROGRAM / CODE

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

prompt = "A futuristic city skyline at sunset, digital art, highly detailed"
image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("generated_city.png")
print("Image generated and saved as generated_city.png")
```

## SAMPLE INPUT

prompt = "A futuristic city skyline at sunset, digital art, highly detailed"

## SAMPLE OUTPUT

```text
Image generated and saved as generated_city.png

(A 512x512 PNG image showing a futuristic city skyline with warm sunset lighting is produced and saved to disk.)
```

## RESULT

An image generation application using the Stable Diffusion diffusion model was successfully implemented, generating a realistic image from a given text prompt.
