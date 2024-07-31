import os
import torch
from diffusers import StableDiffusionPipeline

def generate_images(pipe, prompts, num_images, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  for i, prompt in enumerate(prompts):
    for j in range(num_images):
      image = pipe(prompt, width=256, height=256).images[0]
      image.save(os.path.join(output_dir, f"{prompt.replace(' ', '_')}_{j}.png"))

def main():
  giraffe_prompts = ["a photograph of a giraffe"]
  zebra_prompts = ["a photograph of a zebra"]

  device = "cuda" if torch.cuda.is_available() else "cpu"
  pipe = StableDiffusionPipeline.from_pretrained("lambdalabs/miniSD-diffusers").to(device)

  # Disable safety checker
  pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

  generate_images(pipe, giraffe_prompts, 100, 'generated_images/giraffes')
  generate_images(pipe, zebra_prompts, 100, 'generated_images/zebras')

if __name__ == "__main__":
  main()
