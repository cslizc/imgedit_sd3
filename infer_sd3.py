import PIL
import requests
import torch
from hf_examples.instp2p_sd3_pl import StableDiffusion3InstructPix2PixPipeline
from diffusers import SD3Transformer2DModel

model_path = "./saved_models/magicBrush_sd3/checkpoint-3500/transformer/"
transformer = SD3Transformer2DModel.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
)
#print(transformer.pos_embed.proj.weight.size())


pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
	"./dl_models/stable-diffusion-3-medium-diffusers", 
	transformer=transformer,
	torch_dtype=torch.float16
	).to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/test_pix2pix_4.png"


def download_image(url):
   image = PIL.Image.open(requests.get(url, stream=True).raw)
   image = PIL.ImageOps.exif_transpose(image)
   image = image.convert("RGB")
   return image

image = download_image(url)
prompt = "wipe out the lake."
num_inference_steps = 28
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipe(prompt,
   image=image,
   #num_inference_steps=num_inference_steps,
   image_guidance_scale=image_guidance_scale,
   guidance_scale=guidance_scale,
   generator=generator,
).images[0]
edited_image.save(f"./output/magicBrush_sd3-{prompt.replace(" ", "_").replace(".", "")}.png")