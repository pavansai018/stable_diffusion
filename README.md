# Stable Diffusion from Scratch

A PyTorch implementation of Stable Diffusion v1.5 built from the ground up — every component written by hand, no diffusers library. This was a learning project to understand exactly what's happening inside a latent diffusion model.

## What's implemented

The full SD v1.5 inference pipeline:

- **CLIP text encoder** — 12-layer transformer, 768-dim embeddings, 77-token context window
- **VAE encoder/decoder** — compresses 512×512 RGB images into a 64×64×4 latent space and back
- **U-Net** — the denoising network, with residual blocks, self-attention, cross-attention (where the text conditioning happens), and skip connections
- **DDPM sampler** — linear noise schedule over 1000 training steps, with configurable inference steps
- **Pipeline** — ties everything together, supports text-to-image and image-to-image with classifier-free guidance

The model weights are loaded directly from the official `v1-5-pruned-emaonly.ckpt` checkpoint — no conversion needed beyond what `model_converter.py` handles.

## Setup

```bash
pip install torch torchvision transformers pillow tqdm
```

Download the weights and tokenizer files and place them in `data/`:
- `v1-5-pruned-emaonly.ckpt` — the standard SD 1.5 checkpoint
- `vocab.json` and `merges.txt` — BPE tokenizer files (same ones used by CLIP/OpenAI)

## Usage

Open `sd/demo.ipynb`. The two main modes:

**Text to image**
```python
input_image = None
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens."
uncond_prompt = ""  # negative prompt
```

**Image to image**
```python
input_image = Image.open("../images/dog.jpg")
strength = 0.9  # how much to deviate from the input image (0 = keep original, 1 = ignore it)
```

Parameters worth knowing:
- `cfg_scale` — classifier-free guidance scale, roughly controls how closely the output follows the prompt. 7–8 is a good starting point, higher values push harder toward the prompt but can oversaturate
- `num_inference_steps` — more steps = slower but generally cleaner. 50 is the default
- `seed` — set this for reproducibility

## File structure

```
sd/
  pipeline.py        — main generate() function, orchestrates everything
  ddpm.py            — DDPM noise schedule and sampler step
  diffusion.py       — U-Net + time embedding
  clip.py            — CLIP text encoder
  vae_encoder.py     — VAE encoder (image → latent)
  vae_decoder.py     — VAE decoder (latent → image)
  attention.py       — SelfAttention and CrossAttention
  model_loader.py    — loads all four models from the .ckpt file
  model_converter.py — maps checkpoint keys to our model's parameter names
  demo.ipynb         — example notebook
data/
  v1-5-pruned-emaonly.ckpt
  vocab.json
  merges.txt
images/
  dog.jpg            — sample input for image-to-image
```

## How it works (briefly)

Text-to-image in a nutshell: the prompt and negative prompt both get encoded by CLIP into context vectors. Random noise is sampled in latent space. Then the U-Net iteratively denoises it over N steps — at each step it predicts the noise given the current latent and the text context (via cross-attention). CFG combines the conditional and unconditional predictions to steer the result toward the prompt. The final latent gets decoded by the VAE back into a pixel image.

Image-to-image is the same except you start from a noised version of the input image rather than pure noise. The `strength` parameter controls how many denoising steps to skip — lower strength means you start closer to the original image and fewer steps run.

## Notes

All models run in fp16 and are moved to CPU when idle to reduce VRAM usage. The pipeline assumes CUDA is available but falls back to CPU if not.
