import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def rescale(x: torch.Tensor, old_range: tuple[int, int], new_range: tuple[int, int], clamp=True):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x = x - old_min
    x = x * (new_max - new_min) / (old_max - old_min)
    x = x + new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # shape: (160,)
    freqs = torch.pow(input=10000, exponent=-torch.arange(start=0, end=160, dtype=torch.float32)/160)
    # shape(1, 160)
    x = torch.tensor(data=[timestep], dtype=torch.float32)[:, None] * freqs[None]

    # shape (1, 160 * 2)
    return torch.cat([torch.cos(input=x), torch.sin(input=x)], dim=-1)



def generate(prompt: str, uncond_prompt: str, input_image: None | torch.Tensor, strength: float = 0.8, 
             do_cfg: bool = True, cfg_scale: float = 7.5, sampler_name: str='ddpm', n_inference_steps: int = 50, 
             models: dict={}, seed: int|None = None, device:str|None=None, idle_device:str|None=None,
             tokenizer: str | None = None
        ) -> torch.Tensor:
    
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError('Strength must be between 0 and 1')
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models['clip']
        clip.to(device) 

        if do_cfg:
            # convert the prompt into tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            # convert input ids into tensor (batch_size, seq len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            # (batch size, seq len) -> (batch size, seq len, dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (batch size, seq len) -> (batch size, seq len, dim)
            uncond_context = clip(uncond_tokens)

            # (2, seq len, dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])

        else:
            # convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)

        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f'unknown sampler: {sampler_name}')
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # height, width, channels
            input_image_tensor = np.array(input_image_tensor)
            # convert to tensor
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # unet accepts image in the range [-1, 1]
            input_image_tensor = rescale(x=input_image_tensor, old_range=(0, 255), new_range=(-1, 1))
            # height, width, channels -> batch_size, height, width, channels
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # batch_size, height, width, channels -> batch_size, channels, height, width
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # batch_size, 4, latent height, latent width
            encoder_noise = torch.randn(size=latents_shape, generator=generator, device=device)
            # batch_size, 4, latent_height, latent_width
            latents = encoder(input_image_tensor, encoder_noise)

            # add noise to the latents
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            to_idle(encoder)

        else:
            # if we are running text-to-image, start with random noise N(0, I)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(size=latents_shape, generator=generator, device=device)
        
        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep=timestep).to(device)

            # (batch_szie, 4, latent_height, latent_width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)
            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(x=images, old_range=(-1, 1), new_range=(0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]