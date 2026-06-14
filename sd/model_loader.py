from clip import CLIP
from vae_encoder import VAE_Encoder
from vae_decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter

def preload_models_from_standard_weights(ckpt_path: str, device: str):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    device = 'cpu'
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)
    encoder.half()

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)
    decoder.half()

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)
    diffusion.half()

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)
    clip.half()

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }