import torch
from diffusers import StableDiffusionXLPipeline

from ip_adapter.ip_adapter import IPAdapterPlusXL, IPAdapterXL


def load_ip(type: str = "plus"):
    sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
    ).to(dtype=torch.float16)
    if type == "plus":
        ip_model = IPAdapterPlusXL(
            sd_pipe=sdxl_pipeline,
            image_encoder_path="models/image_encoder",
            ip_ckpt="weights/ip_adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
            device="cuda",
            num_tokens=16,
        )
    elif type == "base":
        ip_model = IPAdapterXL(
            sd_pipe=sdxl_pipeline,
            image_encoder_path="sdxl_models/image_encoder",
            ip_ckpt="weights/ip_adapter/sdxl_models/ip-adapter_sdxl.bin",
            device="cuda",
        )

    return ip_model


def get_image_ip_plus_tokens(ip_model, pil_image):
    image_processed = ip_model.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
    clip_image = image_processed.to("cuda", dtype=torch.float32)
    clip_image_embeds = ip_model.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
    image_prompt_embeds = ip_model.image_proj_model(clip_image_embeds)
    return image_prompt_embeds.cpu().detach(), clip_image_embeds.cpu().detach()


def get_image_ip_base_tokens(ip_model, pil_image):
    image_processed = ip_model.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
    clip_image = image_processed.to("cuda", dtype=torch.float32)
    clip_image_embeds = ip_model.image_encoder(clip_image).image_embeds
    image_prompt_embeds = ip_model.image_proj_model(clip_image_embeds)
    return image_prompt_embeds.cpu().detach(), clip_image_embeds.cpu().detach()
