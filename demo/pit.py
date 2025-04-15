import random
from pathlib import Path
from typing import Optional

import numpy as np
import pyrallis
import torch
from diffusers import (
    StableDiffusionXLPipeline,
)
from huggingface_hub import hf_hub_download
from PIL import Image

from ip_adapter import IPAdapterPlusXL
from model.dit import DiT_Llama
from model.pipeline_pit import PiTPipeline
from training.train_config import TrainConfig


def paste_on_background(image, background, min_scale=0.4, max_scale=0.8, scale=None):
    # Calculate aspect ratio and determine resizing based on the smaller dimension of the background
    aspect_ratio = image.width / image.height
    scale = random.uniform(min_scale, max_scale) if scale is None else scale
    new_width = int(min(background.width, background.height * aspect_ratio) * scale)
    new_height = int(new_width / aspect_ratio)

    # Resize image and calculate position
    image = image.resize((new_width, new_height), resample=Image.LANCZOS)
    pos_x = random.randint(0, background.width - new_width)
    pos_y = random.randint(0, background.height - new_height)

    # Paste the image using its alpha channel as mask if present
    background.paste(image, (pos_x, pos_y), image if "A" in image.mode else None)
    return background


def set_seed(seed: int):
    """Ensures reproducibility across multiple libraries."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU random seed
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random seed
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmarking to avoid randomness


class PiTDemoPipeline:
    def __init__(self, prior_repo: str, prior_path: str):
        # Download model and config
        prior_ckpt_path = hf_hub_download(
            repo_id=prior_repo,
            filename=str(prior_path),
            local_dir="pretrained_models",
        )
        prior_cfg_path = hf_hub_download(
            repo_id=prior_repo, filename=str(Path(prior_path).parent / "cfg.yaml"), local_dir="pretrained_models"
        )
        self.model_cfg: TrainConfig = pyrallis.load(TrainConfig, open(prior_cfg_path, "r"))

        self.weight_dtype = torch.float32
        self.device = "cuda:0"
        prior = DiT_Llama(
            embedding_dim=2048,
            hidden_dim=self.model_cfg.hidden_dim,
            n_layers=self.model_cfg.num_layers,
            n_heads=self.model_cfg.num_attention_heads,
        )
        prior.load_state_dict(torch.load(prior_ckpt_path))
        image_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        ip_ckpt_path = hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename="ip-adapter-plus_sdxl_vit-h.bin",
            subfolder="sdxl_models",
            local_dir="pretrained_models",
        )

        self.ip_model = IPAdapterPlusXL(
            image_pipe,
            "models/image_encoder",
            ip_ckpt_path,
            self.device,
            num_tokens=16,
        )
        self.image_processor = self.ip_model.clip_image_processor

        empty_image = Image.new("RGB", (256, 256), (255, 255, 255))
        zero_image = torch.Tensor(self.image_processor(empty_image)["pixel_values"][0])
        self.zero_image_embeds = self.ip_model.get_image_embeds(zero_image.unsqueeze(0), skip_uncond=True)

        prior_pipeline = PiTPipeline(
            prior=prior,
        )
        self.prior_pipeline = prior_pipeline.to(self.device)
        set_seed(42)

    def run(self, crops_paths: list[str], scale: float = 2.0, seed: Optional[int] = None, n_images: int = 1):
        if seed is not None:
            set_seed(seed)
        processed_crops = []
        input_images = []

        crops_paths = [None] + crops_paths
        # Extend to >3 with Nones
        while len(crops_paths) < 3:
            crops_paths.append(None)

        for path_ind, path in enumerate(crops_paths):
            if path is None:
                image = Image.new("RGB", (224, 224), (255, 255, 255))
            else:
                image = Image.open(path).convert("RGB")
                if path_ind > 0 or not self.model_cfg.use_ref:
                    background = Image.new("RGB", (1024, 1024), (255, 255, 255))
                    image = paste_on_background(image, background, scale=0.92)
                else:
                    image = image.resize((1024, 1024))
                input_images.append(image)
                # Name should be parent directory name
            processed_image = (
                torch.Tensor(self.image_processor(image)["pixel_values"][0])
                .to(self.device)
                .unsqueeze(0)
                .to(self.weight_dtype)
            )
            processed_crops.append(processed_image)

        image_embed_inputs = []
        for crop_ind in range(len(processed_crops)):
            image_embed_inputs.append(self.ip_model.get_image_embeds(processed_crops[crop_ind], skip_uncond=True))
        crops_input_sequence = torch.cat(image_embed_inputs, dim=1)
        generated_images = []
        for _ in range(n_images):
            seed = random.randint(0, 1000000)
            for curr_scale in [scale]:
                negative_cond_sequence = torch.zeros_like(crops_input_sequence)
                embeds_len = self.zero_image_embeds.shape[1]
                for i in range(0, negative_cond_sequence.shape[1], embeds_len):
                    negative_cond_sequence[:, i : i + embeds_len] = self.zero_image_embeds.detach()

                img_emb = self.prior_pipeline(
                    cond_sequence=crops_input_sequence,
                    negative_cond_sequence=negative_cond_sequence,
                    num_inference_steps=25,
                    num_images_per_prompt=1,
                    guidance_scale=curr_scale,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                ).image_embeds

                for seed_2 in range(1):
                    images = self.ip_model.generate(
                        image_prompt_embeds=img_emb,
                        num_samples=1,
                        num_inference_steps=50,
                    )
                    generated_images += images

        return generated_images
