from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pyrallis
import torch
from PIL import Image

from ip_plus_space_exploration.ip_model_utils import get_image_ip_base_tokens, get_image_ip_plus_tokens, load_ip


def get_images_with_ip_direction(ip_model, img_tokens, ip_direction, seed):
    scales_images = {k: [] for k in [-1, -0.5, 0, 0.5, 1]}

    for step_size in scales_images.keys():
        new_ip_tokens = img_tokens + step_size * ip_direction
        image = ip_model.generate(
            pil_image=None,
            scale=1.0,
            image_prompt_embeds=new_ip_tokens.cuda(),
            num_samples=1,
            num_inference_steps=50,
            seed=seed,
        )
        scales_images[step_size] = image
    return scales_images


def get_images_with_clip_direction(ip_model, clip_features, clip_direction, seed):
    scales_images = {k: [] for k in [-1, -0.5, 0, 0.5, 1]}

    for step_size in scales_images.keys():
        new_clip_features = clip_features + step_size * clip_direction
        new_ip_tokens = ip_model.image_proj_model(new_clip_features.cuda())
        image = ip_model.generate(
            pil_image=None,
            scale=1.0,
            image_prompt_embeds=new_ip_tokens.cuda(),
            num_samples=1,
            num_inference_steps=50,
            seed=seed,
        )
        scales_images[step_size] = image
    return scales_images


@dataclass
class EditByDirectionConfig:
    ip_model_type: str
    image_path: Path
    direction_path: Path
    direction_type: str
    output_dir: Path
    seed: Optional[int] = None

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        assert self.direction_type.lower() in ["ip", "clip"]


@torch.inference_mode()
@pyrallis.wrap()
def main(cfg: EditByDirectionConfig):
    ip_model = load_ip(cfg.ip_model_type)
    edit_direction = torch.load(cfg.direction_path)
    if cfg.image_path.is_dir():
        image_paths = list(cfg.image_path.glob("*.jpg")) + list(cfg.image_path.glob("*.png"))
    else:
        image_paths = [cfg.image_path]
    for img in image_paths:
        if cfg.ip_model_type == "plus":
            img_tokens, img_clip_embeds = get_image_ip_plus_tokens(ip_model, Image.open(img))
        elif cfg.ip_model_type == "base":
            img_tokens, img_clip_embeds = get_image_ip_base_tokens(ip_model, Image.open(img))
        if cfg.direction_type.upper() == "IP":
            scales_images = get_images_with_ip_direction(
                ip_model=ip_model, img_tokens=img_tokens, ip_direction=edit_direction, seed=cfg.seed
            )
        elif cfg.direction_type.upper() == "CLIP":
            scales_images = get_images_with_clip_direction(
                ip_model=ip_model, clip_features=img_clip_embeds, clip_direction=edit_direction, seed=cfg.seed
            )
        # Plot all images in a single row
        fig, axes = plt.subplots(1, len(scales_images) + 1, figsize=(20, 5))  # Increased height from 4 to 5

        # Sort the scales for consistent left-to-right ordering
        sorted_scales = sorted(scales_images.keys())

        axes[0].imshow(Image.open(img))
        axes[0].set_title(f"original", pad=15)  # Added padding to the title
        axes[0].axis("off")

        for i, scale in enumerate(sorted_scales):
            axes[i + 1].imshow(scales_images[scale][0])
            axes[i + 1].set_title(f"dist from boundary={scale}", pad=15)  # Added padding to the title
            axes[i + 1].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Added rect parameter to leave more room at the top

        plt.savefig(cfg.output_dir / f"{img.stem}_{cfg.direction_type}.png")
        for scale in scales_images.keys():
            scales_images[scale][0].save(str(cfg.output_dir / f"{img.stem}_{cfg.direction_type}_{scale}.jpg"))
        plt.close()


if __name__ == "__main__":
    main()
