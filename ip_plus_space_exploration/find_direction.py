from dataclasses import dataclass
from pathlib import Path

import pyrallis
import torch
from PIL import Image
from tqdm import tqdm

from ip_plus_space_exploration.ip_model_utils import get_image_ip_base_tokens, get_image_ip_plus_tokens, load_ip


def get_class_tokens(ip_model, class_dir, ip_model_type: str = "plus"):
    image_prompt_embeds_list = []
    clip_image_embeds_list = []
    images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
    for _, image in tqdm(enumerate(images), total=len(images)):
        pil_image = Image.open(image)
        if ip_model_type == "plus":
            image_prompt_embeds, clip_image_embeds = get_image_ip_plus_tokens(ip_model, pil_image)
        elif ip_model_type == "base":
            image_prompt_embeds, clip_image_embeds = get_image_ip_base_tokens(ip_model, pil_image)
        image_prompt_embeds_list.append(image_prompt_embeds.detach().cpu())
        clip_image_embeds_list.append(clip_image_embeds.detach().cpu())
    return torch.cat(image_prompt_embeds_list), torch.cat(clip_image_embeds_list)


@dataclass
class FindDirectionConfig:
    class1_dir: Path
    class2_dir: Path
    output_dir: Path
    ip_model_type: str

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        assert self.ip_model_type in ["plus", "base"]


@torch.inference_mode()
@pyrallis.wrap()
def main(cfg: FindDirectionConfig):
    ip_model = load_ip(cfg.ip_model_type)
    class_1_image_prompt_embeds, class_1_clip_image_embeds = get_class_tokens(
        ip_model=ip_model, class_dir=cfg.class1_dir, ip_model_type=cfg.ip_model_type
    )
    class_2_image_prompt_embeds, class_2_clip_image_embeds = get_class_tokens(
        ip_model=ip_model, class_dir=cfg.class2_dir, ip_model_type=cfg.ip_model_type
    )
    clip_direction = class_2_clip_image_embeds.mean(dim=0) - class_1_clip_image_embeds.mean(dim=0)
    ip_direction = class_2_image_prompt_embeds.mean(dim=0) - class_1_image_prompt_embeds.mean(dim=0)
    torch.save(clip_direction, cfg.output_dir / "clip_direction.pt")
    torch.save(ip_direction, cfg.output_dir / "ip_direction.pt")


if __name__ == "__main__":
    main()
