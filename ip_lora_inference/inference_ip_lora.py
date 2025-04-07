from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pyrallis
import torch
from diffusers import UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from ip_lora_train.ip_adapter_for_lora import IPAdapterPlusXLLoRA
from ip_lora_train.sdxl_ip_lora_pipeline import StableDiffusionXLIPLoRAPipeline


@dataclass
class ExperimentConfig:
    lora_type: str = "character_sheet"
    lora_path: Path = Path("weights/character_sheet/pytorch_lora_weights.safetensors")
    prompt: str = "a character sheet displaying a creature, from several angles with 1 large front view in the middle, clean white background. In the background we can see half-completed, partially colored, sketches of different parts of the object"
    output_dir: Path = Path("ip_lora_inference/character_sheet/")
    ref_images_paths: Union[list[Path], Path] = Path("assets/character_sheet_default_ref.jpg")
    ip_adapter_path: Path = Path("weights/ip_adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin")
    seed: Optional[int] = None
    num_inference_steps: int = 50
    remove_background: bool = True

    def __post_init__(self):
        assert self.lora_type in ["character_sheet", "background_generation"]
        assert self.lora_path.exists(), f"Lora path {self.lora_path} does not exist"
        assert self.ip_adapter_path.exists(), f"IP adapter path {self.ip_adapter_path} does not exist"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.ref_images_paths, Path):
            self.ref_images_paths = [self.ref_images_paths]
        for ref_image_path in self.ref_images_paths:
            assert ref_image_path.exists(), f"Reference image path {ref_image_path} does not exist"


def load_rmbg_model():
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-2.0", trust_remote_code=True)
    torch.set_float32_matmul_precision(["high", "highest"][0])
    model.to("cuda")
    model.eval()
    return model


def remove_background(model, image: Image.Image) -> Image.Image:
    assert image.size == (1024, 1024)
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    input_images = transform_image(image).unsqueeze(0).to("cuda")

    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    mask = transforms.ToPILImage()(pred).resize(image_size)

    # Create white background
    white_bg = Image.new("RGB", image_size, (255, 255, 255))
    # Paste original image using mask
    white_bg.paste(image, mask=mask)
    return white_bg


@torch.inference_mode()
@pyrallis.wrap()
def main(cfg: ExperimentConfig):
    pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
    print("loading unet")
    unet = UNet2DConditionModel.from_pretrained(pipe_id, subfolder="unet")
    print("loading ip model")
    ip_model = IPAdapterPlusXLLoRA(
        unet=unet,
        image_encoder_path="models/image_encoder",
        ip_ckpt=cfg.ip_adapter_path,
        device="cuda",
        num_tokens=16,
    )
    print("loading pipeline")
    pipe = StableDiffusionXLIPLoRAPipeline.from_pretrained(
        pipe_id,
        unet=unet,
        variant=None,
        torch_dtype=torch.float32,
    )
    print("loading lora")
    pipe.load_lora_weights(
        pretrained_model_name_or_path_or_dict=cfg.lora_path,
        adapter_name="lora1",
    )
    pipe.set_adapters(["lora1"], adapter_weights=[1.0])

    pipe.to("cuda")
    print("running inference")
    if cfg.remove_background:
        rmbg_model = load_rmbg_model()
    else:
        rmbg_model = None
    for ref_image_path in cfg.ref_images_paths:
        if cfg.seed is not None:
            generator = torch.Generator("cuda").manual_seed(cfg.seed)
        else:
            generator = None
        ref_image = Image.open(ref_image_path).convert("RGB")
        if cfg.remove_background:
            rmbg_model.cuda()
            ref_image = remove_background(model=rmbg_model, image=ref_image)
            rmbg_model.cpu()
        image_name = ref_image_path.stem
        image = pipe(
            cfg.prompt,
            ip_adapter_image=ref_image,
            ip_model=ip_model,
            num_inference_steps=cfg.num_inference_steps,
            generator=generator,
        ).images[0]
        image.save(cfg.output_dir / f"{image_name}_pred.jpg")
        ref_image.save(cfg.output_dir / f"{image_name}_ref.jpg")

        # Create side-by-side plot
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(ref_image)
        ax[0].axis("off")
        ax[1].imshow(image)
        ax[1].axis("off")
        fig.suptitle(cfg.prompt, fontsize=24, wrap=True)
        plt.tight_layout()
        plt.savefig(cfg.output_dir / f"{image_name}_side_by_side.jpeg", bbox_inches="tight", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
