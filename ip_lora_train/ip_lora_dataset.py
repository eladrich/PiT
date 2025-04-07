from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

CHARACTER_SHEET_PROMPT = "a character sheet displaying a creature, from several angles with 1 large front view in the middle, clean white background. In the background we can see half-completed, partially colored, sketches of different parts of the object"


def _transform():
    return Compose(
        [
            ToTensor(),
            Normalize(
                [0.5],
                [0.5],
            ),
        ]
    )


class IPLoraDataset(Dataset):
    def __init__(
        self,
        tokenizer1,
        tokenizer2,
        image_processor,
        target_resolution: int = 1024,
        base_dir: Path = Path("dataset/"),
        prompt_mode: str = "character_sheet",
    ):
        super().__init__()
        self.base_dir = base_dir
        self.samples = self.init_samples(base_dir)
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.image_processor = image_processor
        self.target_resolution = target_resolution
        self.prompt_mode = prompt_mode

    def init_samples(self, base_dir):
        ref_dir = base_dir / "refs"
        targets_dir = base_dir / "targets"
        prompt_dir = base_dir / "targets"
        ref_files = list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg"))
        targets_files = list(targets_dir.glob("*.png")) + list(targets_dir.glob("*.jpg"))
        prompt_files = list(prompt_dir.glob("*.txt"))
        ref_prefixes = [f.stem.split("_ref")[0] for f in ref_files]
        targets_prefixes = [f.stem for f in targets_files]
        prompt_prefixes = [f.stem for f in prompt_files]
        valid_prefixes = list(set(ref_prefixes) & set(targets_prefixes) & set(prompt_prefixes))
        ref_png_paths = [ref_dir / f"{prefix}_ref.png" for prefix in valid_prefixes]
        ref_jpg_paths = [ref_dir / f"{prefix}_ref.jpg" for prefix in valid_prefixes]
        targets_png_paths = [targets_dir / f"{prefix}.png" for prefix in valid_prefixes]
        targets_jpg_paths = [targets_dir / f"{prefix}.jpg" for prefix in valid_prefixes]
        prompt_paths = [prompt_dir / f"{prefix}.txt" for prefix in valid_prefixes]
        samples = [
            {
                "ref": ref_png_paths[i] if ref_png_paths[i].exists() else ref_jpg_paths[i],
                "sheet": targets_png_paths[i] if targets_png_paths[i].exists() else targets_jpg_paths[i],
                "prompt": prompt_paths[i],
            }
            for i in range(len(valid_prefixes))
        ]
        print(f"lora_dataset.py:: found {len(samples)} samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def get_prompt(self, prompt_text):
        if self.prompt_mode == "character_sheet":
            return CHARACTER_SHEET_PROMPT
        elif self.prompt_mode == "creature_in_scene":
            return prompt_text.split(",")[0] + " an imaginary fantasy creature in" + prompt_text.split("in a")[1]
        else:
            raise ValueError(f"Prompt mode {self.prompt_mode} is not supported.")

    def __getitem__(self, i: int):
        sample = self.samples[i]
        ref_path = sample["ref"]
        sheet_path = sample["sheet"]
        prompt_path = sample["prompt"]
        ref_image = Image.open(ref_path)
        sheet_image = Image.open(sheet_path)
        prompt_text = open(prompt_path, "r").read()
        sample_prompt = self.get_prompt(prompt_text)
        out_dict = {}
        input_ids1 = self.tokenizer1(
            [sample_prompt],
            max_length=self.tokenizer1.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        input_ids2 = self.tokenizer2(
            [sample_prompt],
            max_length=self.tokenizer2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        out_dict["input_ids1"] = input_ids1
        out_dict["input_ids2"] = input_ids2

        out_dict["text"] = prompt_text

        processed_image_prompt = self.image_processor(images=[ref_image], return_tensors="pt").pixel_values
        out_dict["image_prompt"] = processed_image_prompt
        target_image_torch = _transform()(sheet_image)
        out_dict["target_image"] = target_image_torch
        out_dict["original_sizes"] = (self.target_resolution, self.target_resolution)
        out_dict["crop_top_lefts"] = (0, 0)
        return out_dict


def ip_lora_collate_fn(batch):
    return_batch = {}
    return_batch["input_ids_one"] = torch.stack([item["input_ids1"] for item in batch])
    return_batch["input_ids_two"] = torch.stack([item["input_ids2"] for item in batch])
    return_batch["text"] = [item["text"] for item in batch]
    image_prompt = torch.stack([item["image_prompt"] for item in batch])
    image_prompt = image_prompt.to(memory_format=torch.contiguous_format).float()
    return_batch["image_prompt"] = image_prompt
    target_image = torch.stack([item["target_image"] for item in batch])
    target_image = target_image.to(memory_format=torch.contiguous_format).float()

    return_batch["target_image"] = target_image
    original_sizes = [item["original_sizes"] for item in batch]
    crop_top_lefts = [item["crop_top_lefts"] for item in batch]
    return_batch["original_sizes"] = original_sizes
    return_batch["crop_top_lefts"] = crop_top_lefts
    return return_batch
