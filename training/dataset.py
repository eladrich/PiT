import random
import traceback
from pathlib import Path

import einops
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import bezier_utils


class PartsDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        clip_image_size: int = 224,
        image_processor=None,
        max_crops=3,
        use_ref: bool = True,
        ref_as_grid: bool = True,
        grid_size: int = 2,
        sketch_prob: float = 0.0,
    ):
        subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

        all_paths = []
        self.subdir_dict = {}
        for subdir in tqdm(subdirs):
            current_paths = list(subdir.glob("*.jpg"))
            current_target_paths = [p for p in current_paths if len(str(p.name).split("_")) == 2]
            if use_ref and len(current_target_paths) < 9:
                # Skip if not enough target images
                continue
            all_paths.extend(current_paths)
            self.subdir_dict[subdir] = current_target_paths

        print(f"Percentile of valid subdirs: {len(self.subdir_dict) / len(subdirs)}")
        self.target_paths = [p for p in all_paths if len(str(p.name).split("_")) == 2]
        source_paths = [p for p in all_paths if len(str(p.name).split("_")) == 3]
        self.source_target_mappings = {path: [] for path in self.target_paths}
        for source_path in source_paths:
            # Remove last part of the path
            target_path = Path("_".join(str(source_path).split("_")[:-1]) + ".jpg")
            if target_path in self.source_target_mappings:
                self.source_target_mappings[target_path].append(source_path)
        print(f"Loaded {len(self.target_paths)} target images")

        self.clip_image_size = clip_image_size

        self.image_processor = image_processor

        self.max_crops = max_crops

        self.use_ref = use_ref

        self.ref_as_grid = ref_as_grid

        self.grid_size = grid_size

        self.sketch_prob = sketch_prob

    def __len__(self):
        return len(self.target_paths)

    def paste_on_background(self, image, background, min_scale=0.4, max_scale=0.8):
        # Calculate aspect ratio and determine resizing based on the smaller dimension of the background
        aspect_ratio = image.width / image.height
        scale = random.uniform(min_scale, max_scale)
        new_width = int(min(background.width, background.height * aspect_ratio) * scale)
        new_height = int(new_width / aspect_ratio)

        # Resize image and calculate position
        image = image.resize((new_width, new_height), resample=Image.LANCZOS)
        pos_x = random.randint(0, background.width - new_width)
        pos_y = random.randint(0, background.height - new_height)

        # Paste the image using its alpha channel as mask if present
        background.paste(image, (pos_x, pos_y), image if "A" in image.mode else None)
        return background

    def get_random_crop(self, image):
        crop_percent_x = random.uniform(0.8, 1.0)
        crop_percent_y = random.uniform(0.8, 1.0)
        # crop_percent_y = random.uniform(0.1, 0.7)
        crop_x = int(image.width * crop_percent_x)
        crop_y = int(image.height * crop_percent_y)
        x = random.randint(0, image.width - crop_x)
        y = random.randint(0, image.height - crop_y)
        return image.crop((x, y, x + crop_x, y + crop_y))

    def get_empty_image(self):
        empty_image = Image.new("RGB", (self.clip_image_size, self.clip_image_size), (255, 255, 255))
        return self.image_processor(empty_image)["pixel_values"][0]

    def __getitem__(self, i: int):

        out_dict = {}

        try:
            target_path = self.target_paths[i]
            image = Image.open(target_path).convert("RGB")

            input_parts = []

            source_paths = self.source_target_mappings[target_path]
            n_samples = random.randint(1, len(source_paths))

            n_samples = min(n_samples, self.max_crops)
            source_paths = random.sample(source_paths, n_samples)

            if random.random() < 0.1:
                # Use empty image, but maybe still pass reference
                source_paths = []

            if self.use_ref:
                subdir = target_path.parent
                # Take something from same dir
                potential_refs = list(set(self.subdir_dict[subdir]) - {target_path})
                # Choose 4 refs
                reference_paths = random.sample(potential_refs, self.grid_size**2)
                reference_images = [
                    np.array(Image.open(reference_path).convert("RGB")) for reference_path in reference_paths
                ]
                # Concat all images as grid of 2x2
                reference_grid = np.stack(reference_images)
                grid_image = einops.rearrange(
                    reference_grid,
                    "(h w) h1 w1 c -> (h h1) (w w1) c",
                    h=self.grid_size,
                )
                reference_image = Image.fromarray(grid_image).resize((512, 512))

                # Always add the reference image
                input_parts.append(reference_image)

            # Sample a subset
            for source_path in source_paths:
                source_image = Image.open(source_path).convert("RGB")
                if random.random() < 0.2:
                    # Instead of using the source image, use a random crop from the target
                    source_image = self.get_random_crop(source_image)
                if random.random() < 0.2:
                    source_image = T.v2.RandomRotation(degrees=30, expand=True, fill=255)(source_image)
                object_with_background = Image.new("RGB", image.size, (255, 255, 255))
                self.paste_on_background(source_image, object_with_background, min_scale=0.8, max_scale=0.95)
                if self.sketch_prob > 0 and random.random() < self.sketch_prob:
                    num_lines = random.randint(8, 15)
                    object_with_background = bezier_utils.get_sketch(
                        object_with_background,
                        total_curves=num_lines,
                        drop_line_prob=0.1,
                    )
                input_parts.append(object_with_background)

            # Always pad to three parts for now
            actual_max_crops = self.max_crops + 1 if self.use_ref else self.max_crops
            while len(input_parts) < actual_max_crops:
                input_parts.append(
                    Image.new(
                        "RGB",
                        (self.clip_image_size, self.clip_image_size),
                        (255, 255, 255),
                    )
                )

        except Exception as e:
            print(f"Error processing image: {e}")
            traceback.print_exc()
            empty_image = Image.new("RGB", (self.clip_image_size, self.clip_image_size), (255, 255, 255))
            image = empty_image
            actual_max_crops = self.max_crops + 1 if self.use_ref else self.max_crops
            input_parts = [empty_image] * (actual_max_crops)

        clip_target_image = self.image_processor(image)["pixel_values"][0]
        clip_parts = [self.image_processor(part)["pixel_values"][0] for part in input_parts]

        out_dict["crops"] = clip_parts

        return clip_target_image, out_dict
