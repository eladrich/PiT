import random
from pathlib import Path

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers import FluxPipeline
from scipy import ndimage
from transformers import pipeline

from utils import words_bank


@dataclass
class RunConfig:
    out_dir: Path = Path("datasets/generated/products")
    n_images: int = 1000000
    vis_data: bool = False
    n_samples_in_dir: int = 1000


def crop_from_mask(image, mask: np.ndarray):
    # Apply mask and crop a tight box
    mask = mask.astype(np.uint8)
    mask = mask * 255
    mask = Image.fromarray(mask)
    bbox = mask.getbbox()

    # Create a new image with a white background
    white_background = Image.new("RGB", image.size, (255, 255, 255))

    # Apply the mask to the original image
    masked_image = Image.composite(image, white_background, mask)

    # Crop the image to the bounding box
    cropped_image = masked_image.crop(bbox)

    return cropped_image


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_masks_on_image(raw_image, masks, caption=None):
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    if caption:
        plt.title(caption)
    plt.show()


@pyrallis.wrap()
def generate(cfg: RunConfig):
    cfg.out_dir.mkdir(exist_ok=True, parents=True)

    flux_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentor = pipeline("mask-generation", model="facebook/sam-vit-base", device=device)

    with open("assets/openimages_classes.txt", "r") as f:
        objects = f.read().splitlines()
        objects = ["".join(char if char.isalnum() else " " for char in object_name) for object_name in objects]

    tot_generated = 0
    for _ in range(cfg.n_images):
        new_dir_name = f"set_{tot_generated}_{random.randint(0, 1000000)}"
        out_dir = cfg.out_dir / new_dir_name
        out_dir.mkdir(exist_ok=True, parents=True)
        monster_grid = []

        for _ in range(1000):
            try:
                character_count = random.randint(0, 3)
                if character_count == 0:
                    character_txt = ""
                else:
                    characters = random.sample(objects, character_count)
                    # For each character text take only one random word
                    characters = [random.choice(character.split()) for character in characters]
                    characters = [f"{c}-like" for c in characters]
                    character_txt = " ".join(characters)

                attributes_count = random.randint(1, 3)
                material_count = random.randint(1, 2)
                attributes = random.sample(words_bank.object_attributes, attributes_count)
                materials = random.sample(words_bank.product_materials, material_count)
                features = random.sample(words_bank.product_defining_attributes, 1)
                attributes_and_materials_txt = " ".join(attributes + materials + features)

                prompt = f"A product design photo of a {attributes_and_materials_txt}product with {character_txt} attributes, integrated together to create one seamless product.  It is set against a light gray background with a soft gradient, creating a neutral and elegant backdrop that emphasizes the  contemporary design. The soft, even lighting highlights the contours and textures, lending a professional, polished quality to the composition"
                seed = random.randint(0, 1000000)

                print(prompt)
                base_image = flux_pipe(
                    prompt,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                ).images[0]

                if cfg.vis_data:
                    plt.imshow(base_image)
                    plt.title(f"{attributes_and_materials_txt} {character_txt}")
                    plt.show()
                # continue
                all_masks = segmentor(base_image, points_per_batch=64)["masks"]

                if len(all_masks) == 0:
                    continue

                # Sort by area
                all_masks = sorted(all_masks, key=lambda mask: mask.sum(), reverse=False)
                # Remove the last item
                masks = all_masks[:-1]

                if len(all_masks) < 3:
                    # For now take only things with at least 3 parts to keep the data interesting
                    continue

                # Remove masks that intersect with image boundary
                mask_boundary = np.zeros_like(masks[0])
                mask_boundary[0, :] = 1
                mask_boundary[-1, :] = 1
                mask_boundary[:, 0] = 1
                mask_boundary[:, -1] = 1

                masks = [mask for mask in masks if (mask * mask_boundary).sum() == 0]

                masks = [mask for mask in masks if 0.015 < mask.sum() / mask.flatten().shape[0] < 0.3]

                for m_ind in range(len(masks)):
                    # Apply dilate and erode to the mask
                    mask = masks[m_ind].astype(np.uint8)
                    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
                    mask = cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=1)
                    # Now do the reverse to get rid of the small holes
                    mask = cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=1)
                    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
                    if True or random.random() < 0.5:
                        # Close mask
                        mask = ndimage.binary_fill_holes(mask.astype(int))
                    masks[m_ind] = mask == 1

                masks = [mask for mask in masks if 0.015 < mask.sum() / mask.flatten().shape[0] < 0.3]

                if len(masks) == 0:
                    print(f"No masks found for {character_txt}")
                    continue

                # Restrict to 8
                masks = masks[:8]

                if cfg.vis_data:
                    show_masks_on_image(base_image, masks)

                visited_area = np.zeros_like(masks[0])
                prompt_hash = str(abs(hash(prompt)))

                for i, mask in enumerate(masks):
                    # Check if overlaps with visited_area
                    if (visited_area * mask).sum() > 0:
                        continue
                    visited_area += mask

                    cropped_image = crop_from_mask(base_image, mask)
                    cropped_image.thumbnail((256, 256))
                    out_path = out_dir / f"{prompt_hash}_{seed}_{i}.jpg"
                    cropped_image.save(out_path)
                    if cfg.vis_data:
                        plt.imshow(cropped_image)
                        plt.show()

                out_path = out_dir / f"{prompt_hash}_{seed}.jpg"
                base_image.thumbnail((512, 512))
                tot_generated += 1
                base_image.save(out_path)
                if len(monster_grid) < 9:
                    monster_grid.append(base_image)
            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":
    # Use to generate objects or backgrounds
    generate()
