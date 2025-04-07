import cv2
import einops
import math
import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import random
import supervision as sv
import torch
from PIL import Image
from utils import words_bank
from dataclasses import dataclass
from diffusers import StableDiffusionXLPipeline, FluxPipeline
from pathlib import Path
from scipy import ndimage
from transformers import SamModel, SamProcessor, AutoProcessor, AutoModelForCausalLM
from transformers import pipeline
from typing import Tuple, Dict


@dataclass
class RunConfig:
    # Generation mode, should be either 'objects' or 'scenes'
    out_dir: Path = Path("datasets/generated/monsters/")
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


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    facecolor = [random.random(), random.random(), random.random(), 0.3]
    edgecolor = facecolor.copy()
    edgecolor[3] = 1
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=facecolor, lw=2))


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


@pyrallis.wrap()
def generate(cfg: RunConfig):
    cfg.out_dir.mkdir(exist_ok=True, parents=True)

    flux_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    checkpoint = "microsoft/Florence-2-large"
    florence_processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()

    def run_florence_inference(
        image: Image, task: str = "<OPEN_VOCABULARY_DETECTION>", text: str = ""
    ) -> Tuple[str, Dict]:
        prompt = task + text
        inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device)
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            output_scores=True,
        )
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        response = florence_processor.post_process_generation(generated_text, task=task, image_size=image.size)

        detections = sv.Detections.from_lmm(lmm=sv.LMM.FLORENCE_2, result=response, resolution_wh=image.size)
        input_boxes = detections.xyxy
        return input_boxes

    with open("assets/openimages_classes.txt", "r") as f:
        objects = f.read().splitlines()
        objects = ["".join(char if char.isalnum() else " " for char in object_name) for object_name in objects]
    # Duplicate creatures to match the same size as objects
    creatures = words_bank.creatures * (len(objects) // len(words_bank.creatures) + 1) + objects * 10

    tot_generated = 0
    for _ in range(cfg.n_images):
        try:
            new_dir_name = f"set_{tot_generated}_{random.randint(0, 1000000)}"
            out_dir = cfg.out_dir / new_dir_name
            out_dir.mkdir(exist_ok=True, parents=True)
            monster_grid = []

            for _ in range(cfg.n_samples_in_dir):

                adjective_count = random.randint(2, 6)
                adjectives = random.sample(words_bank.adjectives, adjective_count)
                if len(adjectives) > 0:
                    adjectives_txt = " ".join(adjectives) + " "

                character_count = random.randint(1, 3)
                characters = random.sample(creatures, character_count)
                characters = [f"{c}-like" for c in characters]
                character_txt = " ".join(characters)

                prompt = f"studio photo pixar style concept art, An imaginary fantasy {adjectives_txt} {character_txt} creature with eyes arms legs mouth , white background studio photo pixar style asset"
                seed = random.randint(0, 1000000)

                print(prompt)
                base_image = flux_pipe(
                    prompt,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                ).images[0]

                input_boxes = []
                keywords = words_bank.keywords
                for keyword in keywords:
                    current_boxes = list(run_florence_inference(base_image, text=keyword))
                    # Randomly choose one
                    if len(current_boxes) > 0:
                        input_boxes.extend(random.sample(current_boxes, 1))

                # convert to ints
                input_boxes = [[int(x) for x in box] for box in input_boxes]

                inputs = processor(base_image, input_boxes=[input_boxes], return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
                )[0]
                #
                masks = [mask[0].cpu().detach().numpy() for mask in masks]

                masks = sorted(masks, key=lambda mask: mask.sum(), reverse=False)

                # Filter the masks
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

                if cfg.vis_data:
                    plt.imshow(base_image)
                    plt.show()
                    show_masks_on_image(base_image, masks)

                visited_area = np.zeros_like(masks[0])
                prompt_hash = str(abs(hash(prompt)))

                for i, mask in enumerate(masks):
                    # Check if overlaps with visited_area
                    if (visited_area * mask).sum() > 0:
                        continue
                    visited_area += mask

                    cropped_image = crop_from_mask(base_image, mask)
                    out_path = out_dir / f"{prompt_hash}_{seed}_{i}.jpg"
                    cropped_image.save(out_path)
                    if cfg.vis_data:
                        plt.imshow(cropped_image)
                        plt.show()

                out_path = out_dir / f"{prompt_hash}_{seed}.jpg"
                tot_generated += 1
                base_image.save(out_path)
                monster_grid.append(base_image)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    # Use to generate objects or backgrounds
    generate()
