import random
import sys
from pathlib import Path

import diffusers
import pyrallis
import torch
import torch.utils.checkpoint
import transformers
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
from torchvision import transforms
from tqdm import tqdm

from ip_adapter import IPAdapterPlusXL
from model.dit import DiT_Llama
from model.pipeline_pit import PiTPipeline
from training.dataset import (
    PartsDataset,
)
from training.train_config import TrainConfig
from utils import vis_utils

logger = get_logger(__name__, log_level="INFO")


class Coach:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.cfg.output_dir.mkdir(exist_ok=True, parents=True)
        (self.cfg.output_dir / "cfg.yaml").write_text(pyrallis.dump(self.cfg))
        (self.cfg.output_dir / "run.sh").write_text(f'python {Path(__file__).name} {" ".join(sys.argv)}')

        self.logging_dir = self.cfg.output_dir / "logs"
        accelerator_project_config = ProjectConfiguration(
            total_limit=2, project_dir=self.cfg.output_dir, logging_dir=self.logging_dir
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision=self.cfg.mixed_precision,
            log_with=self.cfg.report_to,
            project_config=accelerator_project_config,
        )

        self.device = "cuda"

        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)

        if self.accelerator.is_main_process:
            self.logging_dir.mkdir(exist_ok=True, parents=True)

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.prior = DiT_Llama(
            embedding_dim=2048,
            hidden_dim=self.cfg.hidden_dim,
            n_layers=self.cfg.num_layers,
            n_heads=self.cfg.num_attention_heads,
        )
        # pretty print total number of parameters in Billions
        num_params = sum(p.numel() for p in self.prior.parameters())
        print(f"Number of parameters: {num_params / 1e9:.2f}B")

        self.image_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(self.device)

        ip_ckpt_path = hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename="ip-adapter-plus_sdxl_vit-h.bin",
            subfolder="sdxl_models",
            local_dir="pretrained_models",
        )

        self.ip_model = IPAdapterPlusXL(
            self.image_pipe,
            "models/image_encoder",
            ip_ckpt_path,
            self.device,
            num_tokens=16,
        )

        self.image_processor = self.ip_model.clip_image_processor

        empty_image = Image.new("RGB", (256, 256), (255, 255, 255))
        zero_image = torch.Tensor(self.image_processor(empty_image)["pixel_values"][0])
        self.zero_image_embeds = self.ip_model.get_image_embeds(zero_image.unsqueeze(0), skip_uncond=True)

        self.prior_pipeline = PiTPipeline(prior=self.prior)
        self.prior_pipeline = self.prior_pipeline.to(self.accelerator.device)

        params_to_optimize = list(self.prior.parameters())

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.cfg.lr,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )

        self.train_dataloader, self.validation_dataloader = self.get_dataloaders()

        self.prior, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.prior, self.optimizer, self.train_dataloader
        )

        self.train_step = 0 if self.cfg.resume_from_step is None else self.cfg.resume_from_step
        print(self.train_step)

        if self.cfg.resume_from_path is not None:
            prior_state_dict = torch.load(self.cfg.resume_from_path, map_location=self.device)
            msg = self.prior.load_state_dict(prior_state_dict, strict=False)
            print(msg)

    def save_model(self, save_path):
        save_path.mkdir(exist_ok=True, parents=True)
        prior_state_dict = self.prior.state_dict()
        torch.save(prior_state_dict, save_path / "prior.ckpt")

    def unnormalize_and_pil(self, tensor):
        unnormed = tensor * torch.tensor(self.image_processor.image_std).view(3, 1, 1).to(tensor.device) + torch.tensor(
            self.image_processor.image_mean
        ).view(3, 1, 1).to(tensor.device)
        return transforms.ToPILImage()(unnormed)

    def save_images(self, image, conds, cond_sequence, target_embeds, label="", save_path=""):
        self.prior.eval()
        input_images = []
        captions = []
        for i in range(len(conds)):
            pil_image = self.unnormalize_and_pil(conds[i]).resize((self.cfg.img_size, self.cfg.img_size))
            input_images.append(pil_image)
            captions.append("Condition")
        if image is not None:
            input_images.append(self.unnormalize_and_pil(image).resize((self.cfg.img_size, self.cfg.img_size)))
            captions.append(f"Target {label}")

        seeds = range(2)
        output_images = []
        embebds_to_vis = []
        embeds_captions = []
        embebds_to_vis += [target_embeds]
        embeds_captions += ["Target Reconstruct" if image is not None else "Source Reconstruct"]
        if self.cfg.use_ref:
            embebds_to_vis += [cond_sequence[:, :16]]
            embeds_captions += ["Grid Reconstruct"]
        for embs in embebds_to_vis:
            direct_from_emb = self.ip_model.generate(image_prompt_embeds=embs, num_samples=1, num_inference_steps=50)
            output_images = output_images + direct_from_emb
        captions += embeds_captions

        for seed in seeds:
            for scale in [1, 4]:
                negative_cond_sequence = torch.zeros_like(cond_sequence)
                embeds_len = self.zero_image_embeds.shape[1]
                for i in range(0, negative_cond_sequence.shape[1], embeds_len):
                    negative_cond_sequence[:, i : i + embeds_len] = self.zero_image_embeds.detach()
                img_emb = self.prior_pipeline(
                    cond_sequence=cond_sequence,
                    negative_cond_sequence=negative_cond_sequence,
                    num_inference_steps=25,
                    num_images_per_prompt=1,
                    guidance_scale=scale,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                ).image_embeds

                for seed_2 in range(1):
                    images = self.ip_model.generate(
                        image_prompt_embeds=img_emb,
                        num_samples=1,
                        num_inference_steps=50,
                    )
                    output_images += images
                    captions.append(f"prior_s {seed}, cfg {scale}, unet_s {seed_2}")

        all_images = input_images + output_images
        gen_images = vis_utils.create_table_plot(images=all_images, captions=captions)
        gen_images.save(save_path)
        self.prior.train()

    def get_dataloaders(self) -> torch.utils.data.DataLoader:
        dataset_path = self.cfg.dataset_path
        if not isinstance(self.cfg.dataset_path, list):
            dataset_path = [self.cfg.dataset_path]
        datasets = []
        for path in dataset_path:
            datasets.append(
                PartsDataset(
                    dataset_dir=path,
                    image_processor=self.image_processor,
                    use_ref=self.cfg.use_ref,
                    max_crops=self.cfg.max_crops,
                    sketch_prob=self.cfg.sketch_prob,
                )
            )
        dataset = torch.utils.data.ConcatDataset(datasets)
        print(f"Total number of samples: {len(dataset)}")
        dataset_weights = []
        for single_dataset in datasets:
            dataset_weights.extend([len(dataset) / len(single_dataset)] * len(single_dataset))
        sampler_train = torch.utils.data.WeightedRandomSampler(
            weights=dataset_weights, num_samples=len(dataset_weights)
        )

        validation_dataset = PartsDataset(
            dataset_dir=self.cfg.val_dataset_path,
            image_processor=self.image_processor,
            use_ref=self.cfg.use_ref,
            max_crops=self.cfg.max_crops,
            sketch_prob=self.cfg.sketch_prob,
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.train_batch_size,
            shuffle=sampler_train is None,
            num_workers=self.cfg.num_workers,
            sampler=sampler_train,
        )

        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )
        return train_dataloader, validation_dataloader

    def train(self):
        pbar = tqdm(range(self.train_step, self.cfg.max_train_steps + 1))
        # self.log_validation()

        while self.train_step < self.cfg.max_train_steps:
            train_loss = 0.0
            self.prior.train()
            lossbin = {i: 0 for i in range(10)}
            losscnt = {i: 1e-6 for i in range(10)}

            for sample_idx, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.prior):
                    image, cond = batch

                    image = image.to(self.weight_dtype).to(self.accelerator.device)
                    if "crops" in cond:
                        for crop_ind in range(len(cond["crops"])):
                            cond["crops"][crop_ind] = (
                                cond["crops"][crop_ind].to(self.weight_dtype).to(self.accelerator.device)
                            )
                    for key in cond.keys():
                        if isinstance(cond[key], torch.Tensor):
                            cond[key] = cond[key].to(self.accelerator.device)

                    with torch.no_grad():
                        image_embeds = self.ip_model.get_image_embeds(image, skip_uncond=True)

                        b = image_embeds.size(0)
                        nt = torch.randn((b,)).to(image_embeds.device)
                        t = torch.sigmoid(nt)
                        texp = t.view([b, *([1] * len(image_embeds.shape[1:]))])
                        z_1 = torch.randn_like(image_embeds)
                        noisy_latents = (1 - texp) * image_embeds + texp * z_1

                        target = image_embeds

                        # At some prob uniformly sample across the entire batch so the model also learns to work with unpadded inputs
                        if random.random() < 0.5:
                            crops_to_keep = random.randint(1, len(cond["crops"]))
                            cond["crops"] = cond["crops"][:crops_to_keep]
                        cond_crops = cond["crops"]

                        image_embed_inputs = []
                        for crop_ind in range(len(cond_crops)):
                            image_embed_inputs.append(
                                self.ip_model.get_image_embeds(cond_crops[crop_ind], skip_uncond=True)
                            )
                        input_sequence = torch.cat(image_embed_inputs, dim=1)

                    loss = 0
                    image_feat_seq = input_sequence

                    model_pred = self.prior(
                        noisy_latents,
                        t=t,
                        cond=image_feat_seq,
                    )

                    batchwise_prior_loss = ((z_1 - target.float() - model_pred.float()) ** 2).mean(
                        dim=list(range(1, len(target.shape)))
                    )
                    tlist = batchwise_prior_loss.detach().cpu().reshape(-1).tolist()
                    ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]

                    # count based on t
                    for t, l in ttloss:
                        lossbin[int(t * 10)] += l
                        losscnt[int(t * 10)] += 1

                    loss += batchwise_prior_loss.mean()
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.cfg.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.cfg.gradient_accumulation_steps

                    # Backprop
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.prior.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    pbar.update(1)
                    self.train_step += 1
                    train_loss = 0.0

                    if self.accelerator.is_main_process:

                        if self.train_step % self.cfg.checkpointing_steps == 1:
                            if self.accelerator.is_main_process:
                                save_path = self.cfg.output_dir  # / f"learned_prior.pth"
                                self.save_model(save_path)
                                logger.info(f"Saved state to {save_path}")
                        pbar.set_postfix(**{"loss": loss.cpu().detach().item()})

                        if self.cfg.log_image_frequency > 0 and (self.train_step % self.cfg.log_image_frequency == 1):
                            image_save_path = self.cfg.output_dir / "images" / f"{self.train_step}_step_images.jpg"
                            image_save_path.parent.mkdir(exist_ok=True, parents=True)
                            # Apply the full diffusion process
                            conds_list = []
                            for crop_ind in range(len(cond["crops"])):
                                conds_list.append(cond["crops"][crop_ind][0])

                            self.save_images(
                                image=image[0],
                                conds=conds_list,
                                cond_sequence=image_feat_seq[:1],
                                target_embeds=target[:1],
                                save_path=image_save_path,
                            )

                    if self.cfg.log_validation > 0 and (self.train_step % self.cfg.log_validation == 0):
                        # Run validation
                        self.log_validation()

                    if self.train_step >= self.cfg.max_train_steps:
                        break

            self.train_dataloader, self.validation_dataloader = self.get_dataloaders()
        pbar.close()

    def log_validation(self):
        for sample_idx, batch in tqdm(enumerate(self.validation_dataloader)):
            image, cond = batch
            image = image.to(self.weight_dtype).to(self.accelerator.device)
            if "crops" in cond:
                for crop_ind in range(len(cond["crops"])):
                    cond["crops"][crop_ind] = cond["crops"][crop_ind].to(self.weight_dtype).to(self.accelerator.device)
            for key in cond.keys():
                if isinstance(cond[key], torch.Tensor):
                    cond[key] = cond[key].to(self.accelerator.device)

            with torch.no_grad():
                target_embeds = self.ip_model.get_image_embeds(image, skip_uncond=True)
                crops_to_keep = random.randint(1, len(cond["crops"]))
                cond["crops"] = cond["crops"][:crops_to_keep]
                cond_crops = cond["crops"]
                image_embed_inputs = []
                for crop_ind in range(len(cond_crops)):
                    image_embed_inputs.append(self.ip_model.get_image_embeds(cond_crops[crop_ind], skip_uncond=True))
                input_sequence = torch.cat(image_embed_inputs, dim=1)

            image_save_path = self.cfg.output_dir / "val_images" / f"{self.train_step}_step_{sample_idx}_images.jpg"
            image_save_path.parent.mkdir(exist_ok=True, parents=True)

            save_target_image = image[0]
            conds_list = []
            for crop_ind in range(len(cond["crops"])):
                conds_list.append(cond["crops"][crop_ind][0])

            # Apply the full diffusion process
            self.save_images(
                image=save_target_image,
                conds=conds_list,
                cond_sequence=input_sequence[:1],
                target_embeds=target_embeds[:1],
                save_path=image_save_path,
            )

            if sample_idx == self.cfg.n_val_images:
                break
