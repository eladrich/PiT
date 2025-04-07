import math
from typing import List, Optional, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils import (
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from dataclasses import dataclass
from model.dit import DiT_Llama

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class PiTPipelineOutput(BaseOutput):
    image_embeds: torch.Tensor


class PiTPipeline(DiffusionPipeline):

    def __init__(self, prior: DiT_Llama):
        super().__init__()

        self.register_modules(
            prior=prior,
        )

    def prepare_latents(self, shape, dtype, device, generator, latents):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        cond_sequence: torch.FloatTensor,
        negative_cond_sequence: torch.FloatTensor,
        num_images_per_prompt: int = 1,
        num_inference_steps: int = 25,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        init_latents: Optional[torch.FloatTensor] = None,
        strength: Optional[float] = None,
        guidance_scale: float = 1.0,
        output_type: Optional[str] = "pt",  # pt only
        return_dict: bool = True,
    ):

        do_classifier_free_guidance = guidance_scale > 1.0

        device = self._execution_device

        batch_size = cond_sequence.shape[0]
        batch_size = batch_size * num_images_per_prompt

        embedding_dim = self.prior.config.embedding_dim

        latents = self.prepare_latents(
            (batch_size, 16, embedding_dim),
            self.prior.dtype,
            device,
            generator,
            latents,
        )

        if init_latents is not None:
            init_latents = init_latents.to(latents.device)
            latents = (strength) * latents + (1 - strength) * init_latents

        # Rectified Flow
        dt = 1.0 / num_inference_steps
        dt = torch.tensor([dt] * batch_size).to(latents.device).view([batch_size, *([1] * len(latents.shape[1:]))])
        start_inference_step = (
            math.ceil(num_inference_steps * (strength)) if strength is not None else num_inference_steps
        )
        for i in range(start_inference_step, 0, -1):
            t = i / num_inference_steps
            t = torch.tensor([t] * batch_size).to(latents.device)

            vc = self.prior(latents, t, cond_sequence)
            if do_classifier_free_guidance:
                vu = self.prior(latents, t, negative_cond_sequence)
                vc = vu + guidance_scale * (vc - vu)

            latents = latents - dt * vc

        image_embeddings = latents

        if output_type not in ["pt", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            image_embeddings = image_embeddings.cpu().numpy()

        if not return_dict:
            return image_embeddings

        return PiTPipelineOutput(image_embeds=image_embeddings)
