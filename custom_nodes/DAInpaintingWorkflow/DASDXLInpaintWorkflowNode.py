import torch

from nodes import (
    CLIPTextEncode,
    VAEEncodeForInpaint,
    VAEDecode,
    KSampler,
)

import comfy.samplers


class DASDXLInpaintWorkflowNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),

                "image": ("IMAGE",),
                "mask": ("MASK",),

                "positive_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),

                "grow_mask_by": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 256
                }),

                "seed": ("INT", {
                    "default": 12345,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler"
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "simple"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("image", "latent", "positive", "negative")
    FUNCTION = "process"
    CATEGORY = "DA Nodes/Inpaint"

    def process(
        self,
        model,
        clip,
        vae,
        image,
        mask,
        positive_prompt,
        negative_prompt,
        grow_mask_by,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise
    ):
        positive = CLIPTextEncode().encode(
            clip,
            positive_prompt
        )[0]

        negative = CLIPTextEncode().encode(
            clip,
            negative_prompt
        )[0]

        latent = VAEEncodeForInpaint().encode(
            vae,
            image,
            mask,
            grow_mask_by
        )[0]

        sampled_latent = KSampler().sample(
            model=model,
            seed=seed,
            steps=steps,
            cfg=cfg,
                sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=denoise
        )[0]

        final_image = VAEDecode().decode(
            vae,
            sampled_latent
        )[0]

        return final_image, sampled_latent, positive, negative