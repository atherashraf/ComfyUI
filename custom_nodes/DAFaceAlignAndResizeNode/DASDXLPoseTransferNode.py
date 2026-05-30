import torch
import comfy.sample
import comfy.utils

from nodes import (
    CLIPTextEncode,
    ControlNetApplyAdvanced,
    VAEEncode,
    VAEDecode,
    KSampler,
)


class DASDXLPoseTransferNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "control_net": ("CONTROL_NET",),

                "source_image": ("IMAGE",),
                "pose_image": ("IMAGE",),

                "positive_prompt": ("STRING", {
                    "default": "a high quality photo of the same person, same clothing, same identity, matching the pose reference",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "low quality, blurry, distorted, bad anatomy, extra limbs, missing limbs, bad hands, bad face, duplicate body",
                    "multiline": True,
                }),

                "seed": ("INT", {
                    "default": 12345,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 100,
                }),
                "cfg": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.1,
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal",
                }),
                "denoise": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "control_strength": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "control_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "control_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("final_image", "pose_image")
    FUNCTION = "process"
    CATEGORY = "DA Nodes/Pose"

    def process(
        self,
        model,
        clip,
        vae,
        control_net,
        source_image,
        pose_image,
        positive_prompt,
        negative_prompt,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        control_strength,
        control_start,
        control_end,
    ):
        # Encode prompts
        positive = CLIPTextEncode().encode(
            clip,
            positive_prompt
        )[0]

        negative = CLIPTextEncode().encode(
            clip,
            negative_prompt
        )[0]

        # Apply ControlNet using OpenPose/DWPose image
        positive, negative = ControlNetApplyAdvanced().apply_controlnet(
            positive,
            negative,
            control_net,
            pose_image,
            control_strength,
            control_start,
            control_end
        )

        # Use source image as img2img latent
        latent = VAEEncode().encode(
            vae,
            source_image
        )[0]

        # Sample
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

        # Decode
        final_image = VAEDecode().decode(
            vae,
            sampled_latent
        )[0]

        return final_image, pose_image