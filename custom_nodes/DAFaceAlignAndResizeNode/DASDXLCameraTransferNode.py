from nodes import CLIPTextEncode, VAEEncode, VAEDecode, KSampler
import comfy.samplers


class DASDXLCameraTransferNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "source_image": ("IMAGE",),

                "camera_view": ([
                    "front view",
                    "left side view",
                    "right side view",
                    "three-quarter view",
                    "back view",
                    "top view",
                    "high angle",
                    "low angle",
                    "close-up",
                    "wide shot",
                    "full body shot"
                ], {"default": "three-quarter view"}),

                "positive_prompt": ("STRING", {
                    "default": "high quality realistic photo of the same subject, same identity, same clothing, same style",
                    "multiline": True
                }),
                "negative_prompt": ("STRING", {
                    "default": "low quality, blurry, distorted, bad anatomy, bad face, changed identity, changed clothes, duplicate subject, wrong perspective",
                    "multiline": True
                }),

                "seed": ("INT", {"default": 12345, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_image",)
    FUNCTION = "process"
    CATEGORY = "DA Nodes/Camera"

    def process(
        self,
        model,
        clip,
        vae,
        source_image,
        camera_view,
        positive_prompt,
        negative_prompt,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise
    ):
        final_positive = (
            f"{positive_prompt}, {camera_view}, changed camera angle, realistic perspective, "
            f"same subject, same outfit, consistent character details"
        )

        final_negative = (
            f"{negative_prompt}, incorrect camera angle, inconsistent face, inconsistent body, "
            f"warped perspective, deformed geometry"
        )

        positive = CLIPTextEncode().encode(clip, final_positive)[0]
        negative = CLIPTextEncode().encode(clip, final_negative)[0]

        latent = VAEEncode().encode(vae, source_image)[0]

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

        final_image = VAEDecode().decode(vae, sampled_latent)[0]

        return (final_image,)