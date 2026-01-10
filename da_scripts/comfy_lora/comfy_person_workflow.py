# comfy_person_workflow.py
import nodes
import json
from comfy_lora_trainer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


class PersonGenerationWorkflow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_lora": ("LORA",),
                "base_model": ("MODEL",),
                "positive_prompt": ("STRING",
                                    {"multiline": True, "default": "shs_person, professional photo, sharp focus"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "deformed, blurry, bad anatomy"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048}),
            },
            "optional": {
                "style_lora": ("LORA",),
                "controlnet": ("CONTROL_NET",),
                "pose_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
    RETURN_NAMES = ("image", "latent", "metadata")
    FUNCTION = "generate"
    CATEGORY = "generation/person"

    def generate(self, person_lora, base_model, positive_prompt, negative_prompt,
                 seed, steps, cfg, width, height, style_lora=None, controlnet=None,
                 pose_image=None, reference_image=None):

        # Apply LoRAs to model
        model_with_lora = self.apply_loras(base_model, person_lora, style_lora)

        # Encode prompt
        clip = nodes.CLIPTextEncode()
        positive_encoded = clip.encode(positive_prompt, base_model.clip)[0]
        negative_encoded = clip.encode(negative_prompt, base_model.clip)[0]

        # Generate latent
        if controlnet and pose_image is not None:
            # Use ControlNet for pose control
            latent = self.generate_with_controlnet(
                model_with_lora, positive_encoded, negative_encoded,
                seed, steps, cfg, width, height, controlnet, pose_image
            )
        else:
            # Standard generation
            empty_latent = nodes.EmptyLatentImage()
            latent = empty_latent.generate(width, height, 1, seed)[0]

            ksampler = nodes.KSampler()
            latent = ksampler.sample(
                model_with_lora, seed, steps, cfg,
                "dpmpp_2m", "normal",
                positive_encoded, negative_encoded, latent
            )[0]

        # Decode image
        vae_decode = nodes.VAEDecode()
        image = vae_decode.decode(latent, base_model.vae)[0]

        # Prepare metadata
        metadata = self.create_metadata(
            positive_prompt, negative_prompt, seed, steps, cfg,
            person_lora, style_lora
        )

        return (image, latent, metadata)

    def apply_loras(self, model, person_lora, style_lora=None):
        """Apply multiple LoRAs to model"""
        # Load person LoRA
        lora_loader = nodes.LoraLoader()
        model, _ = lora_loader.load_lora(model, None, person_lora, 1.0, 1.0)

        # Apply style LoRA if provided
        if style_lora:
            model, _ = lora_loader.load_lora(model, None, style_lora, 0.7, 0.7)

        return model

    def generate_with_controlnet(self, model, positive, negative, seed, steps, cfg,
                                 width, height, controlnet, pose_image):
        """Generate with ControlNet pose control"""
        # Apply ControlNet
        controlnet_apply = nodes.ControlNetApplyAdvanced()
        positive_with_cn = controlnet_apply.apply_controlnet(
            positive, controlnet, pose_image, 1.0, 0, 1
        )[0]

        # Generate latent
        empty_latent = nodes.EmptyLatentImage()
        latent = empty_latent.generate(width, height, 1, seed)[0]

        # Sample with ControlNet
        ksampler = nodes.KSampler()
        latent = ksampler.sample(
            model, seed, steps, cfg,
            "dpmpp_2m", "normal",
            positive_with_cn, negative, latent
        )[0]

        return latent

    def create_metadata(self, positive, negative, seed, steps, cfg, person_lora, style_lora):
        """Create generation metadata"""
        metadata = {
            "prompt": positive,
            "negative_prompt": negative,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "person_lora": str(person_lora),
            "style_lora": str(style_lora) if style_lora else None,
            "workflow": "Person Generation with Pose Control"
        }
        return json.dumps(metadata)


class BatchStyleGenerator:
    """Generate person in multiple styles"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_lora": ("LORA",),
                "base_model": ("MODEL",),
                "styles": ("STRING",
                           {"multiline": True, "default": "anime style\ncyberpunk\nwatercolor painting\npixel art"}),
                "num_variations": ("INT", {"default": 3, "min": 1, "max": 10}),
                "seed_start": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "generate_batch"
    CATEGORY = "generation/batch"

    def generate_batch(self, person_lora, base_model, styles, num_variations, seed_start):
        styles_list = [s.strip() for s in styles.split('\n') if s.strip()]

        all_images = []
        seed = seed_start

        for style in styles_list:
            for i in range(num_variations):
                # Create prompt with style
                prompt = f"shs_person, {style}, masterpiece, high quality"

                # Generate single image
                workflow = PersonGenerationWorkflow()
                image, _, _ = workflow.generate(
                    person_lora=person_lora,
                    base_model=base_model,
                    positive_prompt=prompt,
                    negative_prompt="deformed, blurry, bad anatomy",
                    seed=seed,
                    steps=30,
                    cfg=7.5,
                    width=512,
                    height=512
                )

                all_images.append(image)
                seed += 1

        return (all_images,)


# Register workflow nodes
NODE_CLASS_MAPPINGS.update({
    "PersonGenerationWorkflow": PersonGenerationWorkflow,
    "BatchStyleGenerator": BatchStyleGenerator,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "PersonGenerationWorkflow": "Person Generation with LoRA",
    "BatchStyleGenerator": "Batch Style Generator",
})