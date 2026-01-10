# web_ui_extension.py
import os

import folder_paths
import server
from aiohttp import web
import json

from da_scripts.comfy_lora.comfy_lora_trainer import LoRATrainer, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


@server.PromptServer.instance.routes.post("/train_lora")
async def train_lora(request):
    """API endpoint to train LoRA"""
    data = await request.json()

    # Extract parameters
    images = data.get("images", [])
    instance_token = data.get("instance_token", "person")
    training_params = data.get("params", {})

    # Start training
    trainer = LoRATrainer()
    result = await trainer.train_async(images, instance_token, **training_params)

    return web.json_response({"status": "success", "lora_path": result})


@server.PromptServer.instance.routes.get("/list_loras")
async def list_loras(request):
    """List all available LoRAs"""
    lora_dir = os.path.join(folder_paths.models_dir, "loras")
    loras = []

    for file in os.listdir(lora_dir):
        if file.endswith('.safetensors'):
            loras.append({
                "name": file,
                "path": os.path.join(lora_dir, file),
                "size": os.path.getsize(os.path.join(lora_dir, file))
            })

    return web.json_response({"loras": loras})


# Web UI extension
WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']