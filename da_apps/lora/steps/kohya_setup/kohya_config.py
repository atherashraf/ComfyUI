from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml


@dataclass
class KohyaTrainConfig:
    resolution: str = "1024,1024"
    train_batch_size: int = 1
    max_train_epochs: int = 30
    max_token_length: int = 225
    mixed_precision: str = "fp16"
    save_every_n_epochs: int = 5

    # LoRA
    unet_lr: float = 4e-4
    text_encoder_lr: float = 4e-4
    network_dim: int = 64
    network_alpha: int = 32
    train_unet_only: bool = True

    # optimizer
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine_with_restarts"
    max_grad_norm: float = 1.0

    # resume
    save_state: bool = True
    save_last_n_epochs_state: int = 1

    # perf
    xformers: bool = True
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = True
    cache_text_encoder_outputs_to_disk: bool = True
    noise_offset: float = 0.1
    no_half_vae: bool = True


def generate_kohya_config(
    output_file: str,
    train_data_dir: Path,
    output_dir: Path,
    logging_dir: Path,
    model_path: Path,
    concept_name: str,
    cfg: KohyaTrainConfig = KohyaTrainConfig(),
) -> Path:
    model_path_str = str(model_path).replace("\\", "/")
    out_path = Path(output_file)
    if out_path.suffix.lower() != ".toml":
        out_path = out_path.with_suffix(".toml")

    config = {
        "model_arguments": {"pretrained_model_name_or_path": model_path_str},
        "additional_network_arguments": {
            "unet_lr": cfg.unet_lr,
            "text_encoder_lr": cfg.text_encoder_lr,
            "network_module": "networks.lora",
            "network_dim": cfg.network_dim,
            "network_alpha": cfg.network_alpha,
            "network_train_unet_only": cfg.train_unet_only,
            "network_train_text_encoder_only": False,
        },
        "optimizer_arguments": {
            "optimizer_type": cfg.optimizer_type,
            "learning_rate": cfg.unet_lr,
            "max_grad_norm": cfg.max_grad_norm,
            "lr_scheduler": cfg.lr_scheduler,
            "lr_warmup_steps": 0,
        },
        "dataset_arguments": {
            "cache_latents": cfg.cache_latents,
            "debug_dataset": False,
            "resolution": cfg.resolution,
            "train_data_dir": str(train_data_dir),
            "caption_extension": ".txt",
        },
        "training_arguments": {
            "output_dir": str(output_dir),
            "output_name": f"{concept_name}_sdxl_lora",
            "save_precision": cfg.mixed_precision,
            "save_every_n_epochs": cfg.save_every_n_epochs,
            "save_state": cfg.save_state,
            "save_last_n_epochs_state": cfg.save_last_n_epochs_state,
            "train_batch_size": cfg.train_batch_size,
            "max_token_length": cfg.max_token_length,
            "mem_eff_attn": False,
            "xformers": cfg.xformers,
            "max_train_epochs": cfg.max_train_epochs,
            "max_data_loader_n_workers": 0,
            "persistent_data_loader_workers": False,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "mixed_precision": cfg.mixed_precision,
            "logging_dir": str(logging_dir),
            "noise_offset": cfg.noise_offset,
        },
        "sdxl_arguments": {
            "cache_text_encoder_outputs": cfg.cache_text_encoder_outputs,
            "cache_text_encoder_outputs_to_disk": cfg.cache_text_encoder_outputs_to_disk,
            "no_half_vae": cfg.no_half_vae,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(toml.dumps(config), encoding="utf-8")
    print(f"✓ Training config saved to: {out_path}")
    return out_path
